<#
.SYNOPSIS
  Host-side autonomy supervisor for the dense_cnn Model 1 (target_96x6) run.
  (Copy of supervise_scratch64.ps1 retargeted to the 96x6+P7+512-sim fresh run;
  identical guardrails: crash-artifact freeze, resume-from-latest, circuit
  breaker, no-progress guard. Only the default config + process-match differ.)

  Keeps the training run advancing overnight WITHOUT relying on any chat session:
    - Adopts an already-running trainer (never double-launches / never kills it).
    - On each exit: FIRST freezes crash artifacts, THEN bumps resume_from to the
      newest checkpoint (so relaunches ADVANCE, not reset to epoch 9), THEN
      relaunches via the existing launcher with fault-handler env vars.
    - Circuit breaker stops a crash-loop from burning the whole night.
    - Stops cleanly (not as a crash) once the configured epoch count is reached.

  Signals (so monitors / a chat session can observe without being depended on):
    diagnostics\supervisor.log               lifecycle log (ADOPT/LAUNCH/EXIT/RELAUNCH/HALT/COMPLETED)
    diagnostics\supervisor.pid               current child trainer PID
    diagnostics\supervisor.self.pid          this supervisor's PID (single-instance guard)
    diagnostics\supervisor_halted.flag       written on circuit-breaker halt (reason inside)
    diagnostics\supervisor_completed.flag    written on clean completion of all epochs
    diagnostics\crashlog.md                  one signature block appended per exit
    diagnostics\crash_artifacts\<ts>\        frozen logs + dumps per exit
#>
param(
    [string]$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$ConfigPath = "configs\dense_cnn_model1_target_96x6.toml",
    [string]$SealBotPath = "E:\SealBot",
    [string]$PythonExe = "C:\Python314\python.exe",
    [int]$FastCrashSeconds = 180,            # exit sooner than this == "fast crash"
    [int]$MaxConsecutiveFastCrashes = 3,     # 3 fast crashes in a row -> halt
    [int]$MaxCrashesPerHour = 6,             # >6 exits within 60 min -> halt
    [int]$MaxNoProgressRelaunches = 5,       # N relaunches with no new epoch checkpoint -> halt (slow-loop guard).
                                             # Bumped 3->5 for games_per_epoch=512: this guard counts RELAUNCHES (crashes)
                                             # without a new epoch checkpoint, NOT wall-clock — a healthy long epoch is one
                                             # launch->exit-with-progress and resets the counter, so it cannot false-trip on
                                             # epoch length (the supervisor proc.WaitForExit has no timeout). The bump gives a
                                             # crash mid-(now-2x-longer)-epoch a couple more resume attempts before halting,
                                             # since each epoch is more valuable. (~512-game epoch est. ~25-35 min; see
                                             # analysis/throughput_understanding.md. The backstop-watcher "no new shard in N min"
                                             # human heuristic should be relaxed to > one epoch's self-play time.)
    [switch]$ValidateOnly                    # parse + probe state, then exit (no adopt/launch)
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path -LiteralPath $RepositoryRoot).Path
$config = if ([System.IO.Path]::IsPathRooted($ConfigPath)) {
    (Resolve-Path -LiteralPath $ConfigPath).Path
} else {
    (Resolve-Path -LiteralPath (Join-Path $repo $ConfigPath)).Path
}
$launcher = Join-Path $repo "scripts\start_model1_training.ps1"

# ---- config readers (section-aware, no external deps) -----------------------
function Get-ConfigString {
    param([string]$Section, [string]$Key, [string]$Default = "")
    $cur = ""
    foreach ($line in Get-Content -LiteralPath $config) {
        $t = $line.Trim()
        if ($t -match '^\[(.+)\]$') { $cur = $Matches[1]; continue }
        if ($cur -eq $Section -and $t -match ('^' + [regex]::Escape($Key) + '\s*=\s*"?''?([^"''#]+)')) {
            return $Matches[1].Trim()
        }
    }
    return $Default
}
$runName = Get-ConfigString -Section "run" -Key "name" -Default "dense_cnn_model1_target_96x6"
$loopEpochs = [int](Get-ConfigString -Section "loop" -Key "epochs" -Default "30")
$pointerRel = Get-ConfigString -Section "selfplay" -Key "checkpoint_pointer" -Default ""

$runDir   = Join-Path $repo ("runs\" + $runName)
$ckptDir  = Join-Path $runDir "checkpoints"
$diag     = Join-Path $runDir "diagnostics"
New-Item -ItemType Directory -Force -Path $diag | Out-Null
$crashDumps    = Join-Path $diag "crashdumps"
$crashArtRoot  = Join-Path $diag "crash_artifacts"
$supLog        = Join-Path $diag "supervisor.log"
$pidFile       = Join-Path $diag "supervisor.pid"
$selfPidFile   = Join-Path $diag "supervisor.self.pid"
$haltFlag      = Join-Path $diag "supervisor_halted.flag"
$doneFlag      = Join-Path $diag "supervisor_completed.flag"
$crashLog      = Join-Path $diag "crashlog.md"
$eventsFile    = Join-Path $diag "events.jsonl"
$watchdogFile  = Join-Path $diag "resource_watchdog.jsonl"

if ($pointerRel) {
    $pointerFile = if ([System.IO.Path]::IsPathRooted($pointerRel)) { $pointerRel } else { Join-Path (Split-Path $config) $pointerRel }
} else {
    $pointerFile = Join-Path $repo ("data\checkpoints\" + $runName + "_latest.txt")
}

# ---- helpers ----------------------------------------------------------------
function Now-Stamp { (Get-Date).ToString("yyyy-MM-dd HH:mm:ss") }
function Write-SupLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Now-Stamp), $Message
    Add-Content -LiteralPath $supLog -Value $line
    Write-Host $line
}

function Write-Utf8NoBom {
    # PowerShell 5.1 `Set-Content -Encoding UTF8` writes a BOM, which breaks
    # tomllib (TOMLDecodeError at line 1) and [int] casts of pidfiles. Write
    # UTF-8 WITHOUT a BOM for any file Python or numeric parsing consumes.
    param([string]$Path, $Lines)
    $enc = New-Object System.Text.UTF8Encoding($false)
    if ($Lines -is [string]) { [System.IO.File]::WriteAllText($Path, $Lines, $enc) }
    else { [System.IO.File]::WriteAllLines($Path, [string[]]$Lines, $enc) }
}

function Get-Trainer {
    Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match "python" -and
        $_.CommandLine -match "hexo_train\.cli\.train_model" -and
        $_.CommandLine -match "target_96x6"
    } | Select-Object -First 1
}

function Get-RunningSupervisorPid {
    # Pidfile-based single-instance lock. Only a real prior supervisor writes its
    # PID to $selfPidFile, so this never trips on a wrapper shell whose command
    # line merely mentions this script's path. Validates the PID is still a live
    # powershell running this script (guards against a stale pidfile / PID reuse).
    if (-not (Test-Path $selfPidFile)) { return $null }
    $old = (Get-Content -LiteralPath $selfPidFile -Raw).Trim()
    if (-not $old -or [int]$old -eq $PID) { return $null }
    $p = Get-CimInstance Win32_Process -Filter ("ProcessId={0}" -f $old) -ErrorAction SilentlyContinue
    if ($p -and $p.Name -match "powershell" -and $p.CommandLine -match "supervise_scratch64\.ps1") {
        return [int]$old
    }
    return $null
}

function Get-NewestErrLog {
    $f = Get-ChildItem (Join-Path $diag "trainer.*.err.log") -ErrorAction SilentlyContinue |
         Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($f) { return $f.FullName } else { return $null }
}
function Get-NewestOutLog {
    $f = Get-ChildItem (Join-Path $diag "trainer.*.out.log") -ErrorAction SilentlyContinue |
         Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($f) { return $f.FullName } else { return $null }
}

function Get-LatestCheckpoint {
    # Prefer the run's auto-updated pointer; fall back to highest epoch on disk.
    if (Test-Path $pointerFile) {
        $p = (Get-Content -LiteralPath $pointerFile -Raw).Trim()
        if ($p -and (Test-Path $p)) { return (Resolve-Path -LiteralPath $p).Path }
    }
    $c = Get-ChildItem (Join-Path $ckptDir "epoch_*.pt") -ErrorAction SilentlyContinue |
         Sort-Object { [int]([regex]::Match($_.Name, 'epoch_(\d+)').Groups[1].Value) } -Descending |
         Select-Object -First 1
    if ($c) { return $c.FullName } else { return $null }
}
function Get-EpochOfCheckpoint {
    param([string]$Path)
    if (-not $Path) { return -1 }
    $m = [regex]::Match([System.IO.Path]::GetFileName($Path), 'epoch_(\d+)')
    if ($m.Success) { return [int]$m.Groups[1].Value } else { return -1 }
}

function Set-ResumeFrom {
    # Idempotently set [checkpoint] resume_from to $CheckpointPath (forward-slash,
    # double-quoted -> valid TOML on Windows). Removes any prior resume_from.
    param([string]$CheckpointPath)
    $fwd = $CheckpointPath -replace '\\', '/'
    $lines = Get-Content -LiteralPath $config
    $out = New-Object System.Collections.Generic.List[string]
    $inCk = $false; $inserted = $false
    foreach ($line in $lines) {
        $t = $line.Trim()
        if ($t -match '^\[(.+)\]$') {
            $inCk = ($Matches[1] -eq 'checkpoint')
            $out.Add($line)
            if ($inCk) { $out.Add('resume_from = "' + $fwd + '"'); $inserted = $true }
            continue
        }
        if ($inCk -and $t -match '^resume_from\s*=') { continue }   # drop old line
        $out.Add($line)
    }
    if (-not $inserted) {
        $out.Add('[checkpoint]')
        $out.Add('resume_from = "' + $fwd + '"')
    }
    Write-Utf8NoBom $config $out
}

$script:ArchivedDumps = @{}
function Save-CrashArtifacts {
    param([string]$ErrLog, [string]$OutLog, [string]$Reason, $ExitCode, [int]$UptimeSec)
    $ts = (Get-Date).ToString("yyyyMMdd_HHmmss")
    $dir = Join-Path $crashArtRoot $ts
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    if ($ErrLog -and (Test-Path $ErrLog)) { Copy-Item $ErrLog $dir -ErrorAction SilentlyContinue }
    if ($OutLog -and (Test-Path $OutLog)) { Copy-Item $OutLog $dir -ErrorAction SilentlyContinue }
    if (Test-Path $eventsFile)   { Get-Content $eventsFile   -Tail 80 | Set-Content (Join-Path $dir "events.tail.jsonl")   -Encoding UTF8 }
    if (Test-Path $watchdogFile) { Get-Content $watchdogFile -Tail 30 | Set-Content (Join-Path $dir "watchdog.tail.jsonl") -Encoding UTF8 }

    $copiedDump = $null
    if (Test-Path $crashDumps) {
        foreach ($d in (Get-ChildItem (Join-Path $crashDumps "*.dmp") -ErrorAction SilentlyContinue)) {
            if (-not $script:ArchivedDumps.ContainsKey($d.Name)) {
                Copy-Item $d.FullName $dir -ErrorAction SilentlyContinue
                $script:ArchivedDumps[$d.Name] = $true
                $copiedDump = $d.Name
            }
        }
    }

    $sigPat = 'Fatal Python error|Current thread|panicked|stack backtrace|SIGSEGV|SIGABRT|access violation|0xc0000005|fatal exception|Traceback|STATUS_'
    $sig = $null
    if ($ErrLog -and (Test-Path $ErrLog)) {
        $sig = (Select-String -LiteralPath $ErrLog -Pattern $sigPat -ErrorAction SilentlyContinue |
                Select-Object -Last 1).Line
    }
    $sigText = if ($sig) { $sig.Trim() } elseif ($copiedDump) { "native crash, no stderr text (see dump $copiedDump)" } else { "no fault text (clean or external stop)" }
    $block = @(
        ("## {0} -- {1} (exit={2}, uptime={3}s)" -f $ts, $Reason, $ExitCode, $UptimeSec),
        ("- artifacts: {0}" -f $dir),
        ("- dump: {0}" -f $(if ($copiedDump) { $copiedDump } else { "none" })),
        ("- signature: {0}" -f $sigText),
        ""
    )
    Add-Content -LiteralPath $crashLog -Value $block
    Write-SupLog ("CAPTURE -> {0} | sig: {1}" -f $dir, $sigText)
    return $dir
}

function Launch-Trainer {
    $env:PYTHONFAULTHANDLER = "1"
    $env:PYTHONUNBUFFERED = "1"
    $env:RUST_BACKTRACE = "full"
    $json = & $launcher -RepositoryRoot $repo -ConfigPath $config -SealBotPath $SealBotPath -PythonExe $PythonExe |
            Out-String | ConvertFrom-Json
    return $json
}

# ---- validate-only: prove parsing/adoption logic without side effects -------
if ($ValidateOnly) {
    Write-Host "=== ValidateOnly ==="
    Write-Host ("runName       = {0}" -f $runName)
    Write-Host ("loopEpochs    = {0}" -f $loopEpochs)
    Write-Host ("config        = {0}" -f $config)
    Write-Host ("pointerFile   = {0} (exists={1})" -f $pointerFile, (Test-Path $pointerFile))
    $lc = Get-LatestCheckpoint
    Write-Host ("latest ckpt   = {0} (epoch {1} -> resume starts epoch {2})" -f $lc, (Get-EpochOfCheckpoint $lc), ((Get-EpochOfCheckpoint $lc) + 1))
    $t = Get-Trainer
    if ($t) { Write-Host ("live trainer  = PID {0} (would ADOPT)" -f $t.ProcessId) } else { Write-Host "live trainer  = none (would LAUNCH)" }
    Write-Host ("newest errlog = {0}" -f (Get-NewestErrLog))
    $other = Get-RunningSupervisorPid
    if ($other) { Write-Host ("other supervisor (pidfile lock) = PID {0}" -f $other) } else { Write-Host "other supervisor (pidfile lock) = none" }
    # Dry Set-ResumeFrom on a throwaway copy; verify it parses as TOML.
    if ($lc) {
        $tmp = Join-Path $env:TEMP ("scratch64_resume_test_" + $PID + ".toml")
        Copy-Item $config $tmp -Force
        $savedConfig = $config; Set-Variable -Name config -Value $tmp
        Set-ResumeFrom $lc
        Set-Variable -Name config -Value $savedConfig
        Write-Host "--- injected [checkpoint] block in test copy ---"
        Get-Content $tmp | Select-String -Pattern '^\[checkpoint\]|resume_from|initialize_from|save_name' | ForEach-Object { Write-Host ("    " + $_.Line.Trim()) }
        Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    }
    return
}

# ---- single-instance + halt guards -----------------------------------------
$other = Get-RunningSupervisorPid
if ($other) {
    Write-SupLog ("ABORT: another supervisor already running (PID {0}). Exiting." -f $other)
    return
}
Write-Utf8NoBom $selfPidFile ([string]$PID)
if (Test-Path $haltFlag) {
    Write-SupLog ("ABORT: halt flag present ({0}). Clear it to resume. Exiting." -f $haltFlag)
    return
}
if (Test-Path $doneFlag) { Remove-Item $doneFlag -Force -ErrorAction SilentlyContinue }

Write-SupLog ("SUPERVISOR start (pid={0}) run={1} config={2} epochs={3}" -f $PID, $runName, $config, $loopEpochs)
Write-SupLog ("breaker: fast<{0}s, {1} consecutive OR >{2}/hour -> halt" -f $FastCrashSeconds, $MaxConsecutiveFastCrashes, $MaxCrashesPerHour)

# ---- establish the initial child (adopt or launch) --------------------------
$childPid = $null; $errLog = $null; $outLog = $null; $launchTime = $null
$existing = Get-Trainer
if ($existing) {
    $childPid = [int]$existing.ProcessId
    $errLog = Get-NewestErrLog
    $outLog = Get-NewestOutLog
    try { $launchTime = (Get-Process -Id $childPid -ErrorAction Stop).StartTime } catch { $launchTime = $existing.CreationDate }
    $lc = Get-LatestCheckpoint
    Write-SupLog ("ADOPT existing trainer pid={0} (latest checkpoint epoch {1}); will manage relaunches from its next exit. NOT modifying its config/process." -f $childPid, (Get-EpochOfCheckpoint $lc))
} else {
    $lc = Get-LatestCheckpoint
    if ($lc) { Set-ResumeFrom $lc; Write-SupLog ("resume_from -> {0} (will start epoch {1})" -f $lc, ((Get-EpochOfCheckpoint $lc) + 1)) }
    else { Write-SupLog "no checkpoint found; first launch will use initialize_from" }
    $res = Launch-Trainer
    $childPid = [int]$res.trainer_pid; $errLog = $res.trainer_err_log; $outLog = $res.trainer_out_log
    $launchTime = Get-Date
    Write-SupLog ("LAUNCH pid={0} err={1}" -f $childPid, $errLog)
}
Set-Content -LiteralPath $pidFile -Value $childPid -Encoding UTF8

# ---- supervise loop ---------------------------------------------------------
$consecFast = 0
$crashTimes = New-Object System.Collections.Generic.List[datetime]
# No-progress guard: track the highest epoch checkpoint seen. If relaunches stop
# advancing it, we are in a slow no-progress loop the fast-crash rule cannot catch
# (e.g. a watchdog kill every ~20 min that never saves a new epoch).
$lastProgressEpoch = Get-EpochOfCheckpoint (Get-LatestCheckpoint)
$noProgress = 0
while ($true) {
    $proc = Get-Process -Id $childPid -ErrorAction SilentlyContinue
    if ($proc) {
        try { $null = $proc.Handle } catch {}     # cache handle so ExitCode is readable
        $proc.WaitForExit()
    }
    $exitTime = Get-Date
    $code = $null
    if ($proc) { try { $code = $proc.ExitCode } catch { $code = $null } }
    $uptime = [int](($exitTime - $launchTime).TotalSeconds)
    Write-SupLog ("EXIT pid={0} code={1} uptime={2}s" -f $childPid, $code, $uptime)
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue

    # 1) FREEZE ARTIFACTS FIRST
    Save-CrashArtifacts -ErrLog $errLog -OutLog $outLog -Reason "trainer exit" -ExitCode $code -UptimeSec $uptime | Out-Null

    # 2) COMPLETION GUARD (clean finish of all epochs -> stop, not a crash)
    $lc = Get-LatestCheckpoint
    $latestEpoch = Get-EpochOfCheckpoint $lc
    if (($latestEpoch + 1) -gt $loopEpochs) {
        $msg = "Completed through epoch $latestEpoch (config loop.epochs=$loopEpochs). Nothing further to run."
        Write-Utf8NoBom $doneFlag ("[{0}] {1}" -f (Now-Stamp), $msg)
        Write-SupLog ("COMPLETED {0}" -f $msg)
        return
    }

    # 2b) NO-PROGRESS GUARD: count consecutive relaunches that don't advance the epoch
    if ($latestEpoch -gt $lastProgressEpoch) { $lastProgressEpoch = $latestEpoch; $noProgress = 0 }
    else { $noProgress++ }

    # 3) CIRCUIT BREAKER
    $crashTimes.Add($exitTime)
    $cutoff = $exitTime.AddMinutes(-60)
    $recent = @($crashTimes | Where-Object { $_ -ge $cutoff })
    $crashTimes = New-Object System.Collections.Generic.List[datetime]
    $recent | ForEach-Object { $crashTimes.Add($_) }
    if ($uptime -lt $FastCrashSeconds) { $consecFast++ } else { $consecFast = 0 }
    Write-SupLog ("breaker state: consecutiveFast={0} crashesLastHour={1} noProgress={2}/{3} (latest epoch {4})" -f $consecFast, $crashTimes.Count, $noProgress, $MaxNoProgressRelaunches, $latestEpoch)

    if ($noProgress -ge $MaxNoProgressRelaunches -or $consecFast -ge $MaxConsecutiveFastCrashes -or $crashTimes.Count -gt $MaxCrashesPerHour) {
        $reason = if ($noProgress -ge $MaxNoProgressRelaunches) {
            "no checkpoint progress across $noProgress relaunches (latest still epoch $latestEpoch) -- likely a stuck phase (e.g. shuffle), not advancing"
        } elseif ($consecFast -ge $MaxConsecutiveFastCrashes) {
            "$consecFast consecutive fast crashes (<$FastCrashSeconds s each)"
        } else {
            "$($crashTimes.Count) crashes within 60 min (limit $MaxCrashesPerHour)"
        }
        $lastSig = (Select-String -LiteralPath $errLog -Pattern 'Fatal Python error|panicked|access violation|0xc0000005|Traceback|STATUS_' -ErrorAction SilentlyContinue | Select-Object -Last 1).Line
        $flagBody = @(
            ("[{0}] SUPERVISOR HALTED" -f (Now-Stamp)),
            ("reason: {0}" -f $reason),
            ("last child pid: {0} (exit code {1}, uptime {2}s)" -f $childPid, $code, $uptime),
            ("last err.log: {0}" -f $errLog),
            ("last signature: {0}" -f $(if ($lastSig) { $lastSig.Trim() } else { "<none in stderr>" })),
            "Crash artifacts + crashlog.md hold the evidence. Root-cause, fix, then DELETE this flag to allow the supervisor to be restarted."
        )
        Write-Utf8NoBom $haltFlag $flagBody
        Write-SupLog ("HALT: {0}. Wrote {1}. Not relaunching." -f $reason, $haltFlag)
        return
    }

    # 4) BUMP RESUME -> newest checkpoint, then RELAUNCH
    if ($lc) {
        Set-ResumeFrom $lc
        Write-SupLog ("resume_from -> {0} (start epoch {1})" -f $lc, ($latestEpoch + 1))
    } else {
        Write-SupLog "WARN: no checkpoint to resume from; relaunch will use initialize_from"
    }
    $res = Launch-Trainer
    $childPid = [int]$res.trainer_pid; $errLog = $res.trainer_err_log; $outLog = $res.trainer_out_log
    $launchTime = Get-Date
    Write-Utf8NoBom $pidFile ([string]$childPid)
    Write-SupLog ("RELAUNCH pid={0} err={1}" -f $childPid, $errLog)
}
