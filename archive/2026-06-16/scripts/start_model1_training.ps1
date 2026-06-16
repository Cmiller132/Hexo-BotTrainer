param(
    [string]$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$ConfigPath = "configs\dense_cnn_model1.toml",
    [string]$RunName = "",
    [string]$PythonExe = "C:\Python314\python.exe",
    [string]$SealBotPath = "E:\SealBot",
    [double]$MinFreeRamGb = 4.0,
    [double]$MinFreeVirtualGb = 12.0,
    [double]$MinGpuFreeGb = 2.5,
    [double]$MaxTrainerPrivateGb = 18.0,
    [switch]$NoWatchdog,
    [switch]$RestartWatchdog,
    [switch]$StopExistingTrainer,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path -LiteralPath $RepositoryRoot).Path
$config = if ([System.IO.Path]::IsPathRooted($ConfigPath)) {
    (Resolve-Path -LiteralPath $ConfigPath).Path
} else {
    (Resolve-Path -LiteralPath (Join-Path $repo $ConfigPath)).Path
}

function Get-RunNameFromConfig {
    param([string]$Path)

    $section = ""
    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if ($trimmed -match '^\[(.+)\]$') {
            $section = $Matches[1]
            continue
        }
        if ($section -eq "run" -and $trimmed -match '^name\s*=\s*"([^"]+)"') {
            return $Matches[1]
        }
    }
    return "dense_cnn_model1"
}

$resolvedRunName = if ($RunName) { $RunName } else { Get-RunNameFromConfig -Path $config }
$diagnosticsDir = Join-Path $repo ("runs\" + $resolvedRunName + "\diagnostics")
New-Item -ItemType Directory -Force -Path $diagnosticsDir | Out-Null

$packagePaths = @(
    "packages\hexo_models\python",
    "packages\hexo_train\python",
    "packages\hexo_runner\python",
    "packages\hexo_engine\python",
    "packages\hexo_utils\python"
) | ForEach-Object { (Resolve-Path -LiteralPath (Join-Path $repo $_)).Path }

$existingPythonPath = @()
if ($env:PYTHONPATH) {
    $existingPythonPath = $env:PYTHONPATH -split ";" | Where-Object { $_ }
}
$env:PYTHONPATH = (@($packagePaths) + $existingPythonPath | Select-Object -Unique) -join ";"
if ($SealBotPath) {
    $env:SEALBOT_PATH = $SealBotPath
}

function Get-Model1TrainerProcess {
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -match "python" -and
            $_.CommandLine -match "hexo_train\.cli\.train_model" -and
            $_.CommandLine -match "dense_cnn.*\.toml"
        }
}

function Get-Model1WatchdogProcess {
    $currentPid = $PID
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.ProcessId -ne $currentPid -and
            $_.Name -match "powershell" -and
            $_.CommandLine -match "-File .*watch_model1_resources\.ps1"
        }
}

if ($StopExistingTrainer) {
    Get-Model1TrainerProcess | ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

$watchdog = $null
if (-not $NoWatchdog) {
    $watchers = @(Get-Model1WatchdogProcess)
    if ($RestartWatchdog) {
        $watchers | ForEach-Object {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }
        $watchers = @()
    }
    if ($watchers.Count -gt 0) {
        $watchdog = $watchers[0]
    } else {
        $watchArgs = @(
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            (Join-Path $repo "scripts\watch_model1_resources.ps1"),
            "-RepositoryRoot",
            $repo,
            "-RunName",
            $resolvedRunName,
            "-IntervalSeconds",
            "6",
            "-MinFreeRamGb",
            ([string]$MinFreeRamGb),
            "-MinFreeVirtualGb",
            ([string]$MinFreeVirtualGb),
            "-MinGpuFreeGb",
            ([string]$MinGpuFreeGb),
            "-MaxTrainerPrivateGb",
            ([string]$MaxTrainerPrivateGb)
        )
        if (-not $DryRun) {
            $watchdog = Start-Process -FilePath "powershell.exe" -ArgumentList $watchArgs -WorkingDirectory $repo -WindowStyle Hidden -PassThru
        }
    }
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outLog = Join-Path $diagnosticsDir "trainer.$stamp.out.log"
$errLog = Join-Path $diagnosticsDir "trainer.$stamp.err.log"
$trainArgs = @("-m", "hexo_train.cli.train_model", $config)
$trainer = $null
if (-not $DryRun) {
    $trainer = Start-Process -FilePath $PythonExe -ArgumentList $trainArgs -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
}

[pscustomobject]@{
    repository_root = $repo
    run_name = $resolvedRunName
    config_path = $config
    python = $PythonExe
    pythonpath = $env:PYTHONPATH
    sealbot_path = $env:SEALBOT_PATH
    watchdog_pid = if ($watchdog) { [int]$watchdog.ProcessId } else { $null }
    trainer_pid = if ($trainer) { [int]$trainer.Id } else { $null }
    trainer_out_log = $outLog
    trainer_err_log = $errLog
    dry_run = [bool]$DryRun
} | ConvertTo-Json -Depth 4
