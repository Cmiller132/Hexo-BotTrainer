<#
.SYNOPSIS
Captures the complete PRIMARY-only DOM-B2 checkpoint after the V3 queue ends.

.DESCRIPTION
This fail-closed capture invokes only the frozen read-only analyzer and chain
auditor. It never invokes Cargo. It refuses to start until the terminal PRIMARY
DONE, wrapper-process, lock, Cargo-process, source, and zero-replica gates pass.
All outputs use create-new semantics and UTF-8 without BOM with LF line endings.
Partial outputs are preserved on any later failure.
#>
[CmdletBinding(DefaultParameterSetName = 'Capture')]
param(
    [Parameter(Mandatory = $true, ParameterSetName = 'SelfTest')]
    [switch] $SelfTest,

    [Parameter(ParameterSetName = 'Capture')]
    [int[]] $AdditionalPrimaryWrapperPid = @()
)

class DomB2D4CaptureHost : System.Management.Automation.Host.PSHost {
    [bool] $ShouldExit = $false
    [int] $ExitCode = 0
    hidden [guid] $HostId = [guid]::NewGuid()
    hidden [Globalization.CultureInfo] $HostCulture
    hidden [Globalization.CultureInfo] $HostUICulture

    DomB2D4CaptureHost(
        [Globalization.CultureInfo] $culture,
        [Globalization.CultureInfo] $uiCulture
    ) {
        $this.HostCulture = $culture
        $this.HostUICulture = $uiCulture
    }

    [guid] get_InstanceId() { return $this.HostId }
    [string] get_Name() { return 'DOM-B2 D4 checkpoint isolated capture host' }
    [version] get_Version() { return [version]'1.0' }
    [System.Management.Automation.Host.PSHostUserInterface] get_UI() { return $null }
    [Globalization.CultureInfo] get_CurrentCulture() { return $this.HostCulture }
    [Globalization.CultureInfo] get_CurrentUICulture() { return $this.HostUICulture }
    [void] SetShouldExit([int] $exitCode) {
        $this.ShouldExit = $true
        $this.ExitCode = $exitCode
    }
    [void] EnterNestedPrompt() { throw 'nested prompts are unsupported in checkpoint capture' }
    [void] ExitNestedPrompt() { throw 'nested prompts are unsupported in checkpoint capture' }
    [void] NotifyBeginApplication() {}
    [void] NotifyEndApplication() {}
}

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$invariant = [Globalization.CultureInfo]::InvariantCulture
$repoRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$scriptPath = [IO.Path]::GetFullPath($MyInvocation.MyCommand.Path)

$script:version = 1
$script:knownPrimaryWrapperPid = 57144
$script:primaryConfig = 'd6:true,tt_bytes:536870912'
$script:d6OffConfig = 'd6:false,tt_bytes:536870912'
$script:secondTtConfig = 'd6:true,tt_bytes:268435456'
$script:primaryRunnerSha = '25C6AB359B4BCB3F2207F426BF958A7DB588E1B5BFE7FC0D67FF4AC3EC385D8E'
$script:replicaRunnerSha = 'B9E61444E1555E648575336B8B6E8CD406CB0FB24C406557106F639E9B1953C8'
$script:sourceSnapshotSha = '1D4FBB37638668D0F2ED1972D27CDDD833826721A2EFD5500BFDC09DCF81B746'
$script:censusSha = '436E9F6C4A93CDB611EEF6495A01F510615174A2897271CA1092A0E7422DD7BE'
$script:verifierSha = '9990D38618DA2204351E328CA0143BE2AEF98BB3001E4A0462CF346B707F2CE8'
$script:binarySha = '56B8FA5563D5CDE397133B8328DEB3B79D072E2577573C4C0A94619AA4750A14'
$script:analyzerSha = '862BCF23125EA63C66334C8DCAD192FA8A7528267A8954BB8C42D5AE4BD8BED5'
$script:chainSha = 'E8991F499D90FEF81B9CCF492CC17047ED24863159F3F83279A7634A9DE383C8'
$script:childWrapperSha = 'A9F8AF43DB7AD2D2D321DBB0E1BCCD5149175885D91A1FEC9F4DB12BC4CA06BC'
$script:codeId = 'DOM_B2_D4_PRIMARY_V3'
$script:analyzerInvocationScript = '& $domCaptureAnalyzerPath -CensusPath $domCaptureCensusPath -ShardPath $domCaptureShardPaths -ApprovedRunnerMapping $domCaptureMappings 2>&1'
$script:chainInvocationScript = '& $domCaptureChainPath -PrimaryJournalPath $domCaptureJournalPaths 2>&1'
$script:widths = @(329, 312, 312, 330, 313, 330, 313, 330, 313, 383, 383)
$script:fingerprints = @(
    '827EEB0FCB78C698', '7C4092D562D3E619', '319A510062631E51',
    '27C689B6D4D0DC33', '09E733EF333378BA', '530A55C8F49F0911',
    '324FB3CEA1CDCA7E', 'CCE0D5A475F109B6', 'F9FC2F2E9BB41D72',
    'FF353A8E0556E088', 'B57A000A7F5C6800'
)
$script:primaryShardRegex = [regex]::new(
    '^DOM_B2_D4_SHARD_C(?<case>(?:0[0-9]|10))_S(?<start>[0-9]{4})(?:_A(?<attempt>[0-9]{2}))?_RAW[.]log$',
    [Text.RegularExpressions.RegexOptions]::CultureInvariant
)
$script:creditedPrimaryRawRegex = [regex]::new(
    '^DOM_B2_D4_SHARD_C(?<case>(?:0[0-9]|10))_S(?<start>[0-9]{4})_A(?<attempt>[0-9]{2})_RAW[.]log$',
    [Text.RegularExpressions.RegexOptions]::CultureInvariant
)
$script:primaryJournalRegex = [regex]::new(
    '^DOM_B2_D4_QUEUE_V3_RUN(?<run>[0-9]{2,})_RAW[.]log$',
    [Text.RegularExpressions.RegexOptions]::CultureInvariant
)
$script:captureSentinelProperty = 'DOM_B2_D4_CAPTURE_INTERNAL_SENTINEL'
$script:replicaMeasurementRegex = [regex]::new(
    '^DOM_B2_D4_REPLICA_(?:D6_OFF|SECOND_TT)_(?:SHARD|QUEUE)',
    [Text.RegularExpressions.RegexOptions]::CultureInvariant -bor
        [Text.RegularExpressions.RegexOptions]::IgnoreCase
)

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string] $LiteralPath)
    return (Get-FileHash -LiteralPath $LiteralPath -Algorithm SHA256).Hash.ToUpperInvariant()
}

function Get-RelativePath {
    param([Parameter(Mandatory = $true)][string] $FullPath)
    $full = [IO.Path]::GetFullPath($FullPath)
    $prefix = $repoRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    if (-not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "path escapes repository root: $full"
    }
    return $full.Substring($prefix.Length).Replace([IO.Path]::DirectorySeparatorChar, '/')
}

function New-LfWriter {
    param([Parameter(Mandatory = $true)][string] $LiteralPath)
    $stream = [IO.FileStream]::new(
        $LiteralPath,
        [IO.FileMode]::CreateNew,
        [IO.FileAccess]::Write,
        [IO.FileShare]::Read,
        4096,
        [IO.FileOptions]::WriteThrough
    )
    $writer = [IO.StreamWriter]::new($stream, [Text.UTF8Encoding]::new($false, $true))
    $writer.NewLine = "`n"
    $writer.AutoFlush = $true
    return $writer
}

function Write-NormalizedObject {
    param(
        [Parameter(Mandatory = $true)][IO.StreamWriter] $Writer,
        [AllowNull()][object] $Value
    )
    $text = if ($null -eq $Value) { '' } else { [string]$Value }
    $text = $text.Replace("`r`n", "`n").Replace("`r", "`n")
    $parts = @($text -split "`n", -1)
    $limit = $parts.Count
    if ($limit -gt 1 -and $parts[$limit - 1] -ceq '') { $limit-- }
    if ($limit -eq 0) {
        $Writer.WriteLine('')
        return
    }
    for ($index = 0; $index -lt $limit; $index++) {
        $Writer.WriteLine($parts[$index])
    }
}

function Assert-PureLfUtf8File {
    param([Parameter(Mandatory = $true)][string] $LiteralPath)
    $bytes = [IO.File]::ReadAllBytes($LiteralPath)
    if ($bytes.Count -eq 0) { throw "empty capture file: $LiteralPath" }
    if ($bytes[$bytes.Count - 1] -ne 10) { throw "capture lacks terminal LF: $LiteralPath" }
    if (@($bytes | Where-Object { $_ -eq 13 }).Count -ne 0) {
        throw "capture contains CR bytes: $LiteralPath"
    }
    if ($bytes.Count -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
        throw "capture contains UTF-8 BOM: $LiteralPath"
    }
    $decoder = [Text.UTF8Encoding]::new($false, $true)
    try { [void]$decoder.GetString($bytes) }
    catch { throw "capture is not strict UTF-8: $LiteralPath" }
    return [pscustomobject]@{
        Bytes = [int64]$bytes.Count
        Sha256 = Get-Sha256 -LiteralPath $LiteralPath
    }
}

function Read-PureLfLines {
    param([Parameter(Mandatory = $true)][string] $LiteralPath)
    [void](Assert-PureLfUtf8File -LiteralPath $LiteralPath)
    $decoder = [Text.UTF8Encoding]::new($false, $true)
    $text = $decoder.GetString([IO.File]::ReadAllBytes($LiteralPath))
    $text = $text.Substring(0, $text.Length - 1)
    if ($text.Length -eq 0) { return @() }
    return @($text.Split([char]10))
}

function Convert-ToFieldMap {
    param(
        [Parameter(Mandatory = $true)][string] $Line,
        [Parameter(Mandatory = $true)][string] $Prefix
    )
    if ($Line -cne $Prefix -and -not $Line.StartsWith("$Prefix ", [StringComparison]::Ordinal)) {
        throw "record prefix mismatch expected=$Prefix line=$Line"
    }
    $map = @{}
    $tail = $Line.Substring($Prefix.Length).Trim()
    if ($tail.Length -eq 0) { return $map }
    foreach ($token in @($tail.Split([char]' ', [StringSplitOptions]::RemoveEmptyEntries))) {
        $equals = $token.IndexOf('=')
        if ($equals -le 0) { throw "malformed field token=$token prefix=$Prefix" }
        $key = $token.Substring(0, $equals)
        $value = $token.Substring($equals + 1)
        if ($map.ContainsKey($key)) { throw "duplicate field=$key prefix=$Prefix" }
        $map[$key] = $value
    }
    return $map
}

function Get-RequiredField {
    param(
        [Parameter(Mandatory = $true)][hashtable] $Map,
        [Parameter(Mandatory = $true)][string] $Name,
        [Parameter(Mandatory = $true)][string] $Context
    )
    if (-not $Map.ContainsKey($Name)) { throw "$Context missing field=$Name" }
    return [string]$Map[$Name]
}

function Convert-ToInt64 {
    param(
        [Parameter(Mandatory = $true)][string] $Value,
        [Parameter(Mandatory = $true)][string] $Context
    )
    $parsed = [int64]0
    if (-not [int64]::TryParse(
            $Value,
            [Globalization.NumberStyles]::Integer,
            $invariant,
            [ref]$parsed
        )) {
        throw "$Context is not Int64: $Value"
    }
    return $parsed
}

function Convert-ToDecimal {
    param(
        [Parameter(Mandatory = $true)][string] $Value,
        [Parameter(Mandatory = $true)][string] $Context
    )
    $parsed = [decimal]0
    if (-not [decimal]::TryParse(
            $Value,
            [Globalization.NumberStyles]::AllowDecimalPoint,
            $invariant,
            [ref]$parsed
        )) {
        throw "$Context is not a non-exponential decimal: $Value"
    }
    return $parsed
}

function Get-OneRecord {
    param(
        [Parameter(Mandatory = $true)][string[]] $Lines,
        [Parameter(Mandatory = $true)][string] $Prefix
    )
    $matches = @($Lines | Where-Object {
        $_ -ceq $Prefix -or $_.StartsWith("$Prefix ", [StringComparison]::Ordinal)
    })
    if ($matches.Count -ne 1) { throw "expected exactly one $Prefix record; found $($matches.Count)" }
    return $matches[0]
}

function Get-Records {
    param(
        [Parameter(Mandatory = $true)][string[]] $Lines,
        [Parameter(Mandatory = $true)][string] $Prefix
    )
    return @($Lines | Where-Object {
        $_ -ceq $Prefix -or $_.StartsWith("$Prefix ", [StringComparison]::Ordinal)
    })
}

function Assert-FileIdentity {
    param(
        [Parameter(Mandatory = $true)][string] $RelativePath,
        [Parameter(Mandatory = $true)][int64] $ExpectedBytes,
        [Parameter(Mandatory = $true)][string] $ExpectedSha256,
        [string] $Root = $repoRoot
    )
    if ([IO.Path]::IsPathRooted($RelativePath) -or $RelativePath.Contains('..')) {
        throw "unsafe bound relative path: $RelativePath"
    }
    $full = Join-Path $Root $RelativePath
    if (-not (Test-Path -LiteralPath $full -PathType Leaf)) { throw "missing bound file: $RelativePath" }
    $item = Get-Item -LiteralPath $full
    $sha = Get-Sha256 -LiteralPath $full
    if ($item.Length -ne $ExpectedBytes -or $sha -cne $ExpectedSha256) {
        throw "bound file mismatch path=$RelativePath expected=$ExpectedSha256/$ExpectedBytes actual=$sha/$($item.Length)"
    }
    return [pscustomobject]@{ Path = $RelativePath; Bytes = [int64]$item.Length; Sha256 = $sha }
}

function Get-SupportSnapshot {
    $fixed = @(
        @('DOM_B2_D4_PRIMARY_CHECKPOINT_PREREG_RAW.log', 18454, 'C65E1849E8A0DDF34501458F22DF6A00ECC94B487367A9BEC876ADE316D19AF5'),
        @('DOM_B2_D4_PREREG_RAW.log', 7618, '9A76F6F9E24F551E69A8A513FC33031D5EB5F350B129B99354C6FD81959B1981'),
        @('scripts/dom_b2_d4_run_queue.ps1', 49762, $script:primaryRunnerSha),
        @('scripts/dom_b2_d4_run_replica_queue.ps1', 56516, $script:replicaRunnerSha),
        @('DOM_B2_D4_PRIMARY_V3_CODE_SNAPSHOT_RAW.log', 3399, $script:sourceSnapshotSha),
        @('DOM_B2_D4_PRIMARY_V3_CODE_SNAPSHOT_SHA_RAW.log', 396, '38BB0351A021971A4F4CD67F20217BACC7C98A1E14E8E8A59F84CDC78A0527A1'),
        @('DOM_B2_D4_CENSUS_RAW.log', 292756, $script:censusSha),
        @('.target-hunt/x86_64-pc-windows-msvc/release/deps/hexfield_eq-de26e3778420c4c2.exe', 3290112, $script:binarySha),
        @('packages/hexfield_eq/rust/src/tss_verify.rs', 78741, $script:verifierSha),
        @('scripts/dom_b2_d4_aggregate.ps1', 77182, $script:analyzerSha),
        @('DOM_B2_D4_ANALYZER_STATIC_TEST_RAW.log', 6641, 'C1853193844BA45AE35C8E7126711B703A294AF7FDF67AB53485A6F8DBCE95E5'),
        @('scripts/dom_b2_d4_chain_audit.ps1', 116000, $script:chainSha),
        @('DOM_B2_D4_CHAIN_AUDIT_PREREG_RAW.log', 10321, '24F87552069BAE910EC87656CD7EF7A40BD40D504A10A984E7472D2361B42A34'),
        @('DOM_B2_D4_CHAIN_AUDIT_STATIC_RAW.log', 4715, '85AF8E3F55C02FD27D2D6A5649A9A04915E1ED5CF104D07E87B165293A21D9C7')
    )
    $records = New-Object 'Collections.Generic.List[object]'
    foreach ($entry in $fixed) {
        $records.Add((Assert-FileIdentity -RelativePath $entry[0] -ExpectedBytes $entry[1] -ExpectedSha256 $entry[2]))
    }
    $selfItem = Get-Item -LiteralPath $scriptPath
    $records.Add([pscustomobject]@{
        Path = Get-RelativePath -FullPath $scriptPath
        Bytes = [int64]$selfItem.Length
        Sha256 = Get-Sha256 -LiteralPath $scriptPath
    })

    $snapshotPath = Join-Path $repoRoot 'DOM_B2_D4_PRIMARY_V3_CODE_SNAPSHOT_RAW.log'
    $codeCount = 0
    $binaryCount = 0
    $smokeCount = 0
    $verifierCount = 0
    foreach ($line in Get-Content -LiteralPath $snapshotPath) {
        $relative = $null
        $bytes = $null
        $sha = $null
        if ($line -cmatch '^CODE_FILE path=([^ ]+) bytes=([0-9]+) sha256=([0-9A-F]{64})$') {
            $relative = $matches[1]
            $bytes = [int64]$matches[2]
            $sha = $matches[3]
            $codeCount++
        }
        elseif ($line -cmatch '^BINARY path=([^ ]+) bytes=([0-9]+) mtime_utc=[^ ]+ sha256=([0-9A-F]{64})$') {
            $relative = $matches[1]
            $bytes = [int64]$matches[2]
            $sha = $matches[3]
            $binaryCount++
        }
        elseif ($line -cmatch '^SCHEMA_SMOKE path=([^ ]+) bytes=([0-9]+) sha256=([0-9A-F]{64}) code_id=[A-Za-z0-9_.-]+ result=PASS$') {
            $relative = $matches[1]
            $bytes = [int64]$matches[2]
            $sha = $matches[3]
            $smokeCount++
        }
        elseif ($line -cmatch '^STRICT_VERIFIER path=([^ ]+) sha256=([0-9A-F]{64}) untouched=true$') {
            $relative = $matches[1]
            $sha = $matches[2]
            $full = Join-Path $repoRoot $relative
            if (-not (Test-Path -LiteralPath $full -PathType Leaf)) { throw "missing snapshot verifier: $relative" }
            $bytes = (Get-Item -LiteralPath $full).Length
            $verifierCount++
        }
        if ($null -ne $relative) {
            $records.Add((Assert-FileIdentity -RelativePath $relative -ExpectedBytes $bytes -ExpectedSha256 $sha))
        }
    }
    if ($codeCount -ne 17 -or $binaryCount -ne 1 -or $smokeCount -ne 1 -or $verifierCount -ne 1) {
        throw "snapshot entry-count mismatch code=$codeCount binary=$binaryCount smoke=$smokeCount verifier=$verifierCount"
    }

    $deduplicated = @{}
    foreach ($record in $records) {
        $key = $record.Path.ToLowerInvariant()
        if ($deduplicated.ContainsKey($key)) {
            $prior = $deduplicated[$key]
            if ($prior.Bytes -ne $record.Bytes -or $prior.Sha256 -cne $record.Sha256) {
                throw "inconsistent duplicate support binding path=$($record.Path)"
            }
        }
        else { $deduplicated[$key] = $record }
    }
    return @($deduplicated.Values | Sort-Object -Property Path)
}

function Compare-Snapshots {
    param(
        [Parameter(Mandatory = $true)][object[]] $Before,
        [Parameter(Mandatory = $true)][object[]] $After,
        [Parameter(Mandatory = $true)][string] $Context
    )
    $beforeMap = @{}
    foreach ($entry in $Before) { $beforeMap[$entry.Path.ToLowerInvariant()] = $entry }
    $afterMap = @{}
    foreach ($entry in $After) { $afterMap[$entry.Path.ToLowerInvariant()] = $entry }
    if ($beforeMap.Count -ne $afterMap.Count) { throw "$Context snapshot count changed" }
    foreach ($key in $beforeMap.Keys) {
        if (-not $afterMap.ContainsKey($key)) { throw "$Context path disappeared: $key" }
        $left = $beforeMap[$key]
        $right = $afterMap[$key]
        if ($left.Bytes -ne $right.Bytes -or $left.Sha256 -cne $right.Sha256) {
            throw "$Context changed path=$($left.Path)"
        }
    }
}

function Get-FileSnapshots {
    param([Parameter(Mandatory = $true)][IO.FileInfo[]] $Files)
    $snapshots = @()
    foreach ($file in $Files) {
        $snapshots += [pscustomobject]@{
            Path = Get-RelativePath -FullPath $file.FullName
            Bytes = [int64]$file.Length
            Sha256 = Get-Sha256 -LiteralPath $file.FullName
        }
    }
    return $snapshots
}

function Get-CanonicalPrimaryShards {
    param([Parameter(Mandatory = $true)][string] $Root)
    $files = @(Get-ChildItem -LiteralPath $Root -File -Force | Where-Object {
        $script:primaryShardRegex.IsMatch($_.Name)
    })
    $names = [string[]]@($files | ForEach-Object Name)
    [Array]::Sort($names, [StringComparer]::Ordinal)
    $seen = @{}
    $result = @()
    foreach ($name in $names) {
        $key = $name.ToLowerInvariant()
        if ($seen.ContainsKey($key)) { throw "case-insensitive duplicate primary shard name=$name" }
        $seen[$key] = $true
        $result += Get-Item -LiteralPath (Join-Path $Root $name)
    }
    if ($result.Count -eq 0) { throw 'no canonical PRIMARY shard paths found' }
    return $result
}

function Get-PrimaryJournals {
    param([Parameter(Mandatory = $true)][string] $Root)
    $records = @()
    foreach ($file in Get-ChildItem -LiteralPath $Root -File -Force) {
        $match = $script:primaryJournalRegex.Match($file.Name)
        if (-not $match.Success) { continue }
        $run = 0
        if (-not [int]::TryParse(
                $match.Groups['run'].Value,
                [Globalization.NumberStyles]::None,
                $invariant,
                [ref]$run
            )) {
            throw "PRIMARY journal run id is outside Int32: $($file.Name)"
        }
        if ($run -lt 3) { continue }
        $records += [pscustomobject]@{ Run = $run; File = $file }
    }
    if ($records.Count -eq 0 -or @($records | Where-Object { $_.Run -eq 3 }).Count -ne 1) {
        throw 'frozen PRIMARY journal family RUN03+ is missing RUN03'
    }
    $seen = @{}
    $seenRuns = @{}
    foreach ($record in $records) {
        $key = $record.File.Name.ToLowerInvariant()
        if ($seen.ContainsKey($key)) { throw "duplicate journal name=$($record.File.Name)" }
        $seen[$key] = $true
        if ($seenRuns.ContainsKey($record.Run)) {
            throw "duplicate numeric PRIMARY journal run id=$($record.Run) files=$($seenRuns[$record.Run]),$($record.File.Name)"
        }
        $seenRuns[$record.Run] = $record.File.Name
    }
    $names = [string[]]@($records | ForEach-Object { $_.File.Name })
    [Array]::Sort($names, [StringComparer]::Ordinal)
    return @($names | ForEach-Object { Get-Item -LiteralPath (Join-Path $Root $_) })
}

function Assert-TerminalJournals {
    param([Parameter(Mandatory = $true)][IO.FileInfo[]] $Journals)
    $doneRecords = @()
    $terminalPath = $null
    foreach ($journal in $Journals) {
        $lines = @(Get-Content -LiteralPath $journal.FullName | Where-Object { $_.Length -gt 0 })
        foreach ($line in $lines) {
            if ($line -cmatch '^QUEUE timestamp=[^ ]+ DONE result=PASS invocations=([0-9]+) cases=11 total_roots=3648 code_id=DOM_B2_D4_PRIMARY_V3 source_snapshot_sha256=1D4FBB37638668D0F2ED1972D27CDDD833826721A2EFD5500BFDC09DCF81B746$') {
                $doneRecords += [pscustomobject]@{ Journal = $journal; Line = $line; Invocations = [int]$matches[1] }
            }
        }
    }
    if ($doneRecords.Count -ne 1) { throw "expected exactly one PRIMARY DONE record; found $($doneRecords.Count)" }
    $terminal = $doneRecords[0]
    $terminalRun = [int]$script:primaryJournalRegex.Match($terminal.Journal.Name).Groups['run'].Value
    $maxRun = @($Journals | ForEach-Object {
        [int]$script:primaryJournalRegex.Match($_.Name).Groups['run'].Value
    } | Measure-Object -Maximum)[0].Maximum
    if ($terminalRun -ne $maxRun) {
        throw "PRIMARY DONE is not in the numerically latest RUN03+ journal terminal_run=$terminalRun max_run=$maxRun"
    }
    $terminalLines = @(Get-Content -LiteralPath $terminal.Journal.FullName | Where-Object { $_.Length -gt 0 })
    if ($terminalLines[$terminalLines.Count - 1] -cne $terminal.Line) {
        throw 'PRIMARY DONE is not the terminal nonempty journal line'
    }
    $orderedCaseDone = @($terminalLines | Where-Object {
        $_ -cmatch '^QUEUE timestamp=[^ ]+ CASE_DONE case=([0-9]+) roots=([0-9]+) code_id=DOM_B2_D4_PRIMARY_V3$'
    })
    if ($orderedCaseDone.Count -ne 11) {
        throw "terminal journal ordered CASE_DONE count mismatch count=$($orderedCaseDone.Count)"
    }
    for ($case = 0; $case -lt 11; $case++) {
        $pattern = "^QUEUE timestamp=[^ ]+ CASE_DONE case=$case roots=$($script:widths[$case]) code_id=DOM_B2_D4_PRIMARY_V3$"
        $count = @($terminalLines | Where-Object { $_ -cmatch $pattern }).Count
        if ($count -ne 1) { throw "terminal journal CASE_DONE count mismatch case=$case count=$count" }
        if ($orderedCaseDone[$case] -cnotmatch " CASE_DONE case=$case roots=$($script:widths[$case]) ") {
            throw "terminal journal CASE_DONE order mismatch index=$case line=$($orderedCaseDone[$case])"
        }
    }
    return $terminal
}

function Get-ReplicaMeasurementArtifacts {
    param([Parameter(Mandatory = $true)][string] $Root)
    return @(Get-ChildItem -LiteralPath $Root -File -Force | Where-Object {
        $script:replicaMeasurementRegex.IsMatch($_.Name)
    })
}

function Get-LivePidTree {
    param(
        [Parameter(Mandatory = $true)][int[]] $RootPid,
        [Parameter(Mandatory = $true)][object[]] $Processes
    )
    $tree = @{}
    foreach ($pidValue in $RootPid) { $tree[[int]$pidValue] = $true }
    $changed = $true
    while ($changed) {
        $changed = $false
        foreach ($process in $Processes) {
            $pidValue = [int]$process.ProcessId
            $parent = [int]$process.ParentProcessId
            if ($tree.ContainsKey($parent) -and -not $tree.ContainsKey($pidValue)) {
                $tree[$pidValue] = $true
                $changed = $true
            }
        }
    }
    return @($Processes | Where-Object { $tree.ContainsKey([int]$_.ProcessId) })
}

function Assert-TerminalProcessGate {
    param([Parameter(Mandatory = $true)][int[]] $WrapperPids)
    $lockPath = Join-Path $repoRoot '.dom_b2_d4_run_queue.lock'
    if (Test-Path -LiteralPath $lockPath) { throw 'PRIMARY queue lock is still present' }
    foreach ($replicaLock in @(
        '.dom_b2_d4_replica_queue.lock',
        '.dom_b2_d4_replica_d6_off_queue.lock',
        '.dom_b2_d4_replica_second_tt_queue.lock'
    )) {
        if (Test-Path -LiteralPath (Join-Path $repoRoot $replicaLock)) {
            throw "replica lock is present: $replicaLock"
        }
    }
    $processes = @(Get-CimInstance -ClassName Win32_Process)
    $cargo = @($processes | Where-Object { [string]$_.Name -ieq 'cargo.exe' })
    if ($cargo.Count -ne 0) { throw "host-wide cargo.exe count is $($cargo.Count), expected 0" }
    $liveTree = @(Get-LivePidTree -RootPid $WrapperPids -Processes $processes)
    if ($liveTree.Count -ne 0) {
        $detail = @($liveTree | ForEach-Object { "$($_.ProcessId):$($_.Name):ppid=$($_.ParentProcessId)" }) -join ','
        throw "PRIMARY wrapper PID/descendant still present: $detail"
    }
    return [pscustomobject]@{ CargoCount = 0; LiveTreeCount = 0; ProcessCount = $processes.Count }
}

function Assert-ZeroReplicaArtifacts {
    $artifacts = @(Get-ReplicaMeasurementArtifacts -Root $repoRoot)
    if ($artifacts.Count -ne 0) {
        throw "replica measurement artifacts exist: $(@($artifacts | ForEach-Object Name) -join ',')"
    }
    return 0
}

function Validate-AnalyzerOutput {
    param(
        [Parameter(Mandatory = $true)][string[]] $Lines,
        [string[]] $ExpectedShardNames = @()
    )
    if ($Lines.Count -eq 0) { throw 'analyzer output is empty' }
    if (@($Lines | Where-Object { $_.StartsWith('DOM_B2_D4_VALIDATION_ERROR ', [StringComparison]::Ordinal) }).Count -ne 0) {
        throw 'analyzer emitted VALIDATION_ERROR'
    }
    $setup = Convert-ToFieldMap -Line (Get-OneRecord -Lines $Lines -Prefix 'DOM_B2_D4_ANALYSIS_SETUP') -Prefix 'DOM_B2_D4_ANALYSIS_SETUP'
    if ((Get-RequiredField $setup 'census_sha256' 'analysis setup') -cne $script:censusSha -or
        (Get-RequiredField $setup 'census_hard_bind' 'analysis setup') -cne 'PASS' -or
        (Get-RequiredField $setup 'required_code_id' 'analysis setup') -cne $script:codeId -or
        (Get-RequiredField $setup 'source_snapshot_sha256' 'analysis setup') -cne $script:sourceSnapshotSha -or
        (Get-RequiredField $setup 'approval_configs' 'analysis setup') -cne '3' -or
        (Get-RequiredField $setup 'discovery_ignored' 'analysis setup') -cne '0') {
        throw 'analyzer setup binding mismatch'
    }
    $reportedShardFiles = [int](Convert-ToInt64 (Get-RequiredField $setup 'shard_files' 'analysis setup') 'analysis shard_files')
    if ($reportedShardFiles -le 0) { throw 'analyzer reported no shard inputs' }
    if ($ExpectedShardNames.Count -gt 0) {
        if ($reportedShardFiles -ne $ExpectedShardNames.Count) {
            throw "analyzer shard input count mismatch expected=$($ExpectedShardNames.Count) reported=$reportedShardFiles"
        }
        $expectedInputs = @{}
        foreach ($name in $ExpectedShardNames) {
            $key = $name.ToLowerInvariant()
            if ($expectedInputs.ContainsKey($key)) { throw "duplicate expected analyzer shard input=$name" }
            $expectedInputs[$key] = $name
        }
        $classified = @(
            @(Get-Records -Lines $Lines -Prefix 'DOM_B2_D4_CREDITED_SHARD_VALID') +
            @(Get-Records -Lines $Lines -Prefix 'DOM_B2_D4_LEGACY_SHARD_VALID') +
            @(Get-Records -Lines $Lines -Prefix 'DOM_B2_D4_V3_UNTRUSTED_SHARD_VALID')
        )
        if ($classified.Count -ne $ExpectedShardNames.Count) {
            throw "analyzer one-record-per-input classification mismatch expected=$($ExpectedShardNames.Count) classified=$($classified.Count)"
        }
        foreach ($line in $classified) {
            $prefix = @(@(
                    'DOM_B2_D4_CREDITED_SHARD_VALID',
                    'DOM_B2_D4_LEGACY_SHARD_VALID',
                    'DOM_B2_D4_V3_UNTRUSTED_SHARD_VALID'
                ) | Where-Object { $line.StartsWith("$_ ", [StringComparison]::Ordinal) })
            if ($prefix.Count -ne 1) { throw 'could not classify analyzer shard record prefix' }
            $map = Convert-ToFieldMap -Line $line -Prefix $prefix[0]
            $file = Get-RequiredField $map 'file' 'analyzer classified shard'
            $key = $file.ToLowerInvariant()
            if (-not $expectedInputs.ContainsKey($key)) { throw "analyzer classified unexpected shard=$file" }
            [void]$expectedInputs.Remove($key)
        }
        if ($expectedInputs.Count -ne 0) { throw 'analyzer omitted an expected shard input' }
    }

    $approvalLines = @(Get-Records -Lines $Lines -Prefix 'DOM_B2_D4_RUNNER_APPROVAL')
    if ($approvalLines.Count -ne 3) { throw "analyzer approval count=$($approvalLines.Count), expected 3" }
    $expectedApprovals = @{
        'PRIMARY' = @($script:primaryConfig, $script:primaryRunnerSha)
        'D6_OFF' = @($script:d6OffConfig, $script:replicaRunnerSha)
        'SECOND_TT' = @($script:secondTtConfig, $script:replicaRunnerSha)
    }
    foreach ($line in $approvalLines) {
        $map = Convert-ToFieldMap -Line $line -Prefix 'DOM_B2_D4_RUNNER_APPROVAL'
        $lane = Get-RequiredField $map 'lane' 'runner approval'
        if (-not $expectedApprovals.ContainsKey($lane)) { throw "unexpected approval lane=$lane" }
        if ((Get-RequiredField $map 'config' 'runner approval') -cne $expectedApprovals[$lane][0] -or
            (Get-RequiredField $map 'runner_sha256' 'runner approval') -cne $expectedApprovals[$lane][1] -or
            (Get-RequiredField $map 'source' 'runner approval') -cne 'EXPLICIT_MAPPING') {
            throw "runner approval mismatch lane=$lane"
        }
        [void]$expectedApprovals.Remove($lane)
    }
    if ($expectedApprovals.Count -ne 0) { throw 'missing runner approval lane' }

    $caseLines = @(Get-Records -Lines $Lines -Prefix 'DOM_B2_D4_LANE_CASE')
    if ($caseLines.Count -ne 33) { throw "analyzer LANE_CASE count=$($caseLines.Count), expected 33" }
    $seenCases = @{}
    foreach ($line in $caseLines) {
        $map = Convert-ToFieldMap -Line $line -Prefix 'DOM_B2_D4_LANE_CASE'
        $lane = Get-RequiredField $map 'lane' 'lane case'
        $case = [int](Convert-ToInt64 (Get-RequiredField $map 'case' 'lane case') 'lane case')
        if ($case -lt 0 -or $case -gt 10) { throw "lane case outside 0..10: $case" }
        $key = "$lane/$case"
        if ($seenCases.ContainsKey($key)) { throw "duplicate lane case=$key" }
        $seenCases[$key] = $true
        $expectedConfig = switch ($lane) {
            'PRIMARY' { $script:primaryConfig }
            'D6_OFF' { $script:d6OffConfig }
            'SECOND_TT' { $script:secondTtConfig }
            default { throw "unexpected lane case lane=$lane" }
        }
        if ((Get-RequiredField $map 'config' 'lane case') -cne $expectedConfig -or
            [int](Get-RequiredField $map 'root_count' 'lane case') -ne $script:widths[$case] -or
            (Get-RequiredField $map 'fingerprint' 'lane case') -cne $script:fingerprints[$case]) {
            throw "lane case frozen identity mismatch lane=$lane case=$case"
        }
        $complete = [int](Get-RequiredField $map 'unique_complete' 'lane case')
        $missing = [int](Get-RequiredField $map 'missing' 'lane case')
        $incompleteOnly = [int](Get-RequiredField $map 'incomplete_only' 'lane case')
        $duplicates = [int](Get-RequiredField $map 'duplicate_complete' 'lane case')
        $win = [int](Get-RequiredField $map 'win' 'lane case')
        $unknown = [int](Get-RequiredField $map 'unknown' 'lane case')
        $loss = [int](Get-RequiredField $map 'loss' 'lane case')
        $recurrence = Get-RequiredField $map 'recurrence' 'lane case'
        $gate = Get-RequiredField $map 'gate_status' 'lane case'
        $coverage = Get-RequiredField $map 'coverage' 'lane case'
        if ($lane -ceq 'PRIMARY') {
            if ($complete -ne $script:widths[$case] -or $missing -ne 0 -or $incompleteOnly -ne 0 -or
                $duplicates -ne 0 -or ($win + $unknown + $loss) -ne $script:widths[$case] -or
                $coverage -cne 'EXACT' -or $gate -cnotmatch '^(WIN|UNKNOWN|LOSS)$' -or
                $recurrence -cne $gate) {
                throw "PRIMARY case is not exact case=$case"
            }
        }
        else {
            if ($complete -ne 0 -or $missing -ne $script:widths[$case] -or $incompleteOnly -ne 0 -or
                $duplicates -ne 0 -or $win -ne 0 -or $unknown -ne 0 -or $loss -ne 0 -or
                $coverage -cne 'INCOMPLETE' -or $gate -cne 'INCOMPLETE' -or $recurrence -cne 'INCOMPLETE') {
                throw "replica lane unexpectedly has measurement credit lane=$lane case=$case"
            }
        }
    }

    $matrixLines = @(Get-Records -Lines $Lines -Prefix 'DOM_B2_D4_LANE_MATRIX')
    if ($matrixLines.Count -ne 3) { throw "analyzer lane matrix count=$($matrixLines.Count), expected 3" }
    $seenMatrices = @{}
    foreach ($line in $matrixLines) {
        $map = Convert-ToFieldMap -Line $line -Prefix 'DOM_B2_D4_LANE_MATRIX'
        $lane = Get-RequiredField $map 'lane' 'lane matrix'
        if ($seenMatrices.ContainsKey($lane)) { throw "duplicate analyzer lane matrix=$lane" }
        $seenMatrices[$lane] = $true
        $exactCases = [int](Get-RequiredField $map 'exact_cases' 'lane matrix')
        $completeRoots = [int](Get-RequiredField $map 'unique_complete_roots' 'lane matrix')
        $missingRoots = [int](Get-RequiredField $map 'missing_roots' 'lane matrix')
        $incompleteOnly = [int](Get-RequiredField $map 'incomplete_only_roots' 'lane matrix')
        $duplicates = [int](Get-RequiredField $map 'duplicate_complete_rows' 'lane matrix')
        $gapless = Get-RequiredField $map 'gapless' 'lane matrix'
        if ((Get-RequiredField $map 'total_cases' 'lane matrix') -cne '11' -or
            (Get-RequiredField $map 'total_roots' 'lane matrix') -cne '3648') {
            throw "lane matrix denominator mismatch lane=$lane"
        }
        if ($lane -ceq 'PRIMARY') {
            if ($exactCases -ne 11 -or $completeRoots -ne 3648 -or $missingRoots -ne 0 -or
                $incompleteOnly -ne 0 -or $duplicates -ne 0 -or $gapless -cne 'true') {
                throw 'PRIMARY lane matrix is not complete'
            }
        }
        elseif ($lane -ceq 'D6_OFF' -or $lane -ceq 'SECOND_TT') {
            if ($exactCases -ne 0 -or $completeRoots -ne 0 -or $missingRoots -ne 3648 -or
                $incompleteOnly -ne 0 -or $duplicates -ne 0 -or $gapless -cne 'false') {
                throw "replica lane matrix is not empty/partial lane=$lane"
            }
        }
        else { throw "unexpected lane matrix=$lane" }
    }
    foreach ($lane in @('PRIMARY', 'D6_OFF', 'SECOND_TT')) {
        if (-not $seenMatrices.ContainsKey($lane)) { throw "missing analyzer lane matrix=$lane" }
    }

    $credited = @{}
    foreach ($line in Get-Records -Lines $Lines -Prefix 'DOM_B2_D4_CREDITED_SHARD_VALID') {
        $map = Convert-ToFieldMap -Line $line -Prefix 'DOM_B2_D4_CREDITED_SHARD_VALID'
        if ((Get-RequiredField $map 'lane' 'credited analyzer shard') -cne 'PRIMARY' -or
            (Get-RequiredField $map 'config' 'credited analyzer shard') -cne $script:primaryConfig -or
            (Get-RequiredField $map 'role' 'credited analyzer shard') -cne 'PRIMARY_INCLUDED' -or
            (Get-RequiredField $map 'runner_sha256' 'credited analyzer shard') -cne $script:primaryRunnerSha) {
            throw 'analyzer credited non-PRIMARY or wrong-provenance shard'
        }
        $file = Get-RequiredField $map 'file' 'credited analyzer shard'
        $rawMatch = $script:creditedPrimaryRawRegex.Match($file)
        if (-not $rawMatch.Success) { throw "credited analyzer filename mismatch=$file" }
        $case = [int](Get-RequiredField $map 'case' 'credited analyzer shard')
        $start = [int](Get-RequiredField $map 'start' 'credited analyzer shard')
        $rawBytes = [int64](Get-RequiredField $map 'bytes' 'credited analyzer shard')
        $rawSha = Get-RequiredField $map 'sha256' 'credited analyzer shard'
        $meta = Get-RequiredField $map 'meta' 'credited analyzer shard'
        $metaSha = Get-RequiredField $map 'meta_sha256' 'credited analyzer shard'
        if ($case -ne [int]$rawMatch.Groups['case'].Value -or
            $start -ne [int]$rawMatch.Groups['start'].Value -or
            $rawBytes -le 0 -or $rawSha -cnotmatch '^[0-9A-F]{64}$' -or
            $meta -cne [regex]::Replace($file, '_RAW[.]log$', '_META_RAW.log') -or
            $metaSha -cnotmatch '^[0-9A-F]{64}$') {
            throw "credited analyzer identity fields mismatch=$file"
        }
        $key = $file.ToLowerInvariant()
        if ($credited.ContainsKey($key)) { throw "duplicate analyzer credited raw=$file" }
        $credited[$key] = [pscustomobject]@{
            File = $file
            Case = $case
            Start = $start
            Result = Get-RequiredField $map 'result' 'credited analyzer shard'
            RawBytes = $rawBytes
            RawSha256 = $rawSha
            Meta = $meta
            MetaSha256 = $metaSha
        }
    }
    if ($credited.Count -eq 0) { throw 'analyzer has no credited PRIMARY shards' }
    $primarySet = Convert-ToFieldMap -Line (Get-OneRecord -Lines $Lines -Prefix 'DOM_B2_D4_PRIMARY_SET') -Prefix 'DOM_B2_D4_PRIMARY_SET'
    if ([int](Get-RequiredField $primarySet 'files' 'primary set') -ne $credited.Count -or
        (Get-RequiredField $primarySet 'config' 'primary set') -cne $script:primaryConfig) {
        throw 'analyzer PRIMARY_SET does not match credited shard map'
    }
    $replicaSetLines = @(Get-Records -Lines $Lines -Prefix 'DOM_B2_D4_REPLICA_SET')
    if ($replicaSetLines.Count -ne 2) { throw "replica set count=$($replicaSetLines.Count), expected 2" }
    $expectedReplicaSets = @{
        'D6_OFF' = $script:d6OffConfig
        'SECOND_TT' = $script:secondTtConfig
    }
    foreach ($line in $replicaSetLines) {
        $map = Convert-ToFieldMap -Line $line -Prefix 'DOM_B2_D4_REPLICA_SET'
        $lane = Get-RequiredField $map 'lane' 'replica set'
        if (-not $expectedReplicaSets.ContainsKey($lane)) { throw "unexpected/duplicate replica set lane=$lane" }
        if ((Get-RequiredField $map 'files' 'replica set') -cne '0' -or
            (Get-RequiredField $map 'records' 'replica set') -cne '0' -or
            (Get-RequiredField $map 'unique_complete_roots' 'replica set') -cne '0' -or
            (Get-RequiredField $map 'config' 'replica set') -cne $expectedReplicaSets[$lane]) {
            throw "replica set contains credit lane=$lane"
        }
        [void]$expectedReplicaSets.Remove($lane)
    }
    if ($expectedReplicaSets.Count -ne 0) { throw 'missing expected replica set lane' }
    $comparisons = @(Get-Records -Lines $Lines -Prefix 'DOM_B2_D4_COMPARISON')
    if ($comparisons.Count -ne 4) { throw "PRIMARY comparison count=$($comparisons.Count), expected 4" }
    $expectedComparisons = @{
        'K1_32F' = $true
        'K1_19B' = $true
        'K1_498A' = $true
        'K1_FD68' = $true
    }
    foreach ($line in $comparisons) {
        $map = Convert-ToFieldMap -Line $line -Prefix 'DOM_B2_D4_COMPARISON'
        $name = Get-RequiredField $map 'name' 'comparison'
        if (-not $expectedComparisons.ContainsKey($name)) { throw "unexpected/duplicate comparison=$name" }
        if ((Get-RequiredField $map 'status' 'comparison') -cnotmatch '^(SATISFIED|REVERSED)$') {
            throw 'PRIMARY comparison is not exact'
        }
        [void]$expectedComparisons.Remove($name)
    }
    if ($expectedComparisons.Count -ne 0) { throw 'missing PRIMARY comparison' }
    $control = Convert-ToFieldMap -Line (Get-OneRecord -Lines $Lines -Prefix 'DOM_B2_D4_CONTROL') -Prefix 'DOM_B2_D4_CONTROL'
    if ((Get-RequiredField $control 'status' 'control') -cnotmatch '^(MATCH|MISMATCH)$') { throw 'control is not exact' }
    $history = Convert-ToFieldMap -Line (Get-OneRecord -Lines $Lines -Prefix 'DOM_B2_D4_HISTORY_CHECK') -Prefix 'DOM_B2_D4_HISTORY_CHECK'
    if ((Get-RequiredField $history 'status' 'history') -cnotmatch '^(MATCH|DISCREPANCY)$') { throw 'history is not exact' }
    [void](Get-OneRecord -Lines $Lines -Prefix 'DOM_B2_D4_LOSS_QUALIFICATION')
    $chainFence = Convert-ToFieldMap -Line (Get-OneRecord -Lines $Lines -Prefix 'DOM_B2_D4_CHAIN_AUDIT_FENCE') -Prefix 'DOM_B2_D4_CHAIN_AUDIT_FENCE'
    if ((Get-RequiredField $chainFence 'status' 'chain fence') -cne 'EXTERNAL_MECHANICAL_AUDIT_REQUIRED' -or
        (Get-RequiredField $chainFence 'satisfied' 'chain fence') -cne 'false') {
        throw 'analyzer external chain fence is not pending'
    }
    $replicaFence = Convert-ToFieldMap -Line (Get-OneRecord -Lines $Lines -Prefix 'DOM_B2_D4_REPLICA_FENCE') -Prefix 'DOM_B2_D4_REPLICA_FENCE'
    if ((Get-RequiredField $replicaFence 'd6_off_gapless' 'replica fence') -cne 'false' -or
        (Get-RequiredField $replicaFence 'second_tt_gapless' 'replica fence') -cne 'false' -or
        (Get-RequiredField $replicaFence 'satisfied' 'replica fence') -cne 'false') {
        throw 'analyzer replica fence unexpectedly satisfied'
    }
    $matrix = Convert-ToFieldMap -Line (Get-OneRecord -Lines $Lines -Prefix 'DOM_B2_D4_MATRIX') -Prefix 'DOM_B2_D4_MATRIX'
    if ((Get-RequiredField $matrix 'replica_files' 'matrix') -cne '0' -or
        (Get-RequiredField $matrix 'exact_cases' 'matrix') -cne '11' -or
        (Get-RequiredField $matrix 'unique_complete_roots' 'matrix') -cne '3648' -or
        (Get-RequiredField $matrix 'missing_roots' 'matrix') -cne '0' -or
        (Get-RequiredField $matrix 'incomplete_only_roots' 'matrix') -cne '0' -or
        (Get-RequiredField $matrix 'duplicate_complete_rows' 'matrix') -cne '0' -or
        (Get-RequiredField $matrix 'replica_fence_satisfied' 'matrix') -cne 'false') {
        throw 'analyzer matrix is not PRIMARY-complete/replica-pending'
    }
    $doneLine = Get-OneRecord -Lines $Lines -Prefix 'DOM_B2_D4_ANALYSIS_DONE'
    if ($Lines[$Lines.Count - 1] -cne $doneLine) { throw 'ANALYSIS_DONE is not terminal' }
    $done = Convert-ToFieldMap -Line $doneLine -Prefix 'DOM_B2_D4_ANALYSIS_DONE'
    if ((Get-RequiredField $done 'validation' 'analysis done') -cne 'PASS' -or
        (Get-RequiredField $done 'chain_audit_fence' 'analysis done') -cne 'EXTERNAL_REQUIRED' -or
        (Get-RequiredField $done 'authoritative_scope' 'analysis done') -cne 'fixed_completed_pairs_not_exhaustive_F4') {
        throw 'ANALYSIS_DONE binding mismatch'
    }
    $primaryMatrix = Get-RequiredField $done 'primary_matrix' 'analysis done'
    if ($primaryMatrix -cnotmatch '^(REOPEN_EVIDENCE_PENDING_REPLICAS|SCOPED_KILL_CANDIDATE_PENDING_REPLICAS|NULL_FENCE_UNSATISFIED)$') {
        throw "unexpected PRIMARY-only matrix token=$primaryMatrix"
    }
    return [pscustomobject]@{ Credited = $credited; PrimaryMatrix = $primaryMatrix; Lines = $Lines.Count }
}

function Validate-ChainOutput {
    param(
        [Parameter(Mandatory = $true)][string[]] $Lines,
        [Parameter(Mandatory = $true)][IO.FileInfo[]] $ExpectedJournals
    )
    if ($Lines.Count -eq 0) { throw 'chain output is empty' }
    $header = Convert-ToFieldMap -Line (Get-OneRecord -Lines $Lines -Prefix 'DOM_B2_D4_CHAIN_AUDIT') -Prefix 'DOM_B2_D4_CHAIN_AUDIT'
    if ((Get-RequiredField $header 'version' 'chain header') -cne '1' -or
        (Get-RequiredField $header 'mode' 'chain header') -cne 'PARTIAL' -or
        [int](Get-RequiredField $header 'explicit_journals' 'chain header') -ne $ExpectedJournals.Count -or
        [int](Get-RequiredField $header 'frozen_journals' 'chain header') -ne $ExpectedJournals.Count -or
        (Get-RequiredField $header 'open_uncredited_runs' 'chain header') -cne '0' -or
        (Get-RequiredField $header 'filesystem_writes' 'chain header') -cne '0' -or
        (Get-RequiredField $header 'cargo_invocations' 'chain header') -cne '0') {
        throw 'chain header is not PRIMARY-complete PARTIAL mode'
    }
    $binding = Convert-ToFieldMap -Line (Get-OneRecord -Lines $Lines -Prefix 'BINDING') -Prefix 'BINDING'
    if ((Get-RequiredField $binding 'primary_runner_sha256' 'chain binding') -cne $script:primaryRunnerSha -or
        (Get-RequiredField $binding 'replica_runner_sha256' 'chain binding') -cne $script:replicaRunnerSha -or
        (Get-RequiredField $binding 'child_wrapper_sha256' 'chain binding') -cne $script:childWrapperSha -or
        (Get-RequiredField $binding 'source_snapshot_sha256' 'chain binding') -cne $script:sourceSnapshotSha -or
        (Get-RequiredField $binding 'source_entries' 'chain binding') -cne '20' -or
        (Get-RequiredField $binding 'census_sha256' 'chain binding') -cne $script:censusSha -or
        (Get-RequiredField $binding 'strict_verifier_sha256' 'chain binding') -cne $script:verifierSha -or
        (Get-RequiredField $binding 'binary_sha256' 'chain binding') -cne $script:binarySha -or
        (Get-RequiredField $binding 'code_id' 'chain binding') -cne $script:codeId -or
        (Get-RequiredField $binding 'deadline_ms' 'chain binding') -cne '480000' -or
        (Get-RequiredField $binding 'cargo_wall_timeout_ms' 'chain binding') -cne '540000' -or
        (Get-RequiredField $binding 'target' 'chain binding') -cne 'x86_64-pc-windows-msvc' -or
        (Get-RequiredField $binding 'release' 'chain binding') -cne 'true' -or
        (Get-RequiredField $binding 'test_threads' 'chain binding') -cne '1') {
        throw 'chain hard binding mismatch'
    }
    $events = Convert-ToFieldMap -Line (Get-OneRecord -Lines $Lines -Prefix 'JOURNAL_EVENT_SUMMARY') -Prefix 'JOURNAL_EVENT_SUMMARY'
    if ((Get-RequiredField $events 'aborts' 'journal summary') -cne '0' -or
        (Get-RequiredField $events 'grammar' 'journal summary') -cne 'PASS' -or
        (Get-RequiredField $events 'state_machine' 'journal summary') -cne 'PASS') {
        throw 'chain journal grammar/state/abort fence failed'
    }
    $journalLines = @(Get-Records -Lines $Lines -Prefix 'JOURNAL')
    if ($journalLines.Count -ne $ExpectedJournals.Count) { throw 'chain JOURNAL record count mismatch' }
    $expectedJournalNames = @{}
    foreach ($journal in $ExpectedJournals) { $expectedJournalNames[$journal.Name.ToLowerInvariant()] = $journal }
    foreach ($line in $journalLines) {
        $map = Convert-ToFieldMap -Line $line -Prefix 'JOURNAL'
        $path = Get-RequiredField $map 'path' 'journal record'
        $key = $path.ToLowerInvariant()
        if ((Get-RequiredField $map 'lane' 'journal record') -cne 'PRIMARY' -or
            (Get-RequiredField $map 'result' 'journal record') -cne 'PASS' -or
            -not $expectedJournalNames.ContainsKey($key)) {
            throw "unexpected chain journal record path=$path"
        }
        $expected = $expectedJournalNames[$key]
        if ([int64](Get-RequiredField $map 'bytes' 'journal record') -ne $expected.Length -or
            (Get-RequiredField $map 'sha256' 'journal record') -cne (Get-Sha256 -LiteralPath $expected.FullName)) {
            throw "chain journal bytes/hash mismatch path=$path"
        }
        [void]$expectedJournalNames.Remove($key)
    }
    if ($expectedJournalNames.Count -ne 0) { throw 'chain omitted expected journal record' }

    $credited = @{}
    $coverage = @{}
    for ($case = 0; $case -lt 11; $case++) { $coverage[$case] = @{} }
    foreach ($line in Get-Records -Lines $Lines -Prefix 'CREDIT') {
        $map = Convert-ToFieldMap -Line $line -Prefix 'CREDIT'
        if ((Get-RequiredField $map 'lane' 'chain credit') -cne 'PRIMARY' -or
            (Get-RequiredField $map 'chain' 'chain credit') -cne 'PASS') {
            throw 'chain credited non-PRIMARY or failed chain'
        }
        $file = Get-RequiredField $map 'raw' 'chain credit'
        $match = $script:creditedPrimaryRawRegex.Match($file)
        if (-not $match.Success) { throw "chain credited filename mismatch=$file" }
        $case = [int](Get-RequiredField $map 'case' 'chain credit')
        $start = [int](Get-RequiredField $map 'start' 'chain credit')
        $complete = [int](Get-RequiredField $map 'complete' 'chain credit')
        $next = [int](Get-RequiredField $map 'next_start' 'chain credit')
        if ($case -ne [int]$match.Groups['case'].Value -or $start -ne [int]$match.Groups['start'].Value -or
            $complete -le 0 -or $next -ne ($start + $complete) -or $next -gt $script:widths[$case]) {
            throw "chain credit prefix mismatch file=$file"
        }
        $elapsed = Convert-ToDecimal (Get-RequiredField $map 'elapsed_s' 'chain credit') 'chain elapsed_s'
        $wall = Convert-ToDecimal (Get-RequiredField $map 'wall_s' 'chain credit') 'chain wall_s'
        if ($elapsed -lt 0 -or $wall -lt 0 -or $elapsed -ge 600 -or $wall -ge 600 -or
            [Math]::Abs([double]($elapsed - $wall)) -gt 5.0) {
            throw "chain credit timing fence failed file=$file elapsed=$elapsed wall=$wall"
        }
        for ($root = $start; $root -lt $next; $root++) {
            if ($coverage[$case].ContainsKey($root)) { throw "overlapping chain credit case=$case root=$root" }
            $coverage[$case][$root] = $true
        }
        $key = $file.ToLowerInvariant()
        if ($credited.ContainsKey($key)) { throw "duplicate chain credited raw=$file" }
        $meta = [regex]::Replace($file, '_RAW[.]log$', '_META_RAW.log')
        $cargoExit = [regex]::Replace($file, '_RAW[.]log$', '_CARGO_EXIT_RAW.log')
        $rawBytes = [int64](Get-RequiredField $map 'raw_bytes' 'chain credit')
        $rawSha = Get-RequiredField $map 'raw_sha256' 'chain credit'
        $metaBytes = [int64](Get-RequiredField $map 'meta_bytes' 'chain credit')
        $metaSha = Get-RequiredField $map 'meta_sha256' 'chain credit'
        $cargoExitBytes = [int64](Get-RequiredField $map 'cargo_exit_bytes' 'chain credit')
        $cargoExitSha = Get-RequiredField $map 'cargo_exit_sha256' 'chain credit'
        if ($rawBytes -le 0 -or $metaBytes -le 0 -or $cargoExitBytes -le 0 -or
            $rawSha -cnotmatch '^[0-9A-F]{64}$' -or $metaSha -cnotmatch '^[0-9A-F]{64}$' -or
            $cargoExitSha -cnotmatch '^[0-9A-F]{64}$') {
            throw "chain credit artifact identity malformed file=$file"
        }
        $credited[$key] = [pscustomobject]@{
            File = $file
            Case = $case
            Start = $start
            Result = Get-RequiredField $map 'shard_result' 'chain credit'
            RawBytes = $rawBytes
            RawSha256 = $rawSha
            Meta = $meta
            MetaBytes = $metaBytes
            MetaSha256 = $metaSha
            CargoExit = $cargoExit
            CargoExitBytes = $cargoExitBytes
            CargoExitSha256 = $cargoExitSha
        }
    }
    if ($credited.Count -eq 0) { throw 'chain has no credited PRIMARY invocations' }
    if ([int](Get-RequiredField $header 'credited_invocations' 'chain header') -ne $credited.Count -or
        [int](Get-RequiredField $header 'discovered_meta' 'chain header') -ne $credited.Count) {
        throw 'chain header credit/META counts do not match reconstructed credit map'
    }
    for ($case = 0; $case -lt 11; $case++) {
        if ($coverage[$case].Count -ne $script:widths[$case]) {
            throw "chain coverage is not complete case=$case roots=$($coverage[$case].Count)"
        }
        for ($root = 0; $root -lt $script:widths[$case]; $root++) {
            if (-not $coverage[$case].ContainsKey($root)) { throw "chain coverage gap case=$case root=$root" }
        }
    }
    if (@($Lines | Where-Object { $_.StartsWith('UNMATCHED_RUN ', [StringComparison]::Ordinal) }).Count -ne 0) {
        throw 'chain output contains unmatched/open RUN'
    }
    $summaries = @(Get-Records -Lines $Lines -Prefix 'LANE_SUMMARY')
    if ($summaries.Count -ne 3) { throw 'chain lane summary count is not 3' }
    $seenSummaries = @{}
    foreach ($line in $summaries) {
        $map = Convert-ToFieldMap -Line $line -Prefix 'LANE_SUMMARY'
        $lane = Get-RequiredField $map 'lane' 'chain lane summary'
        if ($seenSummaries.ContainsKey($lane)) { throw "duplicate chain lane summary=$lane" }
        $seenSummaries[$lane] = $true
        if ($lane -ceq 'PRIMARY') {
            if ((Get-RequiredField $map 'credited_roots' 'primary summary') -cne '3648' -or
                (Get-RequiredField $map 'complete_cases' 'primary summary') -cne '11' -or
                (Get-RequiredField $map 'queue_done' 'primary summary') -cne 'true' -or
                (Get-RequiredField $map 'expected_roots' 'primary summary') -cne '3648' -or
                (Get-RequiredField $map 'result' 'primary summary') -cne 'COMPLETE') {
                throw 'chain PRIMARY summary is not COMPLETE'
            }
        }
        elseif ($lane -ceq 'D6_OFF' -or $lane -ceq 'SECOND_TT') {
            if ((Get-RequiredField $map 'credited_roots' 'replica summary') -cne '0' -or
                (Get-RequiredField $map 'complete_cases' 'replica summary') -cne '0' -or
                (Get-RequiredField $map 'queue_done' 'replica summary') -cne 'false' -or
                (Get-RequiredField $map 'expected_roots' 'replica summary') -cne '3648' -or
                (Get-RequiredField $map 'result' 'replica summary') -cne 'PARTIAL') {
                throw "chain replica summary is not empty/partial lane=$lane"
            }
        }
        else { throw "unexpected chain summary lane=$lane" }
    }
    foreach ($lane in @('PRIMARY', 'D6_OFF', 'SECOND_TT')) {
        if (-not $seenSummaries.ContainsKey($lane)) { throw "missing chain lane summary=$lane" }
    }
    $doneLine = Get-OneRecord -Lines $Lines -Prefix 'DOM_B2_D4_CHAIN_AUDIT_DONE'
    if ($Lines[$Lines.Count - 1] -cne $doneLine) { throw 'chain DONE is not terminal' }
    $done = Convert-ToFieldMap -Line $doneLine -Prefix 'DOM_B2_D4_CHAIN_AUDIT_DONE'
    if ((Get-RequiredField $done 'mode' 'chain done') -cne 'PARTIAL' -or
        [int](Get-RequiredField $done 'credited_invocations' 'chain done') -ne $credited.Count -or
        [int](Get-RequiredField $done 'credited_meta' 'chain done') -ne $credited.Count -or
        (Get-RequiredField $done 'open_uncredited_runs' 'chain done') -cne '0' -or
        [int](Get-RequiredField $done 'invocation_intervals' 'chain done') -ne $credited.Count -or
        (Get-RequiredField $done 'chronology' 'chain done') -cne 'PASS' -or
        (Get-RequiredField $done 'final_fence' 'chain done') -cne 'NOT_REQUESTED' -or
        (Get-RequiredField $done 'race_check' 'chain done') -cne 'PASS' -or
        (Get-RequiredField $done 'result' 'chain done') -cne 'PASS') {
        throw 'chain terminal footer mismatch'
    }
    return [pscustomobject]@{ Credited = $credited; Lines = $Lines.Count }
}

function Compare-PrimaryJoin {
    param(
        [Parameter(Mandatory = $true)][hashtable] $Analyzer,
        [Parameter(Mandatory = $true)][hashtable] $Chain
    )
    if ($Analyzer.Count -ne $Chain.Count) {
        throw "analyzer/chain credited count mismatch analyzer=$($Analyzer.Count) chain=$($Chain.Count)"
    }
    $joined = @()
    foreach ($key in @($Analyzer.Keys | Sort-Object)) {
        if (-not $Chain.ContainsKey($key)) { throw "chain missing analyzer raw=$($Analyzer[$key].File)" }
        $left = $Analyzer[$key]
        $right = $Chain[$key]
        foreach ($field in @('File', 'Case', 'Start', 'Result', 'RawBytes', 'RawSha256', 'Meta', 'MetaSha256')) {
            if ([string]$left.$field -cne [string]$right.$field) {
                throw "analyzer/chain join mismatch raw=$($left.File) field=$field analyzer=$($left.$field) chain=$($right.$field)"
            }
        }
        $joined += $left
    }
    foreach ($key in $Chain.Keys) {
        if (-not $Analyzer.ContainsKey($key)) { throw "analyzer missing chain raw=$($Chain[$key].File)" }
    }
    return $joined
}

function Assert-CreditedArtifactsCurrent {
    param(
        [Parameter(Mandatory = $true)][hashtable] $Credited,
        [string] $Root = $repoRoot
    )
    $snapshots = New-Object 'Collections.Generic.List[object]'
    foreach ($key in @($Credited.Keys | Sort-Object)) {
        $entry = $Credited[$key]
        if (-not $script:creditedPrimaryRawRegex.IsMatch($entry.File)) {
            throw "credited artifact raw filename is not canonical: $($entry.File)"
        }
        $expectedMeta = [regex]::Replace($entry.File, '_RAW[.]log$', '_META_RAW.log')
        $expectedExit = [regex]::Replace($entry.File, '_RAW[.]log$', '_CARGO_EXIT_RAW.log')
        if ($entry.Meta -cne $expectedMeta -or $entry.CargoExit -cne $expectedExit) {
            throw "credited sidecar derivation mismatch raw=$($entry.File) meta=$($entry.Meta) exit=$($entry.CargoExit)"
        }
        $snapshots.Add((Assert-FileIdentity `
            -RelativePath $entry.File `
            -ExpectedBytes $entry.RawBytes `
            -ExpectedSha256 $entry.RawSha256 `
            -Root $Root))
        $snapshots.Add((Assert-FileIdentity `
            -RelativePath $entry.Meta `
            -ExpectedBytes $entry.MetaBytes `
            -ExpectedSha256 $entry.MetaSha256 `
            -Root $Root))
        $snapshots.Add((Assert-FileIdentity `
            -RelativePath $entry.CargoExit `
            -ExpectedBytes $entry.CargoExitBytes `
            -ExpectedSha256 $entry.CargoExitSha256 `
            -Root $Root))
    }
    if ($snapshots.Count -ne (3 * $Credited.Count)) {
        throw 'credited raw/META/Cargo-exit snapshot count mismatch'
    }
    return @($snapshots | Sort-Object -Property Path)
}

function Write-IsolatedCaptureItem {
    param(
        [AllowNull()][psobject] $Item,
        [Parameter(Mandatory = $true)][IO.StreamWriter] $Writer,
        [Parameter(Mandatory = $true)][string] $Sentinel,
        [Parameter(Mandatory = $true)][ref] $SentinelSeen,
        [Parameter(Mandatory = $true)][ref] $CommandSucceeded,
        [Parameter(Mandatory = $true)][ref] $OutputObjects
    )
    $marker = if ($null -eq $Item) { $null } else {
        $Item.PSObject.Properties[$script:captureSentinelProperty]
    }
    if ($null -ne $marker -and [string]$marker.Value -ceq $Sentinel) {
        if ($SentinelSeen.Value) { throw 'isolated capture emitted duplicate internal sentinel' }
        $successProperty = $Item.PSObject.Properties['CommandSucceeded']
        if ($null -eq $successProperty -or $successProperty.Value -isnot [bool]) {
            throw 'isolated capture sentinel lacks typed CommandSucceeded'
        }
        $SentinelSeen.Value = $true
        $CommandSucceeded.Value = [bool]$successProperty.Value
        return
    }
    if ($SentinelSeen.Value) { throw 'isolated capture received producer output after internal sentinel' }
    Write-NormalizedObject -Writer $Writer -Value $Item
    $OutputObjects.Value++
}

function Invoke-IsolatedPowerShellCapture {
    param(
        [Parameter(Mandatory = $true)][string] $OutputPath,
        [Parameter(Mandatory = $true)][hashtable] $Variables,
        [Parameter(Mandatory = $true)][string] $InvocationScript
    )
    $writer = New-LfWriter -LiteralPath $OutputPath
    $captureHost = [DomB2D4CaptureHost]::new(
        [Globalization.CultureInfo]::CurrentCulture,
        [Globalization.CultureInfo]::CurrentUICulture
    )
    $runspace = $null
    $powerShell = $null
    $input = $null
    $output = $null
    $sentinel = [Guid]::NewGuid().ToString('N')
    $sentinelSeen = $false
    $commandSucceeded = $false
    $outputObjects = 0
    $invokeFailure = $null
    $state = 'NOT_STARTED'
    $errorCount = 0
    $next = 0
    try {
        $runspace = [RunspaceFactory]::CreateRunspace($captureHost)
        $runspace.Open()
        foreach ($name in $Variables.Keys) {
            if ([string]$name -cnotmatch '^domCapture[A-Za-z0-9]+$' -or
                [string]$name -ceq 'domCaptureSentinel') {
                throw "unsafe isolated capture variable name=$name"
            }
            $runspace.SessionStateProxy.SetVariable([string]$name, $Variables[$name])
        }
        $runspace.SessionStateProxy.SetVariable('domCaptureSentinel', $sentinel)
        $powerShell = [PowerShell]::Create()
        $powerShell.Runspace = $runspace
        $payload = $InvocationScript + "`n" +
            '$domCaptureCommandSucceeded = $?' + "`n" +
            "[pscustomobject]@{ $($script:captureSentinelProperty) = `$domCaptureSentinel; CommandSucceeded = [bool]`$domCaptureCommandSucceeded }"
        [void]$powerShell.AddScript($payload, $false)

        $input = [Management.Automation.PSDataCollection[psobject]]::new()
        $output = [Management.Automation.PSDataCollection[psobject]]::new()
        $input.Complete()
        $begin = @([PowerShell].GetMethods() | Where-Object {
            $_.Name -ceq 'BeginInvoke' -and $_.IsGenericMethodDefinition -and
            $_.GetGenericArguments().Count -eq 2 -and $_.GetParameters().Count -eq 2
        })
        if ($begin.Count -ne 1) { throw "could not resolve typed BeginInvoke overload count=$($begin.Count)" }
        $typedBegin = $begin[0].MakeGenericMethod([psobject], [psobject])
        $async = $typedBegin.Invoke($powerShell, @($input, $output))
        while (-not $async.IsCompleted) {
            while ($next -lt $output.Count) {
                Write-IsolatedCaptureItem `
                    -Item $output[$next] `
                    -Writer $writer `
                    -Sentinel $sentinel `
                    -SentinelSeen ([ref]$sentinelSeen) `
                    -CommandSucceeded ([ref]$commandSucceeded) `
                    -OutputObjects ([ref]$outputObjects)
                $next++
            }
            [Threading.Thread]::Sleep(15)
        }
        [void]$powerShell.EndInvoke($async)
        while ($next -lt $output.Count) {
            Write-IsolatedCaptureItem `
                -Item $output[$next] `
                -Writer $writer `
                -Sentinel $sentinel `
                -SentinelSeen ([ref]$sentinelSeen) `
                -CommandSucceeded ([ref]$commandSucceeded) `
                -OutputObjects ([ref]$outputObjects)
            $next++
        }
        $state = [string]$powerShell.InvocationStateInfo.State
    }
    catch {
        $invokeFailure = $_
        if ($null -ne $powerShell) {
            try { $powerShell.Stop() } catch {}
            $state = [string]$powerShell.InvocationStateInfo.State
        }
    }
    finally {
        if ($null -ne $output) {
            while ($next -lt $output.Count) {
                try {
                    Write-IsolatedCaptureItem `
                        -Item $output[$next] `
                        -Writer $writer `
                        -Sentinel $sentinel `
                        -SentinelSeen ([ref]$sentinelSeen) `
                        -CommandSucceeded ([ref]$commandSucceeded) `
                        -OutputObjects ([ref]$outputObjects)
                    $next++
                }
                catch {
                    if ($null -eq $invokeFailure) { $invokeFailure = $_ }
                    break
                }
            }
        }
        if ($null -ne $powerShell) {
            $errorCount = $powerShell.Streams.Error.Count
            foreach ($errorRecord in $powerShell.Streams.Error) {
                Write-NormalizedObject -Writer $writer -Value $errorRecord
                $outputObjects++
            }
            if ($null -ne $invokeFailure -and $errorCount -eq 0) {
                $failureRecord = if ($null -ne $powerShell.InvocationStateInfo.Reason) {
                    $powerShell.InvocationStateInfo.Reason
                }
                else { $invokeFailure }
                Write-NormalizedObject -Writer $writer -Value $failureRecord
                $outputObjects++
                $errorCount++
            }
        }
        $writer.Dispose()
        if ($null -ne $input) { $input.Dispose() }
        if ($null -ne $output) { $output.Dispose() }
        if ($null -ne $powerShell) { $powerShell.Dispose() }
        if ($null -ne $runspace) { $runspace.Dispose() }
    }
    $success = (
        $null -eq $invokeFailure -and $state -ceq 'Completed' -and
        $sentinelSeen -and $commandSucceeded -and -not $captureHost.ShouldExit -and
        $errorCount -eq 0
    )
    return [pscustomobject]@{
        Success = [bool]$success
        State = $state
        SentinelSeen = [bool]$sentinelSeen
        CommandSucceeded = [bool]$commandSucceeded
        ExplicitExit = [bool]$captureHost.ShouldExit
        ExitCode = if ($captureHost.ShouldExit) { [int]$captureHost.ExitCode } else { $null }
        ErrorCount = [int]$errorCount
        OutputObjects = [int]$outputObjects
        Failure = $invokeFailure
    }
}

function Invoke-AnalyzerCapture {
    param(
        [Parameter(Mandatory = $true)][string] $OutputPath,
        [Parameter(Mandatory = $true)][string[]] $ShardPaths,
        [Parameter(Mandatory = $true)][string[]] $Mappings
    )
    $variables = @{
        domCaptureAnalyzerPath = Join-Path $repoRoot 'scripts/dom_b2_d4_aggregate.ps1'
        domCaptureCensusPath = Join-Path $repoRoot 'DOM_B2_D4_CENSUS_RAW.log'
        domCaptureShardPaths = [string[]]$ShardPaths
        domCaptureMappings = [string[]]$Mappings
    }
    return Invoke-IsolatedPowerShellCapture `
        -OutputPath $OutputPath `
        -Variables $variables `
        -InvocationScript $script:analyzerInvocationScript
}

function Invoke-ChainCapture {
    param(
        [Parameter(Mandatory = $true)][string] $OutputPath,
        [Parameter(Mandatory = $true)][string[]] $JournalPaths
    )
    $variables = @{
        domCaptureChainPath = Join-Path $repoRoot 'scripts/dom_b2_d4_chain_audit.ps1'
        domCaptureJournalPaths = [string[]]$JournalPaths
    }
    return Invoke-IsolatedPowerShellCapture `
        -OutputPath $OutputPath `
        -Variables $variables `
        -InvocationScript $script:chainInvocationScript
}

function Convert-ErrorToToken {
    param([Parameter(Mandatory = $true)][string] $Text)
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes($Text)
    return [Convert]::ToBase64String($bytes)
}

function Write-PreflightBlocker {
    param(
        [Parameter(Mandatory = $true)][string] $Stage,
        [Parameter(Mandatory = $true)][Management.Automation.ErrorRecord] $ErrorRecord,
        [Parameter(Mandatory = $true)][int[]] $WrapperPids,
        [string] $Root = $repoRoot
    )
    $capturedUtc = [DateTime]::UtcNow
    $stamp = $capturedUtc.ToString('yyyyMMddTHHmmssfffffffZ', $invariant)
    $nonce = [Guid]::NewGuid().ToString('N')
    $name = "DOM_B2_D4_PRIMARY_CAPTURE_PREFLIGHT_BLOCKER_${stamp}_PID$PID`_${nonce}_RAW.log"
    $path = Join-Path $Root $name
    $writer = New-LfWriter -LiteralPath $path
    try {
        $writer.WriteLine("DOM_B2_D4_PRIMARY_CAPTURE_PREFLIGHT_BLOCKER version=$($script:version) captured_utc=$($capturedUtc.ToString('o')) stage=$Stage result=BLOCKED")
        $writer.WriteLine("SCOPE analyzer_started=false chain_started=false cargo_invocations=0 canonical_success_wrapper_consumed=false partial_outputs_preserved=true")
        $writer.WriteLine("INTENDED_OUTPUT analysis=DOM_B2_D4_PRIMARY_ANALYSIS_RAW.log chain=DOM_B2_D4_CHAIN_AUDIT_PRIMARY_RAW.log wrapper=DOM_B2_D4_PRIMARY_CAPTURE_RUN_RAW.log create_new=true")
        $writer.WriteLine("WRAPPER_PID_GATE required_pids=$($WrapperPids -join ',')")
        $scriptItem = Get-Item -LiteralPath $scriptPath
        $writer.WriteLine("CAPTURE_SCRIPT path=$(Get-RelativePath -FullPath $scriptPath) bytes=$($scriptItem.Length) sha256=$(Get-Sha256 -LiteralPath $scriptPath)")
        $writer.WriteLine("BLOCKER_ERROR error_utf8_b64=$(Convert-ErrorToToken -Text $ErrorRecord.Exception.ToString())")
        $writer.WriteLine('DOM_B2_D4_PRIMARY_CAPTURE_PREFLIGHT_BLOCKER_DONE disposition=REFUSED_RETRY_AFTER_GATE_CORRECTION result=BLOCKED')
    }
    finally { $writer.Dispose() }
    $identity = Assert-PureLfUtf8File -LiteralPath $path
    return [pscustomobject]@{
        Path = $name
        Bytes = $identity.Bytes
        Sha256 = $identity.Sha256
    }
}

function Invoke-LiveCapture {
    Set-Location -LiteralPath $repoRoot
    $analysisPath = Join-Path $repoRoot 'DOM_B2_D4_PRIMARY_ANALYSIS_RAW.log'
    $chainPath = Join-Path $repoRoot 'DOM_B2_D4_CHAIN_AUDIT_PRIMARY_RAW.log'
    $wrapperPath = Join-Path $repoRoot 'DOM_B2_D4_PRIMARY_CAPTURE_RUN_RAW.log'
    $wrapperPids = @($script:knownPrimaryWrapperPid) + @($AdditionalPrimaryWrapperPid)
    $wrapperPids = @($wrapperPids | Sort-Object -Unique)
    $preflightStage = 'output_absence'
    try {
        foreach ($path in @($analysisPath, $chainPath, $wrapperPath)) {
            if (Test-Path -LiteralPath $path) { throw "capture output already exists: $(Split-Path -Leaf $path)" }
        }
        $preflightStage = 'wrapper_pid_arguments'
        if (@($wrapperPids | Where-Object { $_ -le 0 }).Count -ne 0) { throw 'wrapper PIDs must be positive' }

        $preflightStage = 'support_identity'
        $supportBefore = @(Get-SupportSnapshot)
        $preflightStage = 'replica_zero'
        [void](Assert-ZeroReplicaArtifacts)
        $preflightStage = 'terminal_process_gate'
        $processGateBefore = Assert-TerminalProcessGate -WrapperPids $wrapperPids
        $preflightStage = 'primary_journal_terminal'
        $journals = @(Get-PrimaryJournals -Root $repoRoot)
        $terminal = Assert-TerminalJournals -Journals $journals
        $preflightStage = 'primary_shard_census'
        $shards = @(Get-CanonicalPrimaryShards -Root $repoRoot)
        $journalBefore = @(Get-FileSnapshots -Files $journals)
        $shardBefore = @(Get-FileSnapshots -Files $shards)

        $preflightStage = 'approved_runner_mapping'
        $approvedMappings = @(
            "d6=true,tt_bytes=536870912,runner_sha256=$($script:primaryRunnerSha)",
            "d6=false,tt_bytes=536870912,runner_sha256=$($script:replicaRunnerSha)",
            "d6=true,tt_bytes=268435456,runner_sha256=$($script:replicaRunnerSha)"
        )
        $expectedPrimaryMap = "d6=true,tt_bytes=536870912,runner_sha256=$($script:primaryRunnerSha)"
        if ($approvedMappings[0] -cne $expectedPrimaryMap) { throw 'internal PRIMARY mapping transcription mismatch' }
    }
    catch {
        try {
            $blocker = Write-PreflightBlocker `
                -Stage $preflightStage `
                -ErrorRecord $_ `
                -WrapperPids $wrapperPids
            throw "PRIMARY checkpoint preflight refused stage=$preflightStage blocker=$($blocker.Path) blocker_sha256=$($blocker.Sha256)"
        }
        catch {
            if ($_.Exception.Message.StartsWith('PRIMARY checkpoint preflight refused ', [StringComparison]::Ordinal)) {
                throw
            }
            throw "PRIMARY checkpoint preflight refused stage=$preflightStage and blocker emission failed: $($_.Exception.Message)"
        }
    }

    $analyzerCommand = '& .\scripts\dom_b2_d4_aggregate.ps1 -CensusPath .\DOM_B2_D4_CENSUS_RAW.log -ShardPath $primaryShardPaths -ApprovedRunnerMapping $approvedMappings'
    $chainCommand = '& .\scripts\dom_b2_d4_chain_audit.ps1 -PrimaryJournalPath $primaryJournals'

    $stage = 'wrapper_create'
    try { $wrapper = New-LfWriter -LiteralPath $wrapperPath }
    catch {
        try {
            $blocker = Write-PreflightBlocker `
                -Stage $stage `
                -ErrorRecord $_ `
                -WrapperPids $wrapperPids
            throw "PRIMARY checkpoint wrapper create refused blocker=$($blocker.Path) blocker_sha256=$($blocker.Sha256)"
        }
        catch {
            if ($_.Exception.Message.StartsWith('PRIMARY checkpoint wrapper create refused ', [StringComparison]::Ordinal)) {
                throw
            }
            throw "PRIMARY checkpoint wrapper create failed and blocker emission failed: $($_.Exception.Message)"
        }
    }
    $stage = 'wrapper_setup'
    try {
        $wrapper.WriteLine("DOM_B2_D4_PRIMARY_CAPTURE version=$($script:version) captured_utc=$([DateTime]::UtcNow.ToString('o')) code_id=$($script:codeId) cargo_invocations=0 create_new=true encoding=UTF8_NO_BOM line_endings=LF")
        $wrapper.WriteLine("TERMINAL_GATE result=PASS wrapper_pids=$($wrapperPids -join ',') live_pid_tree=0 cargo_count=$($processGateBefore.CargoCount) primary_lock=false replica_artifacts=0 terminal_journal=$($terminal.Journal.Name) terminal_invocations=$($terminal.Invocations)")
        $selfRelative = Get-RelativePath -FullPath $scriptPath
        $selfBefore = @($supportBefore | Where-Object { $_.Path -ceq $selfRelative })
        if ($selfBefore.Count -ne 1) { throw 'capture script is absent/duplicated in support pre-snapshot' }
        $wrapper.WriteLine("CAPTURE_SCRIPT_PRE path=$($selfBefore[0].Path) bytes=$($selfBefore[0].Bytes) sha256=$($selfBefore[0].Sha256) result=PASS")
        foreach ($entry in $supportBefore) {
            $wrapper.WriteLine("SUPPORT_PRE path=$($entry.Path) bytes=$($entry.Bytes) sha256=$($entry.Sha256) result=PASS")
        }
        for ($index = 0; $index -lt $shardBefore.Count; $index++) {
            $entry = $shardBefore[$index]
            $wrapper.WriteLine("SHARD_INPUT index=$index path=$($entry.Path) bytes=$($entry.Bytes) sha256=$($entry.Sha256)")
        }
        for ($index = 0; $index -lt $journalBefore.Count; $index++) {
            $entry = $journalBefore[$index]
            $wrapper.WriteLine("JOURNAL_INPUT index=$index path=$($entry.Path) bytes=$($entry.Bytes) sha256=$($entry.Sha256)")
        }
        for ($index = 0; $index -lt $approvedMappings.Count; $index++) {
            $wrapper.WriteLine("APPROVED_MAPPING index=$index value=$($approvedMappings[$index])")
        }

        $stage = 'analyzer_capture'
        [void](Assert-ZeroReplicaArtifacts)
        $analyzerProcessGate = Assert-TerminalProcessGate -WrapperPids $wrapperPids
        $wrapper.WriteLine("PRE_TOOL_GATE tool=analyzer live_pid_tree=$($analyzerProcessGate.LiveTreeCount) cargo_count=$($analyzerProcessGate.CargoCount) primary_lock=false replica_artifacts=0 result=PASS")
        $wrapper.WriteLine("COMMAND tool=analyzer execution=IN_PROCESS_ISOLATED_RUNSPACE arrays=TYPED_SAME_PROCESS shard_paths=$($shards.Count) output=DOM_B2_D4_PRIMARY_ANALYSIS_RAW.log script_sha256=$($script:analyzerSha) frozen_command_utf8_b64=$(Convert-ErrorToToken -Text $analyzerCommand) executed_bootstrap_utf8_b64=$(Convert-ErrorToToken -Text $script:analyzerInvocationScript)")
        $analysisInvocation = Invoke-AnalyzerCapture `
            -OutputPath $analysisPath `
            -ShardPaths @($shards | ForEach-Object FullName) `
            -Mappings $approvedMappings
        $analysisExitCode = if ($analysisInvocation.ExplicitExit) { [string]$analysisInvocation.ExitCode } else { 'NONE' }
        $wrapper.WriteLine("IN_PROCESS_RESULT tool=analyzer success=$($analysisInvocation.Success.ToString().ToLowerInvariant()) state=$($analysisInvocation.State) explicit_exit=$($analysisInvocation.ExplicitExit.ToString().ToLowerInvariant()) exit_code=$analysisExitCode success_sentinel=$($analysisInvocation.SentinelSeen.ToString().ToLowerInvariant()) command_success=$($analysisInvocation.CommandSucceeded.ToString().ToLowerInvariant()) error_records=$($analysisInvocation.ErrorCount) emitted_objects=$($analysisInvocation.OutputObjects) outer_host_continued=true")
        if (-not $analysisInvocation.Success) {
            $failureDetail = if ($null -eq $analysisInvocation.Failure) { 'NONE' } else {
                Convert-ErrorToToken -Text $analysisInvocation.Failure.Exception.ToString()
            }
            throw "analyzer isolated invocation failed detail_utf8_b64=$failureDetail"
        }
        $analysisFile = Assert-PureLfUtf8File -LiteralPath $analysisPath
        $wrapper.WriteLine("OUTPUT tool=analyzer path=DOM_B2_D4_PRIMARY_ANALYSIS_RAW.log bytes=$($analysisFile.Bytes) sha256=$($analysisFile.Sha256) encoding=PASS")
        $analysisValidation = Validate-AnalyzerOutput `
            -Lines @(Read-PureLfLines -LiteralPath $analysisPath) `
            -ExpectedShardNames @($shards | ForEach-Object Name)
        $wrapper.WriteLine("VALIDATION tool=analyzer result=PASS primary_matrix=$($analysisValidation.PrimaryMatrix) credited_shards=$($analysisValidation.Credited.Count) replicas=EMPTY external_chain=PENDING")

        $stage = 'chain_journal_census'
        $chainJournals = @(Get-PrimaryJournals -Root $repoRoot)
        [void](Assert-TerminalJournals -Journals $chainJournals)
        $chainJournalSnapshot = @(Get-FileSnapshots -Files $chainJournals)
        Compare-Snapshots -Before $journalBefore -After $chainJournalSnapshot -Context 'pre-chain journal'
        for ($index = 0; $index -lt $chainJournalSnapshot.Count; $index++) {
            $entry = $chainJournalSnapshot[$index]
            $wrapper.WriteLine("CHAIN_JOURNAL_INPUT index=$index path=$($entry.Path) bytes=$($entry.Bytes) sha256=$($entry.Sha256)")
        }

        $stage = 'chain_capture'
        [void](Assert-ZeroReplicaArtifacts)
        $chainProcessGate = Assert-TerminalProcessGate -WrapperPids $wrapperPids
        $wrapper.WriteLine("PRE_TOOL_GATE tool=chain_auditor live_pid_tree=$($chainProcessGate.LiveTreeCount) cargo_count=$($chainProcessGate.CargoCount) primary_lock=false replica_artifacts=0 result=PASS")
        $wrapper.WriteLine("COMMAND tool=chain_auditor execution=IN_PROCESS_ISOLATED_RUNSPACE arrays=TYPED_SAME_PROCESS primary_journals=$($chainJournals.Count) final_switch=false output=DOM_B2_D4_CHAIN_AUDIT_PRIMARY_RAW.log script_sha256=$($script:chainSha) frozen_command_utf8_b64=$(Convert-ErrorToToken -Text $chainCommand) executed_bootstrap_utf8_b64=$(Convert-ErrorToToken -Text $script:chainInvocationScript)")
        $chainInvocation = Invoke-ChainCapture `
            -OutputPath $chainPath `
            -JournalPaths @($chainJournals | ForEach-Object FullName)
        $chainExitCode = if ($chainInvocation.ExplicitExit) { [string]$chainInvocation.ExitCode } else { 'NONE' }
        $wrapper.WriteLine("IN_PROCESS_RESULT tool=chain_auditor success=$($chainInvocation.Success.ToString().ToLowerInvariant()) state=$($chainInvocation.State) explicit_exit=$($chainInvocation.ExplicitExit.ToString().ToLowerInvariant()) exit_code=$chainExitCode success_sentinel=$($chainInvocation.SentinelSeen.ToString().ToLowerInvariant()) command_success=$($chainInvocation.CommandSucceeded.ToString().ToLowerInvariant()) error_records=$($chainInvocation.ErrorCount) emitted_objects=$($chainInvocation.OutputObjects) outer_host_continued=true")
        if (-not $chainInvocation.Success) {
            $failureDetail = if ($null -eq $chainInvocation.Failure) { 'NONE' } else {
                Convert-ErrorToToken -Text $chainInvocation.Failure.Exception.ToString()
            }
            throw "chain auditor isolated invocation failed detail_utf8_b64=$failureDetail"
        }
        $chainFile = Assert-PureLfUtf8File -LiteralPath $chainPath
        $wrapper.WriteLine("OUTPUT tool=chain_auditor path=DOM_B2_D4_CHAIN_AUDIT_PRIMARY_RAW.log bytes=$($chainFile.Bytes) sha256=$($chainFile.Sha256) encoding=PASS")
        $chainValidation = Validate-ChainOutput `
            -Lines @(Read-PureLfLines -LiteralPath $chainPath) `
            -ExpectedJournals $chainJournals
        $wrapper.WriteLine("VALIDATION tool=chain_auditor result=PASS credited_invocations=$($chainValidation.Credited.Count) primary_roots=3648 primary_cases=11 replicas=EMPTY mode=PARTIAL")
        $creditedArtifactBefore = @(Assert-CreditedArtifactsCurrent -Credited $chainValidation.Credited)
        foreach ($entry in $creditedArtifactBefore) {
            $wrapper.WriteLine("CREDIT_ARTIFACT_VALIDATED path=$($entry.Path) bytes=$($entry.Bytes) sha256=$($entry.Sha256) result=PASS")
        }

        $stage = 'cross_join'
        $joined = @(Compare-PrimaryJoin -Analyzer $analysisValidation.Credited -Chain $chainValidation.Credited)
        foreach ($entry in $joined) {
            $wrapper.WriteLine("PRIMARY_JOIN raw=$($entry.File) case=$($entry.Case) start=$($entry.Start) result=$($entry.Result) raw_bytes=$($entry.RawBytes) raw_sha256=$($entry.RawSha256) meta=$($entry.Meta) meta_sha256=$($entry.MetaSha256) result_join=PASS")
        }
        $wrapper.WriteLine("PRIMARY_JOIN_SUMMARY analyzer=$($analysisValidation.Credited.Count) chain=$($chainValidation.Credited.Count) joined=$($joined.Count) mismatches=0 result=PASS")

        $stage = 'post_capture_fences'
        [void](Assert-ZeroReplicaArtifacts)
        $processGateAfter = Assert-TerminalProcessGate -WrapperPids $wrapperPids
        $supportAfter = @(Get-SupportSnapshot)
        $journalsAfterFiles = @(Get-PrimaryJournals -Root $repoRoot)
        [void](Assert-TerminalJournals -Journals $journalsAfterFiles)
        $shardsAfterFiles = @(Get-CanonicalPrimaryShards -Root $repoRoot)
        $journalAfter = @(Get-FileSnapshots -Files $journalsAfterFiles)
        $shardAfter = @(Get-FileSnapshots -Files $shardsAfterFiles)
        $creditedArtifactAfter = @(Assert-CreditedArtifactsCurrent -Credited $chainValidation.Credited)
        Compare-Snapshots -Before $supportBefore -After $supportAfter -Context 'support'
        Compare-Snapshots -Before $journalBefore -After $journalAfter -Context 'journal'
        Compare-Snapshots -Before $shardBefore -After $shardAfter -Context 'shard'
        Compare-Snapshots -Before $creditedArtifactBefore -After $creditedArtifactAfter -Context 'credited raw/META/Cargo-exit'
        $selfAfter = @($supportAfter | Where-Object { $_.Path -ceq $selfRelative })
        if ($selfAfter.Count -ne 1) { throw 'capture script is absent/duplicated in support post-snapshot' }

        $analysisFinal = Assert-PureLfUtf8File -LiteralPath $analysisPath
        if ($analysisFinal.Bytes -ne $analysisFile.Bytes -or $analysisFinal.Sha256 -cne $analysisFile.Sha256) {
            throw 'analyzer output changed after semantic validation'
        }
        $chainFinal = Assert-PureLfUtf8File -LiteralPath $chainPath
        if ($chainFinal.Bytes -ne $chainFile.Bytes -or $chainFinal.Sha256 -cne $chainFile.Sha256) {
            throw 'chain output changed after semantic validation'
        }
        foreach ($entry in $supportAfter) {
            $wrapper.WriteLine("SUPPORT_POST path=$($entry.Path) bytes=$($entry.Bytes) sha256=$($entry.Sha256) result=PASS")
        }
        foreach ($entry in $shardAfter) {
            $wrapper.WriteLine("SHARD_POST path=$($entry.Path) bytes=$($entry.Bytes) sha256=$($entry.Sha256) result=PASS")
        }
        foreach ($entry in $journalAfter) {
            $wrapper.WriteLine("JOURNAL_POST path=$($entry.Path) bytes=$($entry.Bytes) sha256=$($entry.Sha256) result=PASS")
        }
        $wrapper.WriteLine("CAPTURE_SCRIPT_POST path=$($selfAfter[0].Path) bytes=$($selfAfter[0].Bytes) sha256=$($selfAfter[0].Sha256) matches_pre=true result=PASS")
        foreach ($entry in $creditedArtifactAfter) {
            $wrapper.WriteLine("CREDIT_ARTIFACT_POST path=$($entry.Path) bytes=$($entry.Bytes) sha256=$($entry.Sha256) result=PASS")
        }
        $wrapper.WriteLine("OUTPUT_POST tool=analyzer path=DOM_B2_D4_PRIMARY_ANALYSIS_RAW.log bytes=$($analysisFinal.Bytes) sha256=$($analysisFinal.Sha256) semantic_snapshot=UNCHANGED result=PASS")
        $wrapper.WriteLine("OUTPUT_POST tool=chain_auditor path=DOM_B2_D4_CHAIN_AUDIT_PRIMARY_RAW.log bytes=$($chainFinal.Bytes) sha256=$($chainFinal.Sha256) semantic_snapshot=UNCHANGED result=PASS")
        $wrapper.WriteLine("POST_GATE result=PASS live_pid_tree=$($processGateAfter.LiveTreeCount) cargo_count=$($processGateAfter.CargoCount) primary_lock=false replica_artifacts=0 source_race=PASS shard_race=PASS journal_race=PASS credited_artifact_race=PASS output_race=PASS")
        $wrapper.WriteLine("DOM_B2_D4_PRIMARY_CAPTURE_DONE analysis_sha256=$($analysisFinal.Sha256) chain_sha256=$($chainFinal.Sha256) primary_roots=3648 primary_cases=11 replicas=0 checkpoint_verdict=NO_VERDICT result=PASS")
    }
    catch {
        $message = Convert-ErrorToToken -Text $_.Exception.ToString()
        $wrapper.WriteLine("DOM_B2_D4_PRIMARY_CAPTURE_FAIL stage=$stage error_utf8_b64=$message partial_outputs_preserved=true result=FAIL")
        throw
    }
    finally { $wrapper.Dispose() }
    [void](Assert-PureLfUtf8File -LiteralPath $wrapperPath)
}

function New-SyntheticAnalysisLines {
    $lines = New-Object 'Collections.Generic.List[string]'
    $lines.Add("DOM_B2_D4_ANALYSIS_SETUP census_sha256=$($script:censusSha) census_path=SYNTHETIC census_hard_bind=PASS shard_name_regex=SYNTHETIC shard_files=11 discovery_ignored=0 required_code_id=$($script:codeId) meta_version=1 meta_required_for_credit=true missing_meta_disposition=V3_UNTRUSTED_NO_META_EXCLUDED source_snapshot_sha256=$($script:sourceSnapshotSha) approval_configs=3 primary_config=$($script:primaryConfig) d6_off_config=$($script:d6OffConfig) second_tt_config=$($script:secondTtConfig)")
    $lines.Add("DOM_B2_D4_RUNNER_APPROVAL lane=PRIMARY config=$($script:primaryConfig) runner_sha256=$($script:primaryRunnerSha) source=EXPLICIT_MAPPING")
    $lines.Add("DOM_B2_D4_RUNNER_APPROVAL lane=D6_OFF config=$($script:d6OffConfig) runner_sha256=$($script:replicaRunnerSha) source=EXPLICIT_MAPPING")
    $lines.Add("DOM_B2_D4_RUNNER_APPROVAL lane=SECOND_TT config=$($script:secondTtConfig) runner_sha256=$($script:replicaRunnerSha) source=EXPLICIT_MAPPING")
    for ($case = 0; $case -lt 11; $case++) {
        $sha = ('{0:X2}' -f ($case + 1)) * 32
        $metaSha = ('{0:X2}' -f ($case + 33)) * 32
        $name = 'DOM_B2_D4_SHARD_C{0:D2}_S0000_A00_RAW.log' -f $case
        $meta = [regex]::Replace($name, '_RAW[.]log$', '_META_RAW.log')
        $lines.Add("DOM_B2_D4_CREDITED_SHARD_VALID file=$name sha256=$sha bytes=$($case + 100) meta=$meta meta_sha256=$metaSha runner_sha256=$($script:primaryRunnerSha) attempt=A00 case=$case start=0 end=$($script:widths[$case]) result=PASS code_id=$($script:codeId) lane=PRIMARY config=$($script:primaryConfig) role=PRIMARY_INCLUDED")
    }
    foreach ($lane in @('PRIMARY', 'D6_OFF', 'SECOND_TT')) {
        $config = if ($lane -ceq 'PRIMARY') { $script:primaryConfig } elseif ($lane -ceq 'D6_OFF') { $script:d6OffConfig } else { $script:secondTtConfig }
        $runner = if ($lane -ceq 'PRIMARY') { $script:primaryRunnerSha } else { $script:replicaRunnerSha }
        $files = if ($lane -ceq 'PRIMARY') { 11 } else { 0 }
        $records = if ($lane -ceq 'PRIMARY') { 3648 } else { 0 }
        $lines.Add("DOM_B2_D4_LANE_SET lane=$lane config=$config approval=SUPPLIED runner_sha256=$runner files=$files records=$records code_id=$($script:codeId) conflict_free=true")
        for ($case = 0; $case -lt 11; $case++) {
            if ($lane -ceq 'PRIMARY') {
                $lines.Add("DOM_B2_D4_LANE_CASE lane=$lane config=$config case=$case root_count=$($script:widths[$case]) fingerprint=$($script:fingerprints[$case]) unique_complete=$($script:widths[$case]) missing=0 incomplete_only=0 incomplete_rows=0 duplicate_complete=0 attempt_rows=$($script:widths[$case]) win=0 unknown=$($script:widths[$case]) loss=0 recurrence=UNKNOWN gate_status=UNKNOWN coverage=EXACT unique_nodes=1 attempt_nodes=1 attempt_wall_s=0.000001")
            }
            else {
                $lines.Add("DOM_B2_D4_LANE_CASE lane=$lane config=$config case=$case root_count=$($script:widths[$case]) fingerprint=$($script:fingerprints[$case]) unique_complete=0 missing=$($script:widths[$case]) incomplete_only=0 incomplete_rows=0 duplicate_complete=0 attempt_rows=0 win=0 unknown=0 loss=0 recurrence=INCOMPLETE gate_status=INCOMPLETE coverage=INCOMPLETE unique_nodes=0 attempt_nodes=0 attempt_wall_s=0.000000")
            }
        }
        if ($lane -ceq 'PRIMARY') {
            $lines.Add("DOM_B2_D4_LANE_MATRIX lane=$lane config=$config exact_cases=11 total_cases=11 unique_complete_roots=3648 total_roots=3648 missing_roots=0 incomplete_only_roots=0 duplicate_complete_rows=0 gapless=true conflict_free=true")
        }
        else {
            $lines.Add("DOM_B2_D4_LANE_MATRIX lane=$lane config=$config exact_cases=0 total_cases=11 unique_complete_roots=0 total_roots=3648 missing_roots=3648 incomplete_only_roots=0 duplicate_complete_rows=0 gapless=false conflict_free=true")
        }
    }
    $lines.Add("DOM_B2_D4_PRIMARY_SET files=11 code_id=$($script:codeId) config=$($script:primaryConfig) included_records=3648")
    $lines.Add("DOM_B2_D4_REPLICA_SET code_id=$($script:codeId) lane=D6_OFF config=$($script:d6OffConfig) approval=SUPPLIED files=0 records=0 unique_complete_roots=0 incomplete_only_roots=0 primary_coverage_effect=NONE")
    $lines.Add("DOM_B2_D4_REPLICA_SET code_id=$($script:codeId) lane=SECOND_TT config=$($script:secondTtConfig) approval=SUPPLIED files=0 records=0 unique_complete_roots=0 incomplete_only_roots=0 primary_coverage_effect=NONE")
    foreach ($name in @('K1_32F', 'K1_19B', 'K1_498A', 'K1_FD68')) {
        $lines.Add("DOM_B2_D4_COMPARISON name=$name split_case=0 h_cases=1 status=SATISFIED split=UNKNOWN split_rank=1 best_h=UNKNOWN best_h_rank=1 relation=split_rank_le_best_h_rank replication_required=true")
    }
    $lines.Add('DOM_B2_D4_CONTROL name=D7_HIT_ANY_EQUALITY cases=9,10 status=MATCH case9=UNKNOWN case10=UNKNOWN replication_required=true')
    $lines.Add('DOM_B2_D4_HISTORY_CHECK case=0 expected=UNKNOWN observed=UNKNOWN status=MATCH replica_d6_off_required=false replica_second_tt_required=false')
    $lines.Add('DOM_B2_D4_LOSS_QUALIFICATION loss_cases= required=false status=NOT_REQUIRED satisfied=true analyzer_scope=NO_UNVERIFIED_ATTESTATION_ACCEPTED')
    $lines.Add('DOM_B2_D4_CHAIN_AUDIT_FENCE scope=CREDITED_RAW_META_TO_RUNNER_CHAIN required_records=RESULT,GATE,SOURCE_FENCE_POST,BINARY_FENCE,CARGO_EXIT requirement=ONE_TO_ONE_PER_CREDITED_RAW status=EXTERNAL_MECHANICAL_AUDIT_REQUIRED satisfied=false analyzer_inputs=RAW_META_CENSUS_ONLY verdict_effect=PENDING')
    $lines.Add("DOM_B2_D4_REPLICA_FENCE code_id=$($script:codeId) compared_inequalities=4 equality_controls=1 d6_off_required=true d6_off_approval=true d6_off_gapless=false d6_off_root_exact_agreement=false second_tt_required=true second_tt_bytes=268435456 second_tt_approval=true second_tt_gapless=false second_tt_root_exact_agreement=false runner_mappings_complete=true runner_hashes_distinct_diagnostic=true comparison_fence=false control_fence=false history_fence=true non_unknown_exact_cases=0 non_unknown_fence=true loss_cases=0 stock_fast_fence=NOT_REQUIRED structural_fence=false external_chain_audit_fence=EXTERNAL_REQUIRED satisfied_without_loss_qualification=false satisfied=false")
    $lines.Add("DOM_B2_D4_MATRIX primary_files=11 replica_files=0 untrusted_v3_excluded_files=0 legacy_excluded_files=0 required_code_id=$($script:codeId) primary_config=$($script:primaryConfig) exact_cases=11 total_cases=11 unique_complete_roots=3648 total_roots=3648 missing_roots=0 incomplete_only_roots=0 duplicate_complete_rows=0 primary_matrix=REOPEN_EVIDENCE_PENDING_REPLICAS replica_fence_satisfied=false")
    $lines.Add('DOM_B2_D4_ANALYSIS_DONE validation=PASS census_hard_bind=PASS census_identity=PASS shard_identity=PASS lane_conflict_fence=PASS code_id_fence=PASS meta_fence=CREDITED_RAWS_PASS untrusted_v3_excluded=0 runner_approval_fence=CREDITED_RAWS_PASS chain_audit_fence=EXTERNAL_REQUIRED primary_role_fence=PASS primary_matrix=REOPEN_EVIDENCE_PENDING_REPLICAS authoritative_scope=fixed_completed_pairs_not_exhaustive_F4')
    return $lines.ToArray()
}

function New-SyntheticChainLines {
    param([Parameter(Mandatory = $true)][IO.FileInfo] $Journal)
    $lines = New-Object 'Collections.Generic.List[string]'
    $lines.Add("DOM_B2_D4_CHAIN_AUDIT version=1 mode=PARTIAL explicit_journals=1 frozen_journals=1 discovered_meta=11 credited_invocations=11 open_uncredited_runs=0 filesystem_writes=0 cargo_invocations=0")
    $lines.Add("BINDING primary_runner_sha256=$($script:primaryRunnerSha) replica_runner_sha256=$($script:replicaRunnerSha) child_wrapper_sha256=A9F8AF43DB7AD2D2D321DBB0E1BCCD5149175885D91A1FEC9F4DB12BC4CA06BC source_snapshot_sha256=$($script:sourceSnapshotSha) source_entries=20 strict_verifier_sha256=$($script:verifierSha) census_sha256=$($script:censusSha) binary_sha256=$($script:binarySha) code_id=$($script:codeId) deadline_ms=480000 cargo_wall_timeout_ms=540000 target=x86_64-pc-windows-msvc release=true test_threads=1")
    $lines.Add('JOURNAL_EVENT_SUMMARY waits_no_launch=0 clean_stops=0 aborts=0 grammar=PASS state_machine=PASS')
    $lines.Add("JOURNAL lane=PRIMARY path=$($Journal.Name) bytes=$($Journal.Length) sha256=$(Get-Sha256 -LiteralPath $Journal.FullName) result=PASS")
    for ($case = 0; $case -lt 11; $case++) {
        $sha = ('{0:X2}' -f ($case + 1)) * 32
        $metaSha = ('{0:X2}' -f ($case + 33)) * 32
        $name = 'DOM_B2_D4_SHARD_C{0:D2}_S0000_A00_RAW.log' -f $case
        $lines.Add("CREDIT lane=PRIMARY case=$case start=0 complete=$($script:widths[$case]) next_start=$($script:widths[$case]) shard_result=PASS raw=$name raw_bytes=$($case + 100) raw_sha256=$sha meta_bytes=500 meta_sha256=$metaSha cargo_exit_bytes=100 cargo_exit_sha256=$('AA' * 32) elapsed_s=1.000 wall_s=1.100 chain=PASS")
    }
    $lines.Add('LANE_SUMMARY lane=PRIMARY credited_roots=3648 complete_cases=11 queue_done=true expected_roots=3648 result=COMPLETE')
    $lines.Add('LANE_SUMMARY lane=D6_OFF credited_roots=0 complete_cases=0 queue_done=false expected_roots=3648 result=PARTIAL')
    $lines.Add('LANE_SUMMARY lane=SECOND_TT credited_roots=0 complete_cases=0 queue_done=false expected_roots=3648 result=PARTIAL')
    $lines.Add('DOM_B2_D4_CHAIN_AUDIT_DONE mode=PARTIAL credited_invocations=11 credited_meta=11 open_uncredited_runs=0 invocation_intervals=11 chronology=PASS final_fence=NOT_REQUESTED race_check=PASS result=PASS')
    return $lines.ToArray()
}

function Write-SyntheticFile {
    param(
        [Parameter(Mandatory = $true)][string] $Path,
        [Parameter(Mandatory = $true)][string[]] $Lines
    )
    $writer = New-LfWriter -LiteralPath $Path
    try { foreach ($line in $Lines) { $writer.WriteLine($line) } }
    finally { $writer.Dispose() }
}

function Invoke-SelfTests {
    $tempRoot = Join-Path ([IO.Path]::GetTempPath()) ('dom_b2_primary_capture_' + [Guid]::NewGuid().ToString('N'))
    [void](New-Item -ItemType Directory -Path $tempRoot)
    $passed = 0
    $failed = 0
    function Run-Test {
        param([string] $Name, [scriptblock] $Body)
        try {
            & $Body
            Write-Output "SELFTEST name=$Name result=PASS"
            $script:localPassed++
        }
        catch {
            Write-Output "SELFTEST name=$Name result=FAIL error=$($_.Exception.Message.Replace(' ', '_'))"
            $script:localFailed++
        }
    }
    $script:localPassed = 0
    $script:localFailed = 0
    try {
        $journalPath = Join-Path $tempRoot 'DOM_B2_D4_QUEUE_V3_RUN03_RAW.log'
        $journalWriter = [IO.StreamWriter]::new(
            [IO.FileStream]::new($journalPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read),
            [Text.UTF8Encoding]::new($false)
        )
        $journalWriter.NewLine = "`r`n"
        try {
            for ($case = 0; $case -lt 11; $case++) {
                $journalWriter.WriteLine("QUEUE timestamp=2026-01-01T00:00:00.0000000Z CASE_DONE case=$case roots=$($script:widths[$case]) code_id=$($script:codeId)")
            }
            $journalWriter.WriteLine("QUEUE timestamp=2026-01-01T00:00:01.0000000Z DONE result=PASS invocations=11 cases=11 total_roots=3648 code_id=$($script:codeId) source_snapshot_sha256=$($script:sourceSnapshotSha)")
        }
        finally { $journalWriter.Dispose() }
        $journal = Get-Item -LiteralPath $journalPath
        $analysisLines = New-SyntheticAnalysisLines
        $chainLines = New-SyntheticChainLines -Journal $journal

        Run-Test 'pure_lf_utf8_create_new' {
            $path = Join-Path $tempRoot 'pure.log'
            Write-SyntheticFile -Path $path -Lines @('one', 'two')
            [void](Assert-PureLfUtf8File -LiteralPath $path)
            $rejected = $false
            try { $writer = New-LfWriter -LiteralPath $path; $writer.Dispose() }
            catch { $rejected = $true }
            if (-not $rejected) { throw 'create-new overwrite was accepted' }
        }
        Run-Test 'isolated_runspace_typed_array_fallthrough_accept' {
            $toolPath = Join-Path $tempRoot 'fixture_success.ps1'
            Write-SyntheticFile -Path $toolPath -Lines @(
                'param([string[]] $Values)',
                'Write-Output ("VALUES count={0} joined={1}" -f $Values.Count, ($Values -join ","))'
            )
            $outputPath = Join-Path $tempRoot 'fixture_success_raw.log'
            $result = Invoke-IsolatedPowerShellCapture `
                -OutputPath $outputPath `
                -Variables @{
                    domCaptureFixturePath = $toolPath
                    domCaptureFixtureValues = [string[]]@('alpha', 'beta', 'gamma')
                } `
                -InvocationScript '& $domCaptureFixturePath -Values $domCaptureFixtureValues 2>&1'
            if (-not $result.Success -or $result.ExplicitExit -or -not $result.SentinelSeen -or
                -not $result.CommandSucceeded -or $result.ErrorCount -ne 0) {
                throw 'fallthrough runspace did not satisfy success/sentinel contract'
            }
            $lines = @(Read-PureLfLines -LiteralPath $outputPath)
            if ($lines.Count -ne 1 -or $lines[0] -cne 'VALUES count=3 joined=alpha,beta,gamma') {
                throw 'typed array was rebound or producer output changed'
            }
        }
        Run-Test 'isolated_nested_exit_zero_scoped_and_continue' {
            $toolPath = Join-Path $tempRoot 'fixture_exit_zero.ps1'
            Write-SyntheticFile -Path $toolPath -Lines @(
                "Write-Output 'before_exit_zero'",
                'exit 0',
                "Write-Output 'after_exit_zero'"
            )
            $outputPath = Join-Path $tempRoot 'fixture_exit_zero_raw.log'
            $result = Invoke-IsolatedPowerShellCapture `
                -OutputPath $outputPath `
                -Variables @{ domCaptureFixturePath = $toolPath } `
                -InvocationScript '& $domCaptureFixturePath 2>&1'
            $outerContinued = $true
            if (-not $result.Success -or $result.ExplicitExit -or $null -ne $result.ExitCode -or
                -not $result.SentinelSeen -or -not $result.CommandSucceeded -or -not $outerContinued) {
                throw "nested exit 0 contract mismatch success=$($result.Success) explicit=$($result.ExplicitExit) code=$($result.ExitCode) sentinel=$($result.SentinelSeen) state=$($result.State) errors=$($result.ErrorCount)"
            }
            $lines = @(Read-PureLfLines -LiteralPath $outputPath)
            if ($lines.Count -ne 1 -or $lines[0] -cne 'before_exit_zero') {
                throw 'exit 0 partial output was not preserved exactly'
            }
        }
        Run-Test 'isolated_nested_exit_nonzero_reject_and_continue' {
            $toolPath = Join-Path $tempRoot 'fixture_exit_two.ps1'
            Write-SyntheticFile -Path $toolPath -Lines @(
                "Write-Output 'before_exit_two'",
                'exit 2',
                "Write-Output 'after_exit_two'"
            )
            $outputPath = Join-Path $tempRoot 'fixture_exit_two_raw.log'
            $result = Invoke-IsolatedPowerShellCapture `
                -OutputPath $outputPath `
                -Variables @{ domCaptureFixturePath = $toolPath } `
                -InvocationScript '& $domCaptureFixturePath 2>&1'
            $outerContinued = $true
            if ($result.Success -or $result.ExplicitExit -or $null -ne $result.ExitCode -or
                -not $result.SentinelSeen -or $result.CommandSucceeded -or -not $outerContinued) {
                throw "nested exit 2 contract mismatch success=$($result.Success) explicit=$($result.ExplicitExit) code=$($result.ExitCode) sentinel=$($result.SentinelSeen) state=$($result.State) errors=$($result.ErrorCount)"
            }
            $lines = @(Read-PureLfLines -LiteralPath $outputPath)
            if ($lines.Count -ne 1 -or $lines[0] -cne 'before_exit_two') {
                throw 'exit 2 partial output was not preserved exactly'
            }
        }
        Run-Test 'isolated_top_level_exit_zero_capture_and_continue' {
            $outputPath = Join-Path $tempRoot 'fixture_direct_exit_zero_raw.log'
            $result = Invoke-IsolatedPowerShellCapture `
                -OutputPath $outputPath `
                -Variables @{ domCaptureFixtureValue = 'unused' } `
                -InvocationScript "Write-Output 'direct_before_exit_zero'; exit 0; Write-Output 'direct_after_exit_zero'"
            $outerContinued = $true
            if ($result.Success -or -not $result.ExplicitExit -or $result.ExitCode -ne 0 -or
                $result.SentinelSeen -or -not $outerContinued) {
                throw "top-level exit 0 capture mismatch success=$($result.Success) explicit=$($result.ExplicitExit) code=$($result.ExitCode) sentinel=$($result.SentinelSeen)"
            }
            $lines = @(Read-PureLfLines -LiteralPath $outputPath)
            if ($lines.Count -ne 1 -or $lines[0] -cne 'direct_before_exit_zero') {
                throw 'top-level exit 0 partial output was not preserved exactly'
            }
        }
        Run-Test 'isolated_top_level_exit_nonzero_capture_and_continue' {
            $outputPath = Join-Path $tempRoot 'fixture_direct_exit_two_raw.log'
            $result = Invoke-IsolatedPowerShellCapture `
                -OutputPath $outputPath `
                -Variables @{ domCaptureFixtureValue = 'unused' } `
                -InvocationScript "Write-Output 'direct_before_exit_two'; exit 2; Write-Output 'direct_after_exit_two'"
            $outerContinued = $true
            if ($result.Success -or -not $result.ExplicitExit -or $result.ExitCode -ne 2 -or
                $result.SentinelSeen -or -not $outerContinued) {
                throw "top-level exit 2 capture mismatch success=$($result.Success) explicit=$($result.ExplicitExit) code=$($result.ExitCode) sentinel=$($result.SentinelSeen)"
            }
            $lines = @(Read-PureLfLines -LiteralPath $outputPath)
            if ($lines.Count -ne 1 -or $lines[0] -cne 'direct_before_exit_two') {
                throw 'top-level exit 2 partial output was not preserved exactly'
            }
        }
        Run-Test 'isolated_runspace_throw_reject_and_continue' {
            $toolPath = Join-Path $tempRoot 'fixture_throw.ps1'
            Write-SyntheticFile -Path $toolPath -Lines @(
                "Write-Output 'before_throw'",
                "throw 'fixture_throw'",
                "Write-Output 'after_throw'"
            )
            $outputPath = Join-Path $tempRoot 'fixture_throw_raw.log'
            $result = Invoke-IsolatedPowerShellCapture `
                -OutputPath $outputPath `
                -Variables @{ domCaptureFixturePath = $toolPath } `
                -InvocationScript '& $domCaptureFixturePath 2>&1'
            $outerContinued = $true
            if ($result.Success -or $result.ExplicitExit -or $result.SentinelSeen -or
                $result.ErrorCount -eq 0 -or -not $outerContinued) {
                throw "throw contract mismatch success=$($result.Success) explicit=$($result.ExplicitExit) code=$($result.ExitCode) sentinel=$($result.SentinelSeen) state=$($result.State) errors=$($result.ErrorCount) failure=$($null -ne $result.Failure)"
            }
            $text = [Text.UTF8Encoding]::new($false, $true).GetString([IO.File]::ReadAllBytes($outputPath))
            if (-not $text.Contains('before_throw') -or -not $text.Contains('fixture_throw') -or
                $text.Contains('after_throw')) {
                throw "throw partial stdout/stderr was not preserved raw_utf8_b64=$(Convert-ErrorToToken -Text $text)"
            }
        }
        Run-Test 'preflight_blocker_temp_only' {
            try { throw 'synthetic preflight blocker' }
            catch {
                $blocker = Write-PreflightBlocker `
                    -Stage 'synthetic_gate' `
                    -ErrorRecord $_ `
                    -WrapperPids @(57144, 60000) `
                    -Root $tempRoot
            }
            $blockerPath = Join-Path $tempRoot $blocker.Path
            $lines = @(Read-PureLfLines -LiteralPath $blockerPath)
            if ($lines[0] -cnotmatch ' stage=synthetic_gate result=BLOCKED$' -or
                $lines[$lines.Count - 1] -cne 'DOM_B2_D4_PRIMARY_CAPTURE_PREFLIGHT_BLOCKER_DONE disposition=REFUSED_RETRY_AFTER_GATE_CORRECTION result=BLOCKED') {
                throw 'preflight blocker artifact contract mismatch'
            }
        }
        Run-Test 'wrapper_create_race_fails_closed_with_blocker' {
            $wrapperFixture = Join-Path $tempRoot 'DOM_B2_D4_PRIMARY_CAPTURE_RUN_RAW.log'
            Write-SyntheticFile -Path $wrapperFixture -Lines @('existing_owner')
            $rejected = $false
            try { $writer = New-LfWriter -LiteralPath $wrapperFixture; $writer.Dispose() }
            catch {
                $rejected = $true
                $blocker = Write-PreflightBlocker `
                    -Stage 'wrapper_create' `
                    -ErrorRecord $_ `
                    -WrapperPids @(57144) `
                    -Root $tempRoot
            }
            if (-not $rejected -or -not (Test-Path -LiteralPath (Join-Path $tempRoot $blocker.Path) -PathType Leaf)) {
                throw 'wrapper create race did not fail closed with separate blocker'
            }
            $lines = @(Read-PureLfLines -LiteralPath $wrapperFixture)
            if ($lines.Count -ne 1 -or $lines[0] -cne 'existing_owner') {
                throw 'wrapper create race overwrote existing owner'
            }
        }
        Run-Test 'terminal_journal_accept' {
            $terminal = Assert-TerminalJournals -Journals @($journal)
            if ($terminal.Invocations -ne 11) { throw 'terminal invocation count mismatch' }
        }
        Run-Test 'terminal_journal_missing_done_reject' {
            $badPath = Join-Path $tempRoot 'DOM_B2_D4_QUEUE_V3_RUN04_RAW.log'
            [IO.File]::WriteAllText($badPath, "QUEUE timestamp=x CASE_DONE case=0 roots=329 code_id=$($script:codeId)`r`n", [Text.UTF8Encoding]::new($false))
            $rejected = $false
            try { [void](Assert-TerminalJournals -Journals @((Get-Item $badPath))) }
            catch { $rejected = $true }
            if (-not $rejected) { throw 'missing DONE was accepted' }
        }
        Run-Test 'terminal_journal_case_order_reject' {
            $badPath = Join-Path $tempRoot 'terminal_bad_order.log'
            $badLines = New-Object 'Collections.Generic.List[string]'
            foreach ($case in @(1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10)) {
                $badLines.Add("QUEUE timestamp=2026-01-01T00:00:00.0000000Z CASE_DONE case=$case roots=$($script:widths[$case]) code_id=$($script:codeId)")
            }
            $badLines.Add("QUEUE timestamp=2026-01-01T00:00:01.0000000Z DONE result=PASS invocations=11 cases=11 total_roots=3648 code_id=$($script:codeId) source_snapshot_sha256=$($script:sourceSnapshotSha)")
            Write-SyntheticFile -Path $badPath -Lines $badLines.ToArray()
            $rejected = $false
            try { [void](Assert-TerminalJournals -Journals @((Get-Item $badPath))) }
            catch { $rejected = $true }
            if (-not $rejected) { throw 'out-of-order CASE_DONE sequence was accepted' }
        }
        Run-Test 'canonical_primary_filename_fence' {
            if (-not $script:primaryShardRegex.IsMatch('DOM_B2_D4_SHARD_C10_S0382_A09_RAW.log')) { throw 'valid primary rejected' }
            foreach ($bad in @(
                'DOM_B2_D4_REPLICA_D6_OFF_SHARD_C10_S0382_A09_RAW.log',
                'DOM_B2_D4_SHARD_C10_S0382_A09_META_RAW.log',
                'DOM_B2_D4_STREAM_SMOKE_RAW.log'
            )) {
                if ($script:primaryShardRegex.IsMatch($bad)) { throw "invalid primary accepted=$bad" }
            }
        }
        Run-Test 'primary_journal_run100_and_numeric_alias_fence' {
            $run100 = Join-Path $tempRoot 'DOM_B2_D4_QUEUE_V3_RUN100_RAW.log'
            Write-SyntheticFile -Path $run100 -Lines @('synthetic')
            $journals = @(Get-PrimaryJournals -Root $tempRoot)
            if (@($journals | Where-Object Name -ceq 'DOM_B2_D4_QUEUE_V3_RUN100_RAW.log').Count -ne 1) {
                throw 'RUN100 was omitted from exact RUN03+ census'
            }
            $alias = Join-Path $tempRoot 'DOM_B2_D4_QUEUE_V3_RUN003_RAW.log'
            Write-SyntheticFile -Path $alias -Lines @('synthetic')
            $rejected = $false
            try { [void](Get-PrimaryJournals -Root $tempRoot) }
            catch { $rejected = $true }
            if (-not $rejected) { throw 'duplicate numeric journal run id was accepted' }
        }
        Run-Test 'replica_hidden_case_insensitive_census' {
            $replicaPath = Join-Path $tempRoot 'dom_b2_d4_replica_d6_off_shard_c00_s0000_a00_raw.log'
            Write-SyntheticFile -Path $replicaPath -Lines @('synthetic')
            $item = Get-Item -LiteralPath $replicaPath
            $item.Attributes = $item.Attributes -bor [IO.FileAttributes]::Hidden
            $artifacts = @(Get-ReplicaMeasurementArtifacts -Root $tempRoot)
            if (@($artifacts | Where-Object { $_.Name -ieq (Split-Path -Leaf $replicaPath) }).Count -ne 1) {
                throw 'hidden/lowercase replica measurement artifact evaded census'
            }
        }
        Run-Test 'analyzer_primary_complete_accept' {
            $expectedNames = @(0..10 | ForEach-Object {
                'DOM_B2_D4_SHARD_C{0:D2}_S0000_A00_RAW.log' -f $_
            })
            $result = Validate-AnalyzerOutput -Lines $analysisLines -ExpectedShardNames $expectedNames
            if ($result.Credited.Count -ne 11) { throw 'synthetic analyzer credit count mismatch' }
        }
        Run-Test 'analyzer_primary_gap_reject' {
            $mutated = @($analysisLines | ForEach-Object {
                if ($_ -cmatch '^DOM_B2_D4_LANE_MATRIX lane=PRIMARY ') { $_.Replace('missing_roots=0', 'missing_roots=1') } else { $_ }
            })
            $rejected = $false
            try { [void](Validate-AnalyzerOutput -Lines $mutated) }
            catch { $rejected = $true }
            if (-not $rejected) { throw 'PRIMARY gap was accepted' }
        }
        Run-Test 'analyzer_replica_credit_reject' {
            $mutated = @($analysisLines | ForEach-Object {
                if ($_ -cmatch '^DOM_B2_D4_LANE_MATRIX lane=D6_OFF ') { $_.Replace('unique_complete_roots=0', 'unique_complete_roots=1') } else { $_ }
            })
            $rejected = $false
            try { [void](Validate-AnalyzerOutput -Lines $mutated) }
            catch { $rejected = $true }
            if (-not $rejected) { throw 'replica credit was accepted' }
        }
        Run-Test 'analyzer_duplicate_lane_matrix_reject' {
            $mutated = @($analysisLines | ForEach-Object {
                if ($_ -cmatch '^DOM_B2_D4_LANE_MATRIX lane=SECOND_TT ') {
                    $_.Replace('lane=SECOND_TT', 'lane=D6_OFF')
                }
                else { $_ }
            })
            $rejected = $false
            try { [void](Validate-AnalyzerOutput -Lines $mutated) }
            catch { $rejected = $true }
            if (-not $rejected) { throw 'duplicate analyzer lane matrix was accepted' }
        }
        Run-Test 'chain_primary_complete_accept' {
            $result = Validate-ChainOutput -Lines $chainLines -ExpectedJournals @($journal)
            if ($result.Credited.Count -ne 11) { throw 'synthetic chain credit count mismatch' }
        }
        Run-Test 'chain_open_run_reject' {
            $mutated = @($chainLines | ForEach-Object {
                if ($_ -cmatch '^DOM_B2_D4_CHAIN_AUDIT ') { $_.Replace('open_uncredited_runs=0', 'open_uncredited_runs=1') } else { $_ }
            })
            $rejected = $false
            try { [void](Validate-ChainOutput -Lines $mutated -ExpectedJournals @($journal)) }
            catch { $rejected = $true }
            if (-not $rejected) { throw 'open chain RUN was accepted' }
        }
        Run-Test 'chain_gap_reject' {
            $mutated = @($chainLines | ForEach-Object {
                if ($_ -cmatch '^CREDIT lane=PRIMARY case=0 ') { $_.Replace('complete=329 next_start=329', 'complete=328 next_start=328') } else { $_ }
            })
            $rejected = $false
            try { [void](Validate-ChainOutput -Lines $mutated -ExpectedJournals @($journal)) }
            catch { $rejected = $true }
            if (-not $rejected) { throw 'chain coverage gap was accepted' }
        }
        Run-Test 'chain_duplicate_lane_summary_reject' {
            $mutated = @($chainLines | ForEach-Object {
                if ($_ -cmatch '^LANE_SUMMARY lane=SECOND_TT ') {
                    $_.Replace('lane=SECOND_TT', 'lane=D6_OFF')
                }
                else { $_ }
            })
            $rejected = $false
            try { [void](Validate-ChainOutput -Lines $mutated -ExpectedJournals @($journal)) }
            catch { $rejected = $true }
            if (-not $rejected) { throw 'duplicate chain lane summary was accepted' }
        }
        Run-Test 'chain_header_count_reject' {
            $mutated = @($chainLines | ForEach-Object {
                if ($_ -cmatch '^DOM_B2_D4_CHAIN_AUDIT ') {
                    $_.Replace('credited_invocations=11', 'credited_invocations=10')
                }
                else { $_ }
            })
            $rejected = $false
            try { [void](Validate-ChainOutput -Lines $mutated -ExpectedJournals @($journal)) }
            catch { $rejected = $true }
            if (-not $rejected) { throw 'chain header credit count mismatch was accepted' }
        }
        Run-Test 'chain_timing_limit_reject' {
            $mutated = @($chainLines | ForEach-Object {
                if ($_ -cmatch '^CREDIT lane=PRIMARY case=0 ') {
                    $_.Replace('elapsed_s=1.000 wall_s=1.100', 'elapsed_s=600.000 wall_s=600.000')
                }
                else { $_ }
            })
            $rejected = $false
            try { [void](Validate-ChainOutput -Lines $mutated -ExpectedJournals @($journal)) }
            catch { $rejected = $true }
            if (-not $rejected) { throw '>=600 second chain timing was accepted' }
        }
        Run-Test 'analyzer_chain_join_accept' {
            $analysis = Validate-AnalyzerOutput -Lines $analysisLines
            $chain = Validate-ChainOutput -Lines $chainLines -ExpectedJournals @($journal)
            $joined = @(Compare-PrimaryJoin -Analyzer $analysis.Credited -Chain $chain.Credited)
            if ($joined.Count -ne 11) { throw 'join count mismatch' }
        }
        Run-Test 'analyzer_chain_split_brain_reject' {
            $analysis = Validate-AnalyzerOutput -Lines $analysisLines
            $chain = Validate-ChainOutput -Lines $chainLines -ExpectedJournals @($journal)
            $first = @($chain.Credited.Keys | Sort-Object)[0]
            $chain.Credited[$first].RawSha256 = 'FF' * 32
            $rejected = $false
            try { [void](Compare-PrimaryJoin -Analyzer $analysis.Credited -Chain $chain.Credited) }
            catch { $rejected = $true }
            if (-not $rejected) { throw 'split analyzer/chain identity was accepted' }
        }
        Run-Test 'credited_raw_meta_exit_identity_and_tamper_fence' {
            $artifactRoot = Join-Path $tempRoot 'credited_artifacts'
            [void](New-Item -ItemType Directory -Path $artifactRoot)
            $rawName = 'DOM_B2_D4_SHARD_C00_S0000_A00_RAW.log'
            $metaName = 'DOM_B2_D4_SHARD_C00_S0000_A00_META_RAW.log'
            $exitName = 'DOM_B2_D4_SHARD_C00_S0000_A00_CARGO_EXIT_RAW.log'
            Write-SyntheticFile -Path (Join-Path $artifactRoot $rawName) -Lines @('raw')
            Write-SyntheticFile -Path (Join-Path $artifactRoot $metaName) -Lines @('meta')
            Write-SyntheticFile -Path (Join-Path $artifactRoot $exitName) -Lines @('exit')
            $rawItem = Get-Item -LiteralPath (Join-Path $artifactRoot $rawName)
            $metaItem = Get-Item -LiteralPath (Join-Path $artifactRoot $metaName)
            $exitItem = Get-Item -LiteralPath (Join-Path $artifactRoot $exitName)
            $credit = [pscustomobject]@{
                File = $rawName
                RawBytes = [int64]$rawItem.Length
                RawSha256 = Get-Sha256 -LiteralPath $rawItem.FullName
                Meta = $metaName
                MetaBytes = [int64]$metaItem.Length
                MetaSha256 = Get-Sha256 -LiteralPath $metaItem.FullName
                CargoExit = $exitName
                CargoExitBytes = [int64]$exitItem.Length
                CargoExitSha256 = Get-Sha256 -LiteralPath $exitItem.FullName
            }
            $credits = @{ $rawName.ToLowerInvariant() = $credit }
            $snapshot = @(Assert-CreditedArtifactsCurrent -Credited $credits -Root $artifactRoot)
            if ($snapshot.Count -ne 3) { throw 'credited artifact snapshot is not raw/META/exit exact set' }
            [IO.File]::AppendAllText($metaItem.FullName, 'tamper', [Text.UTF8Encoding]::new($false))
            $rejected = $false
            try { [void](Assert-CreditedArtifactsCurrent -Credited $credits -Root $artifactRoot) }
            catch { $rejected = $true }
            if (-not $rejected) { throw 'credited META post-snapshot tamper was accepted' }
        }
        Run-Test 'snapshot_set_addition_reject' {
            $before = @([pscustomobject]@{ Path = 'a'; Bytes = 1; Sha256 = 'AA' * 32 })
            $after = @(
                [pscustomobject]@{ Path = 'a'; Bytes = 1; Sha256 = 'AA' * 32 },
                [pscustomobject]@{ Path = 'b'; Bytes = 1; Sha256 = 'BB' * 32 }
            )
            $rejected = $false
            try { Compare-Snapshots -Before $before -After $after -Context 'synthetic addition' }
            catch { $rejected = $true }
            if (-not $rejected) { throw 'post-snapshot set addition was accepted' }
        }
        Run-Test 'pid_descendant_detection' {
            $processes = @(
                [pscustomobject]@{ ProcessId = 200; ParentProcessId = 100; Name = 'child.exe' },
                [pscustomobject]@{ ProcessId = 300; ParentProcessId = 200; Name = 'grandchild.exe' },
                [pscustomobject]@{ ProcessId = 400; ParentProcessId = 1; Name = 'other.exe' }
            )
            $live = @(Get-LivePidTree -RootPid @(100) -Processes $processes)
            if ($live.Count -ne 2) { throw "descendant count=$($live.Count), expected 2" }
        }
    }
    finally {
        $tempFull = [IO.Path]::GetFullPath($tempRoot)
        $tempBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
        if (-not $tempFull.StartsWith($tempBase, [StringComparison]::OrdinalIgnoreCase) -or
            -not (Split-Path -Leaf $tempFull).StartsWith('dom_b2_primary_capture_', [StringComparison]::Ordinal)) {
            throw "unsafe self-test cleanup path=$tempFull"
        }
        Remove-Item -LiteralPath $tempFull -Recurse -Force
    }
    $passed = $script:localPassed
    $failed = $script:localFailed
    Write-Output "DOM_B2_D4_PRIMARY_CAPTURE_SELFTEST version=$($script:version) tests=$($passed + $failed) passed=$passed failures=$failed temp_only=true live_capture=false cargo_invocations=0 result=$(if($failed -eq 0){'PASS'}else{'FAIL'})"
    if ($failed -ne 0) { throw "$failed self-test(s) failed" }
}

if ($SelfTest) {
    Invoke-SelfTests
}
else {
    Invoke-LiveCapture
}
