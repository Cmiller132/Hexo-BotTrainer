<#
.SYNOPSIS
Validates and aggregates the R-CREL-6 DOM-B2 depth-4 root shards.

.DESCRIPTION
The census is checked against all 11 preregistered cases and all 3,648 sorted
root indices. Each shard is then checked against that census, including its
half-open range, coordinates, fingerprint, configuration, stop semantics, and
DONE row. A stopped attempt remains visible, but a later complete row may
discharge the same root. Conflicting complete statuses are a validation error.
Only primary `DOM_B2_D4_SHARD_Ccc_Sssss[_Aaa]_RAW.log` and replica
`DOM_B2_D4_REPLICA_<D6_OFF|SECOND_TT>_SHARD_Ccc_Sssss_Aaa_RAW.log`
names are data raws. RUN logs are never discovered. Primary coverage is fixed
to D6 on with a 512 MiB TT. Every
credited configuration has its own explicitly approved runner SHA-256 and is
aggregated as an independent lane. Replica lanes cannot discharge primary
coverage.
Primary and replica raws must carry `code_id=DOM_B2_D4_PRIMARY_V3` on setup,
every result, and completion. A V3 raw without a META sidecar is retained as
structurally valid but untrusted and receives no coverage credit; a present
invalid META is fatal. Pre-V3 raws with no code_id are validated as a separate
legacy lane and never contribute primary coverage.

This analyzer consumes census, data raws, and exact META sidecars. It does not
consume runner journals or Cargo-exit records, so it emits an unsatisfied
external chain-audit fence for one-to-one RESULT/GATE/SOURCE_FENCE_POST/
BINARY_FENCE/CARGO_EXIT validation. It cannot emit a final PASS or KILL by
itself. Loss stock/fast qualification is likewise external and unattested.

`recurrence=WIN` can be exact despite unfinished siblings because Win is the
maximizing short circuit. `gate_status` remains INCOMPLETE until enumeration is
gapless, because the preregistered matrix PASS/KILL gates require every index.

.EXAMPLE
./scripts/dom_b2_d4_aggregate.ps1 -ApprovedRunnerMapping @(
    'd6=true,tt_bytes=536870912,runner_sha256=25C6AB359B4BCB3F2207F426BF958A7DB588E1B5BFE7FC0D67FF4AC3EC385D8E',
    'd6=false,tt_bytes=536870912,runner_sha256=B9E61444E1555E648575336B8B6E8CD406CB0FB24C406557106F639E9B1953C8',
    'd6=true,tt_bytes=268435456,runner_sha256=B9E61444E1555E648575336B8B6E8CD406CB0FB24C406557106F639E9B1953C8'
) | Set-Content -Encoding utf8 DOM_B2_D4_ANALYSIS_RAW.log

.EXAMPLE
./scripts/dom_b2_d4_aggregate.ps1 -ShardPath @(
    './DOM_B2_D4_SHARD_C00_S0000_RAW.log',
    './DOM_B2_D4_SHARD_C00_S0032_RAW.log'
) -ApprovedRunnerMapping @(
    'd6=true,tt_bytes=536870912,runner_sha256=25C6AB359B4BCB3F2207F426BF958A7DB588E1B5BFE7FC0D67FF4AC3EC385D8E',
    'd6=false,tt_bytes=536870912,runner_sha256=B9E61444E1555E648575336B8B6E8CD406CB0FB24C406557106F639E9B1953C8',
    'd6=true,tt_bytes=268435456,runner_sha256=B9E61444E1555E648575336B8B6E8CD406CB0FB24C406557106F639E9B1953C8'
)
#>
[CmdletBinding()]
param(
    [Parameter()]
    [Alias('ApprovedRunnerByConfig')]
    [string[]] $ApprovedRunnerMapping,

    [Parameter()]
    [string] $CensusPath,

    [Parameter()]
    [string[]] $ShardPath,

    [Parameter()]
    [string] $ShardDirectory,

    [Parameter()]
    [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$invariant = [Globalization.CultureInfo]::InvariantCulture
$repoRoot = Split-Path -Parent $PSScriptRoot
$shardFileRegexText = '^DOM_B2_D4_(?:(?:REPLICA_(?<lane>D6_OFF|SECOND_TT)_))?SHARD_C(?<case>(?:0[0-9]|10))_S(?<start>[0-9]{4})(?:_A(?<attempt>[0-9]{2}))?_RAW[.]log$'
$shardFileRegex = [regex]::new($shardFileRegexText, [Text.RegularExpressions.RegexOptions]::CultureInvariant)
$primaryTtBytes = [uint64]536870912
$secondTtBytes = [uint64]268435456
$requiredPrimaryRunnerSha256 = '25C6AB359B4BCB3F2207F426BF958A7DB588E1B5BFE7FC0D67FF4AC3EC385D8E'
$requiredReplicaRunnerSha256 = 'B9E61444E1555E648575336B8B6E8CD406CB0FB24C406557106F639E9B1953C8'
$requiredCodeId = 'DOM_B2_D4_PRIMARY_V3'
$requiredCensusSha256 = '436E9F6C4A93CDB611EEF6495A01F510615174A2897271CA1092A0E7422DD7BE'
$requiredSourceSnapshotSha256 = '1D4FBB37638668D0F2ED1972D27CDDD833826721A2EFD5500BFDC09DCF81B746'

if ([string]::IsNullOrWhiteSpace($CensusPath)) {
    $CensusPath = Join-Path $repoRoot 'DOM_B2_D4_CENSUS_RAW.log'
}
if ([string]::IsNullOrWhiteSpace($ShardDirectory)) {
    $ShardDirectory = $repoRoot
}

$expectedCases = @(
    [pscustomobject]@{ Case = 0;  Id = '32f44c499244b611:9'; Pair = '(-2,1);(4,1)';   Coverage = 'SPLIT';        Width = 329 },
    [pscustomobject]@{ Case = 1;  Id = '32f44c499244b611:9'; Pair = '(2,1);(-2,1)';   Coverage = 'H_CONTAINING'; Width = 312 },
    [pscustomobject]@{ Case = 2;  Id = '32f44c499244b611:9'; Pair = '(2,1);(4,1)';    Coverage = 'H_CONTAINING'; Width = 312 },
    [pscustomobject]@{ Case = 3;  Id = '19b085e7aa9f6215:9'; Pair = '(-1,0);(5,0)';   Coverage = 'SPLIT';        Width = 330 },
    [pscustomobject]@{ Case = 4;  Id = '19b085e7aa9f6215:9'; Pair = '(3,0);(-1,0)';   Coverage = 'H_CONTAINING'; Width = 313 },
    [pscustomobject]@{ Case = 5;  Id = '498a61ae0b5cf4ef:9'; Pair = '(-2,2);(4,-4)';  Coverage = 'SPLIT';        Width = 330 },
    [pscustomobject]@{ Case = 6;  Id = '498a61ae0b5cf4ef:9'; Pair = '(2,-2);(-2,2)';  Coverage = 'H_CONTAINING'; Width = 313 },
    [pscustomobject]@{ Case = 7;  Id = 'fd688f189544bf72:9'; Pair = '(-2,0);(4,0)';   Coverage = 'SPLIT';        Width = 330 },
    [pscustomobject]@{ Case = 8;  Id = 'fd688f189544bf72:9'; Pair = '(2,0);(-2,0)';   Coverage = 'H_CONTAINING'; Width = 313 },
    [pscustomobject]@{ Case = 9;  Id = 'd7e1b56c925b7f32:19'; Pair = '(-1,0);(-2,3)'; Coverage = 'H_CONTAINING'; Width = 383 },
    [pscustomobject]@{ Case = 10; Id = 'd7e1b56c925b7f32:19'; Pair = '(-1,0);(-1,2)'; Coverage = 'H_CONTAINING'; Width = 383 }
)

function Get-Fields {
    param([Parameter(Mandatory)][string] $Line)
    $fields = @{}
    foreach ($match in [regex]::Matches($Line, '(^|\s)(?<key>[A-Za-z][A-Za-z0-9_]*)=(?<value>\S+)')) {
        $key = $match.Groups['key'].Value
        if ($fields.ContainsKey($key)) {
            throw "duplicate field '$key' in line: $Line"
        }
        $fields[$key] = $match.Groups['value'].Value
    }
    return $fields
}

function Get-RequiredField {
    param(
        [Parameter(Mandatory)][hashtable] $Fields,
        [Parameter(Mandatory)][string] $Name,
        [Parameter(Mandatory)][string] $Context
    )
    if (-not $Fields.ContainsKey($Name)) {
        throw "$Context is missing field '$Name'"
    }
    return [string]$Fields[$Name]
}

function Convert-ToInt {
    param([string] $Value, [string] $Context)
    $parsed = 0
    if (-not [int]::TryParse($Value, [Globalization.NumberStyles]::Integer, $invariant, [ref]$parsed)) {
        throw "$Context is not an Int32: '$Value'"
    }
    return $parsed
}

function Convert-ToUInt64 {
    param([string] $Value, [string] $Context)
    $parsed = [uint64]0
    if (-not [uint64]::TryParse($Value, [Globalization.NumberStyles]::Integer, $invariant, [ref]$parsed)) {
        throw "$Context is not a UInt64: '$Value'"
    }
    return $parsed
}

function Convert-ToDecimal {
    param([string] $Value, [string] $Context)
    $parsed = [decimal]0
    if (-not [decimal]::TryParse($Value, [Globalization.NumberStyles]::Float, $invariant, [ref]$parsed)) {
        throw "$Context is not a finite decimal: '$Value'"
    }
    return $parsed
}

function Convert-ToBoolText {
    param([string] $Value, [string] $Context)
    if ($Value -ceq 'true') { return $true }
    if ($Value -ceq 'false') { return $false }
    throw "$Context must be exactly true or false: '$Value'"
}

function Get-ConfigKey {
    param(
        [Parameter(Mandatory)][bool] $D6,
        [Parameter(Mandatory)][uint64] $TtBytes
    )
    $d6Text = $D6.ToString().ToLowerInvariant()
    return "d6=$d6Text,tt_bytes=$TtBytes"
}

function Read-ApprovalMap {
    param([Parameter()][AllowNull()][string[]] $Entries)

    if ($null -eq $Entries -or $Entries.Count -eq 0) {
        throw 'at least one -ApprovedRunnerMapping entry is required'
    }
    $result = @{}
    $mappingRegex = [regex]::new(
        '^d6=(?<d6>true|false),tt_bytes=(?<tt>[1-9][0-9]*),runner_sha256=(?<sha>[0-9A-F]{64})$',
        [Text.RegularExpressions.RegexOptions]::CultureInvariant
    )
    foreach ($entry in $Entries) {
        $match = $mappingRegex.Match([string]$entry)
        if (-not $match.Success) {
            throw "invalid approved-runner mapping '$entry'; expected d6=<true|false>,tt_bytes=<positive UInt64>,runner_sha256=<64 uppercase hex>"
        }
        $d6 = Convert-ToBoolText $match.Groups['d6'].Value 'approved-runner mapping d6'
        $ttBytes = Convert-ToUInt64 $match.Groups['tt'].Value 'approved-runner mapping tt_bytes'
        if ($ttBytes -eq 0) {
            throw "approved-runner mapping tt_bytes must be positive: '$entry'"
        }
        $key = Get-ConfigKey -D6 $d6 -TtBytes $ttBytes
        if ($result.ContainsKey($key)) {
            throw "duplicate approved-runner mapping for $key"
        }
        $result[$key] = $match.Groups['sha'].Value
    }
    return $result
}

function Assert-ApprovalBindings {
    param([Parameter(Mandatory)][hashtable] $ApprovalMap)

    $primaryKey = Get-ConfigKey -D6 $true -TtBytes $primaryTtBytes
    $d6OffKey = Get-ConfigKey -D6 $false -TtBytes $primaryTtBytes
    $secondTtKey = Get-ConfigKey -D6 $true -TtBytes $secondTtBytes
    Assert-Equal $ApprovalMap.Count 3 'approved-runner mapping count'
    foreach ($key in @($primaryKey, $d6OffKey, $secondTtKey)) {
        if (-not $ApprovalMap.ContainsKey($key)) {
            throw "approval map is missing required configuration $key"
        }
    }
    Assert-Equal ([string]$ApprovalMap[$primaryKey]) $requiredPrimaryRunnerSha256 'primary runner hard bind'
    Assert-Equal ([string]$ApprovalMap[$d6OffKey]) $requiredReplicaRunnerSha256 'D6-off runner hard bind'
    Assert-Equal ([string]$ApprovalMap[$secondTtKey]) $requiredReplicaRunnerSha256 'second-TT runner hard bind'
}

function Get-LaneLabel {
    param([Parameter(Mandatory)][string] $ConfigKey)
    if ($ConfigKey -ceq (Get-ConfigKey -D6 $true -TtBytes $primaryTtBytes)) {
        return 'PRIMARY'
    }
    if ($ConfigKey -ceq (Get-ConfigKey -D6 $false -TtBytes $primaryTtBytes)) {
        return 'D6_OFF'
    }
    if ($ConfigKey -ceq (Get-ConfigKey -D6 $true -TtBytes $secondTtBytes)) {
        return 'SECOND_TT'
    }
    return 'REPLICA_EXTRA'
}

function Assert-FilenameLaneConfiguration {
    param(
        [Parameter(Mandatory)][string] $FilenameLane,
        [Parameter(Mandatory)][bool] $D6,
        [Parameter(Mandatory)][uint64] $TtBytes,
        [Parameter()][AllowNull()] $CodeId,
        [Parameter(Mandatory)][string] $DisplayName
    )
    switch -CaseSensitive ($FilenameLane) {
        'D6_OFF' {
            if ($null -eq $CodeId) { throw "$DisplayName replica filename cannot carry a legacy pre-V3 raw" }
            Assert-Equal $D6 $false "$DisplayName filename lane d6"
            Assert-Equal $TtBytes $primaryTtBytes "$DisplayName filename lane tt_bytes"
        }
        'SECOND_TT' {
            if ($null -eq $CodeId) { throw "$DisplayName replica filename cannot carry a legacy pre-V3 raw" }
            Assert-Equal $D6 $true "$DisplayName filename lane d6"
            Assert-Equal $TtBytes $secondTtBytes "$DisplayName filename lane tt_bytes"
        }
        'PRIMARY' {
            if ($null -ne $CodeId) {
                Assert-Equal $D6 $true "$DisplayName V3 primary filename d6"
                Assert-Equal $TtBytes $primaryTtBytes "$DisplayName V3 primary filename tt_bytes"
            }
        }
        default { throw "$DisplayName has unsupported filename lane '$FilenameLane'" }
    }
}

function Assert-Equal {
    param($Actual, $Expected, [string] $Context)
    if ($Actual -cne $Expected) {
        throw "$Context mismatch: observed='$Actual' expected='$Expected'"
    }
}

function Assert-CodeIdLane {
    param(
        [Parameter(Mandatory)][hashtable] $Fields,
        [Parameter()][AllowNull()] $ShardCodeId,
        [Parameter(Mandatory)][string] $Context
    )
    $hasCodeId = $Fields.ContainsKey('code_id')
    if ($null -eq $ShardCodeId) {
        if ($hasCodeId) {
            throw "$Context mixes code_id into a legacy pre-V3 shard"
        }
        return
    }
    if (-not $hasCodeId) {
        throw "$Context is missing required code_id '$ShardCodeId'"
    }
    Assert-Equal ([string]$Fields['code_id']) $ShardCodeId "$Context code_id"
}

function Get-FileSha256 {
    param([Parameter(Mandatory)][string] $LiteralPath)
    return (Get-FileHash -LiteralPath $LiteralPath -Algorithm SHA256).Hash.ToUpperInvariant()
}

function Read-Census {
    param([Parameter(Mandatory)][string] $LiteralPath)

    $resolved = (Resolve-Path -LiteralPath $LiteralPath).Path
    $observedSha256 = Get-FileSha256 $resolved
    Assert-Equal $observedSha256 $requiredCensusSha256 'census hard-bound SHA256'
    $lines = @(Get-Content -LiteralPath $resolved)
    $universes = @{}
    $roots = @{}
    $doneCount = 0

    foreach ($line in $lines) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            throw "blank line in census '$resolved'"
        }
        $fields = Get-Fields -Line $line
        if ($line.StartsWith('DOM_B2_D4_UNIVERSE ')) {
            $caseIndex = Convert-ToInt (Get-RequiredField $fields 'case' 'census universe') 'census universe case'
            if ($caseIndex -lt 0 -or $caseIndex -ge $expectedCases.Count) {
                throw "census universe has extra case=$caseIndex"
            }
            if ($universes.ContainsKey($caseIndex)) {
                throw "duplicate census universe case=$caseIndex"
            }
            $expected = $expectedCases[$caseIndex]
            $id = Get-RequiredField $fields 'id' "census case=$caseIndex"
            $pair = Get-RequiredField $fields 'pair' "census case=$caseIndex"
            $coverage = Get-RequiredField $fields 'coverage' "census case=$caseIndex"
            $rootCount = Convert-ToInt (Get-RequiredField $fields 'root_count' "census case=$caseIndex") "census case=$caseIndex root_count"
            $fingerprint = Get-RequiredField $fields 'fingerprint' "census case=$caseIndex"
            Assert-Equal $id $expected.Id "census case=$caseIndex id"
            Assert-Equal $pair $expected.Pair "census case=$caseIndex pair"
            Assert-Equal $coverage $expected.Coverage "census case=$caseIndex coverage"
            Assert-Equal $rootCount $expected.Width "census case=$caseIndex root_count"
            if ($fingerprint -cnotmatch '^[0-9A-F]{16}$') {
                throw "census case=$caseIndex has noncanonical fingerprint '$fingerprint'"
            }
            $universes[$caseIndex] = [pscustomobject]@{
                Case = $caseIndex
                Id = $id
                Pair = $pair
                Coverage = $coverage
                RootCount = $rootCount
                Fingerprint = $fingerprint
            }
            $roots[$caseIndex] = @{}
        }
        elseif ($line.StartsWith('DOM_B2_D4_ROOT_MOVE ')) {
            $caseIndex = Convert-ToInt (Get-RequiredField $fields 'case' 'census root') 'census root case'
            if (-not $roots.ContainsKey($caseIndex)) {
                throw "census root precedes or lacks universe case=$caseIndex"
            }
            $rootIndex = Convert-ToInt (Get-RequiredField $fields 'root_index' "census case=$caseIndex root") "census case=$caseIndex root_index"
            if ($roots[$caseIndex].ContainsKey($rootIndex)) {
                throw "duplicate census root case=$caseIndex root_index=$rootIndex"
            }
            $q = Convert-ToInt (Get-RequiredField $fields 'q' "census case=$caseIndex root=$rootIndex") "census q"
            $r = Convert-ToInt (Get-RequiredField $fields 'r' "census case=$caseIndex root=$rootIndex") "census r"
            $fingerprint = Get-RequiredField $fields 'fingerprint' "census case=$caseIndex root=$rootIndex"
            Assert-Equal $fingerprint $universes[$caseIndex].Fingerprint "census case=$caseIndex root=$rootIndex fingerprint"
            $roots[$caseIndex][$rootIndex] = [pscustomobject]@{ Q = $q; R = $r }
        }
        elseif ($line.StartsWith('DOM_B2_D4_CENSUS_DONE ')) {
            $doneCount++
            Assert-Equal (Convert-ToInt (Get-RequiredField $fields 'cases' 'census done') 'census done cases') 11 'census done cases'
            Assert-Equal (Convert-ToInt (Get-RequiredField $fields 'root_actions' 'census done') 'census done root_actions') 3648 'census done root_actions'
            Assert-Equal (Get-RequiredField $fields 'result' 'census done') 'PASS' 'census done result'
        }
        else {
            throw "unknown census line: $line"
        }
    }

    Assert-Equal $doneCount 1 'census done-line count'
    Assert-Equal $universes.Count 11 'census universe count'
    $observedTotal = 0
    for ($caseIndex = 0; $caseIndex -lt $expectedCases.Count; $caseIndex++) {
        if (-not $universes.ContainsKey($caseIndex)) {
            throw "missing census universe case=$caseIndex"
        }
        $universe = $universes[$caseIndex]
        Assert-Equal $roots[$caseIndex].Count $universe.RootCount "census case=$caseIndex root-row count"
        $previousQ = [int]::MinValue
        $previousR = [int]::MinValue
        $coordinateSet = @{}
        for ($rootIndex = 0; $rootIndex -lt $universe.RootCount; $rootIndex++) {
            if (-not $roots[$caseIndex].ContainsKey($rootIndex)) {
                throw "census gap case=$caseIndex root_index=$rootIndex"
            }
            $coord = $roots[$caseIndex][$rootIndex]
            $coordKey = "$($coord.Q),$($coord.R)"
            if ($coordinateSet.ContainsKey($coordKey)) {
                throw "duplicate census coordinate case=$caseIndex coord=$coordKey"
            }
            $coordinateSet[$coordKey] = $true
            if ($rootIndex -gt 0 -and (($coord.Q -lt $previousQ) -or ($coord.Q -eq $previousQ -and $coord.R -le $previousR))) {
                throw "census is not strictly sorted case=$caseIndex root_index=$rootIndex coord=$coordKey"
            }
            $previousQ = $coord.Q
            $previousR = $coord.R
        }
        $observedTotal += $universe.RootCount
    }
    Assert-Equal $observedTotal 3648 'census total root actions'

    return [pscustomobject]@{
        Path = $resolved
        Sha256 = $observedSha256
        Universes = $universes
        Roots = $roots
    }
}

function Read-Shard {
    param(
        [Parameter(Mandatory)][string] $LiteralPath,
        [Parameter(Mandatory)] $Census
    )

    $resolved = (Resolve-Path -LiteralPath $LiteralPath).Path
    $displayName = Split-Path -Leaf $resolved
    $nameMatch = $shardFileRegex.Match($displayName)
    if (-not $nameMatch.Success) {
        throw "$displayName is not a DOM-B2 data raw; expected $shardFileRegexText"
    }
    $nameCase = Convert-ToInt $nameMatch.Groups['case'].Value "$displayName filename case"
    $nameStart = Convert-ToInt $nameMatch.Groups['start'].Value "$displayName filename start"
    if ($nameMatch.Groups['lane'].Success) {
        $filenameLane = $nameMatch.Groups['lane'].Value
    }
    else {
        $filenameLane = 'PRIMARY'
    }
    if ($nameMatch.Groups['attempt'].Success) {
        $nameAttempt = Convert-ToInt $nameMatch.Groups['attempt'].Value "$displayName filename attempt"
        $nameAttemptLabel = ('A{0:D2}' -f $nameAttempt)
    }
    else {
        $nameAttempt = $null
        $nameAttemptLabel = 'PILOT'
    }
    if ($filenameLane -cne 'PRIMARY' -and $null -eq $nameAttempt) {
        throw "$displayName replica filename is missing mandatory _Aaa attempt"
    }
    $lines = @(Get-Content -LiteralPath $resolved)
    if ($lines.Count -lt 3) {
        throw "$displayName has fewer than three rows"
    }
    if (-not $lines[0].StartsWith('DOM_B2_D4_SHARD_SETUP ')) {
        throw "$displayName first row is not SHARD_SETUP"
    }
    if (-not $lines[$lines.Count - 1].StartsWith('DOM_B2_D4_SHARD_DONE ')) {
        throw "$displayName last row is not SHARD_DONE"
    }

    $setupFields = Get-Fields $lines[0]
    if ($setupFields.ContainsKey('code_id')) {
        $codeId = [string]$setupFields['code_id']
        Assert-Equal $codeId $requiredCodeId "$displayName setup code_id"
    }
    else {
        $codeId = $null
    }
    $caseIndex = Convert-ToInt (Get-RequiredField $setupFields 'case' "$displayName setup") "$displayName setup case"
    if (-not $Census.Universes.ContainsKey($caseIndex)) {
        throw "$displayName setup has extra case=$caseIndex"
    }
    $universe = $Census.Universes[$caseIndex]
    $start = Convert-ToInt (Get-RequiredField $setupFields 'start' "$displayName setup") "$displayName setup start"
    $end = Convert-ToInt (Get-RequiredField $setupFields 'end' "$displayName setup") "$displayName setup end"
    $rootCount = Convert-ToInt (Get-RequiredField $setupFields 'root_count' "$displayName setup") "$displayName setup root_count"
    $fingerprint = Get-RequiredField $setupFields 'fingerprint' "$displayName setup"
    $depth = Convert-ToInt (Get-RequiredField $setupFields 'depth' "$displayName setup") "$displayName setup depth"
    $ttBytes = Convert-ToUInt64 (Get-RequiredField $setupFields 'tt_bytes' "$displayName setup") "$displayName setup tt_bytes"
    $d6 = Convert-ToBoolText (Get-RequiredField $setupFields 'd6' "$displayName setup") "$displayName setup d6"
    $deadlineMs = Convert-ToInt (Get-RequiredField $setupFields 'deadline_ms' "$displayName setup") "$displayName setup deadline_ms"
    Assert-Equal (Get-RequiredField $setupFields 'id' "$displayName setup") $universe.Id "$displayName setup id"
    Assert-Equal $caseIndex $nameCase "$displayName filename/setup case"
    Assert-Equal $start $nameStart "$displayName filename/setup start"
    Assert-Equal $rootCount $universe.RootCount "$displayName setup root_count"
    Assert-Equal $fingerprint $universe.Fingerprint "$displayName setup fingerprint"
    Assert-Equal $depth 4 "$displayName setup depth"
    if ($start -lt 0 -or $start -ge $rootCount -or $end -le $start -or $end -gt $rootCount) {
        throw "$displayName has invalid half-open range [$start,$end) for root_count=$rootCount"
    }
    if ($ttBytes -eq 0) { throw "$displayName setup tt_bytes must be positive" }
    if ($deadlineMs -lt 1 -or $deadlineMs -gt 480000) { throw "$displayName setup deadline_ms outside 1..480000" }
    Assert-FilenameLaneConfiguration -FilenameLane $filenameLane -D6 $d6 -TtBytes $ttBytes -CodeId $codeId -DisplayName $displayName

    $records = @()
    for ($lineIndex = 1; $lineIndex -lt $lines.Count - 1; $lineIndex++) {
        $line = $lines[$lineIndex]
        if (-not $line.StartsWith('DOM_B2_D4_ROOT_RESULT ')) {
            throw "$displayName has unknown middle row: $line"
        }
        $fields = Get-Fields $line
        Assert-CodeIdLane -Fields $fields -ShardCodeId $codeId -Context "$displayName result row=$lineIndex"
        $rowCase = Convert-ToInt (Get-RequiredField $fields 'case' "$displayName result") "$displayName result case"
        $rootIndex = Convert-ToInt (Get-RequiredField $fields 'root_index' "$displayName result") "$displayName result root_index"
        Assert-Equal $rowCase $caseIndex "$displayName result case"
        $expectedIndex = $start + ($lineIndex - 1)
        Assert-Equal $rootIndex $expectedIndex "$displayName sequential root_index"
        if ($rootIndex -lt $start -or $rootIndex -ge $end) {
            throw "$displayName result root_index=$rootIndex outside shard [$start,$end)"
        }
        $expectedCoord = $Census.Roots[$caseIndex][$rootIndex]
        $q = Convert-ToInt (Get-RequiredField $fields 'q' "$displayName root=$rootIndex") "$displayName root=$rootIndex q"
        $r = Convert-ToInt (Get-RequiredField $fields 'r' "$displayName root=$rootIndex") "$displayName root=$rootIndex r"
        Assert-Equal $q $expectedCoord.Q "$displayName root=$rootIndex q"
        Assert-Equal $r $expectedCoord.R "$displayName root=$rootIndex r"
        Assert-Equal (Get-RequiredField $fields 'fingerprint' "$displayName root=$rootIndex") $fingerprint "$displayName root=$rootIndex fingerprint"
        Assert-Equal (Convert-ToInt (Get-RequiredField $fields 'root_count' "$displayName root=$rootIndex") "$displayName root_count") $rootCount "$displayName root=$rootIndex root_count"
        Assert-Equal (Convert-ToInt (Get-RequiredField $fields 'root_depth' "$displayName root=$rootIndex") "$displayName root_depth") 4 "$displayName root=$rootIndex root_depth"
        Assert-Equal (Convert-ToInt (Get-RequiredField $fields 'child_depth' "$displayName root=$rootIndex") "$displayName child_depth") 3 "$displayName root=$rootIndex child_depth"
        Assert-Equal (Convert-ToBoolText (Get-RequiredField $fields 'd6' "$displayName root=$rootIndex") "$displayName d6") $d6 "$displayName root=$rootIndex d6"
        Assert-Equal (Convert-ToInt (Get-RequiredField $fields 'deadline_ms' "$displayName root=$rootIndex") "$displayName deadline_ms") $deadlineMs "$displayName root=$rootIndex deadline_ms"
        $status = Get-RequiredField $fields 'status' "$displayName root=$rootIndex"
        if ($status -cnotin @('WIN', 'UNKNOWN', 'LOSS', 'INCOMPLETE')) {
            throw "$displayName root=$rootIndex has invalid status '$status'"
        }
        $terminal = Convert-ToBoolText (Get-RequiredField $fields 'terminal' "$displayName root=$rootIndex") "$displayName terminal"
        $source = Get-RequiredField $fields 'source' "$displayName root=$rootIndex"
        if ($source -cnotin @('direct_outcome', 'bounded_reference')) {
            throw "$displayName root=$rootIndex has invalid source '$source'"
        }
        $nodes = Convert-ToUInt64 (Get-RequiredField $fields 'nodes' "$displayName root=$rootIndex") "$displayName nodes"
        $wall = Convert-ToDecimal (Get-RequiredField $fields 'wall_s' "$displayName root=$rootIndex") "$displayName wall_s"
        if ($wall -lt 0) { throw "$displayName root=$rootIndex has negative wall_s" }
        if ($source -ceq 'direct_outcome') {
            Assert-Equal $terminal $true "$displayName root=$rootIndex direct terminal"
            Assert-Equal $status 'WIN' "$displayName root=$rootIndex direct status"
            Assert-Equal $nodes ([uint64]0) "$displayName root=$rootIndex direct nodes"
        }
        else {
            Assert-Equal $terminal $false "$displayName root=$rootIndex bounded terminal"
        }
        if ($status -ceq 'INCOMPLETE' -and $lineIndex -ne ($lines.Count - 2)) {
            throw "$displayName has rows after INCOMPLETE root=$rootIndex"
        }
        $records += [pscustomobject]@{
            File = $displayName
            FilePath = $resolved
            Case = $caseIndex
            RootIndex = $rootIndex
            Q = $q
            R = $r
            Status = $status
            Nodes = $nodes
            Wall = $wall
            D6 = $d6
            TtBytes = $ttBytes
            DeadlineMs = $deadlineMs
            Source = $source
            CodeId = $codeId
        }
    }

    $doneFields = Get-Fields $lines[$lines.Count - 1]
    Assert-CodeIdLane -Fields $doneFields -ShardCodeId $codeId -Context "$displayName done"
    Assert-Equal (Convert-ToInt (Get-RequiredField $doneFields 'case' "$displayName done") "$displayName done case") $caseIndex "$displayName done case"
    Assert-Equal (Convert-ToInt (Get-RequiredField $doneFields 'start' "$displayName done") "$displayName done start") $start "$displayName done start"
    Assert-Equal (Convert-ToInt (Get-RequiredField $doneFields 'end' "$displayName done") "$displayName done end") $end "$displayName done end"
    Assert-Equal (Get-RequiredField $doneFields 'fingerprint' "$displayName done") $fingerprint "$displayName done fingerprint"
    $doneComplete = Convert-ToInt (Get-RequiredField $doneFields 'complete' "$displayName done") "$displayName done complete"
    $nextStart = Convert-ToInt (Get-RequiredField $doneFields 'next_start' "$displayName done") "$displayName done next_start"
    $doneResult = Get-RequiredField $doneFields 'result' "$displayName done"
    $completeRows = @($records | Where-Object { $_.Status -cne 'INCOMPLETE' }).Count
    $incompleteRows = @($records | Where-Object { $_.Status -ceq 'INCOMPLETE' }).Count
    Assert-Equal $doneComplete $completeRows "$displayName done complete"
    if ($doneResult -ceq 'PASS') {
        Assert-Equal $incompleteRows 0 "$displayName PASS incomplete-row count"
        Assert-Equal $completeRows ($end - $start) "$displayName PASS complete-row count"
        Assert-Equal $nextStart $end "$displayName PASS next_start"
    }
    elseif ($doneResult -ceq 'INCOMPLETE') {
        Assert-Equal $incompleteRows 1 "$displayName INCOMPLETE row count"
        Assert-Equal $records.Count ($completeRows + 1) "$displayName INCOMPLETE total-row count"
        Assert-Equal $nextStart ($start + $completeRows) "$displayName INCOMPLETE next_start"
    }
    else {
        throw "$displayName done has invalid result '$doneResult'"
    }

    return [pscustomobject]@{
        Path = $resolved
        Name = $displayName
        Bytes = [uint64](Get-Item -LiteralPath $resolved).Length
        Attempt = $nameAttempt
        AttemptLabel = $nameAttemptLabel
        FilenameLane = $filenameLane
        CodeId = $codeId
        Sha256 = Get-FileSha256 $resolved
        Case = $caseIndex
        Start = $start
        End = $end
        RootCount = $rootCount
        Fingerprint = $fingerprint
        D6 = $d6
        TtBytes = $ttBytes
        DeadlineMs = $deadlineMs
        Result = $doneResult
        Records = $records
    }
}

function Get-ExpectedMetaLine {
    param(
        [Parameter(Mandatory)] $Shard,
        [Parameter(Mandatory)][string] $RunnerSha256
    )
    $d6Text = $Shard.D6.ToString().ToLowerInvariant()
    return "DOM_B2_D4_SHARD_META version=1 raw=$($Shard.Name) raw_sha256=$($Shard.Sha256) raw_bytes=$($Shard.Bytes) source_snapshot_sha256=$requiredSourceSnapshotSha256 runner_sha256=$RunnerSha256 census_sha256=$requiredCensusSha256 code_id=$requiredCodeId deadline_ms=$($Shard.DeadlineMs) tt_bytes=$($Shard.TtBytes) d6=$d6Text target=x86_64-pc-windows-msvc release=true test_threads=1"
}

function Read-ShardMeta {
    param(
        [Parameter(Mandatory)] $Shard,
        [Parameter(Mandatory)] $Census,
        [Parameter(Mandatory)][hashtable] $ApprovalMap
    )

    Assert-Equal $Shard.CodeId $requiredCodeId "$($Shard.Name) V3 code_id"
    Assert-Equal $Shard.DeadlineMs 480000 "$($Shard.Name) META deadline_ms"
    Assert-Equal $Census.Sha256 $requiredCensusSha256 "$($Shard.Name) META census SHA256"

    $rawSuffix = '_RAW.log'
    if (-not $Shard.Name.EndsWith($rawSuffix, [StringComparison]::Ordinal)) {
        throw "$($Shard.Name) cannot derive META sidecar name"
    }
    $metaName = $Shard.Name.Substring(0, $Shard.Name.Length - $rawSuffix.Length) + '_META_RAW.log'
    $metaPath = Join-Path (Split-Path -Parent $Shard.Path) $metaName
    if (-not (Test-Path -LiteralPath $metaPath)) {
        return $null
    }
    if (-not (Test-Path -LiteralPath $metaPath -PathType Leaf)) {
        throw "$metaName is present but is not a regular META file"
    }

    $configKey = Get-ConfigKey -D6 $Shard.D6 -TtBytes $Shard.TtBytes
    if (-not $ApprovalMap.ContainsKey($configKey)) {
        throw "$metaName is present for unapproved configuration $configKey"
    }
    $approvedRunnerSha256 = [string]$ApprovalMap[$configKey]
    $expectedLine = Get-ExpectedMetaLine -Shard $Shard -RunnerSha256 $approvedRunnerSha256
    [byte[]]$expectedBytes = [Text.Encoding]::ASCII.GetBytes($expectedLine + "`n")
    [byte[]]$actualBytes = [IO.File]::ReadAllBytes($metaPath)
    $matchesExactly = ($actualBytes.Length -eq $expectedBytes.Length)
    if ($matchesExactly) {
        for ($index = 0; $index -lt $actualBytes.Length; $index++) {
            if ($actualBytes[$index] -ne $expectedBytes[$index]) {
                $matchesExactly = $false
                break
            }
        }
    }
    if (-not $matchesExactly) {
        throw "$metaName is not the exact one-LF/no-BOM approved META row for $($Shard.Name)"
    }

    return [pscustomobject]@{
        Path = $metaPath
        Name = $metaName
        Sha256 = Get-FileSha256 $metaPath
        Bytes = [uint64]$actualBytes.Length
        RunnerSha256 = $approvedRunnerSha256
        ConfigKey = $configKey
    }
}

function Get-OutcomeRank {
    param([string] $Status)
    switch -CaseSensitive ($Status) {
        'WIN' { return 0 }
        'UNKNOWN' { return 1 }
        'LOSS' { return 2 }
        default { throw "no defender rank for '$Status'" }
    }
}

function New-LaneSummary {
    param(
        [Parameter(Mandatory)][string] $ConfigKey,
        [Parameter(Mandatory)][string] $Label,
        [Parameter()][AllowEmptyCollection()][object[]] $Shards,
        [Parameter(Mandatory)] $Census
    )

    $laneShards = @($Shards)
    $recordsByRoot = @{}
    foreach ($shard in $laneShards) {
        foreach ($record in $shard.Records) {
            $key = "$($record.Case):$($record.RootIndex)"
            if ($recordsByRoot.ContainsKey($key)) {
                $recordsByRoot[$key] = @($recordsByRoot[$key]) + @($record)
            }
            else {
                $recordsByRoot[$key] = @($record)
            }
        }
    }

    $caseResults = @{}
    $selectedByRoot = @{}
    $totalUniqueComplete = 0
    $totalMissing = 0
    $totalIncompleteOnly = 0
    $totalDuplicateComplete = 0
    $totalRoots = 0
    for ($caseIndex = 0; $caseIndex -lt $expectedCases.Count; $caseIndex++) {
        $universe = $Census.Universes[$caseIndex]
        $win = 0
        $unknown = 0
        $loss = 0
        $uniqueComplete = 0
        $missing = 0
        $incompleteOnly = 0
        $incompleteRows = 0
        $duplicateComplete = 0
        $attemptRows = 0
        $attemptNodes = [decimal]0
        $uniqueNodes = [decimal]0
        $attemptWall = [decimal]0
        $totalRoots += $universe.RootCount
        for ($rootIndex = 0; $rootIndex -lt $universe.RootCount; $rootIndex++) {
            $key = "$caseIndex`:$rootIndex"
            if (-not $recordsByRoot.ContainsKey($key)) {
                $missing++
                continue
            }
            $rows = @($recordsByRoot[$key])
            $attemptRows += $rows.Count
            foreach ($row in $rows) {
                $attemptNodes += [decimal]$row.Nodes
                $attemptWall += [decimal]$row.Wall
                if ($row.Status -ceq 'INCOMPLETE') { $incompleteRows++ }
            }
            $completeRows = @($rows | Where-Object { $_.Status -cne 'INCOMPLETE' } | Sort-Object File, RootIndex)
            $completeStatuses = @($completeRows | Select-Object -ExpandProperty Status -Unique)
            if ($completeStatuses.Count -gt 1) {
                $sources = @($completeRows | ForEach-Object { "$($_.File):$($_.Status)" }) -join ','
                throw "exact-status disagreement inside lane $ConfigKey root=$key sources=$sources"
            }
            if ($completeRows.Count -eq 0) {
                $incompleteOnly++
                continue
            }
            $uniqueComplete++
            if ($completeRows.Count -gt 1) { $duplicateComplete += ($completeRows.Count - 1) }
            $selected = $completeRows[0]
            $selectedByRoot[$key] = $selected
            $uniqueNodes += [decimal]$selected.Nodes
            switch -CaseSensitive ($selected.Status) {
                'WIN' { $win++ }
                'UNKNOWN' { $unknown++ }
                'LOSS' { $loss++ }
            }
        }
        $coverageExact = ($uniqueComplete -eq $universe.RootCount)
        if ($coverageExact) {
            if ($win -gt 0) { $aggregate = 'WIN' }
            elseif ($unknown -gt 0) { $aggregate = 'UNKNOWN' }
            else { $aggregate = 'LOSS' }
            $gateStatus = $aggregate
        }
        else {
            if ($win -gt 0) { $aggregate = 'WIN' } else { $aggregate = 'INCOMPLETE' }
            $gateStatus = 'INCOMPLETE'
        }
        $caseResults[$caseIndex] = [pscustomobject]@{
            Case = $caseIndex
            RootCount = $universe.RootCount
            Fingerprint = $universe.Fingerprint
            UniqueComplete = $uniqueComplete
            Missing = $missing
            IncompleteOnly = $incompleteOnly
            IncompleteRows = $incompleteRows
            DuplicateComplete = $duplicateComplete
            AttemptRows = $attemptRows
            Win = $win
            Unknown = $unknown
            Loss = $loss
            Aggregate = $aggregate
            GateStatus = $gateStatus
            CoverageExact = $coverageExact
            UniqueNodes = $uniqueNodes
            AttemptNodes = $attemptNodes
            AttemptWall = $attemptWall
        }
        $totalUniqueComplete += $uniqueComplete
        $totalMissing += $missing
        $totalIncompleteOnly += $incompleteOnly
        $totalDuplicateComplete += $duplicateComplete
    }
    $exactCases = @($caseResults.Values | Where-Object { $_.CoverageExact }).Count
    return [pscustomobject]@{
        ConfigKey = $ConfigKey
        Label = $Label
        Shards = $laneShards
        RecordsByRoot = $recordsByRoot
        SelectedByRoot = $selectedByRoot
        CaseResults = $caseResults
        ExactCases = $exactCases
        AllExact = ($exactCases -eq $expectedCases.Count)
        TotalRoots = $totalRoots
        TotalUniqueComplete = $totalUniqueComplete
        TotalMissing = $totalMissing
        TotalIncompleteOnly = $totalIncompleteOnly
        TotalDuplicateComplete = $totalDuplicateComplete
    }
}

function Get-LaneEvaluation {
    param([Parameter(Mandatory)] $Summary)

    $specs = @(
        [pscustomobject]@{ Name = 'K1_32F'; Split = 0; H = @(1, 2) },
        [pscustomobject]@{ Name = 'K1_19B'; Split = 3; H = @(4) },
        [pscustomobject]@{ Name = 'K1_498A'; Split = 5; H = @(6) },
        [pscustomobject]@{ Name = 'K1_FD68'; Split = 7; H = @(8) }
    )
    $comparisons = @()
    foreach ($spec in $specs) {
        $involved = @($spec.Split) + @($spec.H)
        $exact = $true
        foreach ($caseIndex in $involved) {
            if (-not $Summary.CaseResults[$caseIndex].CoverageExact) { $exact = $false }
        }
        $splitStatus = 'INCOMPLETE'
        $splitRank = -1
        $bestHStatus = 'INCOMPLETE'
        $bestHRank = -1
        if (-not $exact) {
            $state = 'PENDING_INCOMPLETE'
        }
        else {
            $splitStatus = $Summary.CaseResults[$spec.Split].GateStatus
            $splitRank = Get-OutcomeRank $splitStatus
            foreach ($hCase in $spec.H) {
                $hStatus = $Summary.CaseResults[$hCase].GateStatus
                $hRank = Get-OutcomeRank $hStatus
                if ($hRank -gt $bestHRank) {
                    $bestHRank = $hRank
                    $bestHStatus = $hStatus
                }
            }
            if ($splitRank -le $bestHRank) { $state = 'SATISFIED' } else { $state = 'REVERSED' }
        }
        $comparisons += [pscustomobject]@{
            Name = $spec.Name
            Split = $spec.Split
            H = @($spec.H)
            Involved = $involved
            Exact = $exact
            State = $state
            SplitStatus = $splitStatus
            SplitRank = $splitRank
            BestHStatus = $bestHStatus
            BestHRank = $bestHRank
        }
    }

    if ($Summary.CaseResults[9].CoverageExact -and $Summary.CaseResults[10].CoverageExact) {
        $controlLeft = $Summary.CaseResults[9].GateStatus
        $controlRight = $Summary.CaseResults[10].GateStatus
        if ($controlLeft -ceq $controlRight) { $controlState = 'MATCH' } else { $controlState = 'MISMATCH' }
    }
    else {
        $controlLeft = 'INCOMPLETE'
        $controlRight = 'INCOMPLETE'
        $controlState = 'PENDING_INCOMPLETE'
    }

    if ($Summary.CaseResults[0].CoverageExact) {
        $historyObserved = $Summary.CaseResults[0].GateStatus
        if ($historyObserved -ceq 'UNKNOWN') { $historyState = 'MATCH' } else { $historyState = 'DISCREPANCY' }
    }
    else {
        $historyObserved = 'INCOMPLETE'
        $historyState = 'PENDING_INCOMPLETE'
    }

    return [pscustomobject]@{
        ConfigKey = $Summary.ConfigKey
        Label = $Summary.Label
        Comparisons = $comparisons
        ControlState = $controlState
        ControlLeft = $controlLeft
        ControlRight = $controlRight
        HistoryState = $historyState
        HistoryObserved = $historyObserved
    }
}

function Compare-LaneRoots {
    param(
        [Parameter(Mandatory)] $Primary,
        [Parameter(Mandatory)] $Replica,
        [Parameter(Mandatory)] $Census
    )

    $comparable = 0
    $matches = 0
    $mismatches = 0
    $primaryMissingInReplica = 0
    $replicaMissingInPrimary = 0
    for ($caseIndex = 0; $caseIndex -lt $expectedCases.Count; $caseIndex++) {
        $universe = $Census.Universes[$caseIndex]
        for ($rootIndex = 0; $rootIndex -lt $universe.RootCount; $rootIndex++) {
            $key = "$caseIndex`:$rootIndex"
            $primaryHas = $Primary.SelectedByRoot.ContainsKey($key)
            $replicaHas = $Replica.SelectedByRoot.ContainsKey($key)
            if ($primaryHas -and $replicaHas) {
                $comparable++
                if ($Primary.SelectedByRoot[$key].Status -ceq $Replica.SelectedByRoot[$key].Status) {
                    $matches++
                }
                else {
                    $mismatches++
                }
            }
            elseif ($primaryHas) {
                $primaryMissingInReplica++
            }
            elseif ($replicaHas) {
                $replicaMissingInPrimary++
            }
        }
    }
    return [pscustomobject]@{
        Comparable = $comparable
        Matches = $matches
        Mismatches = $mismatches
        PrimaryMissingInReplica = $primaryMissingInReplica
        ReplicaMissingInPrimary = $replicaMissingInPrimary
        ExactAgreement = ($Primary.AllExact -and $Replica.AllExact -and $matches -eq $Primary.TotalRoots -and $mismatches -eq 0)
    }
}

function Test-CaseReplication {
    param(
        [Parameter(Mandatory)] $Primary,
        [Parameter(Mandatory)] $Replica,
        [Parameter(Mandatory)][int[]] $Cases
    )
    foreach ($caseIndex in $Cases) {
        $primaryCase = $Primary.CaseResults[$caseIndex]
        $replicaCase = $Replica.CaseResults[$caseIndex]
        if (-not $primaryCase.CoverageExact -or -not $replicaCase.CoverageExact -or
            $primaryCase.GateStatus -cne $replicaCase.GateStatus) {
            return $false
        }
    }
    return $true
}

function Invoke-AnalyzerSelfTest {
    $d6OffSha = $requiredReplicaRunnerSha256
    # One frozen replica runner may legitimately serve both exact replica
    # configurations; trust and data coverage remain keyed by configuration.
    $secondTtSha = $d6OffSha
    $map = Read-ApprovalMap @(
        "d6=true,tt_bytes=$primaryTtBytes,runner_sha256=$requiredPrimaryRunnerSha256",
        "d6=false,tt_bytes=$primaryTtBytes,runner_sha256=$d6OffSha",
        "d6=true,tt_bytes=$secondTtBytes,runner_sha256=$secondTtSha"
    )
    Assert-ApprovalBindings -ApprovalMap $map
    $primaryKey = Get-ConfigKey -D6 $true -TtBytes $primaryTtBytes
    $secondKey = Get-ConfigKey -D6 $true -TtBytes $secondTtBytes
    Assert-Equal $map[$primaryKey] $requiredPrimaryRunnerSha256 'self-test primary approval'
    Assert-Equal $map[$secondKey] $secondTtSha 'self-test second-TT approval'

    $primaryNameMatch = $shardFileRegex.Match('DOM_B2_D4_SHARD_C00_S0000_A02_RAW.log')
    $d6OffNameMatch = $shardFileRegex.Match('DOM_B2_D4_REPLICA_D6_OFF_SHARD_C10_S0382_A07_RAW.log')
    $secondTtNameMatch = $shardFileRegex.Match('DOM_B2_D4_REPLICA_SECOND_TT_SHARD_C01_S0000_A00_RAW.log')
    Assert-Equal $primaryNameMatch.Success $true 'self-test primary filename schema'
    Assert-Equal $primaryNameMatch.Groups['lane'].Success $false 'self-test primary filename lane'
    Assert-Equal $d6OffNameMatch.Groups['lane'].Value 'D6_OFF' 'self-test D6-off filename lane'
    Assert-Equal $secondTtNameMatch.Groups['lane'].Value 'SECOND_TT' 'self-test second-TT filename lane'
    Assert-Equal $shardFileRegex.IsMatch('DOM_B2_D4_REPLICA_OTHER_SHARD_C00_S0000_A00_RAW.log') $false 'self-test unsupported replica filename rejection'
    Assert-FilenameLaneConfiguration -FilenameLane PRIMARY -D6 $true -TtBytes $primaryTtBytes -CodeId $requiredCodeId -DisplayName self_test_primary
    Assert-FilenameLaneConfiguration -FilenameLane D6_OFF -D6 $false -TtBytes $primaryTtBytes -CodeId $requiredCodeId -DisplayName self_test_d6_off
    Assert-FilenameLaneConfiguration -FilenameLane SECOND_TT -D6 $true -TtBytes $secondTtBytes -CodeId $requiredCodeId -DisplayName self_test_second_tt
    $laneMismatchRejected = $false
    try {
        Assert-FilenameLaneConfiguration -FilenameLane SECOND_TT -D6 $true -TtBytes $primaryTtBytes -CodeId $requiredCodeId -DisplayName self_test_bad_lane
    }
    catch {
        $laneMismatchRejected = $true
    }
    Assert-Equal $laneMismatchRejected $true 'self-test filename/config mismatch rejection'

    $fakeShard = [pscustomobject]@{
        Name = 'DOM_B2_D4_SHARD_C00_S0000_A00_RAW.log'
        Sha256 = ('C' * 64) -join ''
        Bytes = [uint64]1234
        DeadlineMs = 480000
        TtBytes = $secondTtBytes
        D6 = $true
    }
    $metaLine = Get-ExpectedMetaLine -Shard $fakeShard -RunnerSha256 $map[$secondKey]
    if ($metaLine -cnotmatch "runner_sha256=$secondTtSha" -or
        $metaLine -cnotmatch "tt_bytes=$secondTtBytes d6=true" -or
        $metaLine -cmatch "tt_bytes=$primaryTtBytes d6=true") {
        throw 'self-test generic META construction did not preserve the second-TT lane'
    }
    $duplicateRejected = $false
    try {
        $null = Read-ApprovalMap @(
            "d6=true,tt_bytes=$primaryTtBytes,runner_sha256=$requiredPrimaryRunnerSha256",
            "d6=true,tt_bytes=$primaryTtBytes,runner_sha256=$requiredPrimaryRunnerSha256"
        )
    }
    catch {
        $duplicateRejected = $true
    }
    Assert-Equal $duplicateRejected $true 'self-test duplicate approval rejection'
    $substitutionRejected = $false
    try {
        $substituteSha = ('D' * 64) -join ''
        $substituteMap = Read-ApprovalMap @(
            "d6=true,tt_bytes=$primaryTtBytes,runner_sha256=$requiredPrimaryRunnerSha256",
            "d6=false,tt_bytes=$primaryTtBytes,runner_sha256=$substituteSha",
            "d6=true,tt_bytes=$secondTtBytes,runner_sha256=$requiredReplicaRunnerSha256"
        )
        Assert-ApprovalBindings -ApprovalMap $substituteMap
    }
    catch {
        $substitutionRejected = $true
    }
    Assert-Equal $substitutionRejected $true 'self-test frozen runner substitution rejection'

    $universes = @{}
    $roots = @{}
    $records = @()
    for ($caseIndex = 0; $caseIndex -lt 11; $caseIndex++) {
        $universes[$caseIndex] = [pscustomobject]@{ RootCount = 1; Fingerprint = ('{0:X16}' -f $caseIndex) }
        $roots[$caseIndex] = @{ 0 = [pscustomobject]@{ Q = $caseIndex; R = 0 } }
        $records += [pscustomobject]@{
            File = "synthetic_C$caseIndex"
            Case = $caseIndex
            RootIndex = 0
            Status = 'UNKNOWN'
            Nodes = [uint64]1
            Wall = [decimal]0.1
        }
    }
    $syntheticCensus = [pscustomobject]@{ Universes = $universes; Roots = $roots }
    $primary = New-LaneSummary -ConfigKey $primaryKey -Label PRIMARY -Shards @([pscustomobject]@{ Records = $records }) -Census $syntheticCensus
    Assert-Equal $primary.AllExact $true 'self-test primary exactness'
    Assert-Equal $primary.TotalUniqueComplete 11 'self-test primary unique roots'
    $evaluation = Get-LaneEvaluation -Summary $primary
    Assert-Equal @($evaluation.Comparisons | Where-Object { $_.State -ceq 'SATISFIED' }).Count 4 'self-test comparison count'
    Assert-Equal $evaluation.ControlState 'MATCH' 'self-test equality control'

    $partialRecords = @($records | Where-Object { $_.Case -ne 10 })
    $partial = New-LaneSummary -ConfigKey $secondKey -Label SECOND_TT -Shards @([pscustomobject]@{ Records = $partialRecords }) -Census $syntheticCensus
    Assert-Equal $partial.AllExact $false 'self-test partial replica exactness'
    $rootComparison = Compare-LaneRoots -Primary $primary -Replica $partial -Census $syntheticCensus
    Assert-Equal $rootComparison.Matches 10 'self-test comparable root matches'
    Assert-Equal $rootComparison.PrimaryMissingInReplica 1 'self-test missing replica root'
    Assert-Equal $rootComparison.ExactAgreement $false 'self-test partial root fence'

    Write-Output 'DOM_B2_D4_ANALYZER_SELF_TEST approval_map=PASS frozen_runner_bindings=PASS filename_schema=PASS generic_meta=PASS duplicate_map=PASS lane_aggregation=PASS root_comparison=PASS result=PASS'
}

if ($SelfTest) {
    try {
        Invoke-AnalyzerSelfTest
        exit 0
    }
    catch {
        Write-Output "DOM_B2_D4_ANALYZER_SELF_TEST result=FAIL error=$($_.Exception.Message.Replace(' ', '_'))"
        exit 2
    }
}

try {
    $approvalMap = Read-ApprovalMap -Entries $ApprovedRunnerMapping
    $primaryConfigKey = Get-ConfigKey -D6 $true -TtBytes $primaryTtBytes
    $d6OffConfigKey = Get-ConfigKey -D6 $false -TtBytes $primaryTtBytes
    $secondTtConfigKey = Get-ConfigKey -D6 $true -TtBytes $secondTtBytes
    Assert-ApprovalBindings -ApprovalMap $approvalMap
}
catch {
    Write-Output "DOM_B2_D4_ANALYZER_ABORT stage=APPROVAL_MAP error=$($_.Exception.Message.Replace(' ', '_'))"
    exit 2
}

try {
    $census = Read-Census -LiteralPath $CensusPath
}
catch {
    Write-Output "DOM_B2_D4_ANALYZER_ABORT stage=CENSUS error=$($_.Exception.Message.Replace(' ', '_'))"
    exit 2
}

$discoveryIgnored = @()
if ($null -eq $ShardPath -or $ShardPath.Count -eq 0) {
    $candidates = @(Get-ChildItem -LiteralPath $ShardDirectory -File -Filter 'DOM_B2_D4_*SHARD_C*_RAW.log' |
        Sort-Object -Property FullName)
    $ShardPath = @($candidates |
        Where-Object { $shardFileRegex.IsMatch($_.Name) } |
        ForEach-Object { $_.FullName })
    $discoveryIgnored = @($candidates |
        Where-Object { -not $shardFileRegex.IsMatch($_.Name) } |
        ForEach-Object { $_.Name })
}
else {
    $ShardPath = @($ShardPath | ForEach-Object { (Resolve-Path -LiteralPath $_).Path } | Sort-Object)
}

Write-Output "DOM_B2_D4_ANALYSIS_SETUP census_sha256=$($census.Sha256) census_path=$($census.Path) census_hard_bind=PASS shard_name_regex=$shardFileRegexText shard_files=$($ShardPath.Count) discovery_ignored=$($discoveryIgnored.Count) required_code_id=$requiredCodeId meta_version=1 meta_required_for_credit=true missing_meta_disposition=V3_UNTRUSTED_NO_META_EXCLUDED source_snapshot_sha256=$requiredSourceSnapshotSha256 approval_configs=$($approvalMap.Count) primary_config=$primaryConfigKey d6_off_config=$d6OffConfigKey second_tt_config=$secondTtConfigKey"
foreach ($configKey in @($approvalMap.Keys | Sort-Object)) {
    $label = Get-LaneLabel -ConfigKey $configKey
    Write-Output "DOM_B2_D4_RUNNER_APPROVAL lane=$label config=$configKey runner_sha256=$($approvalMap[$configKey]) source=EXPLICIT_MAPPING"
}
foreach ($name in $discoveryIgnored) {
    if ($name.EndsWith('_META_RAW.log', [StringComparison]::Ordinal)) {
        $ignoreReason = 'META_SIDECAR'
    }
    else {
        $ignoreReason = 'FILENAME_NOT_DATA_RAW'
    }
    Write-Output "DOM_B2_D4_DISCOVERY_IGNORED file=$name reason=$ignoreReason"
}

$creditedShardsByConfig = @{}
$primaryShards = @()
$replicaShards = @()
$legacyShards = @()
$untrustedV3Shards = @()
$validationErrors = @()
$duplicateInputPaths = @($ShardPath | Group-Object | Where-Object { $_.Count -gt 1 })
foreach ($duplicate in $duplicateInputPaths) {
    $validationErrors += "duplicate shard input path=$($duplicate.Name) count=$($duplicate.Count)"
}
foreach ($path in $ShardPath) {
    try {
        $shard = Read-Shard -LiteralPath $path -Census $census
        $meta = $null
        if ($null -ne $shard.CodeId) {
            $meta = Read-ShardMeta -Shard $shard -Census $census -ApprovalMap $approvalMap
        }
        if ($null -eq $shard.CodeId) {
            $legacyShards += $shard
            Write-Output "DOM_B2_D4_LEGACY_SHARD_VALID file=$($shard.Name) sha256=$($shard.Sha256) attempt=$($shard.AttemptLabel) case=$($shard.Case) start=$($shard.Start) end=$($shard.End) result=$($shard.Result) d6=$($shard.D6.ToString().ToLowerInvariant()) tt_bytes=$($shard.TtBytes) code_id=ABSENT meta_required=false role=LEGACY_PRE_V3_EXCLUDED coverage_effect=NONE"
            continue
        }
        if ($null -eq $meta) {
            $untrustedV3Shards += $shard
            $expectedMetaName = [regex]::Replace($shard.Name, '_RAW[.]log$', '_META_RAW.log')
            Write-Output "DOM_B2_D4_V3_UNTRUSTED_SHARD_VALID file=$($shard.Name) sha256=$($shard.Sha256) bytes=$($shard.Bytes) expected_meta=$expectedMetaName meta=ABSENT attempt=$($shard.AttemptLabel) case=$($shard.Case) start=$($shard.Start) end=$($shard.End) result=$($shard.Result) code_id=$($shard.CodeId) d6=$($shard.D6.ToString().ToLowerInvariant()) tt_bytes=$($shard.TtBytes) role=V3_UNTRUSTED_NO_META_EXCLUDED coverage_effect=NONE"
            continue
        }

        $configKey = Get-ConfigKey -D6 $shard.D6 -TtBytes $shard.TtBytes
        Assert-Equal $meta.ConfigKey $configKey "$($shard.Name) META/config lane"
        $shard | Add-Member -NotePropertyName ConfigKey -NotePropertyValue $configKey
        $shard | Add-Member -NotePropertyName RunnerSha256 -NotePropertyValue $meta.RunnerSha256
        if ($creditedShardsByConfig.ContainsKey($configKey)) {
            $creditedShardsByConfig[$configKey] = @($creditedShardsByConfig[$configKey]) + @($shard)
        }
        else {
            $creditedShardsByConfig[$configKey] = @($shard)
        }
        $label = Get-LaneLabel -ConfigKey $configKey
        if ($configKey -ceq $primaryConfigKey) {
            $primaryShards += $shard
            $role = 'PRIMARY_INCLUDED'
        }
        else {
            $replicaShards += $shard
            $role = 'REPLICA_INCLUDED_IN_OWN_LANE'
        }
        Write-Output "DOM_B2_D4_CREDITED_SHARD_VALID file=$($shard.Name) sha256=$($shard.Sha256) bytes=$($shard.Bytes) meta=$($meta.Name) meta_sha256=$($meta.Sha256) runner_sha256=$($meta.RunnerSha256) attempt=$($shard.AttemptLabel) case=$($shard.Case) start=$($shard.Start) end=$($shard.End) result=$($shard.Result) code_id=$($shard.CodeId) lane=$label config=$configKey role=$role"
    }
    catch {
        $leaf = Split-Path -Leaf $path
        $validationErrors += ('{0}: {1}' -f $leaf, $_.Exception.Message)
    }
}

$allCreditedByRoot = @{}
foreach ($configKey in @($creditedShardsByConfig.Keys | Sort-Object)) {
    foreach ($shard in @($creditedShardsByConfig[$configKey])) {
        foreach ($record in $shard.Records) {
            $rootKey = '{0}:{1}' -f $record.Case, $record.RootIndex
            $entry = [pscustomobject]@{
                ConfigKey = $configKey
                File = $record.File
                Status = $record.Status
            }
            if ($allCreditedByRoot.ContainsKey($rootKey)) {
                $allCreditedByRoot[$rootKey] = @($allCreditedByRoot[$rootKey]) + @($entry)
            }
            else {
                $allCreditedByRoot[$rootKey] = @($entry)
            }
        }
    }
}
foreach ($rootKey in @($allCreditedByRoot.Keys | Sort-Object)) {
    $completeRecords = @($allCreditedByRoot[$rootKey] | Where-Object { $_.Status -cne 'INCOMPLETE' })
    $completeStatuses = @($completeRecords | Select-Object -ExpandProperty Status -Unique)
    if ($completeStatuses.Count -gt 1) {
        $sources = @($completeRecords | ForEach-Object { "$($_.ConfigKey)/$($_.File):$($_.Status)" }) -join ','
        $validationErrors += "exact-status disagreement across credited lanes root=$rootKey sources=$sources"
    }
}

if ($validationErrors.Count -gt 0) {
    foreach ($message in @($validationErrors | Sort-Object)) {
        Write-Output "DOM_B2_D4_VALIDATION_ERROR message=$($message.Replace(' ', '_'))"
    }
    Write-Output "DOM_B2_D4_ANALYSIS_DONE validation=FAIL errors=$($validationErrors.Count) matrix=HARD_ABORT"
    exit 2
}

$legacyGroups = @($legacyShards |
    Group-Object -Property { Get-ConfigKey -D6 $_.D6 -TtBytes $_.TtBytes } |
    Sort-Object -Property Name)
foreach ($group in $legacyGroups) {
    $groupRows = @($group.Group | ForEach-Object { $_.Records })
    $completeKeys = @($groupRows |
        Where-Object { $_.Status -cne 'INCOMPLETE' } |
        ForEach-Object { '{0}:{1}' -f $_.Case, $_.RootIndex } |
        Select-Object -Unique)
    $incompleteRows = @($groupRows | Where-Object { $_.Status -ceq 'INCOMPLETE' }).Count
    Write-Output "DOM_B2_D4_LEGACY_SET code_id=ABSENT lane=LEGACY_PRE_V3_EXCLUDED config=$($group.Name) files=$($group.Count) records=$($groupRows.Count) unique_complete_roots=$($completeKeys.Count) incomplete_rows=$incompleteRows coverage_effect=NONE"
}
$untrustedV3Groups = @($untrustedV3Shards |
    Group-Object -Property { Get-ConfigKey -D6 $_.D6 -TtBytes $_.TtBytes } |
    Sort-Object -Property Name)
foreach ($group in $untrustedV3Groups) {
    $groupRows = @($group.Group | ForEach-Object { $_.Records })
    $completeKeys = @($groupRows |
        Where-Object { $_.Status -cne 'INCOMPLETE' } |
        ForEach-Object { '{0}:{1}' -f $_.Case, $_.RootIndex } |
        Select-Object -Unique)
    $incompleteRows = @($groupRows | Where-Object { $_.Status -ceq 'INCOMPLETE' }).Count
    Write-Output "DOM_B2_D4_V3_UNTRUSTED_SET code_id=$requiredCodeId lane=V3_UNTRUSTED_NO_META_EXCLUDED config=$($group.Name) files=$($group.Count) records=$($groupRows.Count) unique_complete_roots=$($completeKeys.Count) incomplete_rows=$incompleteRows coverage_effect=NONE"
}

$laneKeySet = @{}
foreach ($configKey in $approvalMap.Keys) { $laneKeySet[$configKey] = $true }
$laneKeySet[$primaryConfigKey] = $true
$laneKeySet[$d6OffConfigKey] = $true
$laneKeySet[$secondTtConfigKey] = $true
$laneSummaries = @{}
$laneEvaluations = @{}
foreach ($configKey in @($laneKeySet.Keys | Sort-Object)) {
    if ($creditedShardsByConfig.ContainsKey($configKey)) {
        $laneShards = @($creditedShardsByConfig[$configKey])
    }
    else {
        $laneShards = @()
    }
    $label = Get-LaneLabel -ConfigKey $configKey
    $summary = New-LaneSummary -ConfigKey $configKey -Label $label -Shards $laneShards -Census $census
    $evaluation = Get-LaneEvaluation -Summary $summary
    $laneSummaries[$configKey] = $summary
    $laneEvaluations[$configKey] = $evaluation
    $approvalState = if ($approvalMap.ContainsKey($configKey)) { 'SUPPLIED' } else { 'NOT_SUPPLIED' }
    $runnerSha = if ($approvalMap.ContainsKey($configKey)) { [string]$approvalMap[$configKey] } else { 'ABSENT' }
    $recordCount = @($laneShards | ForEach-Object { $_.Records }).Count
    Write-Output "DOM_B2_D4_LANE_SET lane=$label config=$configKey approval=$approvalState runner_sha256=$runnerSha files=$($laneShards.Count) records=$recordCount code_id=$requiredCodeId conflict_free=true"

    for ($caseIndex = 0; $caseIndex -lt $expectedCases.Count; $caseIndex++) {
        $caseResult = $summary.CaseResults[$caseIndex]
        $coverage = if ($caseResult.CoverageExact) { 'EXACT' } else { 'INCOMPLETE' }
        Write-Output ("DOM_B2_D4_LANE_CASE lane={0} config={1} case={2} root_count={3} fingerprint={4} unique_complete={5} missing={6} incomplete_only={7} incomplete_rows={8} duplicate_complete={9} attempt_rows={10} win={11} unknown={12} loss={13} recurrence={14} gate_status={15} coverage={16} unique_nodes={17} attempt_nodes={18} attempt_wall_s={19}" -f
            $label, $configKey, $caseIndex, $caseResult.RootCount, $caseResult.Fingerprint,
            $caseResult.UniqueComplete, $caseResult.Missing, $caseResult.IncompleteOnly,
            $caseResult.IncompleteRows, $caseResult.DuplicateComplete, $caseResult.AttemptRows,
            $caseResult.Win, $caseResult.Unknown, $caseResult.Loss, $caseResult.Aggregate,
            $caseResult.GateStatus, $coverage, $caseResult.UniqueNodes.ToString('0', $invariant),
            $caseResult.AttemptNodes.ToString('0', $invariant),
            $caseResult.AttemptWall.ToString('F6', $invariant))
        if ($configKey -ceq $primaryConfigKey) {
            Write-Output ("DOM_B2_D4_CASE case={0} root_count={1} fingerprint={2} unique_complete={3} missing={4} incomplete_only={5} incomplete_rows={6} duplicate_complete={7} attempt_rows={8} win={9} unknown={10} loss={11} recurrence={12} gate_status={13} coverage={14} unique_nodes={15} attempt_nodes={16} attempt_wall_s={17}" -f
                $caseIndex, $caseResult.RootCount, $caseResult.Fingerprint,
                $caseResult.UniqueComplete, $caseResult.Missing, $caseResult.IncompleteOnly,
                $caseResult.IncompleteRows, $caseResult.DuplicateComplete, $caseResult.AttemptRows,
                $caseResult.Win, $caseResult.Unknown, $caseResult.Loss, $caseResult.Aggregate,
                $caseResult.GateStatus, $coverage, $caseResult.UniqueNodes.ToString('0', $invariant),
                $caseResult.AttemptNodes.ToString('0', $invariant),
                $caseResult.AttemptWall.ToString('F6', $invariant))
        }
    }
    Write-Output "DOM_B2_D4_LANE_MATRIX lane=$label config=$configKey exact_cases=$($summary.ExactCases) total_cases=11 unique_complete_roots=$($summary.TotalUniqueComplete) total_roots=$($summary.TotalRoots) missing_roots=$($summary.TotalMissing) incomplete_only_roots=$($summary.TotalIncompleteOnly) duplicate_complete_rows=$($summary.TotalDuplicateComplete) gapless=$($summary.AllExact.ToString().ToLowerInvariant()) conflict_free=true"

    foreach ($comparison in $evaluation.Comparisons) {
        $hCases = @($comparison.H) -join ','
        if ($comparison.Exact) {
            Write-Output "DOM_B2_D4_LANE_COMPARISON lane=$label config=$configKey name=$($comparison.Name) split_case=$($comparison.Split) h_cases=$hCases status=$($comparison.State) split=$($comparison.SplitStatus) split_rank=$($comparison.SplitRank) best_h=$($comparison.BestHStatus) best_h_rank=$($comparison.BestHRank) relation=split_rank_le_best_h_rank"
        }
        else {
            Write-Output "DOM_B2_D4_LANE_COMPARISON lane=$label config=$configKey name=$($comparison.Name) split_case=$($comparison.Split) h_cases=$hCases status=$($comparison.State)"
        }
        if ($configKey -ceq $primaryConfigKey) {
            if ($comparison.Exact) {
                Write-Output "DOM_B2_D4_COMPARISON name=$($comparison.Name) split_case=$($comparison.Split) h_cases=$hCases status=$($comparison.State) split=$($comparison.SplitStatus) split_rank=$($comparison.SplitRank) best_h=$($comparison.BestHStatus) best_h_rank=$($comparison.BestHRank) relation=split_rank_le_best_h_rank replication_required=true"
            }
            else {
                Write-Output "DOM_B2_D4_COMPARISON name=$($comparison.Name) split_case=$($comparison.Split) h_cases=$hCases status=$($comparison.State) replication_required=true"
            }
        }
    }
    Write-Output "DOM_B2_D4_LANE_CONTROL lane=$label config=$configKey name=D7_HIT_ANY_EQUALITY cases=9,10 status=$($evaluation.ControlState) case9=$($evaluation.ControlLeft) case10=$($evaluation.ControlRight)"
    Write-Output "DOM_B2_D4_LANE_HISTORY lane=$label config=$configKey case=0 expected=UNKNOWN observed=$($evaluation.HistoryObserved) status=$($evaluation.HistoryState)"
    if ($configKey -ceq $primaryConfigKey) {
        Write-Output "DOM_B2_D4_PRIMARY_SET files=$($laneShards.Count) code_id=$requiredCodeId config=$configKey included_records=$recordCount"
        Write-Output "DOM_B2_D4_CONTROL name=D7_HIT_ANY_EQUALITY cases=9,10 status=$($evaluation.ControlState) case9=$($evaluation.ControlLeft) case10=$($evaluation.ControlRight) replication_required=true"
        $historyRequired = ($evaluation.HistoryState -cne 'MATCH').ToString().ToLowerInvariant()
        Write-Output "DOM_B2_D4_HISTORY_CHECK case=0 expected=UNKNOWN observed=$($evaluation.HistoryObserved) status=$($evaluation.HistoryState) replica_d6_off_required=$historyRequired replica_second_tt_required=$historyRequired"
    }
    else {
        Write-Output "DOM_B2_D4_REPLICA_SET code_id=$requiredCodeId lane=$label config=$configKey approval=$approvalState files=$($laneShards.Count) records=$recordCount unique_complete_roots=$($summary.TotalUniqueComplete) incomplete_only_roots=$($summary.TotalIncompleteOnly) primary_coverage_effect=NONE"
    }
}

$primarySummary = $laneSummaries[$primaryConfigKey]
$d6OffSummary = $laneSummaries[$d6OffConfigKey]
$secondTtSummary = $laneSummaries[$secondTtConfigKey]
$primaryEvaluation = $laneEvaluations[$primaryConfigKey]
$d6OffEvaluation = $laneEvaluations[$d6OffConfigKey]
$secondTtEvaluation = $laneEvaluations[$secondTtConfigKey]

$d6OffRootComparison = Compare-LaneRoots -Primary $primarySummary -Replica $d6OffSummary -Census $census
$secondTtRootComparison = Compare-LaneRoots -Primary $primarySummary -Replica $secondTtSummary -Census $census
$d6OffApproval = $approvalMap.ContainsKey($d6OffConfigKey)
$secondTtApproval = $approvalMap.ContainsKey($secondTtConfigKey)
Write-Output "DOM_B2_D4_ROOT_STATUS_FENCE lane=D6_OFF config=$d6OffConfigKey approval_supplied=$($d6OffApproval.ToString().ToLowerInvariant()) comparable=$($d6OffRootComparison.Comparable) matches=$($d6OffRootComparison.Matches) mismatches=$($d6OffRootComparison.Mismatches) primary_complete_missing_in_replica=$($d6OffRootComparison.PrimaryMissingInReplica) replica_complete_missing_in_primary=$($d6OffRootComparison.ReplicaMissingInPrimary) exact_agreement=$($d6OffRootComparison.ExactAgreement.ToString().ToLowerInvariant())"
Write-Output "DOM_B2_D4_ROOT_STATUS_FENCE lane=SECOND_TT config=$secondTtConfigKey approval_supplied=$($secondTtApproval.ToString().ToLowerInvariant()) comparable=$($secondTtRootComparison.Comparable) matches=$($secondTtRootComparison.Matches) mismatches=$($secondTtRootComparison.Mismatches) primary_complete_missing_in_replica=$($secondTtRootComparison.PrimaryMissingInReplica) replica_complete_missing_in_primary=$($secondTtRootComparison.ReplicaMissingInPrimary) exact_agreement=$($secondTtRootComparison.ExactAgreement.ToString().ToLowerInvariant())"

$runnerHashesDistinct = $false
if ($d6OffApproval -and $secondTtApproval) {
    $requiredRunnerHashes = @(
        [string]$approvalMap[$primaryConfigKey],
        [string]$approvalMap[$d6OffConfigKey],
        [string]$approvalMap[$secondTtConfigKey]
    )
    $runnerHashesDistinct = (@($requiredRunnerHashes | Select-Object -Unique).Count -eq 3)
}
$d6RunnerDisplay = if ($d6OffApproval) { $approvalMap[$d6OffConfigKey] } else { 'ABSENT' }
$secondRunnerDisplay = if ($secondTtApproval) { $approvalMap[$secondTtConfigKey] } else { 'ABSENT' }
$allRunnerMappingsSupplied = ($d6OffApproval -and $secondTtApproval)
Write-Output "DOM_B2_D4_RUNNER_INDEPENDENCE primary_runner_sha256=$($approvalMap[$primaryConfigKey]) d6_off_runner_sha256=$d6RunnerDisplay second_tt_runner_sha256=$secondRunnerDisplay all_required_supplied=$($allRunnerMappingsSupplied.ToString().ToLowerInvariant()) pairwise_distinct_diagnostic=$($runnerHashesDistinct.ToString().ToLowerInvariant()) shared_replica_runner_allowed=true independence_basis=EXACT_CONFIG_LANES_AND_SEPARATE_MEASUREMENTS"

$comparisonFenceSatisfied = $primarySummary.AllExact
foreach ($primaryComparison in $primaryEvaluation.Comparisons) {
    $d6Comparison = @($d6OffEvaluation.Comparisons | Where-Object { $_.Name -ceq $primaryComparison.Name })[0]
    $secondComparison = @($secondTtEvaluation.Comparisons | Where-Object { $_.Name -ceq $primaryComparison.Name })[0]
    $d6CasesMatch = Test-CaseReplication -Primary $primarySummary -Replica $d6OffSummary -Cases $primaryComparison.Involved
    $secondCasesMatch = Test-CaseReplication -Primary $primarySummary -Replica $secondTtSummary -Cases $primaryComparison.Involved
    $d6StateMatch = ($primaryComparison.Exact -and $d6Comparison.Exact -and $d6Comparison.State -ceq $primaryComparison.State)
    $secondStateMatch = ($primaryComparison.Exact -and $secondComparison.Exact -and $secondComparison.State -ceq $primaryComparison.State)
    $satisfied = ($d6CasesMatch -and $secondCasesMatch -and $d6StateMatch -and $secondStateMatch)
    if (-not $satisfied) { $comparisonFenceSatisfied = $false }
    Write-Output "DOM_B2_D4_COMPARISON_REPLICA_FENCE name=$($primaryComparison.Name) primary_status=$($primaryComparison.State) d6_off_status=$($d6Comparison.State) d6_off_cases_exact_match=$($d6CasesMatch.ToString().ToLowerInvariant()) d6_off_state_match=$($d6StateMatch.ToString().ToLowerInvariant()) second_tt_status=$($secondComparison.State) second_tt_cases_exact_match=$($secondCasesMatch.ToString().ToLowerInvariant()) second_tt_state_match=$($secondStateMatch.ToString().ToLowerInvariant()) satisfied=$($satisfied.ToString().ToLowerInvariant())"
}

$d6ControlCasesMatch = Test-CaseReplication -Primary $primarySummary -Replica $d6OffSummary -Cases @(9, 10)
$secondControlCasesMatch = Test-CaseReplication -Primary $primarySummary -Replica $secondTtSummary -Cases @(9, 10)
$controlFenceSatisfied = (
    $primaryEvaluation.ControlState -cne 'PENDING_INCOMPLETE' -and
    $d6ControlCasesMatch -and $secondControlCasesMatch -and
    $d6OffEvaluation.ControlState -ceq $primaryEvaluation.ControlState -and
    $secondTtEvaluation.ControlState -ceq $primaryEvaluation.ControlState
)
Write-Output "DOM_B2_D4_CONTROL_REPLICA_FENCE name=D7_HIT_ANY_EQUALITY primary_status=$($primaryEvaluation.ControlState) d6_off_status=$($d6OffEvaluation.ControlState) d6_off_cases_exact_match=$($d6ControlCasesMatch.ToString().ToLowerInvariant()) second_tt_status=$($secondTtEvaluation.ControlState) second_tt_cases_exact_match=$($secondControlCasesMatch.ToString().ToLowerInvariant()) satisfied=$($controlFenceSatisfied.ToString().ToLowerInvariant())"

if ($primaryEvaluation.HistoryState -ceq 'MATCH') {
    $historyFenceState = 'NOT_REQUIRED_PRIMARY_MATCH'
    $historyFenceSatisfied = $true
}
elseif ($primaryEvaluation.HistoryState -ceq 'DISCREPANCY') {
    $d6HistoryMatch = Test-CaseReplication -Primary $primarySummary -Replica $d6OffSummary -Cases @(0)
    $secondHistoryMatch = Test-CaseReplication -Primary $primarySummary -Replica $secondTtSummary -Cases @(0)
    $historyFenceSatisfied = ($d6HistoryMatch -and $secondHistoryMatch)
    if ($historyFenceSatisfied) { $historyFenceState = 'REPLICATED' } else { $historyFenceState = 'PENDING_REPLICAS' }
}
else {
    $historyFenceSatisfied = $false
    $historyFenceState = 'PENDING_PRIMARY'
}
Write-Output "DOM_B2_D4_HISTORY_REPLICA_FENCE primary_status=$($primaryEvaluation.HistoryState) primary_observed=$($primaryEvaluation.HistoryObserved) status=$historyFenceState satisfied=$($historyFenceSatisfied.ToString().ToLowerInvariant())"

$nonUnknownCases = @($primarySummary.CaseResults.Values |
    Where-Object { $_.CoverageExact -and $_.GateStatus -cne 'UNKNOWN' } |
    Sort-Object Case)
$nonUnknownFenceSatisfied = $primarySummary.AllExact
foreach ($caseResult in $nonUnknownCases) {
    $caseIndex = [int]$caseResult.Case
    $d6Match = Test-CaseReplication -Primary $primarySummary -Replica $d6OffSummary -Cases @($caseIndex)
    $secondMatch = Test-CaseReplication -Primary $primarySummary -Replica $secondTtSummary -Cases @($caseIndex)
    $caseSatisfied = ($d6Match -and $secondMatch)
    if (-not $caseSatisfied) { $nonUnknownFenceSatisfied = $false }
    Write-Output "DOM_B2_D4_NON_UNKNOWN_REPLICA_FENCE case=$caseIndex primary_status=$($caseResult.GateStatus) d6_off_exact_match=$($d6Match.ToString().ToLowerInvariant()) second_tt_exact_match=$($secondMatch.ToString().ToLowerInvariant()) satisfied=$($caseSatisfied.ToString().ToLowerInvariant())"
}

$lossCases = @($primarySummary.CaseResults.Values |
    Where-Object { $_.CoverageExact -and $_.GateStatus -ceq 'LOSS' } |
    Sort-Object Case)
if ($lossCases.Count -eq 0) {
    $lossQualificationSatisfied = $true
    $lossQualificationState = 'NOT_REQUIRED'
}
else {
    $lossQualificationSatisfied = $false
    $lossQualificationState = 'PENDING_EXTERNAL_STOCK_FAST_EVIDENCE'
}
$lossCaseText = @($lossCases | ForEach-Object { $_.Case }) -join ','
$lossQualificationRequired = ($lossCases.Count -gt 0)
Write-Output "DOM_B2_D4_LOSS_QUALIFICATION loss_cases=$lossCaseText required=$($lossQualificationRequired.ToString().ToLowerInvariant()) status=$lossQualificationState satisfied=$($lossQualificationSatisfied.ToString().ToLowerInvariant()) analyzer_scope=NO_UNVERIFIED_ATTESTATION_ACCEPTED"

$d6OffLaneFence = ($d6OffApproval -and $d6OffRootComparison.ExactAgreement)
$secondTtLaneFence = ($secondTtApproval -and $secondTtRootComparison.ExactAgreement)
$replicaStructuralFence = (
    $primarySummary.AllExact -and $d6OffLaneFence -and $secondTtLaneFence -and
    $allRunnerMappingsSupplied -and $comparisonFenceSatisfied -and $controlFenceSatisfied -and
    $historyFenceSatisfied -and $nonUnknownFenceSatisfied
)
$chainAuditSatisfied = $false
Write-Output "DOM_B2_D4_CHAIN_AUDIT_FENCE scope=CREDITED_RAW_META_TO_RUNNER_CHAIN required_records=RESULT,GATE,SOURCE_FENCE_POST,BINARY_FENCE,CARGO_EXIT requirement=ONE_TO_ONE_PER_CREDITED_RAW status=EXTERNAL_MECHANICAL_AUDIT_REQUIRED satisfied=false analyzer_inputs=RAW_META_CENSUS_ONLY verdict_effect=PENDING"
$replicaFenceWithoutLoss = ($replicaStructuralFence -and $chainAuditSatisfied)
$replicaFenceSatisfied = ($replicaFenceWithoutLoss -and $lossQualificationSatisfied)
Write-Output "DOM_B2_D4_REPLICA_FENCE code_id=$requiredCodeId compared_inequalities=4 equality_controls=1 d6_off_required=true d6_off_approval=$($d6OffApproval.ToString().ToLowerInvariant()) d6_off_gapless=$($d6OffSummary.AllExact.ToString().ToLowerInvariant()) d6_off_root_exact_agreement=$($d6OffRootComparison.ExactAgreement.ToString().ToLowerInvariant()) second_tt_required=true second_tt_bytes=$secondTtBytes second_tt_approval=$($secondTtApproval.ToString().ToLowerInvariant()) second_tt_gapless=$($secondTtSummary.AllExact.ToString().ToLowerInvariant()) second_tt_root_exact_agreement=$($secondTtRootComparison.ExactAgreement.ToString().ToLowerInvariant()) runner_mappings_complete=$($allRunnerMappingsSupplied.ToString().ToLowerInvariant()) runner_hashes_distinct_diagnostic=$($runnerHashesDistinct.ToString().ToLowerInvariant()) comparison_fence=$($comparisonFenceSatisfied.ToString().ToLowerInvariant()) control_fence=$($controlFenceSatisfied.ToString().ToLowerInvariant()) history_fence=$($historyFenceSatisfied.ToString().ToLowerInvariant()) non_unknown_exact_cases=$($nonUnknownCases.Count) non_unknown_fence=$($nonUnknownFenceSatisfied.ToString().ToLowerInvariant()) loss_cases=$($lossCases.Count) stock_fast_fence=$lossQualificationState structural_fence=$($replicaStructuralFence.ToString().ToLowerInvariant()) external_chain_audit_fence=EXTERNAL_REQUIRED satisfied_without_loss_qualification=$($replicaFenceWithoutLoss.ToString().ToLowerInvariant()) satisfied=$($replicaFenceSatisfied.ToString().ToLowerInvariant())"

$hasReversal = @($primaryEvaluation.Comparisons | Where-Object { $_.State -ceq 'REVERSED' }).Count -gt 0
$hasControlMismatch = ($primaryEvaluation.ControlState -ceq 'MISMATCH')
$allComparisonsSatisfied = @($primaryEvaluation.Comparisons | Where-Object { $_.State -cne 'SATISFIED' }).Count -eq 0
if (-not $primarySummary.AllExact) {
    $primaryMatrix = 'NULL_INCOMPLETE'
}
elseif ($hasReversal -or $hasControlMismatch) {
    if ($replicaFenceSatisfied) {
        $primaryMatrix = 'SCOPED_KILL'
    }
    else {
        $primaryMatrix = 'SCOPED_KILL_CANDIDATE_PENDING_REPLICAS'
    }
}
elseif ($allComparisonsSatisfied -and $primaryEvaluation.ControlState -ceq 'MATCH') {
    if ($replicaFenceSatisfied) {
        $primaryMatrix = 'PASS_REOPEN_EVIDENCE'
    }
    else {
        $primaryMatrix = 'REOPEN_EVIDENCE_PENDING_REPLICAS'
    }
}
else {
    $primaryMatrix = 'NULL_FENCE_UNSATISFIED'
}

Write-Output "DOM_B2_D4_MATRIX primary_files=$($primaryShards.Count) replica_files=$($replicaShards.Count) untrusted_v3_excluded_files=$($untrustedV3Shards.Count) legacy_excluded_files=$($legacyShards.Count) required_code_id=$requiredCodeId primary_config=$primaryConfigKey exact_cases=$($primarySummary.ExactCases) total_cases=11 unique_complete_roots=$($primarySummary.TotalUniqueComplete) total_roots=$($primarySummary.TotalRoots) missing_roots=$($primarySummary.TotalMissing) incomplete_only_roots=$($primarySummary.TotalIncompleteOnly) duplicate_complete_rows=$($primarySummary.TotalDuplicateComplete) primary_matrix=$primaryMatrix replica_fence_satisfied=$($replicaFenceSatisfied.ToString().ToLowerInvariant())"
Write-Output "DOM_B2_D4_ANALYSIS_DONE validation=PASS census_hard_bind=PASS census_identity=PASS shard_identity=PASS lane_conflict_fence=PASS code_id_fence=PASS meta_fence=CREDITED_RAWS_PASS untrusted_v3_excluded=$($untrustedV3Shards.Count) runner_approval_fence=CREDITED_RAWS_PASS chain_audit_fence=EXTERNAL_REQUIRED primary_role_fence=PASS primary_matrix=$primaryMatrix authoritative_scope=fixed_completed_pairs_not_exhaustive_F4"
