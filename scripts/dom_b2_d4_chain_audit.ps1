<#
.SYNOPSIS
Read-only, fail-closed provenance-chain audit for the R-CREL-6 DOM-B2 d=4 queue.

.DESCRIPTION
The caller supplies every queue journal that is allowed to support credit in
each of the three frozen lanes.  The auditor discovers every exact-family META
sidecar in the repository root and requires a one-to-one RESULT in those
explicit journals.  It validates the complete contiguous per-invocation chain,
the raw/META/Cargo-exit bytes and hashes, the frozen source/census/runners and
configuration, and partial or final root coverage.  A RUN without RESULT is
reported as open and uncredited; its raw is deliberately not parsed.

The script never creates, edits, or deletes a file and never invokes Cargo.

.EXAMPLE
./scripts/dom_b2_d4_chain_audit.ps1 `
  -PrimaryJournalPath ./DOM_B2_D4_QUEUE_V3_RUN03_RAW.log

.EXAMPLE
./scripts/dom_b2_d4_chain_audit.ps1 -Final `
  -PrimaryJournalPath ./DOM_B2_D4_QUEUE_V3_RUN03_RAW.log `
  -D6OffJournalPath ./DOM_B2_D4_REPLICA_D6_OFF_QUEUE_RUN01_RAW.log `
  -SecondTtJournalPath ./DOM_B2_D4_REPLICA_SECOND_TT_QUEUE_RUN01_RAW.log
#>
[CmdletBinding(DefaultParameterSetName = 'Audit')]
param(
    [Parameter(ParameterSetName = 'Audit')]
    [string[]] $PrimaryJournalPath = @(),

    [Parameter(ParameterSetName = 'Audit')]
    [string[]] $D6OffJournalPath = @(),

    [Parameter(ParameterSetName = 'Audit')]
    [string[]] $SecondTtJournalPath = @(),

    [Parameter(ParameterSetName = 'Audit')]
    [switch] $Final,

    [Parameter(Mandatory = $true, ParameterSetName = 'SelfTest')]
    [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$invariant = [Globalization.CultureInfo]::InvariantCulture
$repoRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))

$requiredPrimaryRunnerSha256 = '25C6AB359B4BCB3F2207F426BF958A7DB588E1B5BFE7FC0D67FF4AC3EC385D8E'
$requiredReplicaRunnerSha256 = 'B9E61444E1555E648575336B8B6E8CD406CB0FB24C406557106F639E9B1953C8'
$requiredCensusSha256 = '436E9F6C4A93CDB611EEF6495A01F510615174A2897271CA1092A0E7422DD7BE'
$requiredSourceSnapshotSha256 = '1D4FBB37638668D0F2ED1972D27CDDD833826721A2EFD5500BFDC09DCF81B746'
$requiredBinarySha256 = '56B8FA5563D5CDE397133B8328DEB3B79D072E2577573C4C0A94619AA4750A14'
$requiredChildWrapperSha256 = 'A9F8AF43DB7AD2D2D321DBB0E1BCCD5149175885D91A1FEC9F4DB12BC4CA06BC'
$requiredVerifierSha256 = '9990D38618DA2204351E328CA0143BE2AEF98BB3001E4A0462CF346B707F2CE8'
$requiredCodeId = 'DOM_B2_D4_PRIMARY_V3'
$requiredRustc = '1.95.0'
$requiredRustcCommit = '59807616e1fa2540724bfbac14d7976d7e4a3860'
$requiredTarget = 'x86_64-pc-windows-msvc'
$requiredDeadlineMs = 480000
$requiredCargoWallTimeoutMs = 540000
$requiredCount = 64
$requiredSourceFiles = 20
$availableFloor = [int64]10737418240
$freeFloor = [int64]5368709120
$primaryTtBytes = [uint64]536870912
$secondTtBytes = [uint64]268435456
$requiredBinaryRelative = '.target-hunt/x86_64-pc-windows-msvc/release/deps/hexfield_eq-de26e3778420c4c2.exe'

$widths = @(329, 312, 312, 330, 313, 330, 313, 330, 313, 383, 383)
$fingerprints = @(
    '827EEB0FCB78C698', '7C4092D562D3E619', '319A510062631E51',
    '27C689B6D4D0DC33', '09E733EF333378BA', '530A55C8F49F0911',
    '324FB3CEA1CDCA7E', 'CCE0D5A475F109B6', 'F9FC2F2E9BB41D72',
    'FF353A8E0556E088', 'B57A000A7F5C6800'
)
$caseIds = @(
    '32f44c499244b611:9', '32f44c499244b611:9', '32f44c499244b611:9',
    '19b085e7aa9f6215:9', '19b085e7aa9f6215:9',
    '498a61ae0b5cf4ef:9', '498a61ae0b5cf4ef:9',
    'fd688f189544bf72:9', 'fd688f189544bf72:9',
    'd7e1b56c925b7f32:19', 'd7e1b56c925b7f32:19'
)
$casePairs = @(
    '(-2,1);(4,1)', '(2,1);(-2,1)', '(2,1);(4,1)',
    '(-1,0);(5,0)', '(3,0);(-1,0)',
    '(-2,2);(4,-4)', '(2,-2);(-2,2)',
    '(-2,0);(4,0)', '(2,0);(-2,0)',
    '(-1,0);(-2,3)', '(-1,0);(-1,2)'
)
$caseCoverage = @(
    'SPLIT', 'H_CONTAINING', 'H_CONTAINING', 'SPLIT', 'H_CONTAINING',
    'SPLIT', 'H_CONTAINING', 'SPLIT', 'H_CONTAINING',
    'H_CONTAINING', 'H_CONTAINING'
)

$laneSpecs = @{
    PRIMARY = [pscustomobject]@{
        Label = 'PRIMARY'; FileTag = ''; JournalPrefix = 'QUEUE';
        FirstRunId = 3;
        JournalRegex = '^DOM_B2_D4_QUEUE_V3_RUN[0-9]{2,}_RAW[.]log$';
        MetaRegex = '^DOM_B2_D4_SHARD_C(?<case>(?:0[0-9]|10))_S(?<start>[0-9]{4})_A(?<attempt>[0-9]{2})_META_RAW[.]log$';
        RawRegex = '^DOM_B2_D4_SHARD_C(?<case>(?:0[0-9]|10))_S(?<start>[0-9]{4})_A(?<attempt>[0-9]{2})_RAW[.]log$';
        RawTemplate = 'DOM_B2_D4_SHARD_C{0:D2}_S{1:D4}_A{2:D2}_RAW.log';
        D6 = 'true'; TtBytes = $primaryTtBytes; RunnerSha256 = $requiredPrimaryRunnerSha256;
        Replica = $false; ReplicaLane = ''
    }
    D6_OFF = [pscustomobject]@{
        Label = 'D6_OFF'; FileTag = 'D6_OFF'; JournalPrefix = 'DOM_B2_D4_REPLICA_QUEUE';
        FirstRunId = 1;
        JournalRegex = '^DOM_B2_D4_REPLICA_D6_OFF_QUEUE_RUN[0-9]{2,}_RAW[.]log$';
        MetaRegex = '^DOM_B2_D4_REPLICA_D6_OFF_SHARD_C(?<case>(?:0[0-9]|10))_S(?<start>[0-9]{4})_A(?<attempt>[0-9]{2})_META_RAW[.]log$';
        RawRegex = '^DOM_B2_D4_REPLICA_D6_OFF_SHARD_C(?<case>(?:0[0-9]|10))_S(?<start>[0-9]{4})_A(?<attempt>[0-9]{2})_RAW[.]log$';
        RawTemplate = 'DOM_B2_D4_REPLICA_D6_OFF_SHARD_C{0:D2}_S{1:D4}_A{2:D2}_RAW.log';
        D6 = 'false'; TtBytes = $primaryTtBytes; RunnerSha256 = $requiredReplicaRunnerSha256;
        Replica = $true; ReplicaLane = 'd6_off'
    }
    SECOND_TT = [pscustomobject]@{
        Label = 'SECOND_TT'; FileTag = 'SECOND_TT'; JournalPrefix = 'DOM_B2_D4_REPLICA_QUEUE';
        FirstRunId = 1;
        JournalRegex = '^DOM_B2_D4_REPLICA_SECOND_TT_QUEUE_RUN[0-9]{2,}_RAW[.]log$';
        MetaRegex = '^DOM_B2_D4_REPLICA_SECOND_TT_SHARD_C(?<case>(?:0[0-9]|10))_S(?<start>[0-9]{4})_A(?<attempt>[0-9]{2})_META_RAW[.]log$';
        RawRegex = '^DOM_B2_D4_REPLICA_SECOND_TT_SHARD_C(?<case>(?:0[0-9]|10))_S(?<start>[0-9]{4})_A(?<attempt>[0-9]{2})_RAW[.]log$';
        RawTemplate = 'DOM_B2_D4_REPLICA_SECOND_TT_SHARD_C{0:D2}_S{1:D4}_A{2:D2}_RAW.log';
        D6 = 'true'; TtBytes = $secondTtBytes; RunnerSha256 = $requiredReplicaRunnerSha256;
        Replica = $true; ReplicaLane = 'second_tt'
    }
}

function Assert-Equal {
    param($Actual, $Expected, [Parameter(Mandatory = $true)][string] $Context)
    if ($Actual -cne $Expected) {
        throw "$Context mismatch: observed='$Actual' expected='$Expected'"
    }
}

function Assert-True {
    param([bool] $Condition, [Parameter(Mandatory = $true)][string] $Context)
    if (-not $Condition) { throw $Context }
}

function Get-Sha256FromBytes {
    param([Parameter(Mandatory = $true)][byte[]] $Bytes)
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($sha.ComputeHash($Bytes)).Replace('-', '') }
    finally { $sha.Dispose() }
}

function Read-SharedBytes {
    param([Parameter(Mandatory = $true)][string] $Path)
    $full = [IO.Path]::GetFullPath($Path)
    $stream = [IO.FileStream]::new(
        $full,
        [IO.FileMode]::Open,
        [IO.FileAccess]::Read,
        ([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete)
    )
    try {
        if ($stream.Length -gt [int]::MaxValue) { throw "file is too large: $full" }
        $bytes = New-Object byte[] ([int]$stream.Length)
        $offset = 0
        while ($offset -lt $bytes.Length) {
            $read = $stream.Read($bytes, $offset, $bytes.Length - $offset)
            if ($read -eq 0) { throw "short read: $full" }
            $offset += $read
        }
        return ,$bytes
    }
    finally { $stream.Dispose() }
}

function Assert-AsciiBytes {
    param([Parameter(Mandatory = $true)][byte[]] $Bytes, [Parameter(Mandatory = $true)][string] $Context)
    foreach ($value in $Bytes) {
        if ($value -ne 10 -and $value -ne 13 -and ($value -lt 32 -or $value -gt 126)) {
            throw "$Context contains a non-ASCII/control byte: $value"
        }
    }
}

function ConvertFrom-ExactLfBytes {
    param([Parameter(Mandatory = $true)][byte[]] $Bytes, [Parameter(Mandatory = $true)][string] $Context)
    Assert-True ($Bytes.Length -gt 0) "$Context is empty"
    Assert-AsciiBytes -Bytes $Bytes -Context $Context
    Assert-True ($Bytes[$Bytes.Length - 1] -eq 10) "$Context lacks its exact final LF"
    Assert-True (-not ($Bytes -contains [byte]13)) "$Context contains CR; exact LF is required"
    $text = [Text.Encoding]::ASCII.GetString($Bytes)
    $body = $text.Substring(0, $text.Length - 1)
    if ($body.Length -eq 0) { return @() }
    return @($body.Split("`n"))
}

function ConvertFrom-ExactCrlfBytes {
    param([Parameter(Mandatory = $true)][byte[]] $Bytes, [Parameter(Mandatory = $true)][string] $Context)
    Assert-True ($Bytes.Length -ge 2) "$Context is empty/truncated"
    Assert-AsciiBytes -Bytes $Bytes -Context $Context
    Assert-True ($Bytes[$Bytes.Length - 2] -eq 13 -and $Bytes[$Bytes.Length - 1] -eq 10) "$Context lacks its exact final CRLF"
    $text = [Text.Encoding]::ASCII.GetString($Bytes)
    $withoutPairs = $text.Replace("`r`n", '')
    Assert-True (-not $withoutPairs.Contains("`r") -and -not $withoutPairs.Contains("`n")) "$Context has a non-CRLF line ending"
    $body = $text.Substring(0, $text.Length - 2)
    if ($body.Length -eq 0) { return @() }
    return @($body.Split(@("`r`n"), [StringSplitOptions]::None))
}

function Convert-ToInt {
    param([Parameter(Mandatory = $true)][string] $Value, [Parameter(Mandatory = $true)][string] $Context)
    $parsed = 0
    if (-not [int]::TryParse($Value, [Globalization.NumberStyles]::Integer, $invariant, [ref]$parsed)) {
        throw "$Context is not an Int32: '$Value'"
    }
    return $parsed
}

function Convert-ToInt64 {
    param([Parameter(Mandatory = $true)][string] $Value, [Parameter(Mandatory = $true)][string] $Context)
    $parsed = [int64]0
    if (-not [int64]::TryParse($Value, [Globalization.NumberStyles]::Integer, $invariant, [ref]$parsed)) {
        throw "$Context is not an Int64: '$Value'"
    }
    return $parsed
}

function Convert-ToUInt64 {
    param([Parameter(Mandatory = $true)][string] $Value, [Parameter(Mandatory = $true)][string] $Context)
    $parsed = [uint64]0
    if (-not [uint64]::TryParse($Value, [Globalization.NumberStyles]::Integer, $invariant, [ref]$parsed)) {
        throw "$Context is not a UInt64: '$Value'"
    }
    return $parsed
}

function Convert-ToDecimal {
    param([Parameter(Mandatory = $true)][string] $Value, [Parameter(Mandatory = $true)][string] $Context)
    $parsed = [decimal]0
    if (-not [decimal]::TryParse($Value, [Globalization.NumberStyles]::Float, $invariant, [ref]$parsed)) {
        throw "$Context is not a finite decimal: '$Value'"
    }
    return $parsed
}

function Get-Fields {
    param([Parameter(Mandatory = $true)][string] $Text, [Parameter(Mandatory = $true)][string] $Context)
    $result = @{}
    if ($Text.Length -eq 0) { return $result }
    $matches = [regex]::Matches($Text, '(?:^| )(?<key>[A-Za-z][A-Za-z0-9_]*)=(?<value>\S+)')
    $canonical = New-Object Collections.Generic.List[string]
    foreach ($match in $matches) {
        $key = $match.Groups['key'].Value
        if ($result.ContainsKey($key)) { throw "$Context duplicates field '$key'" }
        $value = $match.Groups['value'].Value
        $result[$key] = $value
        $canonical.Add("$key=$value")
    }
    Assert-Equal ($canonical -join ' ') $Text "$Context field grammar"
    return $result
}

function Assert-FieldNames {
    param([Parameter(Mandatory = $true)][hashtable] $Fields, [Parameter(Mandatory = $true)][string[]] $Names, [Parameter(Mandatory = $true)][string] $Context)
    # Hashtable key lookup shadows the intrinsic .Count property when a
    # record legitimately has a `count=` field.  Count keys explicitly.
    Assert-Equal $Fields.Keys.Count $Names.Count "$Context field count"
    foreach ($name in $Names) {
        Assert-True ($Fields.ContainsKey($name)) "$Context is missing field '$name'"
    }
}

function Get-Field {
    param([Parameter(Mandatory = $true)][hashtable] $Fields, [Parameter(Mandatory = $true)][string] $Name, [Parameter(Mandatory = $true)][string] $Context)
    Assert-True ($Fields.ContainsKey($Name)) "$Context is missing field '$Name'"
    return [string]$Fields[$Name]
}

function Resolve-RootLeaf {
    param([Parameter(Mandatory = $true)][string] $Path, [Parameter(Mandatory = $true)][string] $Context)
    $full = [IO.Path]::GetFullPath($Path)
    $parent = [IO.Path]::GetDirectoryName($full)
    Assert-True ($parent.Equals($repoRoot, [StringComparison]::OrdinalIgnoreCase)) "$Context must be a direct repository-root file: $full"
    Assert-True (Test-Path -LiteralPath $full -PathType Leaf) "$Context is missing: $full"
    $item = Get-Item -LiteralPath $full -Force
    Assert-True (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) "$Context may not be a reparse point: $full"
    return $full
}

function Assert-BoundFile {
    param([Parameter(Mandatory = $true)][string] $RelativePath, [Parameter(Mandatory = $true)][string] $ExpectedSha256, [Parameter(Mandatory = $true)][int64] $ExpectedBytes)
    $full = [IO.Path]::GetFullPath((Join-Path $repoRoot $RelativePath))
    Assert-True (Test-Path -LiteralPath $full -PathType Leaf) "bound file missing: $RelativePath"
    $item = Get-Item -LiteralPath $full -Force
    Assert-True (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) "bound file may not be a reparse point: $RelativePath"
    Assert-Equal ([int64]$item.Length) $ExpectedBytes "$RelativePath bytes"
    Assert-Equal ((Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash.ToUpperInvariant()) $ExpectedSha256 "$RelativePath SHA256"
    return $full
}

function Resolve-BoundManifestPath {
    param([Parameter(Mandatory = $true)][string] $RelativePath, [Parameter(Mandatory = $true)][string] $Context)
    Assert-True (-not [IO.Path]::IsPathRooted($RelativePath)) "$Context is rooted: $RelativePath"
    Assert-True ($RelativePath -cnotmatch '(^|[\\/])[.][.]([\\/]|$)') "$Context contains parent traversal: $RelativePath"
    $full = [IO.Path]::GetFullPath((Join-Path $repoRoot $RelativePath))
    $rootPrefix = $repoRoot.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    Assert-True ($full.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) "$Context escapes repository root: $RelativePath"
    Assert-True (Test-Path -LiteralPath $full -PathType Leaf) "$Context file is missing: $RelativePath"
    $item = Get-Item -LiteralPath $full -Force
    Assert-True (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) "$Context may not be a reparse point: $RelativePath"
    return $full
}

function Assert-LiveSourceSnapshot {
    param([Parameter(Mandatory = $true)][string] $SnapshotPath)
    $snapshotBytes = Read-SharedBytes -Path $SnapshotPath
    Assert-Equal $snapshotBytes.Length 3399 'source snapshot live bytes'
    Assert-Equal (Get-Sha256FromBytes $snapshotBytes) $requiredSourceSnapshotSha256 'source snapshot live SHA256'
    $lines = @(ConvertFrom-ExactLfBytes -Bytes $snapshotBytes -Context 'source snapshot')
    Assert-True ($lines.Count -ge 7) 'source snapshot is truncated'
    Assert-Equal $lines[0] 'R-CREL-6 PHASE 3C DOM-B2 FINAL PRIMARY CODE SNAPSHOT' 'source snapshot title'
    Assert-True ($lines[1] -cmatch '^SNAPSHOT_META captured_utc=\S+ input_head=5f5da82a04d14f645fdbf08ea96937a428182cde code_id=DOM_B2_D4_PRIMARY_V3 purpose=mandatory_per_launch_source_and_binary_fence$') 'source snapshot META mismatch'
    Assert-Equal $lines[2] 'TOOLCHAIN rustc=1.95.0 rustc_commit=59807616e1fa2540724bfbac14d7976d7e4a3860 host=x86_64-pc-windows-msvc llvm=22.1.2 profile=release target_dir=.target-hunt' 'source snapshot toolchain'
    $codeCount = 0
    $binaryCount = 0
    $schemaCount = 0
    $verifierCount = 0
    for ($index = 3; $index -lt $lines.Count - 1; $index++) {
        $line = $lines[$index]
        $code = [regex]::Match($line, '^CODE_FILE path=(\S+) bytes=(\d+) sha256=([0-9A-F]{64})$')
        $binary = [regex]::Match($line, '^BINARY path=(\S+) bytes=(\d+) mtime_utc=\S+ sha256=([0-9A-F]{64})$')
        $schema = [regex]::Match($line, '^SCHEMA_SMOKE path=(\S+) bytes=(\d+) sha256=([0-9A-F]{64}) code_id=DOM_B2_D4_PRIMARY_V3 result=PASS$')
        $verifier = [regex]::Match($line, '^STRICT_VERIFIER path=(\S+) sha256=([0-9A-F]{64}) untouched=true$')
        if ($code.Success) {
            $full = Resolve-BoundManifestPath $code.Groups[1].Value "source CODE_FILE line $($index + 1)"
            Assert-Equal ([int64](Get-Item -LiteralPath $full).Length) ([int64]$code.Groups[2].Value) "source CODE_FILE bytes $($code.Groups[1].Value)"
            Assert-Equal ((Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash.ToUpperInvariant()) $code.Groups[3].Value "source CODE_FILE SHA256 $($code.Groups[1].Value)"
            $codeCount++
        }
        elseif ($binary.Success) {
            Assert-Equal $binary.Groups[1].Value $requiredBinaryRelative 'source BINARY path'
            Assert-Equal ([int64]$binary.Groups[2].Value) ([int64]3290112) 'source BINARY bytes'
            Assert-Equal $binary.Groups[3].Value $requiredBinarySha256 'source BINARY SHA256 manifest'
            $full = Resolve-BoundManifestPath $binary.Groups[1].Value 'source BINARY'
            Assert-Equal ([int64](Get-Item -LiteralPath $full).Length) ([int64]3290112) 'source BINARY live bytes'
            Assert-Equal ((Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash.ToUpperInvariant()) $requiredBinarySha256 'source BINARY live SHA256'
            $binaryCount++
        }
        elseif ($schema.Success) {
            $full = Resolve-BoundManifestPath $schema.Groups[1].Value 'source SCHEMA_SMOKE'
            Assert-Equal ([int64](Get-Item -LiteralPath $full).Length) ([int64]$schema.Groups[2].Value) 'source SCHEMA_SMOKE live bytes'
            Assert-Equal ((Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash.ToUpperInvariant()) $schema.Groups[3].Value 'source SCHEMA_SMOKE live SHA256'
            $schemaCount++
        }
        elseif ($verifier.Success) {
            Assert-Equal $verifier.Groups[1].Value 'packages/hexfield_eq/rust/src/tss_verify.rs' 'source STRICT_VERIFIER path'
            Assert-Equal $verifier.Groups[2].Value $requiredVerifierSha256 'source STRICT_VERIFIER manifest SHA256'
            $full = Resolve-BoundManifestPath $verifier.Groups[1].Value 'source STRICT_VERIFIER'
            Assert-Equal ((Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash.ToUpperInvariant()) $requiredVerifierSha256 'source STRICT_VERIFIER live SHA256'
            $verifierCount++
        }
        else { throw "unknown source snapshot entry at line $($index + 1): $line" }
    }
    Assert-Equal $lines[$lines.Count - 1] 'LEGACY_FENCE All status shards lacking exact code_id=DOM_B2_D4_PRIMARY_V3 are retained but excluded from primary/replica completeness and verdict credit.' 'source snapshot footer'
    Assert-Equal $codeCount 17 'source CODE_FILE count'
    Assert-Equal $binaryCount 1 'source BINARY count'
    Assert-Equal $schemaCount 1 'source SCHEMA_SMOKE count'
    Assert-Equal $verifierCount 1 'source STRICT_VERIFIER count'
    Assert-Equal ($codeCount + $binaryCount + $schemaCount + $verifierCount) $requiredSourceFiles 'source bound-entry count'
    return [pscustomobject]@{ Entries=$requiredSourceFiles; CodeFiles=$codeCount; Bytes=$snapshotBytes.Length; Sha256=$requiredSourceSnapshotSha256 }
}

function Parse-JournalLine {
    param([Parameter(Mandatory = $true)][string] $Line, [Parameter(Mandatory = $true)] $Lane, [Parameter(Mandatory = $true)][int] $LineNumber)
    $pattern = if ($Lane.Replica) {
        '^DOM_B2_D4_REPLICA_QUEUE timestamp=(?<timestamp>\S+) lane=(?<lane>d6_off|second_tt) (?<event>[A-Z_]+)(?: (?<fields>.*))?$'
    }
    else {
        '^QUEUE timestamp=(?<timestamp>\S+) (?<event>[A-Z_]+)(?: (?<fields>.*))?$'
    }
    $match = [regex]::Match($Line, $pattern)
    Assert-True $match.Success "malformed $($Lane.Label) journal line $LineNumber"
    if ($Lane.Replica) { Assert-Equal $match.Groups['lane'].Value $Lane.ReplicaLane "$($Lane.Label) journal line $LineNumber lane"
    }
    $timestamp = [DateTimeOffset]::MinValue
    if (-not [DateTimeOffset]::TryParseExact(
        $match.Groups['timestamp'].Value,
        'yyyy-MM-ddTHH:mm:ss.fffffffzzz',
        $invariant,
        [Globalization.DateTimeStyles]::None,
        [ref]$timestamp
    )) { throw "malformed $($Lane.Label) journal timestamp at line $LineNumber" }
    $fieldText = if ($match.Groups['fields'].Success) { $match.Groups['fields'].Value } else { '' }
    $fields = if ($match.Groups['event'].Value -ceq 'ABORT') {
        # ABORT output_tail/message may contain spaces. Capture unambiguous leading key=value tokens only.
        $partial = @{}
        foreach ($fieldMatch in [regex]::Matches($fieldText, '(?:^| )(?<key>[A-Za-z][A-Za-z0-9_]*)=(?<value>\S+)')) {
            $key = $fieldMatch.Groups['key'].Value
            if (-not $partial.ContainsKey($key)) { $partial[$key] = $fieldMatch.Groups['value'].Value }
        }
        $partial
    }
    else { Get-Fields -Text $fieldText -Context "$($Lane.Label) journal line $LineNumber" }
    return [pscustomobject]@{
        Index = $LineNumber - 1; LineNumber = $LineNumber; Timestamp = $timestamp;
        Event = $match.Groups['event'].Value; Fields = $fields; FieldText = $fieldText; Original = $Line
    }
}

function Read-Journal {
    param([Parameter(Mandatory = $true)][string] $Path, [Parameter(Mandatory = $true)] $Lane)
    $full = Resolve-RootLeaf -Path $Path -Context "$($Lane.Label) journal"
    $name = [IO.Path]::GetFileName($full)
    Assert-True ($name -cmatch $Lane.JournalRegex) "$($Lane.Label) journal filename is outside the frozen family: $name"
    $runIdMatch = [regex]::Match($name, 'RUN(?<runid>[0-9]{2,})_RAW[.]log$')
    Assert-True $runIdMatch.Success "$($Lane.Label) journal lacks a frozen RUN id: $name"
    $runId = Convert-ToInt $runIdMatch.Groups['runid'].Value "$($Lane.Label) journal RUN id"
    Assert-True ($runId -ge $Lane.FirstRunId) "$($Lane.Label) journal predates frozen first RUN id $($Lane.FirstRunId): $name"
    $bytes = Read-SharedBytes -Path $full
    $lines = @(ConvertFrom-ExactCrlfBytes -Bytes $bytes -Context $name)
    Assert-True ($lines.Count -gt 0) "$name has no records"
    $events = New-Object Collections.Generic.List[object]
    $priorTimestamp = [DateTimeOffset]::MinValue
    for ($index = 0; $index -lt $lines.Count; $index++) {
        $event = Parse-JournalLine -Line $lines[$index] -Lane $Lane -LineNumber ($index + 1)
        Assert-True ($event.Timestamp -ge $priorTimestamp) "$name timestamps regress at line $($index + 1)"
        $priorTimestamp = $event.Timestamp
        $events.Add($event)
    }
    return [pscustomobject]@{
        Lane = $Lane; Path = $full; Name = $name; RunId = $runId; Bytes = $bytes.Length;
        Sha256 = Get-Sha256FromBytes $bytes; Events = @($events | ForEach-Object { $_ }); OriginalBytes = $bytes
    }
}

function Assert-SetupEvent {
    param([Parameter(Mandatory = $true)] $Journal)
    $setups = @($Journal.Events | Where-Object { $_.Event -ceq 'SETUP' })
    Assert-Equal $setups.Count 1 "$($Journal.Name) SETUP count"
    Assert-Equal $setups[0].Index 0 "$($Journal.Name) SETUP position"
    $lane = $Journal.Lane
    $fields = $setups[0].Fields
    $common = @(
        'mode','cases','total_roots','count','deadline_ms','cargo_wall_timeout_ms','tt_bytes','d6','target','release','target_dir','test_threads',
        'code_id','expected_source_snapshot_sha256','actual_source_snapshot_sha256','source_snapshot_path','source_files','source_algorithm','source_result',
        'census_sha256','runner_sha256','rustc','rustc_commit','cargo_configs','build_overrides','stop_file'
    )
    $names = if ($lane.Replica) {
        @('mode','lane','lane_file_tag') + $common[1..($common.Count - 1)] + @('lane_lock','replica_global_lock','primary_queue_lock')
    }
    else { $common + @('lock_file') }
    Assert-FieldNames -Fields $fields -Names $names -Context "$($Journal.Name) SETUP"
    Assert-Equal (Get-Field $fields 'mode' 'SETUP') $(if ($lane.Replica) { 'replica_v3' } else { 'primary_v3' }) 'SETUP mode'
    if ($lane.Replica) {
        Assert-Equal (Get-Field $fields 'lane' 'SETUP') $lane.ReplicaLane 'SETUP lane'
        Assert-Equal (Get-Field $fields 'lane_file_tag' 'SETUP') $lane.FileTag 'SETUP lane_file_tag'
    }
    Assert-Equal (Get-Field $fields 'cases' 'SETUP') '11' 'SETUP cases'
    Assert-Equal (Get-Field $fields 'total_roots' 'SETUP') '3648' 'SETUP roots'
    Assert-Equal (Get-Field $fields 'count' 'SETUP') ([string]$requiredCount) 'SETUP count'
    Assert-Equal (Get-Field $fields 'deadline_ms' 'SETUP') ([string]$requiredDeadlineMs) 'SETUP deadline'
    Assert-Equal (Get-Field $fields 'cargo_wall_timeout_ms' 'SETUP') ([string]$requiredCargoWallTimeoutMs) 'SETUP watchdog'
    Assert-Equal (Get-Field $fields 'tt_bytes' 'SETUP') ([string]$lane.TtBytes) 'SETUP TT'
    Assert-Equal (Get-Field $fields 'd6' 'SETUP') $lane.D6 'SETUP d6'
    Assert-Equal (Get-Field $fields 'target' 'SETUP') $requiredTarget 'SETUP target'
    Assert-Equal (Get-Field $fields 'release' 'SETUP') 'true' 'SETUP release'
    Assert-Equal (Get-Field $fields 'target_dir' 'SETUP') '.target-hunt' 'SETUP target_dir'
    Assert-Equal (Get-Field $fields 'test_threads' 'SETUP') '1' 'SETUP test_threads'
    Assert-Equal (Get-Field $fields 'code_id' 'SETUP') $requiredCodeId 'SETUP code_id'
    Assert-Equal (Get-Field $fields 'expected_source_snapshot_sha256' 'SETUP') $requiredSourceSnapshotSha256 'SETUP expected source'
    Assert-Equal (Get-Field $fields 'actual_source_snapshot_sha256' 'SETUP') $requiredSourceSnapshotSha256 'SETUP actual source'
    Assert-Equal (Get-Field $fields 'source_snapshot_path' 'SETUP') ([IO.Path]::GetFullPath((Join-Path $repoRoot 'DOM_B2_D4_PRIMARY_V3_CODE_SNAPSHOT_RAW.log'))) 'SETUP snapshot path'
    Assert-Equal (Get-Field $fields 'source_files' 'SETUP') ([string]$requiredSourceFiles) 'SETUP source_files'
    Assert-Equal (Get-Field $fields 'source_algorithm' 'SETUP') 'SEALED_MANIFEST_AND_BOUND_SHA256_V1' 'SETUP source algorithm'
    Assert-Equal (Get-Field $fields 'source_result' 'SETUP') 'PASS' 'SETUP source result'
    Assert-Equal (Get-Field $fields 'census_sha256' 'SETUP') $requiredCensusSha256 'SETUP census'
    Assert-Equal (Get-Field $fields 'runner_sha256' 'SETUP') $lane.RunnerSha256 'SETUP runner'
    Assert-Equal (Get-Field $fields 'rustc' 'SETUP') $requiredRustc 'SETUP rustc'
    Assert-Equal (Get-Field $fields 'rustc_commit' 'SETUP') $requiredRustcCommit 'SETUP rustc commit'
    Assert-Equal (Get-Field $fields 'cargo_configs' 'SETUP') '0' 'SETUP cargo configs'
    Assert-Equal (Get-Field $fields 'build_overrides' 'SETUP') '0' 'SETUP build overrides'
    $expectedStop = if ($lane.Replica) {
        [IO.Path]::GetFullPath((Join-Path $repoRoot "DOM_B2_D4_REPLICA_$($lane.FileTag)_QUEUE.STOP"))
    }
    else { [IO.Path]::GetFullPath((Join-Path $repoRoot 'DOM_B2_D4_QUEUE.STOP')) }
    Assert-Equal (Get-Field $fields 'stop_file' 'SETUP') $expectedStop 'SETUP stop path'
    if ($lane.Replica) {
        $laneLockName = if ($lane.Label -ceq 'D6_OFF') { '.dom_b2_d4_replica_d6_off_queue.lock' } else { '.dom_b2_d4_replica_second_tt_queue.lock' }
        Assert-Equal (Get-Field $fields 'lane_lock' 'SETUP') ([IO.Path]::GetFullPath((Join-Path $repoRoot $laneLockName))) 'SETUP lane lock path'
        Assert-Equal (Get-Field $fields 'replica_global_lock' 'SETUP') ([IO.Path]::GetFullPath((Join-Path $repoRoot '.dom_b2_d4_replica_queue.lock'))) 'SETUP replica global lock path'
        Assert-Equal (Get-Field $fields 'primary_queue_lock' 'SETUP') ([IO.Path]::GetFullPath((Join-Path $repoRoot '.dom_b2_d4_run_queue.lock'))) 'SETUP primary queue lock path'
    }
    else {
        Assert-Equal (Get-Field $fields 'lock_file' 'SETUP') ([IO.Path]::GetFullPath((Join-Path $repoRoot '.dom_b2_d4_run_queue.lock'))) 'SETUP primary lock path'
    }
}

function Assert-CaseStartFields {
    param([Parameter(Mandatory = $true)][hashtable] $Fields, [Parameter(Mandatory = $true)][string] $Context)
    $case = Convert-ToInt (Get-Field $Fields 'case' $Context) "$Context case"
    $start = Convert-ToInt (Get-Field $Fields 'start' $Context) "$Context start"
    Assert-True ($case -ge 0 -and $case -lt 11) "$Context case is out of range: $case"
    Assert-True ($start -ge 0 -and $start -lt $widths[$case]) "$Context start is out of range: case=$case start=$start"
    return [pscustomobject]@{ Case=$case; Start=$start }
}

function Assert-GateEvent {
    param([Parameter(Mandatory = $true)] $Journal, [Parameter(Mandatory = $true)] $Event)
    $lane = $Journal.Lane
    $names = @('case','start','available_bytes','free_physical_bytes','foreign_cargo') + $(if($lane.Replica){@('primary_queue_active')}else{@()}) + @('result')
    Assert-FieldNames $Event.Fields $names 'GATE'
    [void](Assert-CaseStartFields $Event.Fields 'GATE')
    $available = Convert-ToInt64 (Get-Field $Event.Fields 'available_bytes' 'GATE') 'GATE available'
    $free = Convert-ToInt64 (Get-Field $Event.Fields 'free_physical_bytes' 'GATE') 'GATE free'
    $foreign = Convert-ToInt (Get-Field $Event.Fields 'foreign_cargo' 'GATE') 'GATE foreign cargo'
    Assert-True ($available -ge 0 -and $free -ge 0 -and $foreign -ge 0) 'GATE has a negative resource value'
    $primaryInactive = $true
    if ($lane.Replica) {
        $primaryText = Get-Field $Event.Fields 'primary_queue_active' 'GATE'
        Assert-True ($primaryText -ceq 'true' -or $primaryText -ceq 'false') 'GATE primary_queue_active is not Boolean text'
        $primaryInactive = $primaryText -ceq 'false'
    }
    $gatePass = $available -ge $availableFloor -and $free -ge $freeFloor -and $foreign -eq 0 -and $primaryInactive
    $result = Get-Field $Event.Fields 'result' 'GATE'
    Assert-True ($result -ceq 'PASS' -or $result -ceq 'WAIT_NO_LAUNCH') "GATE result is invalid: $result"
    Assert-Equal ($result -ceq 'PASS') $gatePass 'GATE result/resource truth table'
}

function Assert-SourceFenceEvent {
    param([Parameter(Mandatory = $true)] $Journal, [Parameter(Mandatory = $true)] $Event)
    $fields = $Event.Fields
    Assert-FieldNames $fields @('case','start','expected','actual','source_files','runner_sha256','census_sha256','code_id','rustc','cargo_configs','build_overrides','result') 'SOURCE_FENCE'
    [void](Assert-CaseStartFields $fields 'SOURCE_FENCE')
    Assert-Equal (Get-Field $fields 'expected' 'SOURCE_FENCE') $requiredSourceSnapshotSha256 'SOURCE_FENCE expected'
    Assert-Equal (Get-Field $fields 'actual' 'SOURCE_FENCE') $requiredSourceSnapshotSha256 'SOURCE_FENCE actual'
    Assert-Equal (Get-Field $fields 'source_files' 'SOURCE_FENCE') ([string]$requiredSourceFiles) 'SOURCE_FENCE source files'
    Assert-Equal (Get-Field $fields 'runner_sha256' 'SOURCE_FENCE') $Journal.Lane.RunnerSha256 'SOURCE_FENCE runner'
    Assert-Equal (Get-Field $fields 'census_sha256' 'SOURCE_FENCE') $requiredCensusSha256 'SOURCE_FENCE census'
    Assert-Equal (Get-Field $fields 'code_id' 'SOURCE_FENCE') $requiredCodeId 'SOURCE_FENCE code'
    Assert-Equal (Get-Field $fields 'rustc' 'SOURCE_FENCE') $requiredRustc 'SOURCE_FENCE rustc'
    Assert-Equal (Get-Field $fields 'cargo_configs' 'SOURCE_FENCE') '0' 'SOURCE_FENCE cargo configs'
    Assert-Equal (Get-Field $fields 'build_overrides' 'SOURCE_FENCE') '0' 'SOURCE_FENCE overrides'
    Assert-Equal (Get-Field $fields 'result' 'SOURCE_FENCE') 'PASS' 'SOURCE_FENCE result'
}

function Assert-PrelaunchEvent {
    param([Parameter(Mandatory = $true)] $Journal, [Parameter(Mandatory = $true)] $Event)
    Assert-True $Journal.Lane.Replica 'PRELAUNCH_GATE is forbidden in the primary journal'
    $fields = $Event.Fields
    Assert-FieldNames $fields @('case','start','available_bytes','free_physical_bytes','foreign_cargo','primary_queue_active','stop_requested','source_snapshot_sha256','runner_sha256','rustc','cargo_configs','build_overrides','result') 'PRELAUNCH_GATE'
    [void](Assert-CaseStartFields $fields 'PRELAUNCH_GATE')
    $available = Convert-ToInt64 (Get-Field $fields 'available_bytes' 'PRELAUNCH_GATE') 'PRELAUNCH available'
    $free = Convert-ToInt64 (Get-Field $fields 'free_physical_bytes' 'PRELAUNCH_GATE') 'PRELAUNCH free'
    $foreign = Convert-ToInt (Get-Field $fields 'foreign_cargo' 'PRELAUNCH_GATE') 'PRELAUNCH foreign cargo'
    $primary = Get-Field $fields 'primary_queue_active' 'PRELAUNCH_GATE'
    $stop = Get-Field $fields 'stop_requested' 'PRELAUNCH_GATE'
    Assert-True ($primary -ceq 'true' -or $primary -ceq 'false') 'PRELAUNCH primary_queue_active is not Boolean text'
    Assert-True ($stop -ceq 'true' -or $stop -ceq 'false') 'PRELAUNCH stop_requested is not Boolean text'
    Assert-Equal (Get-Field $fields 'rustc' 'PRELAUNCH_GATE') $requiredRustc 'PRELAUNCH rustc'
    Assert-Equal (Get-Field $fields 'cargo_configs' 'PRELAUNCH_GATE') '0' 'PRELAUNCH cargo configs'
    Assert-Equal (Get-Field $fields 'build_overrides' 'PRELAUNCH_GATE') '0' 'PRELAUNCH overrides'
    $pass = $available -ge $availableFloor -and $free -ge $freeFloor -and $foreign -eq 0 -and
        $primary -ceq 'false' -and $stop -ceq 'false' -and
        (Get-Field $fields 'source_snapshot_sha256' 'PRELAUNCH_GATE') -ceq $requiredSourceSnapshotSha256 -and
        (Get-Field $fields 'runner_sha256' 'PRELAUNCH_GATE') -ceq $Journal.Lane.RunnerSha256
    $result = Get-Field $fields 'result' 'PRELAUNCH_GATE'
    Assert-True ($result -ceq 'PASS' -or $result -ceq 'WAIT_NO_LAUNCH') "PRELAUNCH result is invalid: $result"
    Assert-Equal ($result -ceq 'PASS') $pass 'PRELAUNCH result/gate truth table'
}

function Assert-PostFenceEvent {
    param([Parameter(Mandatory = $true)] $Journal, [Parameter(Mandatory = $true)] $Event)
    $fields = $Event.Fields
    Assert-FieldNames $fields @('case','start','expected','actual','source_files','runner_initial','runner_actual','code_id','rustc','cargo_configs','build_overrides','result') 'SOURCE_FENCE_POST'
    [void](Assert-CaseStartFields $fields 'SOURCE_FENCE_POST')
    Assert-Equal (Get-Field $fields 'expected' 'SOURCE_FENCE_POST') $requiredSourceSnapshotSha256 'SOURCE_FENCE_POST expected'
    Assert-Equal (Get-Field $fields 'source_files' 'SOURCE_FENCE_POST') ([string]$requiredSourceFiles) 'SOURCE_FENCE_POST source files'
    Assert-Equal (Get-Field $fields 'runner_initial' 'SOURCE_FENCE_POST') $Journal.Lane.RunnerSha256 'SOURCE_FENCE_POST runner initial'
    Assert-Equal (Get-Field $fields 'code_id' 'SOURCE_FENCE_POST') $requiredCodeId 'SOURCE_FENCE_POST code'
    Assert-Equal (Get-Field $fields 'rustc' 'SOURCE_FENCE_POST') $requiredRustc 'SOURCE_FENCE_POST rustc'
    Assert-Equal (Get-Field $fields 'cargo_configs' 'SOURCE_FENCE_POST') '0' 'SOURCE_FENCE_POST cargo configs'
    Assert-Equal (Get-Field $fields 'build_overrides' 'SOURCE_FENCE_POST') '0' 'SOURCE_FENCE_POST overrides'
    $actual = Get-Field $fields 'actual' 'SOURCE_FENCE_POST'
    $runnerActual = Get-Field $fields 'runner_actual' 'SOURCE_FENCE_POST'
    Assert-True ($actual -cmatch '^[0-9A-F]{64}$') 'SOURCE_FENCE_POST actual SHA grammar'
    Assert-True ($runnerActual -cmatch '^[0-9A-F]{64}$') 'SOURCE_FENCE_POST runner_actual SHA grammar'
    $pass = $actual -ceq $requiredSourceSnapshotSha256 -and $runnerActual -ceq $Journal.Lane.RunnerSha256
    $result = Get-Field $fields 'result' 'SOURCE_FENCE_POST'
    Assert-True ($result -ceq 'PASS' -or $result -ceq 'ABORT') "SOURCE_FENCE_POST invalid result: $result"
    Assert-Equal ($result -ceq 'PASS') $pass 'SOURCE_FENCE_POST result truth table'
}

function Assert-BinaryFenceEvent {
    param([Parameter(Mandatory = $true)] $Event)
    $fields = $Event.Fields
    Assert-FieldNames $fields @('case','start','reported_path','expected_path','result') 'BINARY_FENCE'
    [void](Assert-CaseStartFields $fields 'BINARY_FENCE')
    $expected = [IO.Path]::GetFullPath((Join-Path $repoRoot $requiredBinaryRelative))
    Assert-Equal (Get-Field $fields 'reported_path' 'BINARY_FENCE') $expected 'BINARY_FENCE reported path'
    Assert-Equal (Get-Field $fields 'expected_path' 'BINARY_FENCE') $expected 'BINARY_FENCE expected path'
    Assert-Equal (Get-Field $fields 'result' 'BINARY_FENCE') 'PASS' 'BINARY_FENCE result'
}

function Assert-StopEvent {
    param([Parameter(Mandatory = $true)] $Journal, [Parameter(Mandatory = $true)] $Event)
    $fields = $Event.Fields
    $reason = Get-Field $fields 'reason' 'STOP'
    switch -CaseSensitive ($reason) {
        'max_invocations' {
            Assert-FieldNames $fields @('reason','invocations','case','next_start') 'STOP max_invocations'
            Assert-True ((Convert-ToInt (Get-Field $fields 'invocations' 'STOP') 'STOP invocations') -ge 0) 'STOP invocations is negative'
        }
        { @('operator_stop_file','operator_stop_file_during_gate','operator_stop_file_after_gate','operator_stop_file_at_prelaunch') -ccontains $_ } {
            if ($reason -ceq 'operator_stop_file_at_prelaunch') { Assert-True $Journal.Lane.Replica 'primary journal emitted replica-only prelaunch STOP' }
            Assert-FieldNames $fields @('reason','case','next_start','path') "STOP $reason"
            $expectedStop = if ($Journal.Lane.Replica) { [IO.Path]::GetFullPath((Join-Path $repoRoot "DOM_B2_D4_REPLICA_$($Journal.Lane.FileTag)_QUEUE.STOP")) }
                else { [IO.Path]::GetFullPath((Join-Path $repoRoot 'DOM_B2_D4_QUEUE.STOP')) }
            Assert-Equal (Get-Field $fields 'path' 'STOP') $expectedStop "STOP $reason path"
        }
        'root_requires_recursive_subdivision' {
            Assert-FieldNames $fields @('reason','case','root_index','path') 'STOP recursive subdivision'
            $case = Convert-ToInt (Get-Field $fields 'case' 'STOP') 'STOP case'
            $root = Convert-ToInt (Get-Field $fields 'root_index' 'STOP') 'STOP root'
            Assert-True ($case -ge 0 -and $case -lt 11 -and $root -ge 0 -and $root -lt $widths[$case]) 'STOP recursive root is out of range'
            $rawName = Get-Field $fields 'path' 'STOP'
            $rawMatch = [regex]::Match($rawName, $Journal.Lane.RawRegex)
            Assert-True ($rawMatch.Success -and [int]$rawMatch.Groups['case'].Value -eq $case -and [int]$rawMatch.Groups['start'].Value -eq $root) 'STOP recursive raw path does not match lane/case/root'
        }
        default { throw "unknown STOP reason in $($Journal.Name): $reason" }
    }
    if ($fields.ContainsKey('case')) {
        $case = Convert-ToInt (Get-Field $fields 'case' 'STOP') 'STOP case'
        Assert-True ($case -ge 0 -and $case -lt 11) 'STOP case is out of range'
        if ($fields.ContainsKey('next_start')) {
            $next = Convert-ToInt (Get-Field $fields 'next_start' 'STOP') 'STOP next_start'
            Assert-True ($next -ge 0 -and $next -le $widths[$case]) 'STOP next_start is out of range'
        }
    }
}

function Assert-AbortEvent {
    param([Parameter(Mandatory = $true)] $Journal, [Parameter(Mandatory = $true)] $Event)
    $reason = Get-Field $Event.Fields 'reason' 'ABORT'
    $allowed = @(
        'source_snapshot_mismatch_before_launch','runner_changed_on_disk','source_or_runner_mismatch_at_prelaunch',
        'cargo_wall_watchdog','cargo_wall_watchdog_process_survived_HARD_CONSTRAINT_VIOLATION','cargo_exit',
        'source_or_runner_changed_during_invocation','missing_raw','uncreditable_raw','runner_exception'
    )
    Assert-True ($allowed -ccontains $reason) "unknown ABORT reason in $($Journal.Name): $reason"
    if ($reason -ceq 'source_or_runner_mismatch_at_prelaunch') { Assert-True $Journal.Lane.Replica 'primary journal emitted replica-only prelaunch ABORT' }
    $pattern = switch -CaseSensitive ($reason) {
        'source_snapshot_mismatch_before_launch' { '^reason=source_snapshot_mismatch_before_launch case=\d+ start=\d+ expected=[0-9A-F]{64} actual=[0-9A-F]{64} source_files=\d+$' }
        'runner_changed_on_disk' { '^reason=runner_changed_on_disk case=\d+ start=\d+ initial=[0-9A-F]{64} actual=[0-9A-F]{64}$' }
        'source_or_runner_mismatch_at_prelaunch' { '^reason=source_or_runner_mismatch_at_prelaunch case=\d+ start=\d+ expected_source=[0-9A-F]{64} actual_source=[0-9A-F]{64} runner_initial=[0-9A-F]{64} runner_actual=[0-9A-F]{64}$' }
        'cargo_wall_watchdog' { '^reason=cargo_wall_watchdog case=\d+ start=\d+ elapsed_s=\d+\.\d{3} captured_process_ids=\S* kill_verified=(?:true|false) survivor_process_ids=\S* raw_state=\S+ raw_path=\S+ partial_bytes=\d+ partial_sha256=\S+ cargo_exit_raw_exists=(?:true|false) cargo_exit_bytes=\d+ cargo_exit_sha256=\S+$' }
        'cargo_wall_watchdog_process_survived_HARD_CONSTRAINT_VIOLATION' { '^reason=cargo_wall_watchdog_process_survived_HARD_CONSTRAINT_VIOLATION case=\d+ start=\d+ elapsed_s=\d+\.\d{3} captured_process_ids=\S* kill_verified=(?:true|false) survivor_process_ids=\S* raw_state=\S+ raw_path=\S+ partial_bytes=\d+ partial_sha256=\S+ cargo_exit_raw_exists=(?:true|false) cargo_exit_bytes=\d+ cargo_exit_sha256=\S+$' }
        'cargo_exit' { '^reason=cargo_exit code=-?\d+ case=\d+ start=\d+ elapsed_s=\d+\.\d{3} cargo_exit_raw=\S+ cargo_exit_bytes=\d+ cargo_exit_sha256=[0-9A-F]{64} output_tail=.*$' }
        'source_or_runner_changed_during_invocation' { '^reason=source_or_runner_changed_during_invocation case=\d+ start=\d+ raw_retained_without_provenance=\S+$' }
        'missing_raw' { '^reason=missing_raw case=\d+ start=\d+$' }
        'uncreditable_raw' { '^reason=uncreditable_raw kind=\S+ case=\d+ start=\d+ path=\S+$' }
        'runner_exception' { '^reason=runner_exception message=\S+$' }
    }
    Assert-True ($Event.FieldText -cmatch $pattern) "$reason ABORT grammar mismatch"
    if ($reason -cne 'runner_exception') {
        Assert-True ($Event.Fields.ContainsKey('case') -and $Event.Fields.ContainsKey('start')) "per-invocation ABORT lacks case/start: $($Event.Original)"
        [void](Assert-CaseStartFields $Event.Fields 'ABORT')
    }
    if ($Event.Fields.ContainsKey('elapsed_s')) {
        $elapsed = Convert-ToDecimal (Get-Field $Event.Fields 'elapsed_s' 'ABORT') 'ABORT elapsed_s'
        Assert-True ($elapsed -ge 0 -and $elapsed -lt 600) "ABORT invocation elapsed is not <10m: $elapsed"
    }
}

function Assert-GateSuccessor {
    param([Parameter(Mandatory = $true)] $Gate, [AllowNull()] $NextEvent, [bool] $IsLast)
    $result = Get-Field $Gate.Fields 'result' 'GATE'
    if ($result -ceq 'PASS') {
        Assert-True (-not $IsLast) 'passing GATE is terminal'
        if ($NextEvent.Event -ceq 'STOP') {
            Assert-Equal (Get-Field $NextEvent.Fields 'reason' 'STOP after passing GATE') 'operator_stop_file_after_gate' 'STOP after passing GATE reason'
        }
        else { Assert-True ($NextEvent.Event -in @('SOURCE_FENCE','ABORT')) 'passing GATE is not followed by SOURCE_FENCE/allowed STOP/ABORT' }
    }
    elseif (-not $IsLast) {
        Assert-True ($NextEvent.Event -in @('GATE','STOP','ABORT')) 'WAIT_NO_LAUNCH GATE is followed by a launch-path event'
    }
}

function Assert-JournalGrammar {
    param([Parameter(Mandatory = $true)] $Journal)
    $allowedEvents = @('SETUP','UNTRUSTED_RETAINED','CASE_DONE','GATE','SOURCE_FENCE','PRELAUNCH_GATE','RUN','SOURCE_FENCE_POST','BINARY_FENCE','RESULT','STOP','ABORT','DONE')
    foreach ($event in $Journal.Events) {
        Assert-True ($allowedEvents -ccontains $event.Event) "unknown journal event in $($Journal.Name) line=$($event.LineNumber): $($event.Event)"
        switch -CaseSensitive ($event.Event) {
            'SETUP' { }
            'UNTRUSTED_RETAINED' {
                Assert-FieldNames $event.Fields @('count','raws','resume_credit') 'UNTRUSTED_RETAINED'
                $count = Convert-ToInt (Get-Field $event.Fields 'count' 'UNTRUSTED_RETAINED') 'UNTRUSTED count'
                Assert-True ($count -gt 0) 'UNTRUSTED count is not positive'
                Assert-Equal (Get-Field $event.Fields 'resume_credit' 'UNTRUSTED_RETAINED') '0' 'UNTRUSTED resume credit'
                Assert-Equal @((Get-Field $event.Fields 'raws' 'UNTRUSTED_RETAINED').Split(',')).Count $count 'UNTRUSTED raw-list count'
            }
            'CASE_DONE' {
                Assert-FieldNames $event.Fields @('case','roots','code_id') 'CASE_DONE'
                $case = Convert-ToInt (Get-Field $event.Fields 'case' 'CASE_DONE') 'CASE_DONE case'
                Assert-True ($case -ge 0 -and $case -lt 11) 'CASE_DONE case is out of range'
                Assert-Equal (Get-Field $event.Fields 'roots' 'CASE_DONE') ([string]$widths[$case]) 'CASE_DONE roots'
                Assert-Equal (Get-Field $event.Fields 'code_id' 'CASE_DONE') $requiredCodeId 'CASE_DONE code'
            }
            'GATE' { Assert-GateEvent $Journal $event }
            'SOURCE_FENCE' { Assert-SourceFenceEvent $Journal $event }
            'PRELAUNCH_GATE' { Assert-PrelaunchEvent $Journal $event }
            'RUN' {
                $output = Get-Field $event.Fields 'output' 'RUN'
                Assert-True ($output -cmatch $Journal.Lane.RawRegex) "RUN output is outside lane filename family: $output"
                $prefix = Assert-RunPrefix $Journal $event
                Assert-ChainFields -Journal $Journal -Chain $prefix -ExpectedRawName $output -PrefixOnly
            }
            'SOURCE_FENCE_POST' { Assert-PostFenceEvent $Journal $event }
            'BINARY_FENCE' { Assert-BinaryFenceEvent $event }
            'RESULT' {
                Assert-FieldNames $event.Fields @('case','start','complete','next_start','result','elapsed_s','bytes','sha256','path','cargo_exit_raw','cargo_exit_bytes','cargo_exit_sha256','meta','meta_bytes','meta_sha256','code_id','source_snapshot_sha256') 'RESULT'
                [void](Assert-CaseStartFields $event.Fields 'RESULT')
                Assert-True ((Get-Field $event.Fields 'path' 'RESULT') -cmatch $Journal.Lane.RawRegex) 'RESULT raw filename family mismatch'
                Assert-True ((Get-Field $event.Fields 'sha256' 'RESULT') -cmatch '^[0-9A-F]{64}$') 'RESULT raw SHA grammar'
                Assert-True ((Get-Field $event.Fields 'meta_sha256' 'RESULT') -cmatch '^[0-9A-F]{64}$') 'RESULT META SHA grammar'
                Assert-True ((Get-Field $event.Fields 'cargo_exit_sha256' 'RESULT') -cmatch '^[0-9A-F]{64}$') 'RESULT exit SHA grammar'
                Assert-Equal (Get-Field $event.Fields 'code_id' 'RESULT') $requiredCodeId 'RESULT code'
                Assert-Equal (Get-Field $event.Fields 'source_snapshot_sha256' 'RESULT') $requiredSourceSnapshotSha256 'RESULT source'
            }
            'STOP' { Assert-StopEvent $Journal $event }
            'ABORT' { Assert-AbortEvent $Journal $event }
            'DONE' { }
        }
    }
    for ($index=1; $index -lt $Journal.Events.Count; $index++) {
        Assert-True ($Journal.Events[$index].Event -cne 'SETUP') "$($Journal.Name) has a noninitial SETUP"
    }
    for ($index=0; $index -lt $Journal.Events.Count; $index++) {
        $event = $Journal.Events[$index]
        $last = $index -eq $Journal.Events.Count - 1
        $next = if($last){''}else{$Journal.Events[$index+1].Event}
        switch -CaseSensitive ($event.Event) {
            'SETUP' { if(-not $last){ Assert-True ($next -in @('UNTRUSTED_RETAINED','CASE_DONE','GATE','STOP','ABORT','DONE')) 'invalid event after SETUP' } }
            'UNTRUSTED_RETAINED' { if(-not $last){ Assert-True ($next -in @('CASE_DONE','GATE','STOP','ABORT','DONE')) 'invalid event after UNTRUSTED_RETAINED' } }
            'CASE_DONE' { if(-not $last){ Assert-True ($next -in @('CASE_DONE','GATE','STOP','ABORT','DONE')) 'invalid event after CASE_DONE' } }
            'GATE' { Assert-GateSuccessor $event $(if($last){$null}else{$Journal.Events[$index+1]}) $last }
            'SOURCE_FENCE' { Assert-True (-not $last -and $next -in @($(if($Journal.Lane.Replica){'PRELAUNCH_GATE'}else{'RUN'}),'ABORT')) 'SOURCE_FENCE has invalid successor' }
            'PRELAUNCH_GATE' {
                if ((Get-Field $event.Fields 'result' 'PRELAUNCH_GATE') -ceq 'PASS') { Assert-True (-not $last -and $next -in @('RUN','ABORT')) 'passing PRELAUNCH_GATE is not followed by RUN/ABORT' }
                elseif (-not $last) { Assert-True ($next -in @('GATE','STOP','ABORT')) 'WAIT_NO_LAUNCH PRELAUNCH_GATE is followed by a launch-path event' }
            }
            'RUN' { if(-not $last){ Assert-True ($next -in @('SOURCE_FENCE_POST','ABORT')) 'RUN has invalid successor' } }
            'SOURCE_FENCE_POST' {
                if((Get-Field $event.Fields 'result' 'SOURCE_FENCE_POST') -ceq 'PASS'){ Assert-True (-not $last -and $next -in @('BINARY_FENCE','ABORT')) 'passing SOURCE_FENCE_POST is not followed by BINARY_FENCE/ABORT' }
                else { Assert-True (-not $last -and $next -ceq 'ABORT') 'aborting SOURCE_FENCE_POST is not followed by ABORT' }
            }
            'BINARY_FENCE' { Assert-True (-not $last -and $next -in @('RESULT','ABORT')) 'BINARY_FENCE has invalid successor' }
            'RESULT' { if(-not $last){ Assert-True ($next -in @('GATE','CASE_DONE','STOP','ABORT','DONE')) 'RESULT has invalid successor' } }
            'STOP' { Assert-True $last 'STOP is not terminal in its journal' }
            'ABORT' { Assert-True $last 'ABORT is not terminal in its journal' }
            'DONE' { Assert-True $last 'DONE is not terminal in its journal' }
        }
    }
}

function Load-FrozenCensus {
    param([Parameter(Mandatory = $true)][string] $Path)
    $bytes = Read-SharedBytes -Path $Path
    Assert-Equal (Get-Sha256FromBytes $bytes) $requiredCensusSha256 'census SHA256 at load'
    $lines = @(ConvertFrom-ExactLfBytes -Bytes $bytes -Context 'frozen census')
    $cursor = 0
    $moves = @()
    for ($case = 0; $case -lt 11; $case++) {
        Assert-True ($cursor -lt $lines.Count) "census truncates before case $case"
        $universe = [regex]::Match($lines[$cursor], '^DOM_B2_D4_UNIVERSE case=(\d+) id=(\S+) pair=(\S+) coverage=(SPLIT|H_CONTAINING) root_count=(\d+) fingerprint=([0-9A-F]{16})$')
        Assert-True $universe.Success "malformed census universe for case $case"
        Assert-Equal ([int]$universe.Groups[1].Value) $case "census universe case $case"
        Assert-Equal $universe.Groups[2].Value $caseIds[$case] "census id case $case"
        Assert-Equal $universe.Groups[3].Value $casePairs[$case] "census pair case $case"
        Assert-Equal $universe.Groups[4].Value $caseCoverage[$case] "census coverage case $case"
        Assert-Equal ([int]$universe.Groups[5].Value) $widths[$case] "census width case $case"
        Assert-Equal $universe.Groups[6].Value $fingerprints[$case] "census fingerprint case $case"
        $cursor++
        $caseMoves = @()
        for ($rootIndex = 0; $rootIndex -lt $widths[$case]; $rootIndex++) {
            Assert-True ($cursor -lt $lines.Count) "census truncates in case $case root $rootIndex"
            $move = [regex]::Match($lines[$cursor], '^DOM_B2_D4_ROOT_MOVE case=(\d+) root_index=(\d+) q=(-?\d+) r=(-?\d+) fingerprint=([0-9A-F]{16})$')
            Assert-True $move.Success "malformed census move case $case root $rootIndex"
            Assert-Equal ([int]$move.Groups[1].Value) $case "census move case $case root $rootIndex"
            Assert-Equal ([int]$move.Groups[2].Value) $rootIndex "census root index case $case"
            Assert-Equal $move.Groups[5].Value $fingerprints[$case] "census move fingerprint case $case root $rootIndex"
            $caseMoves += [pscustomobject]@{ Q = [int]$move.Groups[3].Value; R = [int]$move.Groups[4].Value }
            $cursor++
        }
        $moves += ,@($caseMoves)
    }
    Assert-True ($cursor -lt $lines.Count) 'census lacks footer'
    Assert-Equal $lines[$cursor] 'DOM_B2_D4_CENSUS_DONE cases=11 root_actions=3648 result=PASS' 'census footer'
    Assert-Equal ($cursor + 1) $lines.Count 'census trailing lines'
    return [pscustomobject]@{ Moves = $moves; Bytes = $bytes.Length; Sha256 = Get-Sha256FromBytes $bytes }
}

function Read-CreditedRaw {
    param(
        [Parameter(Mandatory = $true)][string] $Path,
        [Parameter(Mandatory = $true)] $Lane,
        [Parameter(Mandatory = $true)][int] $FilenameCase,
        [Parameter(Mandatory = $true)][int] $FilenameStart,
        [Parameter(Mandatory = $true)] $Census
    )
    $bytes = Read-SharedBytes -Path $Path
    $lines = @(ConvertFrom-ExactLfBytes -Bytes $bytes -Context ([IO.Path]::GetFileName($Path)))
    Assert-True ($lines.Count -ge 3) "credited raw is truncated: $Path"
    $setup = [regex]::Match($lines[0], '^DOM_B2_D4_SHARD_SETUP case=(\d+) id=(\S+) start=(\d+) end=(\d+) root_count=(\d+) fingerprint=([0-9A-F]{16}) depth=(\d+) tt_bytes=(\d+) d6=(true|false) deadline_ms=(\d+) code_id=([A-Za-z0-9_.-]+)$')
    Assert-True $setup.Success "malformed credited raw setup: $Path"
    $case = [int]$setup.Groups[1].Value
    $start = [int]$setup.Groups[3].Value
    $end = [int]$setup.Groups[4].Value
    Assert-Equal $case $FilenameCase "raw filename/setup case"
    Assert-Equal $start $FilenameStart "raw filename/setup start"
    Assert-True ($start -ge 0 -and $start -lt $widths[$case] -and $end -gt $start -and $end -le $widths[$case]) "raw range is invalid: $Path"
    Assert-Equal $end ([Math]::Min($start + $requiredCount, $widths[$case])) 'raw frozen Count=64 end'
    Assert-Equal $setup.Groups[2].Value $caseIds[$case] 'raw case id'
    Assert-Equal ([int]$setup.Groups[5].Value) $widths[$case] 'raw root_count'
    Assert-Equal $setup.Groups[6].Value $fingerprints[$case] 'raw fingerprint'
    Assert-Equal $setup.Groups[7].Value '4' 'raw depth'
    Assert-Equal ([uint64]$setup.Groups[8].Value) $Lane.TtBytes 'raw TT config'
    Assert-Equal $setup.Groups[9].Value $Lane.D6 'raw D6 config'
    Assert-Equal ([int]$setup.Groups[10].Value) $requiredDeadlineMs 'raw deadline'
    Assert-Equal $setup.Groups[11].Value $requiredCodeId 'raw code id'

    $statuses = New-Object Collections.Generic.List[string]
    $rootIndices = New-Object Collections.Generic.List[int]
    $sawIncomplete = $false
    for ($lineIndex = 1; $lineIndex -lt $lines.Count - 1; $lineIndex++) {
        $row = [regex]::Match($lines[$lineIndex], '^DOM_B2_D4_ROOT_RESULT case=(\d+) root_index=(\d+) q=(-?\d+) r=(-?\d+) status=(WIN|UNKNOWN|LOSS|INCOMPLETE) terminal=(true|false) source=(direct_outcome|bounded_reference) nodes=(\d+) tt_hits=(\d+) tt_entries=(\d+) tt_bytes=(\d+) tt_clears=(\d+) wall_s=(\d+\.\d{6}) root_count=(\d+) root_depth=(\d+) child_depth=(\d+) d6=(true|false) deadline_ms=(\d+) fingerprint=([0-9A-F]{16}) code_id=([A-Za-z0-9_.-]+)$')
        Assert-True $row.Success "malformed credited raw result line $($lineIndex + 1): $Path"
        $rootIndex = [int]$row.Groups[2].Value
        $status = $row.Groups[5].Value
        Assert-Equal ([int]$row.Groups[1].Value) $case 'raw row case'
        Assert-Equal $rootIndex ($start + $lineIndex - 1) 'raw row order'
        Assert-True ($rootIndex -lt $end) "raw row exceeds setup end: $Path"
        Assert-Equal ([int]$row.Groups[3].Value) $Census.Moves[$case][$rootIndex].Q 'raw row q/census'
        Assert-Equal ([int]$row.Groups[4].Value) $Census.Moves[$case][$rootIndex].R 'raw row r/census'
        Assert-Equal ([int]$row.Groups[14].Value) $widths[$case] 'raw row root_count'
        Assert-Equal $row.Groups[15].Value '4' 'raw row root_depth'
        Assert-Equal $row.Groups[16].Value '3' 'raw row child_depth'
        Assert-Equal $row.Groups[17].Value $Lane.D6 'raw row D6'
        Assert-Equal ([int]$row.Groups[18].Value) $requiredDeadlineMs 'raw row deadline'
        Assert-Equal $row.Groups[19].Value $fingerprints[$case] 'raw row fingerprint'
        Assert-Equal $row.Groups[20].Value $requiredCodeId 'raw row code id'
        if ($sawIncomplete) { throw "raw row follows INCOMPLETE: $Path" }
        if ($status -ceq 'INCOMPLETE') { $sawIncomplete = $true }
        else { $rootIndices.Add($rootIndex) }
        $statuses.Add($status)
    }
    Assert-True ($statuses.Count -gt 0) "credited raw has no result rows: $Path"
    $footer = [regex]::Match($lines[$lines.Count - 1], '^DOM_B2_D4_SHARD_DONE case=(\d+) start=(\d+) end=(\d+) complete=(\d+) next_start=(\d+) fingerprint=([0-9A-F]{16}) result=(PASS|INCOMPLETE) code_id=([A-Za-z0-9_.-]+)$')
    Assert-True $footer.Success "malformed credited raw footer: $Path"
    $complete = [int]$footer.Groups[4].Value
    $nextStart = [int]$footer.Groups[5].Value
    $result = $footer.Groups[7].Value
    Assert-Equal ([int]$footer.Groups[1].Value) $case 'raw footer case'
    Assert-Equal ([int]$footer.Groups[2].Value) $start 'raw footer start'
    Assert-Equal ([int]$footer.Groups[3].Value) $end 'raw footer end'
    Assert-Equal $footer.Groups[6].Value $fingerprints[$case] 'raw footer fingerprint'
    Assert-Equal $footer.Groups[8].Value $requiredCodeId 'raw footer code id'
    Assert-Equal $complete $rootIndices.Count 'raw footer complete count'
    Assert-Equal $nextStart ($start + $complete) 'raw footer next_start'
    if ($result -ceq 'PASS') {
        Assert-True (-not $sawIncomplete -and $nextStart -eq $end -and $statuses.Count -eq ($end - $start)) "PASS raw stop semantics mismatch: $Path"
    }
    else {
        Assert-True ($sawIncomplete -and $statuses[$statuses.Count - 1] -ceq 'INCOMPLETE' -and $statuses.Count -eq ($complete + 1) -and $nextStart -lt $end) "INCOMPLETE raw stop semantics mismatch: $Path"
    }
    return [pscustomobject]@{
        Path = $Path; Name = [IO.Path]::GetFileName($Path); Bytes = $bytes.Length;
        Sha256 = Get-Sha256FromBytes $bytes; Case = $case; Start = $start; End = $end;
        Complete = $complete; NextStart = $nextStart; Result = $result; RootIndices = @($rootIndices | ForEach-Object { $_ })
    }
}

function Assert-EventIdentity {
    param([Parameter(Mandatory = $true)] $Event, [Parameter(Mandatory = $true)][int] $Case, [Parameter(Mandatory = $true)][int] $Start, [Parameter(Mandatory = $true)][string] $Context)
    Assert-Equal (Get-Field $Event.Fields 'case' $Context) ([string]$Case) "$Context case"
    Assert-Equal (Get-Field $Event.Fields 'start' $Context) ([string]$Start) "$Context start"
}

function Assert-RunPrefix {
    param([Parameter(Mandatory = $true)] $Journal, [Parameter(Mandatory = $true)] $Run)
    $lane = $Journal.Lane
    $case = Convert-ToInt (Get-Field $Run.Fields 'case' 'RUN') 'RUN case'
    $start = Convert-ToInt (Get-Field $Run.Fields 'start' 'RUN') 'RUN start'
    $expectedEvents = if ($lane.Replica) { @('GATE','SOURCE_FENCE','PRELAUNCH_GATE','RUN') }
        else { @('GATE','SOURCE_FENCE','RUN') }
    $first = $Run.Index - ($expectedEvents.Count - 1)
    Assert-True ($first -ge 0) "RUN prefix bounds fail for $($Journal.Name) case=$case start=$start"
    $prefix = @($Journal.Events[$first..$Run.Index])
    for ($index=0; $index -lt $expectedEvents.Count; $index++) {
        Assert-Equal $prefix[$index].Event $expectedEvents[$index] "RUN prefix event $index for $($Journal.Name) case=$case start=$start"
        Assert-EventIdentity $prefix[$index] $case $start $prefix[$index].Event
    }
    return [pscustomobject]@{ Case=$case; Start=$start; FirstIndex=$first; Events=$prefix }
}

function Assert-ChainEvents {
    param([Parameter(Mandatory = $true)] $Journal, [Parameter(Mandatory = $true)] $Run, [Parameter(Mandatory = $true)] $Result)
    $lane = $Journal.Lane
    $case = Convert-ToInt (Get-Field $Run.Fields 'case' 'RUN') 'RUN case'
    $start = Convert-ToInt (Get-Field $Run.Fields 'start' 'RUN') 'RUN start'
    $expectedEvents = if ($lane.Replica) {
        @('GATE','SOURCE_FENCE','PRELAUNCH_GATE','RUN','SOURCE_FENCE_POST','BINARY_FENCE','RESULT')
    }
    else { @('GATE','SOURCE_FENCE','RUN','SOURCE_FENCE_POST','BINARY_FENCE','RESULT') }
    $first = $Run.Index - $(if ($lane.Replica) { 3 } else { 2 })
    Assert-True ($first -ge 0 -and $first + $expectedEvents.Count - 1 -lt $Journal.Events.Count) "chain bounds fail for $($Journal.Name) case=$case start=$start"
    $chain = @($Journal.Events[$first..($first + $expectedEvents.Count - 1)])
    for ($index = 0; $index -lt $expectedEvents.Count; $index++) {
        Assert-Equal $chain[$index].Event $expectedEvents[$index] "chain event $index for $($Journal.Name) case=$case start=$start"
        Assert-EventIdentity -Event $chain[$index] -Case $case -Start $start -Context $chain[$index].Event
    }
    Assert-Equal $chain[$expectedEvents.Count - 1].Index $Result.Index 'chain RESULT identity'
    foreach ($abort in @($Journal.Events | Where-Object { $_.Event -ceq 'ABORT' })) {
        $sameIdentity = $abort.Fields.ContainsKey('case') -and $abort.Fields.ContainsKey('start') -and
            $abort.Fields['case'] -ceq ([string]$case) -and $abort.Fields['start'] -ceq ([string]$start)
        if ($sameIdentity -or ($abort.Index -ge $first -and $abort.Index -le $Result.Index)) {
            throw "journal ABORT is relevant to credited run $($Journal.Name) case=$case start=$start line=$($abort.LineNumber)"
        }
    }
    $wallSeconds = ($Result.Timestamp - $Run.Timestamp).TotalSeconds
    Assert-True ($wallSeconds -ge 0 -and $wallSeconds -lt 600) "journal invocation wall is not <10m for $($Journal.Name) case=$case start=$($start): $wallSeconds"
    $elapsed = Convert-ToDecimal (Get-Field $Result.Fields 'elapsed_s' 'RESULT') 'RESULT elapsed_s'
    Assert-True ($elapsed -ge 0 -and $elapsed -lt 600) "reported invocation elapsed is not <10m for $($Journal.Name) case=$case start=$($start): $elapsed"
    Assert-True ([Math]::Abs([double]$elapsed - $wallSeconds) -le 5.0) "reported/journal elapsed differs by >5s for $($Journal.Name) case=$case start=$start"
    return [pscustomobject]@{ Case = $case; Start = $start; FirstIndex = $first; Events = $chain; WallSeconds = $wallSeconds; Elapsed = $elapsed }
}

function Assert-ChainFields {
    param(
        [Parameter(Mandatory = $true)] $Journal,
        [Parameter(Mandatory = $true)] $Chain,
        [Parameter(Mandatory = $true)][string] $ExpectedRawName,
        [switch] $PrefixOnly
    )
    $lane = $Journal.Lane
    $events = @{}
    foreach ($event in $Chain.Events) { $events[$event.Event] = $event }
    $gate = $events['GATE'].Fields
    $gateNames = @('case','start','available_bytes','free_physical_bytes','foreign_cargo') + $(if ($lane.Replica) { @('primary_queue_active') } else { @() }) + @('result')
    Assert-FieldNames $gate $gateNames 'GATE'
    Assert-True ((Convert-ToInt64 (Get-Field $gate 'available_bytes' 'GATE') 'GATE available') -ge $availableFloor) 'GATE available memory is below 10 GiB'
    Assert-True ((Convert-ToInt64 (Get-Field $gate 'free_physical_bytes' 'GATE') 'GATE free') -ge $freeFloor) 'GATE free memory is below 5 GiB'
    Assert-Equal (Get-Field $gate 'foreign_cargo' 'GATE') '0' 'GATE foreign cargo'
    if ($lane.Replica) { Assert-Equal (Get-Field $gate 'primary_queue_active' 'GATE') 'false' 'GATE primary queue active' }
    Assert-Equal (Get-Field $gate 'result' 'GATE') 'PASS' 'GATE result'

    $source = $events['SOURCE_FENCE'].Fields
    Assert-FieldNames $source @('case','start','expected','actual','source_files','runner_sha256','census_sha256','code_id','rustc','cargo_configs','build_overrides','result') 'SOURCE_FENCE'
    Assert-Equal (Get-Field $source 'expected' 'SOURCE_FENCE') $requiredSourceSnapshotSha256 'SOURCE_FENCE expected'
    Assert-Equal (Get-Field $source 'actual' 'SOURCE_FENCE') $requiredSourceSnapshotSha256 'SOURCE_FENCE actual'
    Assert-Equal (Get-Field $source 'source_files' 'SOURCE_FENCE') ([string]$requiredSourceFiles) 'SOURCE_FENCE source files'
    Assert-Equal (Get-Field $source 'runner_sha256' 'SOURCE_FENCE') $lane.RunnerSha256 'SOURCE_FENCE runner'
    Assert-Equal (Get-Field $source 'census_sha256' 'SOURCE_FENCE') $requiredCensusSha256 'SOURCE_FENCE census'
    Assert-Equal (Get-Field $source 'code_id' 'SOURCE_FENCE') $requiredCodeId 'SOURCE_FENCE code'
    Assert-Equal (Get-Field $source 'rustc' 'SOURCE_FENCE') $requiredRustc 'SOURCE_FENCE rustc'
    Assert-Equal (Get-Field $source 'cargo_configs' 'SOURCE_FENCE') '0' 'SOURCE_FENCE cargo configs'
    Assert-Equal (Get-Field $source 'build_overrides' 'SOURCE_FENCE') '0' 'SOURCE_FENCE overrides'
    Assert-Equal (Get-Field $source 'result' 'SOURCE_FENCE') 'PASS' 'SOURCE_FENCE result'

    if ($lane.Replica) {
        $pre = $events['PRELAUNCH_GATE'].Fields
        Assert-FieldNames $pre @('case','start','available_bytes','free_physical_bytes','foreign_cargo','primary_queue_active','stop_requested','source_snapshot_sha256','runner_sha256','rustc','cargo_configs','build_overrides','result') 'PRELAUNCH_GATE'
        Assert-True ((Convert-ToInt64 (Get-Field $pre 'available_bytes' 'PRELAUNCH_GATE') 'PRE available') -ge $availableFloor) 'PRELAUNCH_GATE available memory is below 10 GiB'
        Assert-True ((Convert-ToInt64 (Get-Field $pre 'free_physical_bytes' 'PRELAUNCH_GATE') 'PRE free') -ge $freeFloor) 'PRELAUNCH_GATE free memory is below 5 GiB'
        Assert-Equal (Get-Field $pre 'foreign_cargo' 'PRELAUNCH_GATE') '0' 'PRELAUNCH_GATE foreign cargo'
        Assert-Equal (Get-Field $pre 'primary_queue_active' 'PRELAUNCH_GATE') 'false' 'PRELAUNCH_GATE primary queue'
        Assert-Equal (Get-Field $pre 'stop_requested' 'PRELAUNCH_GATE') 'false' 'PRELAUNCH_GATE stop'
        Assert-Equal (Get-Field $pre 'source_snapshot_sha256' 'PRELAUNCH_GATE') $requiredSourceSnapshotSha256 'PRELAUNCH_GATE source'
        Assert-Equal (Get-Field $pre 'runner_sha256' 'PRELAUNCH_GATE') $lane.RunnerSha256 'PRELAUNCH_GATE runner'
        Assert-Equal (Get-Field $pre 'rustc' 'PRELAUNCH_GATE') $requiredRustc 'PRELAUNCH_GATE rustc'
        Assert-Equal (Get-Field $pre 'cargo_configs' 'PRELAUNCH_GATE') '0' 'PRELAUNCH_GATE cargo configs'
        Assert-Equal (Get-Field $pre 'build_overrides' 'PRELAUNCH_GATE') '0' 'PRELAUNCH_GATE overrides'
        Assert-Equal (Get-Field $pre 'result' 'PRELAUNCH_GATE') 'PASS' 'PRELAUNCH_GATE result'
    }

    $run = $events['RUN'].Fields
    Assert-FieldNames $run @('case','start','count','output','cargo_stdout','cargo_stderr','cargo_exit_raw','child_wrapper_sha256','deadline_ms','cargo_wall_timeout_ms','tt_bytes','d6','code_id','source_snapshot_sha256') 'RUN'
    Assert-Equal (Get-Field $run 'count' 'RUN') ([string]$requiredCount) 'RUN count'
    Assert-Equal (Get-Field $run 'output' 'RUN') $ExpectedRawName 'RUN output'
    $stem = $ExpectedRawName.Substring(0, $ExpectedRawName.Length - '_RAW.log'.Length)
    Assert-Equal (Get-Field $run 'cargo_stdout' 'RUN') ($stem + '_CARGO_STDOUT_RAW.log') 'RUN stdout path'
    Assert-Equal (Get-Field $run 'cargo_stderr' 'RUN') ($stem + '_CARGO_STDERR_RAW.log') 'RUN stderr path'
    Assert-Equal (Get-Field $run 'cargo_exit_raw' 'RUN') ($stem + '_CARGO_EXIT_RAW.log') 'RUN exit path'
    Assert-Equal (Get-Field $run 'child_wrapper_sha256' 'RUN') $requiredChildWrapperSha256 'RUN child wrapper SHA'
    Assert-Equal (Get-Field $run 'deadline_ms' 'RUN') ([string]$requiredDeadlineMs) 'RUN deadline'
    Assert-Equal (Get-Field $run 'cargo_wall_timeout_ms' 'RUN') ([string]$requiredCargoWallTimeoutMs) 'RUN watchdog'
    Assert-Equal (Get-Field $run 'tt_bytes' 'RUN') ([string]$lane.TtBytes) 'RUN TT'
    Assert-Equal (Get-Field $run 'd6' 'RUN') $lane.D6 'RUN d6'
    Assert-Equal (Get-Field $run 'code_id' 'RUN') $requiredCodeId 'RUN code'
    Assert-Equal (Get-Field $run 'source_snapshot_sha256' 'RUN') $requiredSourceSnapshotSha256 'RUN source'

    if ($PrefixOnly) { return }

    $post = $events['SOURCE_FENCE_POST'].Fields
    Assert-FieldNames $post @('case','start','expected','actual','source_files','runner_initial','runner_actual','code_id','rustc','cargo_configs','build_overrides','result') 'SOURCE_FENCE_POST'
    Assert-Equal (Get-Field $post 'expected' 'SOURCE_FENCE_POST') $requiredSourceSnapshotSha256 'SOURCE_FENCE_POST expected'
    Assert-Equal (Get-Field $post 'actual' 'SOURCE_FENCE_POST') $requiredSourceSnapshotSha256 'SOURCE_FENCE_POST actual'
    Assert-Equal (Get-Field $post 'source_files' 'SOURCE_FENCE_POST') ([string]$requiredSourceFiles) 'SOURCE_FENCE_POST source files'
    Assert-Equal (Get-Field $post 'runner_initial' 'SOURCE_FENCE_POST') $lane.RunnerSha256 'SOURCE_FENCE_POST runner initial'
    Assert-Equal (Get-Field $post 'runner_actual' 'SOURCE_FENCE_POST') $lane.RunnerSha256 'SOURCE_FENCE_POST runner actual'
    Assert-Equal (Get-Field $post 'code_id' 'SOURCE_FENCE_POST') $requiredCodeId 'SOURCE_FENCE_POST code'
    Assert-Equal (Get-Field $post 'rustc' 'SOURCE_FENCE_POST') $requiredRustc 'SOURCE_FENCE_POST rustc'
    Assert-Equal (Get-Field $post 'cargo_configs' 'SOURCE_FENCE_POST') '0' 'SOURCE_FENCE_POST cargo configs'
    Assert-Equal (Get-Field $post 'build_overrides' 'SOURCE_FENCE_POST') '0' 'SOURCE_FENCE_POST overrides'
    Assert-Equal (Get-Field $post 'result' 'SOURCE_FENCE_POST') 'PASS' 'SOURCE_FENCE_POST result'

    $binary = $events['BINARY_FENCE'].Fields
    Assert-FieldNames $binary @('case','start','reported_path','expected_path','result') 'BINARY_FENCE'
    $binaryFull = [IO.Path]::GetFullPath((Join-Path $repoRoot $requiredBinaryRelative))
    Assert-Equal (Get-Field $binary 'reported_path' 'BINARY_FENCE') $binaryFull 'BINARY_FENCE reported path'
    Assert-Equal (Get-Field $binary 'expected_path' 'BINARY_FENCE') $binaryFull 'BINARY_FENCE expected path'
    Assert-Equal (Get-Field $binary 'result' 'BINARY_FENCE') 'PASS' 'BINARY_FENCE result'

    $result = $events['RESULT'].Fields
    Assert-FieldNames $result @('case','start','complete','next_start','result','elapsed_s','bytes','sha256','path','cargo_exit_raw','cargo_exit_bytes','cargo_exit_sha256','meta','meta_bytes','meta_sha256','code_id','source_snapshot_sha256') 'RESULT'
    Assert-Equal (Get-Field $result 'path' 'RESULT') $ExpectedRawName 'RESULT raw path'
    Assert-Equal (Get-Field $result 'cargo_exit_raw' 'RESULT') ($stem + '_CARGO_EXIT_RAW.log') 'RESULT exit path'
    Assert-Equal (Get-Field $result 'meta' 'RESULT') ($stem + '_META_RAW.log') 'RESULT meta path'
    Assert-Equal (Get-Field $result 'code_id' 'RESULT') $requiredCodeId 'RESULT code'
    Assert-Equal (Get-Field $result 'source_snapshot_sha256' 'RESULT') $requiredSourceSnapshotSha256 'RESULT source'
}

function Get-MetaSnapshot {
    $result = @{}
    foreach ($file in Get-ChildItem -LiteralPath $repoRoot -File -Force) {
        foreach ($label in @('PRIMARY','D6_OFF','SECOND_TT')) {
            $lane = $laneSpecs[$label]
            $match = [regex]::Match($file.Name, $lane.MetaRegex)
            if ($match.Success) {
                $key = $file.Name.ToUpperInvariant()
                if ($result.ContainsKey($key)) { throw "case-insensitive META filename collision: $($file.Name)" }
                $bytes = Read-SharedBytes -Path $file.FullName
                $result[$key] = [pscustomobject]@{
                    Lane = $lane; File = $file; Bytes = $bytes.Length; Sha256 = Get-Sha256FromBytes $bytes;
                    Case = [int]$match.Groups['case'].Value; Start = [int]$match.Groups['start'].Value;
                    Attempt = [int]$match.Groups['attempt'].Value
                }
            }
        }
    }
    return $result
}

function Get-FrozenJournalSnapshot {
    $result = @{}
    foreach ($file in Get-ChildItem -LiteralPath $repoRoot -File -Force) {
        foreach ($label in @('PRIMARY','D6_OFF','SECOND_TT')) {
            $lane = $laneSpecs[$label]
            if ($file.Name -cmatch $lane.JournalRegex) {
                $runIdMatch = [regex]::Match($file.Name, 'RUN(?<runid>[0-9]{2,})_RAW[.]log$')
                Assert-True $runIdMatch.Success "frozen journal discovery lacks RUN id: $($file.Name)"
                $runId = Convert-ToInt $runIdMatch.Groups['runid'].Value 'discovered journal RUN id'
                if ($runId -ge $lane.FirstRunId) {
                    $key = $file.Name.ToUpperInvariant()
                    Assert-True (-not $result.ContainsKey($key)) "case-insensitive frozen journal collision: $($file.Name)"
                    $bytes = Read-SharedBytes $file.FullName
                    $result[$key] = [pscustomobject]@{ Lane=$lane; File=$file; RunId=$runId; Bytes=$bytes.Length; Sha256=Get-Sha256FromBytes $bytes }
                }
            }
        }
    }
    return $result
}

function Assert-JournalSnapshotsEqual {
    param([Parameter(Mandatory = $true)][hashtable] $Before, [Parameter(Mandatory = $true)][hashtable] $After)
    Assert-Equal $After.Keys.Count $Before.Keys.Count 'frozen journal snapshot count/stability'
    foreach ($key in $Before.Keys) {
        Assert-True ($After.ContainsKey($key)) "frozen journal vanished/renamed during audit: $key"
        Assert-Equal $After[$key].Bytes $Before[$key].Bytes "frozen journal bytes changed during audit: $key"
        Assert-Equal $After[$key].Sha256 $Before[$key].Sha256 "frozen journal hash changed during audit: $key"
    }
}

function Assert-MetaSnapshotsEqual {
    param([Parameter(Mandatory = $true)][hashtable] $Before, [Parameter(Mandatory = $true)][hashtable] $After)
    Assert-Equal $After.Count $Before.Count 'META snapshot count/stability'
    foreach ($key in $Before.Keys) {
        Assert-True ($After.ContainsKey($key)) "META vanished/renamed during audit: $key"
        Assert-Equal $After[$key].Bytes $Before[$key].Bytes "META bytes changed during audit: $key"
        Assert-Equal $After[$key].Sha256 $Before[$key].Sha256 "META hash changed during audit: $key"
    }
}

function Assert-ExactSingleLfBytes {
    param([Parameter(Mandatory = $true)][byte[]] $Bytes, [Parameter(Mandatory = $true)][string] $Expected, [Parameter(Mandatory = $true)][string] $Context)
    $lines = @(ConvertFrom-ExactLfBytes -Bytes $Bytes -Context $Context)
    Assert-Equal $lines.Count 1 "$Context line count"
    Assert-Equal $lines[0] $Expected "$Context exact bytes"
}

function Assert-ExactSingleLfLine {
    param([Parameter(Mandatory = $true)][string] $Path, [Parameter(Mandatory = $true)][string] $Expected, [Parameter(Mandatory = $true)][string] $Context)
    $bytes = Read-SharedBytes -Path $Path
    Assert-ExactSingleLfBytes -Bytes $bytes -Expected $Expected -Context $Context
    return [pscustomobject]@{ Bytes = $bytes.Length; Sha256 = Get-Sha256FromBytes $bytes; OriginalBytes = $bytes }
}

function Assert-DoneInvocationBinding {
    param([Parameter(Mandatory = $true)][hashtable] $Fields, [Parameter(Mandatory = $true)] $Journal, [Parameter(Mandatory = $true)][string] $Context)
    $doneInvocations = Convert-ToInt (Get-Field $Fields 'invocations' $Context) "$Context invocations"
    Assert-True ($doneInvocations -ge 0) "$Context invocations is negative"
    $journalResults = @($Journal.Events | Where-Object { $_.Event -ceq 'RESULT' }).Count
    Assert-Equal $doneInvocations $journalResults "$Context invocations/RESULT count"
}

function Assert-TerminalDoneChronology {
    param([Parameter(Mandatory = $true)] $DoneRecord, [Parameter(Mandatory = $true)][AllowEmptyCollection()][object[]] $Journals, [Parameter(Mandatory = $true)][string] $Label)
    foreach ($journal in $Journals) {
        foreach ($event in $journal.Events) {
            if (-not ($journal.Path -ceq $DoneRecord.Journal.Path -and $event.Index -eq $DoneRecord.Event.Index)) {
                if ($journal.Path -ceq $DoneRecord.Journal.Path) {
                    Assert-True ($event.Index -lt $DoneRecord.Event.Index -and $event.Timestamp -le $DoneRecord.Event.Timestamp) "$Label same-journal activity is not ordered before terminal DONE: journal=$($journal.Name) line=$($event.LineNumber)"
                }
                else {
                    Assert-True ($event.Timestamp -lt $DoneRecord.Event.Timestamp) "$Label cross-journal activity is not before terminal DONE: journal=$($journal.Name) line=$($event.LineNumber)"
                }
            }
        }
    }
}

function Assert-DoneAndCoverage {
    param(
        [Parameter(Mandatory = $true)] $Lane,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][object[]] $Journals,
        [Parameter(Mandatory = $true)] $Coverage,
        [bool] $RequireFinal
    )
    $doneRecords = @()
    foreach ($journal in $Journals) {
        foreach ($event in @($journal.Events | Where-Object { $_.Event -ceq 'DONE' })) {
            $doneRecords += [pscustomobject]@{ Journal = $journal; Event = $event }
        }
    }
    $completeCases = 0
    $rootTotal = 0
    for ($case = 0; $case -lt 11; $case++) {
        $count = $Coverage[$case].Count
        $rootTotal += $count
        if ($count -eq $widths[$case]) { $completeCases++ }
    }
    foreach ($journal in $Journals) {
        $priorDoneCase = -1
        foreach ($caseDone in @($journal.Events | Where-Object { $_.Event -ceq 'CASE_DONE' })) {
            Assert-FieldNames $caseDone.Fields @('case','roots','code_id') "$($Lane.Label) CASE_DONE"
            $case = Convert-ToInt (Get-Field $caseDone.Fields 'case' 'CASE_DONE') 'CASE_DONE case'
            Assert-True ($case -ge 0 -and $case -lt 11) "$($Lane.Label) CASE_DONE case is out of range: $case"
            Assert-True ($case -gt $priorDoneCase) "$($Lane.Label) CASE_DONE order repeats/regresses in $($journal.Name): prior=$priorDoneCase current=$case"
            $priorDoneCase = $case
            Assert-Equal (Get-Field $caseDone.Fields 'roots' 'CASE_DONE') ([string]$widths[$case]) "CASE_DONE roots case=$case"
            Assert-Equal (Get-Field $caseDone.Fields 'code_id' 'CASE_DONE') $requiredCodeId "CASE_DONE code case=$case"
            Assert-Equal $Coverage[$case].Count $widths[$case] "$($Lane.Label) CASE_DONE credited coverage case=$case"
            foreach ($later in @($journal.Events | Where-Object {
                $_.Index -gt $caseDone.Index -and $_.Fields.ContainsKey('case') -and
                $_.Event -in @('GATE','SOURCE_FENCE','PRELAUNCH_GATE','RUN','SOURCE_FENCE_POST','BINARY_FENCE','RESULT','CASE_DONE')
            })) {
                $laterCase = Convert-ToInt (Get-Field $later.Fields 'case' "$($later.Event) after CASE_DONE") "$($later.Event) case"
                Assert-True ($laterCase -gt $case) "$($Lane.Label) activity for case=$laterCase follows CASE_DONE case=$case in $($journal.Name)"
            }
        }
    }
    if ($doneRecords.Count -gt 0) {
        Assert-Equal $doneRecords.Count 1 "$($Lane.Label) DONE count"
        Assert-Equal $rootTotal 3648 "$($Lane.Label) DONE coverage"
        Assert-Equal $completeCases 11 "$($Lane.Label) DONE cases"
        $done = $doneRecords[0].Event
        Assert-Equal $done.Index ($doneRecords[0].Journal.Events.Count - 1) "$($Lane.Label) DONE final position"
        $fields = $done.Fields
        $names = if ($Lane.Replica) { @('result','lane','invocations','cases','total_roots','tt_bytes','d6','code_id','source_snapshot_sha256') }
            else { @('result','invocations','cases','total_roots','code_id','source_snapshot_sha256') }
        Assert-FieldNames $fields $names "$($Lane.Label) DONE"
        Assert-Equal (Get-Field $fields 'result' 'DONE') 'PASS' 'DONE result'
        Assert-DoneInvocationBinding $fields $doneRecords[0].Journal "$($Lane.Label) DONE"
        Assert-Equal (Get-Field $fields 'cases' 'DONE') '11' 'DONE cases'
        Assert-Equal (Get-Field $fields 'total_roots' 'DONE') '3648' 'DONE total_roots'
        Assert-Equal (Get-Field $fields 'code_id' 'DONE') $requiredCodeId 'DONE code'
        Assert-Equal (Get-Field $fields 'source_snapshot_sha256' 'DONE') $requiredSourceSnapshotSha256 'DONE source'
        if ($Lane.Replica) {
            Assert-Equal (Get-Field $fields 'lane' 'DONE') $Lane.ReplicaLane 'DONE lane'
            Assert-Equal (Get-Field $fields 'tt_bytes' 'DONE') ([string]$Lane.TtBytes) 'DONE TT'
            Assert-Equal (Get-Field $fields 'd6' 'DONE') $Lane.D6 'DONE d6'
        }
        for ($case = 0; $case -lt 11; $case++) {
            $caseDone = @($doneRecords[0].Journal.Events | Where-Object {
                $_.Index -lt $done.Index -and $_.Event -ceq 'CASE_DONE' -and
                $_.Fields.ContainsKey('case') -and $_.Fields['case'] -ceq ([string]$case)
            })
            Assert-Equal $caseDone.Count 1 "$($Lane.Label) final journal CASE_DONE case=$case count"
            Assert-FieldNames $caseDone[0].Fields @('case','roots','code_id') "$($Lane.Label) CASE_DONE case=$case"
            Assert-Equal (Get-Field $caseDone[0].Fields 'roots' 'CASE_DONE') ([string]$widths[$case]) "CASE_DONE roots case=$case"
            Assert-Equal (Get-Field $caseDone[0].Fields 'code_id' 'CASE_DONE') $requiredCodeId "CASE_DONE code case=$case"
        }
        Assert-TerminalDoneChronology $doneRecords[0] $Journals $Lane.Label
    }
    if ($RequireFinal) {
        Assert-Equal $doneRecords.Count 1 "$($Lane.Label) final DONE count"
        Assert-Equal $rootTotal 3648 "$($Lane.Label) final roots"
        Assert-Equal $completeCases 11 "$($Lane.Label) final cases"
    }
    return [pscustomobject]@{ Roots = $rootTotal; CompleteCases = $completeCases; Done = ($doneRecords.Count -eq 1) }
}

function Assert-GlobalChronology {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][object[]] $Journals,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][object[]] $Credits,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][object[]] $UnmatchedRuns
    )
    foreach ($label in @('PRIMARY','D6_OFF','SECOND_TT')) {
        $laneJournals = @($Journals | Where-Object { $_.Lane.Label -ceq $label } | Sort-Object RunId)
        for ($index=1; $index -lt $laneJournals.Count; $index++) {
            Assert-True ($laneJournals[$index-1].RunId -lt $laneJournals[$index].RunId) "$label journal RUN ids repeat/regress"
            $priorEnd = $laneJournals[$index-1].Events[$laneJournals[$index-1].Events.Count-1].Timestamp
            $nextStart = $laneJournals[$index].Events[0].Timestamp
            Assert-True ($priorEnd -lt $nextStart) "$label journals overlap/regress: $($laneJournals[$index-1].Name) then $($laneJournals[$index].Name)"
        }
    }

    $intervals = New-Object Collections.Generic.List[object]
    foreach ($credit in $Credits) {
        $runEvent = @($credit.Chain.Events | Where-Object { $_.Event -ceq 'RUN' })[0]
        $resultEvent = @($credit.Chain.Events | Where-Object { $_.Event -ceq 'RESULT' })[0]
        $intervals.Add([pscustomobject]@{ Lane=$credit.Lane.Label; Raw=$credit.Raw.Name; Start=$runEvent.Timestamp; End=$resultEvent.Timestamp; Kind='CREDITED' })
    }
    foreach ($unmatched in $UnmatchedRuns) {
        $end = if($unmatched.Classification -ceq 'ABORTED_UNCREDITED'){$unmatched.Abort.Timestamp}else{[DateTimeOffset]::MaxValue}
        $intervals.Add([pscustomobject]@{ Lane=$unmatched.Journal.Lane.Label; Raw=(Get-Field $unmatched.Run.Fields 'output' 'unmatched RUN'); Start=$unmatched.Run.Timestamp; End=$end; Kind=$unmatched.Classification })
    }
    $allRuns = @($Journals | ForEach-Object { $_.Events } | Where-Object { $_.Event -ceq 'RUN' }).Count
    Assert-Equal $intervals.Count $allRuns 'global RUN/invocation-interval count'
    $ordered = @($intervals | Sort-Object Start, End)
    for ($index=1; $index -lt $ordered.Count; $index++) {
        Assert-True ($ordered[$index-1].End -le $ordered[$index].Start) "Cargo invocation intervals overlap: prior=$($ordered[$index-1].Lane)/$($ordered[$index-1].Raw) next=$($ordered[$index].Lane)/$($ordered[$index].Raw)"
    }

    function Get-LaneDoneRecord([string] $Label) {
        $records = @()
        foreach ($journal in @($Journals | Where-Object { $_.Lane.Label -ceq $Label })) {
            foreach ($event in @($journal.Events | Where-Object { $_.Event -ceq 'DONE' })) { $records += [pscustomobject]@{Journal=$journal;Event=$event} }
        }
        if ($records.Count -eq 1) { return $records[0] }
        return $null
    }
    $primaryDone = Get-LaneDoneRecord 'PRIMARY'
    $d6Done = Get-LaneDoneRecord 'D6_OFF'
    $d6Journals = @($Journals | Where-Object { $_.Lane.Label -ceq 'D6_OFF' })
    $secondJournals = @($Journals | Where-Object { $_.Lane.Label -ceq 'SECOND_TT' })
    if ($d6Journals.Count -gt 0) {
        Assert-True ($null -ne $primaryDone) 'D6_OFF journal activity exists before an attested terminal PRIMARY DONE'
        $firstD6 = @($d6Journals | ForEach-Object { $_.Events[0].Timestamp } | Sort-Object)[0]
        Assert-True ($primaryDone.Event.Timestamp -lt $firstD6) 'PRIMARY DONE is not before first D6_OFF journal activity'
    }
    if ($secondJournals.Count -gt 0) {
        $d6Terminal = $d6Done
        if ($null -eq $d6Terminal) {
            Assert-True ($d6Journals.Count -gt 0) 'SECOND_TT journal activity exists without any prior D6_OFF journal'
            $lastD6Journal = @($d6Journals | Sort-Object RunId)[-1]
            $lastD6Event = $lastD6Journal.Events[$lastD6Journal.Events.Count - 1]
            Assert-True ($lastD6Event.Event -ceq 'STOP') 'SECOND_TT journal activity requires D6_OFF DONE or a terminal clean STOP'
            $d6Terminal = [pscustomobject]@{ Journal=$lastD6Journal; Event=$lastD6Event }
        }
        $firstSecond = @($secondJournals | ForEach-Object { $_.Events[0].Timestamp } | Sort-Object)[0]
        Assert-True ($d6Terminal.Event.Timestamp -lt $firstSecond) 'D6_OFF terminal DONE/clean STOP is not before first SECOND_TT journal activity'
    }
    return [pscustomobject]@{ Intervals=$intervals.Count; Result='PASS' }
}

function Get-TerminalAbortForRun {
    param([Parameter(Mandatory = $true)] $Journal, [Parameter(Mandatory = $true)] $Run)
    $caseText = Get-Field $Run.Fields 'case' 'unmatched RUN'
    $startText = Get-Field $Run.Fields 'start' 'unmatched RUN'
    $candidates = @($Journal.Events | Where-Object {
        if ($_.Index -le $Run.Index -or $_.Event -cne 'ABORT') { return $false }
        $sameIdentity = $_.Fields.ContainsKey('case') -and $_.Fields.ContainsKey('start') -and
            $_.Fields['case'] -ceq $caseText -and $_.Fields['start'] -ceq $startText
        $terminalGlobal = $_.Fields.ContainsKey('reason') -and $_.Fields['reason'] -ceq 'runner_exception' -and
            $_.Index -eq $Journal.Events.Count - 1 -and
            @($Journal.Events | Where-Object { $_.Index -gt $Run.Index -and $_.Index -lt $Journal.Events.Count - 1 -and $_.Event -in @('RUN','RESULT') }).Count -eq 0
        return $sameIdentity -or $terminalGlobal
    })
    Assert-True ($candidates.Count -le 1) "multiple ABORTs follow unmatched RUN $($Journal.Name) case=$caseText start=$startText"
    if ($candidates.Count -eq 1) { return $candidates[0] }
    return $null
}

function Assert-AbortedInvocationTiming {
    param([Parameter(Mandatory = $true)] $Run, [Parameter(Mandatory = $true)] $Abort, [Parameter(Mandatory = $true)][string] $Context)
    $wall = ($Abort.Timestamp - $Run.Timestamp).TotalSeconds
    Assert-True ($wall -ge 0 -and $wall -lt 600) "$Context aborted invocation journal wall is not <10m: $wall"
    if ($Abort.Fields.ContainsKey('elapsed_s')) {
        $elapsed = Convert-ToDecimal (Get-Field $Abort.Fields 'elapsed_s' $Context) "$Context elapsed_s"
        Assert-True ($elapsed -ge 0 -and $elapsed -lt 600) "$Context aborted invocation reported elapsed is not <10m: $elapsed"
        Assert-True ([Math]::Abs([double]$elapsed - $wall) -le 5.0) "$Context aborted reported/journal elapsed differs by >5s"
    }
    return [pscustomobject]@{ WallSeconds=$wall }
}

function Invoke-InMemorySelfTest {
    $tests = New-Object Collections.Generic.List[string]
    function Pass([string] $Name) { $tests.Add($Name); "SELFTEST name=$Name result=PASS" }
    function MustThrow([scriptblock] $Body, [string] $Name) {
        $threw = $false
        try { & $Body } catch { $threw = $true }
        if (-not $threw) { throw "self-test expected rejection: $Name" }
        Pass $Name
    }

    $ascii = [Text.Encoding]::ASCII
    $lf = $ascii.GetBytes("one`n")
    Assert-Equal (@(ConvertFrom-ExactLfBytes $lf 'self LF')).Count 1 'self LF line count'
    Pass 'exact_lf_accept'
    MustThrow { ConvertFrom-ExactLfBytes ($ascii.GetBytes("one`r`n")) 'self CRLF' } 'exact_lf_reject_crlf'
    $canonicalMeta = 'DOM_B2_D4_SHARD_META version=1 raw=X_RAW.log raw_sha256=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA raw_bytes=1 source_snapshot_sha256=BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB runner_sha256=CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC census_sha256=DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD code_id=DOM_B2_D4_PRIMARY_V3 deadline_ms=480000 tt_bytes=536870912 d6=true target=x86_64-pc-windows-msvc release=true test_threads=1'
    Assert-ExactSingleLfBytes ($ascii.GetBytes($canonicalMeta + "`n")) $canonicalMeta 'self canonical META'
    Pass 'canonical_meta_accept'
    MustThrow {
        Assert-ExactSingleLfBytes ($ascii.GetBytes($canonicalMeta + "`n")) ($canonicalMeta.Replace('raw_bytes=1','raw_bytes=2')) 'self META tamper'
    } 'canonical_meta_tamper_rejected'
    $canonicalExit = 'DOM_B2_D4_CARGO_EXIT code=0 runner_sha256=CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC'
    Assert-ExactSingleLfBytes ($ascii.GetBytes($canonicalExit + "`n")) $canonicalExit 'self canonical exit'
    Pass 'canonical_cargo_exit_accept'
    MustThrow { Get-Fields 'a=1 a=2' 'self duplicate' } 'duplicate_field_rejected'

    $baseTime = [DateTimeOffset]::ParseExact('2026-07-18T12:00:00.0000000-04:00','yyyy-MM-ddTHH:mm:ss.fffffffzzz',$invariant)
    function Event([int] $Index, [string] $Name, [int] $OffsetSeconds) {
        [pscustomobject]@{
            Index=$Index; LineNumber=$Index+1; Event=$Name; Timestamp=$baseTime.AddSeconds($OffsetSeconds);
            Fields=@{ case='0'; start='0'; elapsed_s='5.000' }; Original='synthetic'
        }
    }
    $primaryEvents = @((Event 0 'GATE' 0),(Event 1 'SOURCE_FENCE' 0),(Event 2 'RUN' 1),(Event 3 'SOURCE_FENCE_POST' 5),(Event 4 'BINARY_FENCE' 5),(Event 5 'RESULT' 6))
    $primaryJournal = [pscustomobject]@{ Lane=$laneSpecs.PRIMARY; Name='memory-primary'; Events=$primaryEvents }
    [void](Assert-ChainEvents $primaryJournal $primaryEvents[2] $primaryEvents[5])
    Pass 'primary_chain_accept'
    $replicaEvents = @((Event 0 'GATE' 0),(Event 1 'SOURCE_FENCE' 0),(Event 2 'PRELAUNCH_GATE' 1),(Event 3 'RUN' 1),(Event 4 'SOURCE_FENCE_POST' 5),(Event 5 'BINARY_FENCE' 5),(Event 6 'RESULT' 6))
    $replicaJournal = [pscustomobject]@{ Lane=$laneSpecs.D6_OFF; Name='memory-replica'; Events=$replicaEvents }
    [void](Assert-ChainEvents $replicaJournal $replicaEvents[3] $replicaEvents[6])
    Pass 'replica_prelaunch_chain_accept'
    $missingPre = @($replicaEvents[0],$replicaEvents[1],$replicaEvents[3],$replicaEvents[4],$replicaEvents[5],$replicaEvents[6])
    for ($i=0;$i -lt $missingPre.Count;$i++){ $missingPre[$i].Index=$i; $missingPre[$i].LineNumber=$i+1 }
    $missingJournal = [pscustomobject]@{ Lane=$laneSpecs.D6_OFF; Name='memory-missing-pre'; Events=$missingPre }
    MustThrow { Assert-ChainEvents $missingJournal $missingPre[2] $missingPre[5] } 'replica_missing_prelaunch_rejected'
    $longEvents = @((Event 0 'GATE' 0),(Event 1 'SOURCE_FENCE' 0),(Event 2 'RUN' 1),(Event 3 'SOURCE_FENCE_POST' 600),(Event 4 'BINARY_FENCE' 600),(Event 5 'RESULT' 601))
    $longEvents[5].Fields.elapsed_s='600.000'
    $longJournal = [pscustomobject]@{ Lane=$laneSpecs.PRIMARY; Name='memory-long'; Events=$longEvents }
    MustThrow { Assert-ChainEvents $longJournal $longEvents[2] $longEvents[5] } 'ten_minute_boundary_rejected'
    $abortEvents = @((Event 0 'GATE' 0),(Event 1 'SOURCE_FENCE' 0),(Event 2 'RUN' 1),(Event 3 'ABORT' 2),(Event 4 'SOURCE_FENCE_POST' 5),(Event 5 'BINARY_FENCE' 5),(Event 6 'RESULT' 6))
    $abortJournal = [pscustomobject]@{ Lane=$laneSpecs.PRIMARY; Name='memory-abort'; Events=$abortEvents }
    MustThrow { Assert-ChainEvents $abortJournal $abortEvents[2] $abortEvents[6] } 'relevant_abort_rejected'
    MustThrow { Assert-FieldNames @{case='0';start='0';extra='x'} @('case','start') 'self collision' } 'extra_config_field_rejected'

    $coverage = @()
    for ($case=0;$case -lt 11;$case++){ $coverage += ,([Collections.Generic.HashSet[int]]::new()) }
    Assert-True ($coverage[0].Add(0)) 'self coverage first add'
    Assert-True (-not $coverage[0].Add(0)) 'self duplicate add did not reject'
    Pass 'duplicate_root_detectable'
    $syntheticRuns = @([pscustomobject]@{ output='OPEN_RAW.log' })
    $syntheticResults = @()
    Assert-Equal $syntheticResults.Count 0 'self open RUN credit count'
    Assert-Equal $syntheticRuns.Count 1 'self open RUN visibility'
    Pass 'open_run_is_uncredited'
    $syntheticMeta = @{ 'M_META_RAW.LOG' = 1 }
    $syntheticResultMeta = @{}
    MustThrow { Assert-Equal $syntheticResultMeta.Keys.Count $syntheticMeta.Keys.Count 'self orphan META' } 'orphan_meta_rejected'
    $syntheticFrozenJournals = @{ 'RUN04' = 1 }
    $syntheticExplicitJournals = @{}
    MustThrow { Assert-Equal $syntheticExplicitJournals.Keys.Count $syntheticFrozenJournals.Keys.Count 'self omitted frozen journal' } 'omitted_later_journal_rejected'

    $waitGate = [pscustomobject]@{ Fields=@{case='0';start='0';available_bytes='10737418239';free_physical_bytes='5368709120';foreign_cargo='0';result='WAIT_NO_LAUNCH'} }
    $waitJournal = [pscustomobject]@{ Lane=$laneSpecs.PRIMARY; Name='memory-wait' }
    Assert-GateEvent $waitJournal $waitGate
    Pass 'failed_gate_wait_no_launch_accept'
    $falsePassGate = [pscustomobject]@{ Fields=@{case='0';start='0';available_bytes='10737418239';free_physical_bytes='5368709120';foreign_cargo='0';result='PASS'} }
    MustThrow { Assert-GateEvent $waitJournal $falsePassGate } 'below_floor_pass_gate_rejected'
    $passGateSuccessor=[pscustomobject]@{Fields=@{result='PASS'}}
    $afterGateStop=[pscustomobject]@{Event='STOP';Fields=@{reason='operator_stop_file_after_gate'}}
    Assert-GateSuccessor $passGateSuccessor $afterGateStop $false
    Pass 'pass_gate_operator_stop_race_accept'
    $wrongAfterGateStop=[pscustomobject]@{Event='STOP';Fields=@{reason='operator_stop_file'}}
    MustThrow { Assert-GateSuccessor $passGateSuccessor $wrongAfterGateStop $false } 'pass_gate_wrong_stop_reason_rejected'
    MustThrow { Assert-Equal 63 ([Math]::Min(0 + $requiredCount, $widths[0])) 'self frozen raw end' } 'raw_end_not_count64_rejected'
    MustThrow { Resolve-BoundManifestPath '../outside' 'self manifest traversal' } 'source_manifest_traversal_rejected'

    $aRun=Event 0 'RUN' 0; $bRun=Event 1 'RUN' 5; $aResult=Event 2 'RESULT' 10; $bResult=Event 3 'RESULT' 15
    $overlapJournal=[pscustomobject]@{Lane=$laneSpecs.PRIMARY;Name='memory-overlap';Events=@($aRun,$bRun,$aResult,$bResult)}
    $creditA=[pscustomobject]@{Lane=$laneSpecs.PRIMARY;Raw=[pscustomobject]@{Name='A'};Chain=[pscustomobject]@{Events=@($aRun,$aResult)}}
    $creditB=[pscustomobject]@{Lane=$laneSpecs.PRIMARY;Raw=[pscustomobject]@{Name='B'};Chain=[pscustomobject]@{Events=@($bRun,$bResult)}}
    MustThrow { Assert-GlobalChronology @($overlapJournal) @($creditA,$creditB) @() } 'global_invocation_overlap_rejected'
    $slowRun=Event 0 'RUN' 0; $slowAbort=Event 1 'ABORT' 601; $slowAbort.Fields.elapsed_s='570.000'
    MustThrow { Assert-AbortedInvocationTiming $slowRun $slowAbort 'memory slow abort' } 'aborted_wall_over_ten_minutes_rejected'
    $globalRun=Event 0 'RUN' 0; $globalPost=Event 1 'SOURCE_FENCE_POST' 1; $globalAbort=Event 2 'ABORT' 2
    $globalAbort.Fields=@{reason='runner_exception';message='x'}
    $globalJournal=[pscustomobject]@{Name='memory-global-abort';Events=@($globalRun,$globalPost,$globalAbort)}
    Assert-True ($null -ne (Get-TerminalAbortForRun $globalJournal $globalRun)) 'global runner_exception after POST was not associated'
    Pass 'global_abort_after_post_associated'
    $binaryRun=Event 0 'RUN' 0; $binaryPost=Event 1 'SOURCE_FENCE_POST' 1; $binaryFence=Event 2 'BINARY_FENCE' 2; $binaryAbort=Event 3 'ABORT' 3
    $binaryAbort.Fields=@{reason='runner_exception';message='x'}
    $binaryJournal=[pscustomobject]@{Name='memory-binary-abort';Events=@($binaryRun,$binaryPost,$binaryFence,$binaryAbort)}
    Assert-True ($null -ne (Get-TerminalAbortForRun $binaryJournal $binaryRun)) 'global runner_exception after BINARY was not associated'
    Pass 'global_abort_after_binary_associated'
    $d6Setup=Event 0 'SETUP' 20
    $earlyD6=[pscustomobject]@{Lane=$laneSpecs.D6_OFF;Name='memory-early-d6';Events=@($d6Setup)}
    MustThrow { Assert-GlobalChronology @($earlyD6) @() @() } 'reversed_lane_order_rejected'
    $phasePrimaryDone=Event 0 'DONE' 0
    $phasePrimaryJournal=[pscustomobject]@{Lane=$laneSpecs.PRIMARY;Name='memory-phase-primary';RunId=3;Events=@($phasePrimaryDone)}
    $phaseD6Stop=Event 0 'STOP' 10
    $phaseD6Journal=[pscustomobject]@{Lane=$laneSpecs.D6_OFF;Name='memory-phase-d6';RunId=1;Events=@($phaseD6Stop)}
    $phaseSecondSetup=Event 0 'SETUP' 20
    $phaseSecondJournal=[pscustomobject]@{Lane=$laneSpecs.SECOND_TT;Name='memory-phase-second';RunId=1;Events=@($phaseSecondSetup)}
    [void](Assert-GlobalChronology @($phasePrimaryJournal,$phaseD6Journal,$phaseSecondJournal) @() @())
    Pass 'd6_clean_stop_then_second_partial_accept'
    $lateD6Stop=Event 0 'STOP' 21
    $lateD6Journal=[pscustomobject]@{Lane=$laneSpecs.D6_OFF;Name='memory-late-d6';RunId=1;Events=@($lateD6Stop)}
    MustThrow { Assert-GlobalChronology @($phasePrimaryJournal,$lateD6Journal,$phaseSecondJournal) @() @() } 'd6_stop_second_overlap_rejected'
    $unterminatedD6=Event 0 'SETUP' 10
    $unterminatedD6Journal=[pscustomobject]@{Lane=$laneSpecs.D6_OFF;Name='memory-unterminated-d6';RunId=1;Events=@($unterminatedD6)}
    MustThrow { Assert-GlobalChronology @($phasePrimaryJournal,$unterminatedD6Journal,$phaseSecondJournal) @() @() } 'second_without_d6_terminal_rejected'
    $journalOne=[pscustomobject]@{Lane=$laneSpecs.PRIMARY;Name='memory-run04';RunId=4;Events=@((Event 0 'SETUP' 20))}
    $journalTwo=[pscustomobject]@{Lane=$laneSpecs.PRIMARY;Name='memory-run05';RunId=5;Events=@((Event 0 'SETUP' 19))}
    MustThrow { Assert-GlobalChronology @($journalOne,$journalTwo) @() @() } 'journal_runid_time_regression_rejected'

    $doneEvent=Event 0 'DONE' 10
    $doneJournal=[pscustomobject]@{Path='memory-done';Name='memory-done';Events=@($doneEvent)}
    $laterEvent=Event 0 'SETUP' 11
    $laterJournal=[pscustomobject]@{Path='memory-later';Name='memory-later';Events=@($laterEvent)}
    $doneRecord=[pscustomobject]@{Journal=$doneJournal;Event=$doneEvent}
    MustThrow { Assert-TerminalDoneChronology $doneRecord @($doneJournal,$laterJournal) 'memory' } 'post_done_activity_rejected'
    $equalCaseDone=Event 0 'CASE_DONE' 10; $equalDone=Event 1 'DONE' 10
    $equalDoneJournal=[pscustomobject]@{Path='memory-equal-done';Name='memory-equal-done';Events=@($equalCaseDone,$equalDone)}
    $equalDoneRecord=[pscustomobject]@{Journal=$equalDoneJournal;Event=$equalDone}
    Assert-TerminalDoneChronology $equalDoneRecord @($equalDoneJournal) 'memory equal DONE'
    Pass 'equal_timestamp_case_done_then_done_accept'
    $badDoneFields=@{invocations='2'}
    $oneResultJournal=[pscustomobject]@{Events=@((Event 0 'RESULT' 1))}
    MustThrow { Assert-DoneInvocationBinding $badDoneFields $oneResultJournal 'memory DONE' } 'done_invocation_count_rejected'
    MustThrow { Assert-Equal 1 0 'self FINAL unmatched RUN count' } 'final_open_run_rejected'
    MustThrow { Assert-Equal 1 0 'self FINAL ABORT count' } 'final_abort_rejected'
    $badAbort=[pscustomobject]@{FieldText='reason=runner_exception message=x extra=y';Fields=@{reason='runner_exception';message='x';extra='y'};Original='synthetic'}
    MustThrow { Assert-AbortEvent $waitJournal $badAbort } 'abort_extra_field_rejected'
    $mixedCaseStop=[pscustomobject]@{Fields=@{reason='OPERATOR_STOP_FILE';case='0';next_start='0';path='x'}}
    MustThrow { Assert-StopEvent $waitJournal $mixedCaseStop } 'mixed_case_stop_reason_rejected'
    $badPost=[pscustomobject]@{Fields=@{
        case='0';start='0';expected=$requiredSourceSnapshotSha256;actual=('0'*64);source_files='20';
        runner_initial=$requiredPrimaryRunnerSha256;runner_actual=('0'*64);code_id=$requiredCodeId;rustc=$requiredRustc;
        cargo_configs='0';build_overrides='0';result='banana'
    }}
    MustThrow { Assert-PostFenceEvent $waitJournal $badPost } 'post_fence_invalid_result_rejected'
    "DOM_B2_D4_CHAIN_AUDIT_SELFTEST version=1 tests=$($tests.Count) failures=0 filesystem_writes=0 cargo_invocations=0 result=PASS"
}

if ($SelfTest) {
    Invoke-InMemorySelfTest
    exit 0
}

# Audit mode begins here.  No function below this point writes to disk.
$primaryRunnerFull = Assert-BoundFile 'scripts/dom_b2_d4_run_queue.ps1' $requiredPrimaryRunnerSha256 49762
$replicaRunnerFull = Assert-BoundFile 'scripts/dom_b2_d4_run_replica_queue.ps1' $requiredReplicaRunnerSha256 56516
$sourceSnapshotFull = Assert-BoundFile 'DOM_B2_D4_PRIMARY_V3_CODE_SNAPSHOT_RAW.log' $requiredSourceSnapshotSha256 3399
$censusFull = Assert-BoundFile 'DOM_B2_D4_CENSUS_RAW.log' $requiredCensusSha256 292756
$binaryFull = Assert-BoundFile $requiredBinaryRelative $requiredBinarySha256 3290112
[void](Assert-LiveSourceSnapshot $sourceSnapshotFull)
$census = Load-FrozenCensus $censusFull

$journalInputs = @(
    [pscustomobject]@{ Lane=$laneSpecs.PRIMARY; Paths=@($PrimaryJournalPath) },
    [pscustomobject]@{ Lane=$laneSpecs.D6_OFF; Paths=@($D6OffJournalPath) },
    [pscustomobject]@{ Lane=$laneSpecs.SECOND_TT; Paths=@($SecondTtJournalPath) }
)
$frozenJournalBefore = Get-FrozenJournalSnapshot
$journals = New-Object Collections.Generic.List[object]
$journalPathsSeen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($input in $journalInputs) {
    foreach ($path in $input.Paths) {
        Assert-True (-not [string]::IsNullOrWhiteSpace($path)) "$($input.Lane.Label) contains an empty journal path"
        $journal = Read-Journal -Path $path -Lane $input.Lane
        Assert-True ($journalPathsSeen.Add($journal.Path)) "duplicate/cross-lane journal input: $($journal.Path)"
        Assert-SetupEvent $journal
        Assert-JournalGrammar $journal
        $journals.Add($journal)
    }
}
Assert-Equal $journals.Count $frozenJournalBefore.Keys.Count 'explicit/frozen journal one-to-one count'
foreach ($journal in $journals) {
    Assert-True ($frozenJournalBefore.ContainsKey($journal.Name.ToUpperInvariant())) "explicit journal is outside frozen discovery: $($journal.Name)"
}
foreach ($key in $frozenJournalBefore.Keys) {
    Assert-True (@($journals | Where-Object { $_.Name.ToUpperInvariant() -ceq $key }).Count -eq 1) "frozen journal omitted from explicit inputs: $($frozenJournalBefore[$key].File.Name)"
}

$metaBefore = Get-MetaSnapshot
$resultByMeta = @{}
$runOutputs = @{}
$openRuns = New-Object Collections.Generic.List[object]
foreach ($journal in $journals) {
    $runs = @($journal.Events | Where-Object { $_.Event -ceq 'RUN' })
    $results = @($journal.Events | Where-Object { $_.Event -ceq 'RESULT' })
    foreach ($run in $runs) {
        $output = Get-Field $run.Fields 'output' 'RUN'
        $key = $output.ToUpperInvariant()
        Assert-True (-not $runOutputs.ContainsKey($key)) "duplicate RUN output across explicit journals: $output"
        $runOutputs[$key] = [pscustomobject]@{ Journal=$journal; Run=$run }
    }
    foreach ($result in $results) {
        $metaName = Get-Field $result.Fields 'meta' 'RESULT'
        $rawName = Get-Field $result.Fields 'path' 'RESULT'
        $metaKey = $metaName.ToUpperInvariant()
        Assert-True (-not $resultByMeta.ContainsKey($metaKey)) "duplicate RESULT/META credit: $metaName"
        $matches = @($runs | Where-Object {
            $_.Fields.ContainsKey('output') -and $_.Fields['output'] -ceq $rawName -and
            $_.Fields.ContainsKey('case') -and $_.Fields.ContainsKey('start') -and
            $_.Fields['case'] -ceq $result.Fields['case'] -and $_.Fields['start'] -ceq $result.Fields['start']
        })
        Assert-Equal $matches.Count 1 "RESULT-to-RUN match $($journal.Name) raw=$rawName"
        $resultByMeta[$metaKey] = [pscustomobject]@{ Journal=$journal; Run=$matches[0]; Result=$result }
    }
    foreach ($run in $runs) {
        $hasResult = @($results | Where-Object { $_.Fields.ContainsKey('path') -and $_.Fields['path'] -ceq $run.Fields['output'] }).Count -eq 1
        if (-not $hasResult) {
            $terminalAbort = Get-TerminalAbortForRun $journal $run
            $classification = if($null -ne $terminalAbort){'ABORTED_UNCREDITED'}else{'ACTIVE_UNCREDITED'}
            if ($null -ne $terminalAbort) { [void](Assert-AbortedInvocationTiming $run $terminalAbort "$($journal.Name) case=$($run.Fields['case']) start=$($run.Fields['start'])") }
            $openRuns.Add([pscustomobject]@{ Journal=$journal; Run=$run; Classification=$classification; Abort=$terminalAbort })
        }
    }
}

Assert-Equal $resultByMeta.Count $metaBefore.Count 'credited RESULT/exact-family META one-to-one count'
foreach ($key in $metaBefore.Keys) { Assert-True ($resultByMeta.ContainsKey($key)) "unmatched credited META has no RESULT in explicit journals: $($metaBefore[$key].File.Name)" }
foreach ($key in $resultByMeta.Keys) { Assert-True ($metaBefore.ContainsKey($key)) "RESULT references missing/non-family META: $key" }

$coverageByLane = @{}
foreach ($label in @('PRIMARY','D6_OFF','SECOND_TT')) {
    $sets = @()
    for ($case=0;$case -lt 11;$case++){ $sets += ,([Collections.Generic.HashSet[int]]::new()) }
    $coverageByLane[$label] = $sets
}
$creditRecords = New-Object Collections.Generic.List[object]
$creditKeys = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$stableFiles = New-Object Collections.Generic.List[object]
foreach ($key in @($resultByMeta.Keys | Sort-Object)) {
    $binding = $resultByMeta[$key]
    $journal = $binding.Journal
    $lane = $journal.Lane
    $metaInfo = $metaBefore[$key]
    Assert-True ($metaInfo.Lane.Label -ceq $lane.Label) "META/journal lane collision: $($metaInfo.File.Name)"
    $case = Convert-ToInt (Get-Field $binding.Result.Fields 'case' 'RESULT') 'RESULT case'
    $start = Convert-ToInt (Get-Field $binding.Result.Fields 'start' 'RESULT') 'RESULT start'
    Assert-Equal $case $metaInfo.Case 'META filename/result case'
    Assert-Equal $start $metaInfo.Start 'META filename/result start'
    $creditKey = "$($lane.Label)/$case/$start"
    Assert-True ($creditKeys.Add($creditKey)) "duplicate credited lane/case/start: $creditKey"
    $expectedRawName = [string]::Format($invariant, $lane.RawTemplate, $case, $start, $metaInfo.Attempt)
    Assert-Equal (Get-Field $binding.Result.Fields 'path' 'RESULT') $expectedRawName 'RESULT frozen raw filename'
    $chain = Assert-ChainEvents -Journal $journal -Run $binding.Run -Result $binding.Result
    Assert-ChainFields -Journal $journal -Chain $chain -ExpectedRawName $expectedRawName
    $rawPath = Resolve-RootLeaf (Join-Path $repoRoot $expectedRawName) 'credited raw'
    $raw = Read-CreditedRaw -Path $rawPath -Lane $lane -FilenameCase $case -FilenameStart $start -Census $census
    $stem = $expectedRawName.Substring(0, $expectedRawName.Length - '_RAW.log'.Length)
    $metaPath = Resolve-RootLeaf (Join-Path $repoRoot ($stem + '_META_RAW.log')) 'credited META'
    $exitPath = Resolve-RootLeaf (Join-Path $repoRoot ($stem + '_CARGO_EXIT_RAW.log')) 'credited Cargo exit'
    $stdoutPath = Resolve-RootLeaf (Join-Path $repoRoot ($stem + '_CARGO_STDOUT_RAW.log')) 'credited Cargo stdout'
    $stderrPath = Resolve-RootLeaf (Join-Path $repoRoot ($stem + '_CARGO_STDERR_RAW.log')) 'credited Cargo stderr'
    $metaLine = "DOM_B2_D4_SHARD_META version=1 raw=$expectedRawName raw_sha256=$($raw.Sha256) raw_bytes=$($raw.Bytes) source_snapshot_sha256=$requiredSourceSnapshotSha256 runner_sha256=$($lane.RunnerSha256) census_sha256=$requiredCensusSha256 code_id=$requiredCodeId deadline_ms=$requiredDeadlineMs tt_bytes=$($lane.TtBytes) d6=$($lane.D6) target=$requiredTarget release=true test_threads=1"
    $meta = Assert-ExactSingleLfLine -Path $metaPath -Expected $metaLine -Context "META $($metaInfo.File.Name)"
    $exitLine = "DOM_B2_D4_CARGO_EXIT code=0 runner_sha256=$($lane.RunnerSha256)"
    $exit = Assert-ExactSingleLfLine -Path $exitPath -Expected $exitLine -Context "Cargo exit $([IO.Path]::GetFileName($exitPath))"
    $resultFields = $binding.Result.Fields
    Assert-Equal (Convert-ToInt (Get-Field $resultFields 'complete' 'RESULT') 'RESULT complete') $raw.Complete 'RESULT/raw complete'
    Assert-Equal (Convert-ToInt (Get-Field $resultFields 'next_start' 'RESULT') 'RESULT next_start') $raw.NextStart 'RESULT/raw next_start'
    Assert-Equal (Get-Field $resultFields 'result' 'RESULT') $raw.Result 'RESULT/raw result'
    Assert-Equal (Convert-ToInt64 (Get-Field $resultFields 'bytes' 'RESULT') 'RESULT bytes') ([int64]$raw.Bytes) 'RESULT/raw bytes'
    Assert-Equal (Get-Field $resultFields 'sha256' 'RESULT') $raw.Sha256 'RESULT/raw SHA256'
    Assert-Equal (Convert-ToInt64 (Get-Field $resultFields 'cargo_exit_bytes' 'RESULT') 'RESULT exit bytes') ([int64]$exit.Bytes) 'RESULT/exit bytes'
    Assert-Equal (Get-Field $resultFields 'cargo_exit_sha256' 'RESULT') $exit.Sha256 'RESULT/exit SHA256'
    Assert-Equal (Convert-ToInt64 (Get-Field $resultFields 'meta_bytes' 'RESULT') 'RESULT META bytes') ([int64]$meta.Bytes) 'RESULT/META bytes'
    Assert-Equal (Get-Field $resultFields 'meta_sha256' 'RESULT') $meta.Sha256 'RESULT/META SHA256'
    foreach ($rootIndex in $raw.RootIndices) {
        Assert-True ($coverageByLane[$lane.Label][$case].Add($rootIndex)) "overlapping credited root lane=$($lane.Label) case=$case root=$rootIndex"
    }
    $creditRecords.Add([pscustomobject]@{
        Lane=$lane; Journal=$journal; Raw=$raw; Meta=$meta; Exit=$exit;
        MetaPath=$metaPath; ExitPath=$exitPath; StdoutPath=$stdoutPath; StderrPath=$stderrPath;
        Chain=$chain; Result=$binding.Result
    })
    foreach ($stable in @(
        [pscustomobject]@{Path=$rawPath;Bytes=$raw.Bytes;Sha256=$raw.Sha256},
        [pscustomobject]@{Path=$metaPath;Bytes=$meta.Bytes;Sha256=$meta.Sha256},
        [pscustomobject]@{Path=$exitPath;Bytes=$exit.Bytes;Sha256=$exit.Sha256}
    )) { $stableFiles.Add($stable) }
}

# A footerless active raw is never opened above: only a RESULT+META reaches Read-CreditedRaw.
foreach ($open in $openRuns) {
    $rawName = Get-Field $open.Run.Fields 'output' 'open RUN'
    $metaName = $rawName.Substring(0, $rawName.Length - '_RAW.log'.Length) + '_META_RAW.log'
    Assert-True (-not $metaBefore.ContainsKey($metaName.ToUpperInvariant())) "unmatched RUN already has a credited-family META: $metaName"
}

$allAborts = @($journals | ForEach-Object { $_.Events } | Where-Object { $_.Event -ceq 'ABORT' })
if ($Final) {
    Assert-Equal $openRuns.Count 0 'FINAL unmatched RUN count'
    Assert-Equal $allAborts.Count 0 'FINAL ABORT count'
}

$laneSummaries = @{}
foreach ($label in @('PRIMARY','D6_OFF','SECOND_TT')) {
    $lane = $laneSpecs[$label]
    $laneJournals = @($journals | Where-Object { $_.Lane.Label -ceq $label })
    $summary = Assert-DoneAndCoverage -Lane $lane -Journals $laneJournals -Coverage $coverageByLane[$label] -RequireFinal $Final.IsPresent
    $laneSummaries[$label] = $summary
}
$chronology = Assert-GlobalChronology -Journals @($journals | ForEach-Object { $_ }) -Credits @($creditRecords | ForEach-Object { $_ }) -UnmatchedRuns @($openRuns | ForEach-Object { $_ })

# Fail closed on any mutation/race while the read-only snapshot was being checked.
Assert-MetaSnapshotsEqual -Before $metaBefore -After (Get-MetaSnapshot)
Assert-JournalSnapshotsEqual -Before $frozenJournalBefore -After (Get-FrozenJournalSnapshot)
foreach ($journal in $journals) {
    $after = Read-SharedBytes -Path $journal.Path
    Assert-Equal $after.Length $journal.Bytes "$($journal.Name) bytes changed during audit"
    Assert-Equal (Get-Sha256FromBytes $after) $journal.Sha256 "$($journal.Name) hash changed during audit"
}
foreach ($stable in $stableFiles) {
    $after = Read-SharedBytes -Path $stable.Path
    Assert-Equal $after.Length $stable.Bytes "$($stable.Path) bytes changed during audit"
    Assert-Equal (Get-Sha256FromBytes $after) $stable.Sha256 "$($stable.Path) hash changed during audit"
}
[void](Assert-BoundFile 'scripts/dom_b2_d4_run_queue.ps1' $requiredPrimaryRunnerSha256 49762)
[void](Assert-BoundFile 'scripts/dom_b2_d4_run_replica_queue.ps1' $requiredReplicaRunnerSha256 56516)
[void](Assert-BoundFile 'DOM_B2_D4_PRIMARY_V3_CODE_SNAPSHOT_RAW.log' $requiredSourceSnapshotSha256 3399)
[void](Assert-BoundFile 'DOM_B2_D4_CENSUS_RAW.log' $requiredCensusSha256 292756)
[void](Assert-BoundFile $requiredBinaryRelative $requiredBinarySha256 3290112)
[void](Assert-LiveSourceSnapshot $sourceSnapshotFull)

$mode = if ($Final) { 'FINAL' } else { 'PARTIAL' }
$allStops = @($journals | ForEach-Object { $_.Events } | Where-Object { $_.Event -ceq 'STOP' })
$allWaits = @($journals | ForEach-Object { $_.Events } | Where-Object {
    ($_.Event -ceq 'GATE' -or $_.Event -ceq 'PRELAUNCH_GATE') -and $_.Fields.ContainsKey('result') -and $_.Fields['result'] -ceq 'WAIT_NO_LAUNCH'
})
"DOM_B2_D4_CHAIN_AUDIT version=1 mode=$mode explicit_journals=$($journals.Count) frozen_journals=$($frozenJournalBefore.Count) discovered_meta=$($metaBefore.Count) credited_invocations=$($creditRecords.Count) open_uncredited_runs=$($openRuns.Count) filesystem_writes=0 cargo_invocations=0"
"BINDING primary_runner_sha256=$requiredPrimaryRunnerSha256 replica_runner_sha256=$requiredReplicaRunnerSha256 child_wrapper_sha256=$requiredChildWrapperSha256 source_snapshot_sha256=$requiredSourceSnapshotSha256 source_entries=20 strict_verifier_sha256=$requiredVerifierSha256 census_sha256=$requiredCensusSha256 binary_sha256=$requiredBinarySha256 code_id=$requiredCodeId deadline_ms=$requiredDeadlineMs cargo_wall_timeout_ms=$requiredCargoWallTimeoutMs target=$requiredTarget release=true test_threads=1"
"JOURNAL_EVENT_SUMMARY waits_no_launch=$($allWaits.Count) clean_stops=$($allStops.Count) aborts=$($allAborts.Count) grammar=PASS state_machine=PASS"
foreach ($journal in @($journals | Sort-Object { $_.Lane.Label }, Name)) {
    "JOURNAL lane=$($journal.Lane.Label) path=$($journal.Name) bytes=$($journal.Bytes) sha256=$($journal.Sha256) result=PASS"
}
foreach ($credit in @($creditRecords | Sort-Object { $_.Lane.Label }, { $_.Raw.Case }, { $_.Raw.Start })) {
    "CREDIT lane=$($credit.Lane.Label) case=$($credit.Raw.Case) start=$($credit.Raw.Start) complete=$($credit.Raw.Complete) next_start=$($credit.Raw.NextStart) shard_result=$($credit.Raw.Result) raw=$($credit.Raw.Name) raw_bytes=$($credit.Raw.Bytes) raw_sha256=$($credit.Raw.Sha256) meta_bytes=$($credit.Meta.Bytes) meta_sha256=$($credit.Meta.Sha256) cargo_exit_bytes=$($credit.Exit.Bytes) cargo_exit_sha256=$($credit.Exit.Sha256) elapsed_s=$($credit.Chain.Elapsed) wall_s=$($credit.Chain.WallSeconds.ToString('F3',$invariant)) chain=PASS"
}
foreach ($open in @($openRuns | Sort-Object { $_.Journal.Lane.Label }, { $_.Run.Index })) {
    $rawName = Get-Field $open.Run.Fields 'output' 'open RUN'
    $rawExists = Test-Path -LiteralPath (Join-Path $repoRoot $rawName) -PathType Leaf
    "UNMATCHED_RUN classification=$($open.Classification) lane=$($open.Journal.Lane.Label) journal=$($open.Journal.Name) case=$(Get-Field $open.Run.Fields 'case' 'open RUN') start=$(Get-Field $open.Run.Fields 'start' 'open RUN') raw=$rawName raw_exists=$($rawExists.ToString().ToLowerInvariant()) raw_inspected=false credit=0"
}
foreach ($label in @('PRIMARY','D6_OFF','SECOND_TT')) {
    $summary = $laneSummaries[$label]
    "LANE_SUMMARY lane=$label credited_roots=$($summary.Roots) complete_cases=$($summary.CompleteCases) queue_done=$($summary.Done.ToString().ToLowerInvariant()) expected_roots=3648 result=$(if($summary.Done){'COMPLETE'}else{'PARTIAL'})"
}
"DOM_B2_D4_CHAIN_AUDIT_DONE mode=$mode credited_invocations=$($creditRecords.Count) credited_meta=$($metaBefore.Count) open_uncredited_runs=$($openRuns.Count) invocation_intervals=$($chronology.Intervals) chronology=PASS final_fence=$(if($Final){'PASS'}else{'NOT_REQUESTED'}) race_check=PASS result=PASS"
