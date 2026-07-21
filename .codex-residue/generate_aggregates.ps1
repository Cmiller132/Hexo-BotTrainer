$ErrorActionPreference = 'Stop'

$artifactRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$categories = @(
    'D_FORCED_GEN', 'D_UNFORCED_FHW_ELIGIBLE_GEN',
    'D_UNFORCED_NONFHW_GEN', 'D_UNFORCED_UNCLASSIFIED_GEN',
    'A_OR_GEN', 'A_OR_WINNER_PATH', 'A_OR_ORDERING_MISS',
    'A_OR_UNRESOLVED', 'TT_PROBE', 'TT_STORE', 'CENSUS_GATE',
    'SEARCH_BOOKKEEPING', 'CERT_BUILD', 'CERT_VERIFY',
    'HORIZON_LADDER_OVERHEAD', 'CAP_RESUME_OVERHEAD', 'OTHER_MEASURED'
)

$disposition = @{
    D_FORCED_GEN = @('implemented/local', 'NOT_MEASURED', 'ceiling only')
    D_UNFORCED_FHW_ELIGIBLE_GEN = @('unobserved/no selector', 'NOT_MEASURED', 'no eligible events')
    D_UNFORCED_NONFHW_GEN = @('unobserved/no selector', 'NOT_MEASURED', 'no classified events')
    D_UNFORCED_UNCLASSIFIED_GEN = @('OPEN/audit debt', 'NOT_MEASURED', 'ceiling only')
    A_OR_GEN = @('implemented/local', 'NOT_MEASURED', 'ceiling only')
    A_OR_WINNER_PATH = @('productive ceiling', 'NOT_MEASURED', 'ceiling only')
    A_OR_ORDERING_MISS = @('OPEN', 'measured wall', 'direct exclusive wall')
    A_OR_UNRESOLVED = @('OPEN', 'NOT_MEASURED', 'ceiling only')
    TT_PROBE = @('OPEN/local', 'NOT_MEASURED', 'ceiling only')
    TT_STORE = @('OPEN/local', 'NOT_MEASURED', 'ceiling only')
    CENSUS_GATE = @('implemented/gated', 'NOT_MEASURED', 'ceiling only')
    SEARCH_BOOKKEEPING = @('OPEN', 'NOT_MEASURED', 'ceiling only')
    CERT_BUILD = @('verified/mandatory', 'NOT_MEASURED', 'ceiling only')
    CERT_VERIFY = @('verified/mandatory', 'NOT_MEASURED', 'ceiling only')
    HORIZON_LADDER_OVERHEAD = @('protocol', 'NOT_MEASURED', 'ceiling only')
    CAP_RESUME_OVERHEAD = @('protocol', 'NOT_MEASURED', 'ceiling only')
    OTHER_MEASURED = @('audit debt', 'measured wall', 'direct timer')
}

function Median([object[]]$values) {
    $sorted = @($values | Sort-Object)
    return $sorted[[Math]::Floor(($sorted.Count - 1) / 2)]
}

function Percentile([object[]]$values, [int]$percent) {
    $sorted = @($values | Sort-Object)
    $index = [Math]::Min($sorted.Count - 1, [Math]::Ceiling($sorted.Count * $percent / 100.0) - 1)
    return $sorted[$index]
}

function Load-Rows([string[]]$names) {
    return @($names | ForEach-Object {
        Get-Content -LiteralPath (Join-Path $artifactRoot $_) | ForEach-Object { $_ | ConvertFrom-Json }
    })
}

function Write-ProfileTable([System.Collections.Generic.List[string]]$output, [string]$title, [object[]]$rows) {
    $repetitionGroups = @($rows | Group-Object rep)
    $medianWall = [long](Median @($repetitionGroups | ForEach-Object {
        ($_.Group | Measure-Object job_wall_ns -Sum).Sum
    }))
    $logicalGroups = @($rows | Group-Object profile,row,cap_rung,horizon_rung,resume)
    $output.Add("## $title ($($logicalGroups.Count) logical jobs; median $('{0:F3}' -f ($medianWall / 1e6)) ms)")
    $output.Add('')
    $output.Add('| category | median sum ms | wall % | p90 row % | p95 row % | max row %/id | disposition | measured value estimate | estimate method | eliminability upper bound |')
    $output.Add('|---|---:|---:|---:|---:|---|---|---|---|---:|')
    foreach ($category in $categories) {
        $medianSum = [long](Median @($repetitionGroups | ForEach-Object {
            ($_.Group | ForEach-Object { [long]$_.categories.$category } | Measure-Object -Sum).Sum
        }))
        $shares = @($logicalGroups | ForEach-Object {
            $group = $_.Group
            $share = [double](Median @($group | ForEach-Object {
                if ($_.job_wall_ns -eq 0) { 0.0 } else { 100.0 * [double]$_.categories.$category / [double]$_.job_wall_ns }
            }))
            [pscustomobject]@{
                share = $share
                id = ('{0}/{1}/cap{2}/{3}' -f $group[0].profile, $group[0].row, $group[0].cap_rung, $group[0].horizon_rung)
            }
        })
        $maximum = $shares | Sort-Object share -Descending | Select-Object -First 1
        $wallShare = 100.0 * $medianSum / $medianWall
        $meta = $disposition[$category]
        $output.Add(('| {0} | {1:F3} | {2:F6} | {3:F6} | {4:F6} | {5:F6}/{6} | {7} | {8} | {9} | {2:F6} |' -f
            $category, ($medianSum / 1e6), $wallShare,
            (Percentile @($shares.share) 90), (Percentile @($shares.share) 95),
            $maximum.share, $maximum.id, $meta[0], $meta[1], $meta[2]))
    }
    $output.Add('')
}

$f19 = Load-Rows @('f19-rep0.jsonl', 'f19-rep1.jsonl', 'f19-rep2.jsonl')
$s2 = Load-Rows @('s2-rep0.jsonl', 's2-rep1.jsonl', 's2-rep2.jsonl')
$human = Load-Rows @('human160-rep0.jsonl', 'human160-rep1.jsonl', 'human160-rep2.jsonl')
$f19Final = @($f19 | Group-Object rep,row | ForEach-Object {
    $_.Group | Sort-Object cap_rung -Descending | Select-Object -First 1
})

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add('# Residue map aggregate tables')
$lines.Add('')
$lines.Add('Row p90/p95/max values are computed after taking each logical job''s median share across its three repetitions.')
$lines.Add('')
Write-ProfileTable $lines 'F19 protocol wall' $f19
Write-ProfileTable $lines 'F19 final attempts' $f19Final
Write-ProfileTable $lines 'S2' $s2
Write-ProfileTable $lines 'Human 160' $human

$outputPath = Join-Path $artifactRoot 'aggregate-tables.md'
$lines | Set-Content -LiteralPath $outputPath -Encoding utf8
Write-Output ("generated four aggregate tables at {0}" -f $outputPath)
