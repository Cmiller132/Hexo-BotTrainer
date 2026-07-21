$ErrorActionPreference = 'Stop'

$artifactRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$inputNames = @(
    'f19-rep0.jsonl', 'f19-rep1.jsonl', 'f19-rep2.jsonl',
    's2-rep0.jsonl', 's2-rep1.jsonl', 's2-rep2.jsonl',
    'human160-rep0.jsonl', 'human160-rep1.jsonl', 'human160-rep2.jsonl'
)

function Format-Cell([long]$nanoseconds, [long]$wallNanoseconds) {
    $milliseconds = $nanoseconds / 1000000.0
    $share = if ($wallNanoseconds -eq 0) { 0.0 } else { 100.0 * $nanoseconds / $wallNanoseconds }
    return ('{0:F3} ({1:F6}%)' -f $milliseconds, $share)
}

function Escape-Markdown([object]$value) {
    return ([string]$value).Replace('|', '\|')
}

$header = @(
    '# Residue map per-job rows',
    '',
    'Generated from the nine frozen three-repetition JSONL files. Every timing cell is `milliseconds (% job_wall)`.',
    '',
    '| profile | row | rep | cap rung | horizon rung | resume | status/verified | nodes/exp | TT hit/entries/peak | cert n/e | total ms | forced D | unforced FHW | unforced other/unclass | A gen | A winner | A miss | A unresolved | TT probe/store | census | search | cert build/verify | horizon/resume overhead | other | cross-check |',
    '|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'
)

$rows = [System.Collections.Generic.List[string]]::new()
foreach ($name in $inputNames) {
    $path = Join-Path $artifactRoot $name
    foreach ($line in Get-Content -LiteralPath $path) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $job = $line | ConvertFrom-Json
        $wall = [long]$job.job_wall_ns
        $cat = $job.categories
        $verify = if ($null -eq $job.strict_verify_result) { '-' } else { [string]$job.strict_verify_result }
        $ttPeakMiB = [double]$job.peak_tt_bytes / 1MB
        $unforcedOther = (Format-Cell ([long]$cat.D_UNFORCED_NONFHW_GEN) $wall) + '; ' +
            (Format-Cell ([long]$cat.D_UNFORCED_UNCLASSIFIED_GEN) $wall)
        $tt = (Format-Cell ([long]$cat.TT_PROBE) $wall) + '; ' + (Format-Cell ([long]$cat.TT_STORE) $wall)
        $cert = (Format-Cell ([long]$cat.CERT_BUILD) $wall) + '; ' + (Format-Cell ([long]$cat.CERT_VERIFY) $wall)
        $ladder = (Format-Cell ([long]$cat.HORIZON_LADDER_OVERHEAD) $wall) + '; ' +
            (Format-Cell ([long]$cat.CAP_RESUME_OVERHEAD) $wall)
        $values = @(
            $job.profile,
            (Escape-Markdown $job.row),
            $job.rep,
            $job.cap_rung,
            (Escape-Markdown $job.horizon_rung),
            $job.resume,
            ((Escape-Markdown $job.status) + '/' + $verify),
            ($job.nodes.ToString() + '/' + $job.expansions.ToString()),
            ('{0}/{1}/{2:F3} MiB' -f $job.tt_hits, $job.tt_entries, $ttPeakMiB),
            ($job.cert_nodes.ToString() + '/' + $job.cert_edges.ToString()),
            ('{0:F3}' -f ($wall / 1000000.0)),
            (Format-Cell ([long]$cat.D_FORCED_GEN) $wall),
            (Format-Cell ([long]$cat.D_UNFORCED_FHW_ELIGIBLE_GEN) $wall),
            $unforcedOther,
            (Format-Cell ([long]$cat.A_OR_GEN) $wall),
            (Format-Cell ([long]$cat.A_OR_WINNER_PATH) $wall),
            (Format-Cell ([long]$cat.A_OR_ORDERING_MISS) $wall),
            (Format-Cell ([long]$cat.A_OR_UNRESOLVED) $wall),
            $tt,
            (Format-Cell ([long]$cat.CENSUS_GATE) $wall),
            (Format-Cell ([long]$cat.SEARCH_BOOKKEEPING) $wall),
            $cert,
            $ladder,
            (Format-Cell ([long]$cat.OTHER_MEASURED) $wall),
            ('{0}/{1} ns' -f $job.crosscheck_residual_ns, $job.crosscheck_abs_ns)
        )
        $rows.Add('| ' + ($values -join ' | ') + ' |')
    }
}

$output = Join-Path $artifactRoot 'per-job.md'
@($header + $rows) | Set-Content -LiteralPath $output -Encoding utf8
Write-Output ("generated {0} rows at {1}" -f $rows.Count, $output)
