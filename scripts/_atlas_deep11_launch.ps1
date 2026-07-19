$ErrorActionPreference = 'Stop'
$root = 'E:\Hexo-BotTrainer-hexgt\.claude\worktrees\opening-atlas'
Set-Location $root
$exe = Join-Path $root '.target-atlas\x86_64-pc-windows-msvc\release\deps\hexfield_eq-de26e3778420c4c2.exe'
$outdir = Join-Path $root 'corpus11_shards'
New-Item -ItemType Directory -Force -Path $outdir | Out-Null

# DEEP + NARROW profile (proven win-harvesting config), extended to 11 plies:
#  width=vcf_pair_complete, horizon UNBOUNDED, cap 8,000,000, win_line PV emitted.
$env:OPENING_ATLAS_MODE         = 'corpus_first_n'
$env:OPENING_ATLAS_CORPUS       = 'E:\Hexo-BotTrainer-hexgt\data\hexo-bootstrap-corpus\hexo_human_corpus.jsonl'
$env:OPENING_ATLAS_FIRST_N      = '11'
$env:OPENING_ATLAS_WIDTH        = 'vcf_pair_complete'
$env:OPENING_ATLAS_NODE_LADDER  = '8000000'
$env:OPENING_ATLAS_TT_BYTES     = '1073741824'   # 1 GiB cap (vcf fills only a few MB in practice)
$env:OPENING_ATLAS_UNBOUNDED    = '1'
$env:OPENING_ATLAS_WALL_SECONDS = '9000'
$env:SHARD_COUNT                = '16'

$procArgs = @('--exact','tss_opening_atlas::opening_atlas_pass1','--ignored','--nocapture')
$procs = @()
for ($i = 0; $i -lt 16; $i++) {
    $env:SHARD_INDEX = "$i"
    $out = Join-Path $outdir ("shard_{0:D2}.txt" -f $i)
    $err = Join-Path $outdir ("shard_{0:D2}.err" -f $i)
    $p = Start-Process -FilePath $exe -ArgumentList $procArgs -NoNewWindow -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err
    $procs += $p
    Start-Sleep -Milliseconds 150
}
$start = Get-Date
Write-Output "launched 16 deep-11ply workers at $start -> $outdir"
$null = Wait-Process -Id ($procs.Id) -Timeout 9600 -ErrorAction SilentlyContinue
$elapsed = (New-TimeSpan -Start $start -End (Get-Date)).TotalSeconds
Write-Output "all deep-11ply workers exited; elapsed_s=$([math]::Round($elapsed,0))"
