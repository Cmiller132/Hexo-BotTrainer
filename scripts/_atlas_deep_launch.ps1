$ErrorActionPreference = 'Stop'
$root = 'E:\Hexo-BotTrainer-hexgt\.claude\worktrees\opening-atlas'
Set-Location $root
$exe = Join-Path $root '.target-atlas\x86_64-pc-windows-msvc\release\deps\hexfield_eq-de26e3778420c4c2.exe'
$outdir = Join-Path $root 'deep_shards'
New-Item -ItemType Directory -Force -Path $outdir | Out-Null

# Clear any TSS_* env that could perturb solver behavior.
Get-ChildItem Env: | Where-Object Name -Like 'TSS_*' | ForEach-Object { Remove-Item "Env:$($_.Name)" }

# Shared deep-profile config.
$env:OPENING_ATLAS_MODE        = 'corpus_first_n'
$env:OPENING_ATLAS_CORPUS      = 'E:\Hexo-BotTrainer-hexgt\data\hexo-bootstrap-corpus\hexo_human_corpus.jsonl'
$env:OPENING_ATLAS_FIRST_N     = '7'
$env:OPENING_ATLAS_NODE_LADDER = '2000000'
$env:OPENING_ATLAS_TT_BYTES    = '671088640'   # 640 MiB
$env:OPENING_ATLAS_UNBOUNDED   = '1'
$env:OPENING_ATLAS_WALL_SECONDS= '14400'        # 4 hours
$env:SHARD_COUNT               = '16'

$args = @('--exact','tss_opening_atlas::opening_atlas_pass1','--ignored','--nocapture')
for ($i = 0; $i -lt 16; $i++) {
    $env:SHARD_INDEX = "$i"
    $out = Join-Path $outdir ("shard_{0:D2}.txt" -f $i)
    $err = Join-Path $outdir ("shard_{0:D2}.err" -f $i)
    Start-Process -FilePath $exe -ArgumentList $args -NoNewWindow `
        -RedirectStandardOutput $out -RedirectStandardError $err
    Start-Sleep -Milliseconds 150
}
Write-Output "launched 16 workers -> $outdir"
