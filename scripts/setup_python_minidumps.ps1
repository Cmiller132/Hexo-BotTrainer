# Configure Windows Error Reporting LocalDumps to capture a full minidump for
# python.exe on the next hard fault (access violation / stack overflow / native
# abort that produces no Python traceback). Must be run elevated (writes HKLM).
#
# Run elevated, e.g.:
#   Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','E:\Hexo-BotTrainer\scripts\setup_python_minidumps.ps1'

$ErrorActionPreference = 'Stop'

$base   = 'HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps'
$key    = "$base\python.exe"
$folder = 'E:\Hexo-BotTrainer\runs\dense_cnn_model1_scratch_64\diagnostics\crashdumps'

if (-not (Test-Path $folder)) { New-Item -ItemType Directory -Force -Path $folder | Out-Null }
if (-not (Test-Path $base))   { New-Item -Path $base -Force | Out-Null }
if (-not (Test-Path $key))    { New-Item -Path $key  -Force | Out-Null }

# DumpType=2 -> full dump (faulting-module + heap). DumpCount caps stored dumps.
New-ItemProperty -Path $key -Name 'DumpType'   -PropertyType DWord        -Value 2  -Force | Out-Null
New-ItemProperty -Path $key -Name 'DumpCount'  -PropertyType DWord        -Value 10 -Force | Out-Null
New-ItemProperty -Path $key -Name 'DumpFolder' -PropertyType ExpandString -Value $folder -Force | Out-Null

Write-Host '=== LocalDumps\python.exe configured ==='
Get-ItemProperty -Path $key | Select-Object DumpType, DumpCount, DumpFolder | Format-List
Write-Host "DumpFolder exists: $(Test-Path $folder)"
Write-Host 'NOTE: takes effect for processes that crash AFTER this point. The'
Write-Host 'already-running trainer (PID 10672) is covered -- WER reads this key'
Write-Host 'at fault time, not at process start.'
