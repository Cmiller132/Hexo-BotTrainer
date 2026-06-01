param(
    [string]$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$RunName = "dense_cnn_model1",
    [int]$IntervalSeconds = 6,
    [double]$MinFreeRamGb = 8.0,
    [double]$MinFreeVirtualGb = 12.0,
    [double]$MinGpuFreeGb = 2.5,
    [double]$MaxTrainerPrivateGb = 18.0
)

$ErrorActionPreference = "Continue"
$runDir = Join-Path $RepositoryRoot ("runs\" + $RunName)
$diagnosticsDir = Join-Path $runDir "diagnostics"
New-Item -ItemType Directory -Force -Path $diagnosticsDir | Out-Null
$logPath = Join-Path $diagnosticsDir "resource_watchdog.jsonl"
$stopPath = Join-Path $diagnosticsDir "resource_watchdog.stop.json"

function Convert-ToGb($bytes) {
    if ($null -eq $bytes) { return 0.0 }
    return [math]::Round(([double]$bytes / 1GB), 3)
}

function Get-GpuMetrics {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        return @{ available = $false; error = "nvidia-smi not found" }
    }
    try {
        $line = & nvidia-smi --query-gpu=memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>$null | Select-Object -First 1
        if (-not $line) { return @{ available = $false; error = "nvidia-smi returned no data" } }
        $parts = $line -split "," | ForEach-Object { $_.Trim() }
        return @{
            available = $true
            used_gb = [math]::Round(([double]$parts[0] / 1024.0), 3)
            total_gb = [math]::Round(([double]$parts[1] / 1024.0), 3)
            free_gb = [math]::Round(([double]$parts[2] / 1024.0), 3)
            utilization_percent = [int]$parts[3]
            temperature_c = [int]$parts[4]
            error = $null
        }
    } catch {
        return @{ available = $false; error = $_.Exception.Message }
    }
}

function Get-TrainerProcess {
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -match "python" -and
            $_.CommandLine -match "hexo_train\.cli\.train_model|dense_cnn_model1\.toml"
        } |
        Select-Object -First 1
}

while ($true) {
    $os = Get-CimInstance Win32_OperatingSystem
    $gpu = Get-GpuMetrics
    $trainerProc = Get-TrainerProcess
    $trainer = $null
    if ($trainerProc) {
        try {
            $process = Get-Process -Id $trainerProc.ProcessId -ErrorAction Stop
            $trainer = @{
                pid = [int]$process.Id
                private_gb = Convert-ToGb $process.PrivateMemorySize64
                working_set_gb = Convert-ToGb $process.WorkingSet64
                virtual_gb = Convert-ToGb $process.VirtualMemorySize64
                cpu_seconds = [math]::Round([double]$process.CPU, 3)
                command_line = $trainerProc.CommandLine
            }
        } catch {
            $trainer = @{ pid = [int]$trainerProc.ProcessId; error = $_.Exception.Message }
        }
    }

    $sample = @{
        timestamp = (Get-Date).ToUniversalTime().ToString("o")
        status = "ok"
        critical = @()
        memory = @{
            free_ram_gb = [math]::Round(([double]$os.FreePhysicalMemory / 1MB), 3)
            total_ram_gb = [math]::Round(([double]$os.TotalVisibleMemorySize / 1MB), 3)
            free_virtual_gb = [math]::Round(([double]$os.FreeVirtualMemory / 1MB), 3)
            total_virtual_gb = [math]::Round(([double]$os.TotalVirtualMemorySize / 1MB), 3)
        }
        gpu = $gpu
        trainer = $trainer
    }

    if ($sample.memory.free_ram_gb -lt $MinFreeRamGb) {
        $sample.critical += "free_ram_gb < $MinFreeRamGb"
    }
    if ($sample.memory.free_virtual_gb -lt $MinFreeVirtualGb) {
        $sample.critical += "free_virtual_gb < $MinFreeVirtualGb"
    }
    if ($gpu.available -and [double]$gpu.free_gb -lt $MinGpuFreeGb) {
        $sample.critical += "gpu_free_gb < $MinGpuFreeGb"
    }
    if ($trainer -and [double]$trainer.private_gb -gt $MaxTrainerPrivateGb) {
        $sample.critical += "trainer_private_gb > $MaxTrainerPrivateGb"
    }

    if ($sample.critical.Count -gt 0) {
        $sample.status = "stopping_trainer"
        if ($trainer -and $trainer.pid) {
            Stop-Process -Id $trainer.pid -Force -ErrorAction SilentlyContinue
        }
        $sample | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 -Path $stopPath
    }

    $sample | ConvertTo-Json -Compress -Depth 8 | Add-Content -Encoding UTF8 -Path $logPath
    Start-Sleep -Seconds $IntervalSeconds
}
