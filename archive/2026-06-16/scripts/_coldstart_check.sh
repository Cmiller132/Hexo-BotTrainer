#!/usr/bin/env bash
RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x6
echo "=== WSL kernel/uptime ==="; uptime
echo "=== CUDA in WSL (nvidia-smi) ==="; nvidia-smi --query-gpu=name,driver_version,memory.used,utilization.gpu --format=csv,noheader 2>&1 | head -3
echo "=== venv torch CUDA ==="; /root/.venvs/hexo-bottrainer-wsl/bin/python -c "import torch;print('torch',torch.__version__,'cuda_avail',torch.cuda.is_available(),(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-CUDA'))" 2>&1 | tail -2
echo "=== trainer/supervisor procs (expect NONE) ==="; ps -eo pid,etime,cmd | grep -E 'train_model|supervise_target_96x6_wsl' | grep -v grep || echo "NONE running"
echo "=== latest epoch checkpoints ==="; ls -t --time-style=+%FT%TZ "$RD"/checkpoints/epoch_*.pt 2>/dev/null | head -3 || echo "no epoch checkpoints"
echo "=== bootstrap present? ==="; ls -l --time-style=+%FT%TZ "$RD"/checkpoints/bootstrap_sealbot_prefit.pt 2>/dev/null || echo "no bootstrap"
echo "=== halt/completed flags (must be clear to resume) ==="; ls -l "$RD"/diagnostics/*.flag 2>/dev/null || echo "no flags"
echo "=== stale supervisor lock? ==="; if [ -f "$RD/diagnostics/supervisor_wsl.lock" ]; then p=$(cat "$RD/diagnostics/supervisor_wsl.lock"); echo "lock pid=$p alive=$(kill -0 "$p" 2>/dev/null && echo yes || echo no-STALE)"; else echo "no lock"; fi
echo "=== TRT importable in venv? ==="; /root/.venvs/hexo-bottrainer-wsl/bin/python -c "import tensorrt;print('tensorrt',tensorrt.__version__)" 2>&1 | tail -1
