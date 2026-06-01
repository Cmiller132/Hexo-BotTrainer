#!/usr/bin/env bash
echo "bootstrap procs:"; pgrep -af bootstrap_dense_cnn_sealbot || echo "  none"
echo "trainer/supervisor procs:"; pgrep -af 'train_model|supervise_target' || echo "  none"
echo "gpu compute apps:"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || echo "  none"
echo "gpu mem:"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
