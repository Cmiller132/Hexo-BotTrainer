#!/bin/bash
cd /mnt/e/Hexo-BotTrainer-hexgt/scripts
export CUDA_VISIBLE_DEVICES=
P=/root/.venvs/hexgt-build/bin/python
$P - > _wf_s3_npzcheck_out.txt 2>&1 <<'EOF'
import numpy as np, glob
for run, ep in (("dense_cnn_restnet_main_3", 5), ("dense_cnn_restnet_main_3", 9)):
    pats = glob.glob(f"/mnt/e/Hexo-BotTrainer/runs/{run}/shuffleddata/*epoch_{ep:06d}/train/*.npz")
    print(run, f"ep{ep}", "n_shards:", len(pats))
    if not pats:
        continue
    a = np.load(pats[0])
    print(" keys:", sorted(a.files))
    ml = a["moves_left"]
    print(" moves_left: shape", ml.shape, "min", ml.min(), "max", ml.max(),
          "mean", round(float(ml[ml >= 0].mean()), 1),
          "frac_masked(<0)", round(float((ml < 0).mean()), 4),
          "frac>512", round(float((ml > 512).mean()), 4))
    print(" turn_index max:", a["turn_index"].max())
EOF
echo "exit=$?" >> _wf_s3_npzcheck_out.txt
