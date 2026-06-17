#!/bin/bash
cd /mnt/e/Hexo-BotTrainer-hexgt/scripts
/root/.venvs/hexgt-build/bin/python - > _wf_s3_losstraj_out.txt 2>&1 <<'PYEOF'
import json, glob
for run in ("dense_cnn_restnet_main_3", "dense_cnn_restnet_main_4"):
    print("==", run)
    for p in sorted(glob.glob(f"/mnt/e/Hexo-BotTrainer/runs/{run}/diagnostics/epoch_0000*.json")):
        d = json.load(open(p))
        ep = p.rsplit("epoch_", 1)[1][:6]
        def find_ml(obj, path=""):
            hits = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if "moves_left" in k and isinstance(v, (int, float)):
                        hits.append((path + "/" + k, v))
                    else:
                        hits += find_ml(v, path + "/" + k)
            elif isinstance(obj, list) and obj and isinstance(obj[0], (dict, list)):
                for i, v in enumerate(obj[:3]):
                    hits += find_ml(v, path + f"[{i}]")
            return hits
        hits = find_ml(d)
        print(ep, hits[:6] if hits else "no moves_left keys; top keys: " + str(list(d.keys())[:10]))
PYEOF
echo "exit=$?" >> _wf_s3_losstraj_out.txt
