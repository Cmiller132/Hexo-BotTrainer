RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
echo "=== epoch_000015.json: all elapsed/timing/shuffle fields ==="
$V - <<'PY'
import json
d=json.load(open("/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1/diagnostics/epoch_000015.json"))
print("TOP elapsed_seconds:", d.get("elapsed_seconds"))
r=d["metadata"]["result"]
def show(name, blk):
    if isinstance(blk,dict):
        t={k:blk.get(k) for k in blk if any(x in k.lower() for x in ('elapsed','second','status','steps','loss','rows','games','reason'))}
        print(f"  {name}:", t)
show("selfplay", r.get("selfplay"))
show("training", r.get("training"))
show("evaluation", r.get("evaluation"))
sel=(r.get("samples") or {}).get("selection") or {}
print("  samples.selection:", {k:sel.get(k) for k in sel if 'elapsed' in k.lower() or 'status' in k.lower() or 'second' in k.lower()})
sh=sel.get("metadata",{}).get("shuffle",{})
print("  shuffle:", {k:sh.get(k) for k in sh if any(x in k.lower() for x in ('elapsed','second','output_rows','used_rows','total_num','desired','status','validation_rows'))})
fin=(r.get("samples") or {}).get("finalize") or {}
print("  samples.finalize:", {k:fin.get(k) for k in fin if 'elapsed' in k.lower() or 'second' in k.lower() or 'status' in k.lower()})
PY
echo "=== file mtimes for epoch 15-16 (phase boundaries) ==="
ls -lt --time-style=+%H:%M:%S "$RD"/diagnostics/dense_cnn.selfplay.epoch_00001[4-6].json "$RD"/diagnostics/epoch_00001[4-6].json "$RD"/diagnostics/dense_cnn.evaluation.epoch_00001[4-6].json 2>/dev/null | awk '{print $6, $7}' | sort
