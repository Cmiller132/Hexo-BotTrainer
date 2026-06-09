"""Phase 3: cross-epoch statistics — diversity, game length, SealBot, value calibration.

Reads per-epoch selfplay/eval diagnostics (runs/.../eval/epoch_*_{selfplay,eval}.json)
into a trend table, and recomputes value calibration corr(v,z) at sampled epochs by
evaluating that epoch's own model on its own searched positions (value-only, CPU).
"""
import sys, json, glob, math
from pathlib import Path
sys.path.insert(0, "/mnt/e/Hexo-BotTrainer-hexgt/_expl")
import common as C
from hexo_utils.records import HexoRecordFile
from hexo_models.dense_cnn.compact_io import read_compact_shard

EVAL = f"{C.RUN}/eval"
CAL_EPOCHS = [5, 10, 15, 20, 25, 30, 35, 41]
CAL_GAMES = 30

def mean(xs): return (sum(xs) / len(xs)) if xs else float("nan")
def corr(pairs):
    xs = [a for a, b in pairs if b == b]; ys = [b for a, b in pairs if b == b]
    if len(xs) < 3: return float("nan")
    mx, my = mean(xs), mean(ys)
    dx = math.sqrt(sum((a-mx)**2 for a in xs)); dy = math.sqrt(sum((b-my)**2 for b in ys))
    num = sum((a-mx)*(b-my) for a, b in zip(xs, ys))
    return num/(dx*dy) if dx > 0 and dy > 0 else float("nan")

# --- trend table from diagnostics --------------------------------------------
rows = []
for sp in sorted(glob.glob(f"{EVAL}/epoch_*_selfplay.json")):
    d = json.load(open(sp))
    ep = int(d["epoch"])
    ev = {}
    evp = f"{EVAL}/epoch_{ep:06d}_eval.json"
    if Path(evp).exists():
        ej = json.load(open(evp))
        sb = ej.get("vs_sealbot") or {}
        ev = {"sb_win": sb.get("win_rate"), "sb_turns": sb.get("mean_turns")}
    rows.append({
        "epoch": ep,
        "len_med": d.get("game_length_median"),
        "len_mean": d.get("game_length_mean"),
        "open_ent": d.get("opening_entropy"),
        "move2_ent": d.get("move2_entropy"),
        "open_uniq": d.get("opening_unique_fraction"),
        "visitH": d.get("mean_visit_entropy"),
        "priorH": d.get("mean_prior_entropy"),
        "top1": d.get("mean_top_visit_fraction"),
        "decisive": d.get("decisive_fraction"),
        "draw": d.get("draw_fraction"),
        "sb_win": ev.get("sb_win"),
        "sb_turns": ev.get("sb_turns"),
    })

print(f"\n=== PHASE 3: cross-epoch trends (hexgnn_rl_main1, {len(rows)} epochs) ===")
print("epoch len_med len_mean openH move2H visitH priorH top1  decisive draw  SBwin SBturns")
for r in rows:
    def f(x, w=6, p=2):
        return (f"{x:.{p}f}".rjust(w)) if isinstance(x, (int, float)) else str(x).rjust(w)
    print(f"{r['epoch']:5d} {f(r['len_med'],7,1)} {f(r['len_mean'],8,1)} {f(r['open_ent'],5)} "
          f"{f(r['move2_ent'],6)} {f(r['visitH'],6)} {f(r['priorH'],6)} {f(r['top1'],5)} "
          f"{f(r['decisive'],8,2)} {f(r['draw'],5,2)} {f(r['sb_win'],5,3) if r['sb_win'] is not None else '   -- '} {f(r['sb_turns'],7,1) if r['sb_turns'] is not None else '   --  '}")

# --- value calibration corr(v,z) at sampled epochs ---------------------------
print(f"\n=== value calibration corr(v,z) by epoch ({CAL_GAMES} games each, model=that epoch) ===")
cal = {}
for ep in CAL_EPOCHS:
    ckpt = f"{C.CKPT_DIR}/hexgnn_rl_epoch{ep:06d}.pt"
    if not Path(ckpt).exists():
        continue
    net, _ = C.load_model(ep)
    ev = C.Evaluator(net, n=4)
    hxr = f"{C.SP_DIR}/epoch_{ep:06d}.hxr"
    if not Path(hxr).exists():
        continue
    recs = list(HexoRecordFile.open(hxr).iter_records())[:CAL_GAMES]
    vz = []
    for rec in recs:
        idx = C.shard_index_from_game_id(rec.game_id)
        shp = glob.glob(f"{C.SP_DIR}/epoch_{ep:06d}_game_{idx:06d}.npz")
        if not shp:
            continue
        rows_s = read_compact_shard(Path(shp[0]))
        z_by = {int(r.turn_index): float(r.value) for r in rows_s}
        actions = [int(a) for a in rec.action_ids]
        s = C.new_game(seed=int(rec.seed))
        for t, a in enumerate(actions):
            if C.engine.terminal(s) is not None:
                break
            if t in z_by:
                vz.append((ev.value(s), z_by[t]))
            C.apply(s, a)
    cal[ep] = {"corr": corr(vz), "n": len(vz), "mean_v": mean([v for v, z in vz]),
               "mean_z": mean([z for v, z in vz])}
    print(f"  epoch {ep:3d}: corr(v,z) = {cal[ep]['corr']:+.3f}  (n={cal[ep]['n']}, "
          f"mean|model_v|->{mean([abs(v) for v,z in vz]):.2f}, mean z={cal[ep]['mean_z']:+.2f})")

out = {"trends": rows, "calibration": cal}
Path(f"{C.RUN}".replace("/runs/", "/runs/")).mkdir(exist_ok=True, parents=True)
Path("/mnt/e/Hexo-BotTrainer-hexgt/_expl/out").mkdir(exist_ok=True)
json.dump(out, open("/mnt/e/Hexo-BotTrainer-hexgt/_expl/out/phase3.json", "w"), indent=2)
print("\nwrote _expl/out/phase3.json")
