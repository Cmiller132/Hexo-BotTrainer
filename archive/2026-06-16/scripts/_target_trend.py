"""Per-epoch self-play target stats: policy-target entropy + game length.

If policy-target entropy RISES over epochs, the rising reported training loss is
explained as a moving (higher-entropy) target ceiling, not model divergence.
"""
import sys, glob, json
import numpy as np

RD = "/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x6"
sys.path.insert(0, "/mnt/e/Hexo-BotTrainer/packages/hexo_models/dense_cnn/python")
from hexo_models.dense_cnn.replay import POLICY_KEY, VALUE_KEY

print(f"{'ep':>3} {'pos/game':>8} {'tgt_polH':>9} {'supp_mv':>8} {'val|>.9|':>8} {'#rows':>7}")
for e in range(3, 13):
    shard = f"{RD}/selfplay/epoch_{e:06d}_game_000000.npz"
    g = glob.glob(shard)
    if not g:
        # grab any shard for that epoch
        g = sorted(glob.glob(f"{RD}/selfplay/epoch_{e:06d}_game_*.npz"))
    if not g:
        continue
    # aggregate a few shards for a stable estimate
    pols, vals, rows = [], [], 0
    nshards = 0
    for sh in g[:12]:
        try:
            with np.load(sh) as d:
                p = d[POLICY_KEY]
                v = d[VALUE_KEY]
        except Exception:
            continue
        p = p.reshape(p.shape[0], -1).astype(np.float32)
        p = p / np.clip(p.sum(1, keepdims=True), 1e-9, None)
        H = -(p * np.log(np.clip(p, 1e-12, None))).sum(1)
        supp = (p > 1e-6).sum(1)
        pols.append((H.mean(), supp.mean(), p.shape[0]))
        vals.append(np.abs(v.reshape(-1)))
        rows += p.shape[0]
        nshards += 1
    if not pols:
        continue
    # rows here = samples across nshards games; pos/game from epoch json instead
    ej = f"{RD}/diagnostics/epoch_{e:06d}.json"
    posg = float("nan")
    if glob.glob(ej):
        try:
            jd = json.load(open(ej))
            spj = jd["metadata"]["result"]["selfplay"]
            posg = spj.get("effective_samples", 0) / max(1, spj.get("games_finished", 1))
        except Exception:
            pass
    Hm = np.average([x[0] for x in pols], weights=[x[2] for x in pols])
    sm = np.average([x[1] for x in pols], weights=[x[2] for x in pols])
    valc = np.concatenate(vals)
    print(f"{e:>3} {posg:>8.1f} {Hm:>9.3f} {sm:>8.1f} {np.mean(valc>0.9):>8.3f} {rows:>7}")
