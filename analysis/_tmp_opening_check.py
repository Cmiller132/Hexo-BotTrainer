"""Opening-diversity probe. Two signals across many epoch-1 self-play shards:
 (A) search target argmax at rows 0..2 (what MCTS most-visited)
 (B) the actually-PLAYED opening: signature of the row-k input board state
     (identical move sequences => identical input tensors), rows 1,2,4.
Pure numpy on NPZ rows; no engine."""
import glob, os, hashlib, numpy as np, collections

D = "/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8/selfplay"
games = sorted(glob.glob(os.path.join(D, "epoch_000001_game_*.npz")))[:200]
print(f"games sampled: {len(games)}")

tgt0 = []
sig = {1: [], 2: [], 4: []}
for g in games:
    try:
        z = np.load(g)
    except Exception:
        continue
    pol = z["policyTargetsNCHW"]; inp = z["inputNCHW"]
    n = pol.shape[0]
    if n == 0:
        continue
    tgt0.append(int(np.argmax(pol[0].reshape(-1))))
    for k in sig:
        if k < n:
            sig[k].append(hashlib.md5(np.ascontiguousarray(inp[k]).tobytes()).hexdigest()[:10])

c0 = collections.Counter(tgt0)
print(f"\n(A) search target row-0 argmax: {len(c0)} distinct / {len(tgt0)} games; top: {c0.most_common(5)}")
for k in (1, 2, 4):
    c = collections.Counter(sig[k])
    print(f"(B) PLAYED board after move {k}: {len(c)} distinct / {len(sig[k])} games; "
          f"top share {c.most_common(1)[0][1]}/{len(sig[k])}")
