"""Diagnose exploration in 96x8 self-play: played-move spatial dispersion and
policy-target diffuseness by game phase. Pure read of compact self-play shards."""
import glob, os, math, statistics as st
from collections import defaultdict

from hexo_models.dense_cnn import compact_io as cio
from hexo_models.dense_cnn import d6
from hexo_models.dense_cnn.geometry import hex_distance

D = "/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8/selfplay"
shards = sorted(glob.glob(os.path.join(D, "epoch_*_game_*.npz")))[-40:]
print(f"shards sampled: {len(shards)}")

dists = []          # each played move's hex-dist from centroid of prior moves
far = 0             # moves > 6 hex from the prior centroid ("middle of nowhere")
total = 0
ent_by_phase = defaultdict(list)   # 0 early / 1 mid / 2 late -> policy entropy
played_visit_share = []            # (proxy) max policy weight per position
game_lens = []

for path in shards:
    try:
        samples = cio.read_compact_shard(path)
    except Exception:
        continue
    if not samples:
        continue
    # Move order: take the sample with the longest placement history (end of game).
    last = max(samples, key=lambda s: len(list(s.placement_history)))
    hist = sorted(
        [(int(idx), int(q), int(r)) for q, r, _p, _ph, idx, _a, _b in last.placement_history],
        key=lambda t: t[0],
    )
    moves = [(q, r) for _i, q, r in hist]
    game_lens.append(len(moves))
    for k in range(2, len(moves)):
        prior = moves[:k]
        cq = round(sum(p[0] for p in prior) / len(prior))
        cr = round(sum(p[1] for p in prior) / len(prior))
        d = hex_distance(moves[k], (cq, cr))
        dists.append(d)
        total += 1
        if d > 6:
            far += 1
    # Policy-target entropy + concentration, deduped by turn index, bucketed by game phase.
    seen = {}
    n = max((int(s.turn_index) for s in samples), default=1) + 1
    for s in samples:
        ti = int(s.turn_index)
        if ti in seen:
            continue
        seen[ti] = True
        w = [float(x) for _a, x in s.policy]
        sw = sum(w)
        if sw <= 0:
            continue
        w = [x / sw for x in w]
        ent = -sum(x * math.log(x + 1e-12) for x in w if x > 0)
        ent_by_phase[min(2, int(3 * ti / n))].append(ent)
        played_visit_share.append(max(w))

if dists:
    sd = sorted(dists)
    print("\n=== played-move spatial dispersion (hex-dist from centroid of prior moves) ===")
    print(f"  moves={total}  median={st.median(dists):.1f}  p90={sd[int(0.9*len(sd))]}  max={max(dists)}")
    print(f"  'middle of nowhere' (>6 hex): {far} = {100*far/max(1,total):.1f}% of moves")
    print(f"  mean game length: {st.mean(game_lens):.0f} moves")
print("\n=== policy-target entropy (nats; eff_moves = exp(entropy)) by game phase ===")
for ph, label in [(0, "early"), (1, "mid  "), (2, "late ")]:
    e = ent_by_phase.get(ph, [])
    if e:
        print(f"  {label}: mean_ent={st.mean(e):.2f}  eff_moves≈{math.exp(st.mean(e)):.1f}  n={len(e)}")
if played_visit_share:
    print(f"\n  median top-move target prob: {st.median(played_visit_share):.2f} "
          f"(low => diffuse target / the net is taught to spread visits)")
