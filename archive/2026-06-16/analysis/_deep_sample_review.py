"""Deep integrity review of dense_cnn compact self-play shards + shuffled data.

Checks, on REAL self-play shards (not synthetic):
  - schema/range sanity: value/stvalue in [-1,1], policy mass > 0, legal/stone/history counts
  - D6 augmentation correctness: value & short-term-value INVARIANT across symmetries;
    policy mass preserved; input planes actually CHANGE (aug is not silently identity)
  - square-crop fallback rate on real data (how many symmetries empty a policy)
  - expanded-tensor health: shape (13,41,41), finite, policy finite & normalizable
  - the actual training path (expand_shard_to_arrays) on one shard
  - shuffleddata/<gen>/ row counts + train.json presence
Pure CPU read; no CUDA. Run with the WSL venv python + worktree PYTHONPATH.
"""
import glob
import math
import os
import statistics as st

import numpy as np
import torch

from hexo_models.dense_cnn import compact_io as cio
from hexo_models.dense_cnn import d6
from hexo_models.dense_cnn import samples as S

RUN = "/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8"
SP = os.path.join(RUN, "selfplay")
HOR = (1, 4, 8)
D6N = d6.D6_SIZE

shards = sorted(glob.glob(os.path.join(SP, "epoch_*_game_*.npz")))[-30:]
print(f"shards sampled: {len(shards)}  (latest: {os.path.basename(shards[-1]) if shards else 'NONE'})")

vals, stvals, pol_mass, pol_eff = [], [], [], []
legal_n, stones_n, hist_n = [], [], []
n_samples = 0
val_oor = stval_oor = pol_zero = 0

# D6 / crop accumulators
d6_value_mismatch = d6_stval_mismatch = d6_polmass_mismatch = 0
d6_input_identical = d6_input_changed = 0
crop_empty_sym = crop_total_sym = samples_any_empty = d6_subset = 0
bad_shape = nan_input = nonfinite_pol = 0

for path in shards:
    try:
        rows = cio.read_compact_shard(path)
    except Exception as e:  # noqa: BLE001
        print("READ FAIL", os.path.basename(path), repr(e))
        continue
    for s in rows:
        n_samples += 1
        v = float(s.value)
        vals.append(v)
        if not (-1.0001 <= v <= 1.0001):
            val_oor += 1
        for _h, sv in s.short_term_value:
            sv = float(sv)
            stvals.append(sv)
            if not (-1.0001 <= sv <= 1.0001):
                stval_oor += 1
        w = [float(x) for _a, x in s.policy]
        sw = sum(w)
        pol_mass.append(sw)
        if sw <= 0:
            pol_zero += 1
        else:
            ww = [x / sw for x in w]
            ent = -sum(x * math.log(x + 1e-12) for x in ww if x > 0)
            pol_eff.append(math.exp(ent))
        legal_n.append(len(list(s.legal_action_ids)))
        stones_n.append(len(list(s.stones)))
        hist_n.append(len(list(s.placement_history)))

    for s in rows[:20]:  # capped D6 + crop checks
        try:
            base = S.expand_sample(s, symmetry=0)
        except Exception:  # noqa: BLE001
            continue
        bi = base["input"].numpy()
        bp = base["policy"].numpy()
        bv = float(base["value"])
        bst = {h: float(base[f"stvalue_{h}"]) for h in HOR if f"stvalue_{h}" in base}
        if bi.shape != (13, 41, 41):
            bad_shape += 1
        if not np.isfinite(bi).all():
            nan_input += 1
        if not np.isfinite(bp).all():
            nonfinite_pol += 1
        any_empty = False
        for k in range(1, D6N):
            crop_total_sym += 1
            try:
                e = S.expand_sample(s, symmetry=k)
            except ValueError:
                crop_empty_sym += 1
                any_empty = True
                continue
            d6_subset += 1
            if abs(float(e["value"]) - bv) > 1e-5:
                d6_value_mismatch += 1
            for h, sv in bst.items():
                if f"stvalue_{h}" in e and abs(float(e[f"stvalue_{h}"]) - sv) > 1e-5:
                    d6_stval_mismatch += 1
            ep = e["policy"].numpy()
            if abs(float(ep.sum()) - float(bp.sum())) > 1e-4:
                d6_polmass_mismatch += 1
            if np.array_equal(e["input"].numpy(), bi):
                d6_input_identical += 1
            else:
                d6_input_changed += 1
        if any_empty:
            samples_any_empty += 1


def pct(a, b):
    return f"{100.0 * a / max(1, b):.2f}%"


print("\n========== SCHEMA / RANGE SANITY ==========")
print(f"  samples scanned: {n_samples}")
print(f"  value: min={min(vals):.3f} max={max(vals):.3f} mean={st.mean(vals):+.3f}  out-of-range={val_oor}")
print(f"  short_term_value: n={len(stvals)} min={min(stvals):.3f} max={max(stvals):.3f}  out-of-range={stval_oor}")
print(f"  policy mass: min={min(pol_mass):.3f} median={st.median(pol_mass):.3f}  ZERO-mass rows={pol_zero}")
print(f"  policy eff-moves: median={st.median(pol_eff):.2f} p10={sorted(pol_eff)[len(pol_eff)//10]:.2f} max={max(pol_eff):.1f}")
print(f"  legal moves/pos: median={int(st.median(legal_n))} max={max(legal_n)}")
print(f"  stones/pos: median={int(st.median(stones_n))} max={max(stones_n)}   history/pos: median={int(st.median(hist_n))} max={max(hist_n)}")
# value balance: fraction positive vs negative (should be roughly balanced over self-play)
pos = sum(1 for v in vals if v > 0.05)
neg = sum(1 for v in vals if v < -0.05)
zer = len(vals) - pos - neg
print(f"  value balance: pos={pct(pos,len(vals))} neg={pct(neg,len(vals))} ~0={pct(zer,len(vals))}")

print("\n========== EXPANDED-TENSOR HEALTH ==========")
print(f"  wrong shape: {bad_shape}   non-finite input: {nan_input}   non-finite policy: {nonfinite_pol}")

print("\n========== D6 AUGMENTATION CORRECTNESS ==========")
print(f"  D6 expansions checked: {d6_subset}  (over {d6_subset + crop_empty_sym} symmetry attempts)")
print(f"  value INVARIANCE violations:           {d6_value_mismatch}  (expect 0)")
print(f"  short-term-value INVARIANCE violations:{d6_stval_mismatch}  (expect 0)")
print(f"  policy-mass preservation violations:   {d6_polmass_mismatch}  (expect 0)")
print(f"  input planes CHANGED under non-id sym:  {pct(d6_input_changed, d6_input_changed + d6_input_identical)}  (high => aug really transforms)")
print(f"  input planes identical to identity:     {d6_input_identical}  (ok only for board-symmetric positions)")

print("\n========== SQUARE-CROP FALLBACK RATE (real data) ==========")
print(f"  symmetry attempts: {crop_total_sym}   emptied-policy (->identity fallback): {crop_empty_sym} = {pct(crop_empty_sym, crop_total_sym)}")
print(f"  samples with >=1 vulnerable symmetry: {samples_any_empty} = {pct(samples_any_empty, min(n_samples, 30*20))}")

print("\n========== ACTUAL TRAINING PATH (expand_shard_to_arrays) ==========")
try:
    syms = np.array([i % D6N for i in range(cio.compact_row_count(shards[-1]))], dtype=np.int64)
    arr = cio.expand_shard_to_arrays(shards[-1], syms, HOR)
    inp, pol = arr["input"], arr["policy"]
    print(f"  shard {os.path.basename(shards[-1])}: input={inp.shape} policy={pol.shape} value={arr['value'].shape}")
    print(f"  input finite={np.isfinite(inp).all()} range=[{inp.min():.2f},{inp.max():.2f}]")
    print(f"  policy finite={np.isfinite(pol).all()} per-row mass min={pol.sum(1).min():.3f} (all must be > 0)")
    print(f"  value range=[{arr['value'].min():+.2f},{arr['value'].max():+.2f}]  rows={inp.shape[0]} (vs {len(syms)} requested; diff=dropped)")
except Exception as e:  # noqa: BLE001
    print("  EXPAND FAIL", repr(e))

print("\n========== SHUFFLED TRAINING DATA ==========")
SH = os.path.join(RUN, "shuffleddata")
gens = sorted(glob.glob(os.path.join(SH, "*")))
print(f"  shuffle gens present: {[os.path.basename(g) for g in gens[-4:]]}")
for g in gens[-2:]:
    tj = os.path.join(g, "train.json")
    tn = glob.glob(os.path.join(g, "train", "*.npz"))
    note = "train.json OK" if os.path.exists(tj) else "train.json MISSING"
    rc = 0
    for f in tn[:50]:
        try:
            rc += int(cio.compact_row_count(f))
        except Exception:  # noqa: BLE001
            pass
    print(f"  {os.path.basename(g)}: {len(tn)} train shards, ~{rc} rows (first {min(50,len(tn))}), {note}")

print("\nDONE")
