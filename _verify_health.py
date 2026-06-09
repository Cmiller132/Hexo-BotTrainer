"""Value-health (corr/won-lost) on recent checkpoints + STV-vs-marginal recompute. CPU."""
import glob, importlib.util as u
import numpy as np, torch
spec = u.spec_from_file_location("exp", "_lambda0_experiment.py"); m = u.module_from_spec(spec); spec.loader.exec_module(m)
from hexo_models.hexgt.losses import scalar_to_binned_target
torch.set_grad_enabled(False)

# value-health on recent checkpoints (use epoch 18 hxr games as the held-out eval set)
hxr = sorted(glob.glob(f"{m.SP}/epoch_000018.hxr")) + sorted(glob.glob(f"{m.SP}/epoch_000019.hxr"))
print("=== value health (held-out epoch18-19 games) ===")
print(f"{'ckpt':<10} | opening | probe | mean_v corr | v_won/v_lost gap | n")
for ep in [6, 16, 21, 23]:
    p = f"{m.CK}/hexgt_rl_epoch{ep:06d}.pt"
    if not glob.glob(p): continue
    r = m.eval_ckpt(p, hxr)
    print(f"epoch{ep:<5}| {r['opening']:+.3f}  | {r['probe']:+.2f} | {r['mean_v']:+.3f} {r['corr']:+.3f} | {r['v_won']:+.3f}/{r['v_lost']:+.3f} {r['gap']:+.3f} | {r['n']}")

# STV target entropy floor + marginal baseline on recent (epoch 21) shards
print("\n=== STV-vs-marginal (epoch-21 shards) ===")
shs = sorted(glob.glob(f"{m.SP}/epoch_000021_game_*.npz"))[:60]
hz = None; cols = {}
for s in shs:
    d = np.load(s, allow_pickle=True)
    if hz is None: hz = [int(x) for x in d["horizons"]]
    stv = d["stvalue"]; msk = d["stvalue_mask"]
    for ci, h in enumerate(hz):
        cols.setdefault(h, []).append(stv[:, ci][msk[:, ci] > 0])
meas = {4: 2.99, 12: 3.13, 24: 3.22}  # epoch-21 measured stv losses from train log
print(f"{'horizon':<8} | std | 2-bin floor | marginal-baseline CE | measured(ep21) | head-edge(marg-meas)")
for h in hz:
    v = np.concatenate(cols[h]); v = v[np.isfinite(v)]
    t = scalar_to_binned_target(torch.tensor(v, dtype=torch.float32))
    floor = -(t.clamp_min(1e-12) * t.clamp_min(1e-12).log()).sum(-1).mean().item()
    marg = t.mean(0); marg = marg / marg.sum()
    ce_marg = -(t * marg.clamp_min(1e-12).log()).sum(-1).mean().item()
    print(f"stv{h:<5} | {v.std():.3f} | {floor:.3f}      | {ce_marg:.3f}              | {meas.get(h, float('nan')):.3f}        | {ce_marg-meas.get(h,float('nan')):+.3f}")
