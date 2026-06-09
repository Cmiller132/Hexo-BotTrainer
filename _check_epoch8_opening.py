"""CPU check: did the lambda=0 fix hold on the OPENING value (loop gauge) at epoch 8?"""
import glob, importlib.util as u
spec = u.spec_from_file_location("exp", "_lambda0_experiment.py")
m = u.module_from_spec(spec); spec.loader.exec_module(m)
import torch
torch.set_grad_enabled(False)

hxr = sorted(glob.glob(f"{m.SP}/epoch_000006.hxr")) + sorted(glob.glob(f"{m.SP}/epoch_000005.hxr"))
rows = [
    ("epoch6 baseline (lambda0.5)", f"{m.CK}/hexgt_rl_epoch000006.pt"),
    ("epoch7 lambda0_warm (experiment arm)", f"{m.CK}/manual/hexgt_rl_epoch000007_lambda0_warm.pt"),
    ("epoch8 LIVE (1st full lambda0 epoch, hard-z buffer)", f"{m.CK}/hexgt_rl_epoch000008.pt"),
]
print(f"\n{'checkpoint':<52} | OPENING | probe(both) | mean_v  corr | v_won/v_lost gap | n")
print("-" * 120)
for name, path in rows:
    r = m.eval_ckpt(path, hxr)
    print(f"{name:<52} | {r['opening']:+.3f}  | {r['probe']:+.3f}({r['both']:.0%}) | "
          f"{r['mean_v']:+.3f} {r['corr']:+.3f} | {r['v_won']:+.3f}/{r['v_lost']:+.3f} {r['gap']:+.3f} | {r['n']}")
print("\nOPENING is the loop-fix gauge (baseline +0.223 -> should hold ~0 at lambda=0).")
print("probe(optimism_sum) measures swap-antisymmetry/head sharpness; lambda=0 is NOT expected to move it.")
