"""CPU eval of the soft-Z experiment checkpoints (training already done)."""
import glob, importlib.util as u
spec = u.spec_from_file_location("exp", "_lambda0_experiment.py")
m = u.module_from_spec(spec); spec.loader.exec_module(m)
import torch
torch.set_grad_enabled(False)

RUN = m.RUN
hxr = sorted(glob.glob(f"{m.SP}/epoch_000006.hxr")) + sorted(glob.glob(f"{m.SP}/epoch_000005.hxr"))
rows = [
    ("epoch6 baseline (lambda0.5, step3605)", f"{m.CK}/hexgt_rl_epoch000006.pt"),
    ("epoch7 lambda0.5 CONTROL (1024st, fresh-opt, from e6)", f"{m.CK}/manual/hexgt_rl_epoch000007_lambda05.pt"),
    ("epoch7 lambda0   TREATMENT (1024st, fresh-opt, from e6)", f"{m.CK}/hexgt_rl_epoch000007_lambda0.pt"),
    ("epoch7 lambda0.5 (watcher retrain, 512st warm-opt)", f"{m.CK}/hexgt_rl_epoch000007.pt"),
]
print(f"\n{'checkpoint':<54} | opening | probe(both>0) | mean_v  corr | v_won / v_lost  gap | n")
print("-" * 124)
for name, path in rows:
    r = m.eval_ckpt(path, hxr)
    print(f"{name:<54} | {r['opening']:+.3f}  | {r['probe']:+.3f}({r['both']:.0%})  | "
          f"{r['mean_v']:+.3f} {r['corr']:+.3f} | {r['v_won']:+.3f} / {r['v_lost']:+.3f} {r['gap']:+.3f} | {r['n']}")
print("\nopening baseline +0.223; loop hypothesis predicts lambda0 opening drops toward 0.")
