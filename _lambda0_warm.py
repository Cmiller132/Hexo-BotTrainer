"""A/B redo via the WARM-optimizer + production fix path (faithful epoch-7 training).

Reuses _lambda0_experiment.py helpers. Difference vs the prior fresh-opt run:
loads epoch6's optimizer state then applies the SAME reconciliation the production
fix uses (drop per-param state whose moment shape != param shape -> fresh ZERO
moments for exactly the 3 graft-widened STV heads, warm-start everything else).
This exercises the graft+resume bug path end-to-end and is the literal epoch-7
training pass for each lambda arm. 1024 steps x b64, identical Random(7) sampling.
"""
import glob, time, random
import importlib.util as u
import torch

spec = u.spec_from_file_location("exp", "_lambda0_experiment.py")
m = u.module_from_spec(spec); spec.loader.exec_module(m)
from hexo_models.hexgt.trainer import HexgtTrainer


def run_pass_warm(shard_root, tag, device, steps=1024):
    net, ck, arch, grafted = m.load_grafted(m.EP6, device)
    cfg = m.make_cfg(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)
    # WARM-START + production reconciliation (mirrors scripts/_rl_train.py fix)
    restored = drifted = 0
    if "optimizer" in ck:
        try:
            opt.load_state_dict(ck["optimizer"]); restored = 1
        except Exception as e:
            print("opt load failed:", e)
        for group in opt.param_groups:
            for p in group["params"]:
                st = opt.state.get(p)
                if st and st.get("exp_avg") is not None and tuple(st["exp_avg"].shape) != tuple(p.shape):
                    opt.state.pop(p, None); drifted += 1
    trainer = HexgtTrainer(model=net, config=cfg, optimizer=opt)
    trainer.load_train_state(ck.get("train_state"))
    start = trainer.train_state.global_step
    shards = sorted(p for e in m.WINDOW_EPOCHS for p in glob.glob(f"{shard_root}/epoch_{e:06d}_game_*.npz"))
    weights = [0.9 ** (7 - m.epoch_of(p)) for p in shards]
    rng = random.Random(7)
    target = start + steps
    cs, cn, t0 = {}, 0, time.perf_counter()
    while trainer.train_state.global_step < target and shards:
        group = rng.choices(shards, weights=weights, k=30)
        for h in trainer.train_on_shards(group, batch_size=64, max_steps=target - trainer.train_state.global_step):
            for k, v in h.items():
                cs[k] = cs.get(k, 0.0) + v
            cn += 1
    avg = lambda k: cs.get(k, float("nan")) / cn if cn else float("nan")
    print(f"[{tag}] warm-opt restored={restored} reset_drifted={drifted} (expect 3) "
          f"start={start}->{trainer.train_state.global_step} ({cn} steps, {(time.perf_counter()-t0)/60:.1f}min) "
          f"value={avg('value'):.4f} policy={avg('policy'):.4f} total={avg('total'):.4f}", flush=True)
    out = f"{m.CK}/manual/hexgt_rl_epoch000007_{tag}.pt"
    torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(),
                "train_state": trainer.train_state.to_dict(), "arch": arch,
                "step": trainer.train_state.global_step, "rl_epoch": 7,
                "feature_schema_version": int(ck.get("feature_schema_version", 1)),
                "epoch_train_complete": True, "_experiment": f"warm-opt {tag}"}, out)
    print(f"[{tag}] saved {out}", flush=True)
    return out


if __name__ == "__main__":
    torch.set_grad_enabled(True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}", flush=True)
    p05 = run_pass_warm(m.SP, "lambda05_warm", dev)
    p0 = run_pass_warm(m.SP0, "lambda0_warm", dev)
    torch.set_grad_enabled(False)
    hxr = sorted(glob.glob(f"{m.SP}/epoch_000006.hxr")) + sorted(glob.glob(f"{m.SP}/epoch_000005.hxr"))
    rows = [("epoch6 baseline", m.EP6),
            ("epoch7 lambda0.5 WARM (control)", p05),
            ("epoch7 lambda0   WARM (treatment)", p0)]
    print(f"\n{'checkpoint':<36} | opening | probe(both) | mean_v  corr | v_won/v_lost gap | n", flush=True)
    print("-" * 108, flush=True)
    for name, path in rows:
        r = m.eval_ckpt(path, hxr)
        print(f"{name:<36} | {r['opening']:+.3f}  | {r['probe']:+.3f}({r['both']:.0%}) | "
              f"{r['mean_v']:+.3f} {r['corr']:+.3f} | {r['v_won']:+.3f}/{r['v_lost']:+.3f} {r['gap']:+.3f} | {r['n']}", flush=True)
    print("DONE (warm-opt A/B). checkpoints in manual/. latest untouched; run still halted.", flush=True)
