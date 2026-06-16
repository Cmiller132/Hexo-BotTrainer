"""Offline, CPU-only deep analysis of dense_cnn_model1_target_96x8 checkpoints.

Read-only: loads frozen checkpoints + a fixed sample of recorded held-out positions
(epoch-19 self-play shards, not yet trained on), runs each checkpoint's value/policy/
short-term-value heads on the SAME positions, and reports value calibration, policy-prior
sharpness vs the MCTS visit target, a fixed-sample loss decomposition, short-term-value
behavior, and per-phase confidence. No GPU, no new self-play, threads capped so the live
trainer keeps its cores.
"""
import glob
import math
import os
import time

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("DMA_THREADS", "6")))
torch.set_grad_enabled(False)

from hexo_models.dense_cnn.architecture import Model1Network
from hexo_models.dense_cnn.losses import decode_binned_value, scalar_to_binned_target
from hexo_models.dense_cnn import compact_io as cio
from hexo_models.dense_cnn import samples as S

RUN = "/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8"
CKPTS = os.path.join(RUN, "checkpoints")
HOR = (1, 4, 8)
N_POS = int(os.environ.get("DMA_NPOS", "2500"))
BATCH = 128


def build_model():
    return Model1Network(in_channels=13, channels=96, blocks=8, dropout=0.0,
                         short_term_value_horizons=HOR).eval()


def sample_positions(n):
    shards = sorted(glob.glob(os.path.join(RUN, "selfplay", "epoch_000019_game_*.npz")))
    rows = []
    for path in shards:
        try:
            samples = cio.read_compact_shard(path)
        except Exception:  # noqa: BLE001
            continue
        glen = max((int(s.turn_index) for s in samples), default=0) + 1
        for s in samples:
            try:
                e = S.expand_sample(s, symmetry=0)
            except ValueError:
                continue
            rows.append({
                "input": e["input"], "policy": e["policy"].numpy().astype(np.float64),
                "value": float(e["value"]), "legal": e["legal_mask"].numpy(),
                "turn": int(s.turn_index), "glen": glen,
                "stv": {h: float(e[f"stvalue_{h}"]) for h in HOR if f"stvalue_{h}" in e},
            })
            if len(rows) >= n:
                return rows, len(shards)
    return rows, len(shards)


def analyze(model, rows):
    preds, acts = [], []
    p_ent, p_top1, p_eff = [], [], []
    t_ent, t_top1, argmatch, kls = [], [], [], []
    pol_ce, val_ce = [], []
    stv_abs = {h: [] for h in HOR}
    phase = {0: {"ae": [], "pe": [], "n": 0}, 1: {"ae": [], "pe": [], "n": 0}, 2: {"ae": [], "pe": [], "n": 0}}
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        X = torch.stack([r["input"] for r in chunk])
        out = model(X)
        pv = decode_binned_value(out["value"]).numpy()
        logp_v = torch.log_softmax(out["value"], dim=-1)
        vt = scalar_to_binned_target(torch.tensor([r["value"] for r in chunk], dtype=torch.float32))
        vce = -(vt * logp_v).sum(-1).numpy()
        logp_p = torch.log_softmax(out["policy"], dim=-1).numpy().astype(np.float64)
        pol_logit = out["policy"].numpy().astype(np.float64)
        stv_pred = {h: decode_binned_value(out[f"stvalue_{h}"]).numpy() for h in HOR}
        for j, r in enumerate(chunk):
            pvj, av = float(pv[j]), r["value"]
            preds.append(pvj); acts.append(av)
            val_ce.append(float(vce[j]))
            tgt = r["policy"]; tsum = tgt.sum()
            # policy CE (full 1681, training-style) of this checkpoint's logits vs the MCTS target
            pol_ce.append(float(-(tgt * logp_p[j]).sum()))
            # legal-masked prior (what search uses)
            legal = r["legal"]
            lg = np.where(legal, pol_logit[j], -1e30)
            mx = lg.max(); ex = np.exp(lg - mx); ex[~legal] = 0.0; prior = ex / ex.sum()
            pp = prior[prior > 0]
            e = float(-(pp * np.log(pp)).sum())
            p_ent.append(e); p_eff.append(math.exp(e)); p_top1.append(float(prior.max()))
            tnz = tgt[tgt > 0]
            te = float(-(tnz * np.log(tnz)).sum())
            t_ent.append(te); t_top1.append(float(tgt.max()))
            argmatch.append(int(prior.argmax() == tgt.argmax()))
            mt = tgt > 0
            kls.append(float((tgt[mt] * (np.log(tgt[mt]) - np.log(prior[mt] + 1e-12))).sum()))
            for h in HOR:
                if h in r["stv"]:
                    stv_abs[h].append(abs(float(stv_pred[h][j]) - r["stv"][h]))
            frac = r["turn"] / max(1, r["glen"])
            ph = 0 if frac < 0.34 else (1 if frac < 0.67 else 2)
            phase[ph]["ae"].append(abs(pvj - av)); phase[ph]["pe"].append(e); phase[ph]["n"] += 1
    preds, acts = np.array(preds), np.array(acts)
    # value calibration
    p_win = (preds + 1) / 2
    y = np.where(acts > 0.05, 1.0, np.where(acts < -0.05, 0.0, 0.5))
    mae = float(np.mean(np.abs(preds - acts)))
    brier = float(np.mean((p_win - y) ** 2))
    nondraw = np.abs(acts) > 0.05
    sign_acc = float(np.mean(np.sign(preds[nondraw]) == np.sign(acts[nondraw]))) if nondraw.any() else float("nan")
    # reliability buckets by predicted value
    edges = [-1.01, -0.6, -0.2, 0.2, 0.6, 1.01]
    buckets = []
    for a, b in zip(edges, edges[1:]):
        m = (preds >= a) & (preds < b)
        if m.sum() == 0:
            buckets.append((f"[{a:.1f},{b:.1f})", 0, None, None))
            continue
        emp_win = float(np.mean(acts[m] > 0.05))
        buckets.append((f"[{max(a,-1):.1f},{min(b,1):.1f})", int(m.sum()), float(preds[m].mean()), emp_win))
    return {
        "n": len(preds),
        "value": {"mae": mae, "brier": brier, "sign_acc": sign_acc,
                  "mean_pred": float(preds.mean()), "mean_act": float(acts.mean()), "val_ce": float(np.mean(val_ce))},
        "policy": {"prior_ent": float(np.mean(p_ent)), "prior_eff": float(np.mean(p_eff)),
                   "prior_top1": float(np.mean(p_top1)), "tgt_ent": float(np.mean(t_ent)),
                   "tgt_eff": float(np.mean(np.exp(t_ent))), "tgt_top1": float(np.mean(t_top1)),
                   "argmatch": float(np.mean(argmatch)), "kl_tgt_prior": float(np.mean(kls)),
                   "pol_ce": float(np.mean(pol_ce))},
        "stvalue_mae": {h: (float(np.mean(stv_abs[h])) if stv_abs[h] else None) for h in HOR},
        "phase": {k: {"value_mae": (float(np.mean(v["ae"])) if v["ae"] else None),
                      "prior_ent": (float(np.mean(v["pe"])) if v["pe"] else None), "n": v["n"]} for k, v in phase.items()},
        "buckets": buckets,
    }


def main():
    print(f"threads={torch.get_num_threads()} device=cpu n_pos_target={N_POS}")
    t0 = time.perf_counter()
    rows, nsh = sample_positions(N_POS)
    print(f"sampled {len(rows)} positions from {nsh} epoch-19 shards in {time.perf_counter()-t0:.1f}s")
    if not rows:
        print("NO POSITIONS — abort"); return
    model = build_model()
    ckpts = []
    pf = os.path.join(CKPTS, "bootstrap_sealbot_prefit.pt")
    if os.path.exists(pf):
        ckpts.append(("prefit", pf))
    for ep in (1, 3, 5, 7, 10, 14):
        p = os.path.join(CKPTS, f"epoch_{ep:06d}.pt")
        if os.path.exists(p):
            ckpts.append((f"ep{ep}", p))
    latest = sorted(glob.glob(os.path.join(CKPTS, "epoch_0*.pt")))
    if latest:
        ckpts.append((f"latest({os.path.basename(latest[-1])})", latest[-1]))
    results = {}
    for label, path in ckpts:
        tc = time.perf_counter()
        try:
            payload = torch.load(path, map_location="cpu")
            ms = payload.get("model_state", payload)
            model.load_state_dict(ms)
        except Exception as ex:  # noqa: BLE001
            print(f"[{label}] LOAD FAILED: {ex!r}"); continue
        r = analyze(model, rows)
        results[label] = r
        rt = time.perf_counter() - tc
        v, p = r["value"], r["policy"]
        print(f"\n=== {label}  ({rt:.1f}s) ===")
        print(f"  VALUE: MAE={v['mae']:.3f} Brier={v['brier']:.3f} sign_acc={v['sign_acc']:.3f} "
              f"meanPred={v['mean_pred']:+.3f} meanAct={v['mean_act']:+.3f} valCE={v['val_ce']:.3f}")
        print(f"  POLICY prior: entropy={p['prior_ent']:.2f} eff_moves={p['prior_eff']:.1f} top1={p['prior_top1']:.3f}"
              f"  | MCTS target: eff={p['tgt_eff']:.1f} top1={p['tgt_top1']:.3f}"
              f"  | argmaxMatch={p['argmatch']:.3f} KL(tgt||prior)={p['kl_tgt_prior']:.3f} polCE={p['pol_ce']:.3f}")
        print(f"  STV MAE: " + " ".join(f"h{h}={r['stvalue_mae'][h]:.3f}" for h in HOR if r['stvalue_mae'][h] is not None))
        ph = r["phase"]
        print("  PHASE value_MAE / prior_entropy: "
              + " ".join(f"{nm}=({ph[k]['value_mae']:.3f}/{ph[k]['prior_ent']:.2f},n={ph[k]['n']})"
                         for k, nm in [(0, "open"), (1, "mid"), (2, "end")] if ph[k]['value_mae'] is not None))
    # reliability table for latest
    if results:
        last = list(results)[-1]
        print(f"\n=== VALUE RELIABILITY (latest = {last}) ===")
        print("  pred_bucket | n | mean_pred | empirical_winrate")
        for name, n, mp, ew in results[last]["buckets"]:
            print(f"  {name} | {n} | {('%.3f'%mp) if mp is not None else '--'} | {('%.3f'%ew) if ew is not None else '--'}")
        # also epoch-1 reliability for contrast
        if "ep1" in results:
            print(f"\n=== VALUE RELIABILITY (ep1) ===")
            for name, n, mp, ew in results["ep1"]["buckets"]:
                print(f"  {name} | {n} | {('%.3f'%mp) if mp is not None else '--'} | {('%.3f'%ew) if ew is not None else '--'}")


if __name__ == "__main__":
    main()
