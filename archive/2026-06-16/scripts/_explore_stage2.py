#!/usr/bin/env python3
"""STAGE 2 (CPU, read-only, no tree search): per-knob exploration analysis on the
LIVE checkpoint epoch_000030.pt. Loads the net on CPU, forwards REAL self-play
positions to get the raw policy prior, then replicates the EXACT Rust math
(mcts_tree.rs / mcts.rs) for: Dirichlet root noise, root-policy-temperature,
policy-nucleus widening, and KataGo forced playouts -- to measure how each knob
actually reshapes search at this run's branching factor. Also fills the Stage-1
gap: prior <-> search-policy agreement (the v4 shard drops the stored prior).

Zero GPU, zero disturbance to the live run.
"""
from __future__ import annotations
import json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

from dense_cnn_restnet.architecture import RestnetNetwork
from dense_cnn_restnet.constants import INPUT_CHANNELS
from dense_cnn_restnet.compact_io import read_compact_shard
from dense_cnn_restnet.samples import expand_sample

torch.set_num_threads(2)
RUN = Path("/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main1")
CKPT = RUN / "checkpoints" / "epoch_000030.pt"
SP = RUN / "selfplay"
VISITS = 512
FPK = 2.0
EPS = 0.25
ALPHA = 10.83
ROOT_T = 1.1
WIDEN_MASS = 0.95
WIDEN_MAX = 96
WIDEN_MIN = 2
N_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 18
SEEDS = 24

def band(t):
    if t < 10: return "00-09"
    if t < 20: return "10-19"
    if t < 40: return "20-39"
    if t < 70: return "40-69"
    return "70+"

def ent(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if p.size else 0.0

def nucleus_count(prior_sorted, mass, lo, hi):
    total = len(prior_sorted)
    lo = max(1, min(lo, total)); hi = max(lo, min(hi, total))
    if lo >= hi: return hi
    c = np.cumsum(prior_sorted)
    k = int(np.searchsorted(c, mass) + 1)
    return int(min(max(k, lo), hi))

# ---- build net + load live checkpoint on CPU ----
net = RestnetNetwork(
    in_channels=INPUT_CHANNELS, channels=96, blocks_type="R_R_R_T_R_R_T_R",
    attention_heads=4, mlp_ratio=2, embed_kernel_size=3, residual_conv="hex",
    attention_impl="sdpa", attention_scope="disk", dropout=0.0,
    short_term_value_horizons=(2, 6, 16), moves_left_head=True,
)
payload = torch.load(CKPT, map_location="cpu")
net.load_state_dict(payload["model_state"])
net.eval()

# ---- collect real positions from the latest games ----
samples = []  # (turn_index, expanded tensors dict)
npzs = (sorted(SP.glob("epoch_000030_game_*.npz")) +
        sorted(SP.glob("epoch_000031_game_*.npz")))[:N_GAMES]
for npz in npzs:
    try:
        rows = read_compact_shard(npz)
    except Exception:
        continue
    for r in rows:
        if not r.policy:
            continue
        samples.append((r.turn_index, expand_sample(r, symmetry=0)))

# forward in batches
agg = defaultdict(lambda: defaultdict(list))
def softmax_legal(logits, legal):
    z = logits.clone()
    z[legal < 0.5] = -1e30
    z = z - z.max()
    e = torch.exp(z) * (legal > 0.5)
    return (e / e.sum()).numpy()

B = 64
with torch.no_grad():
    for i in range(0, len(samples), B):
        chunk = samples[i:i + B]
        x = torch.stack([c[1]["input"] for c in chunk])
        out = net.forward_policy_value(x)
        pol = out["policy"].reshape(len(chunk), -1).float()
        for j, (t, tens) in enumerate(chunk):
            legal = tens["legal_mask"].reshape(-1)
            visit = tens["policy"].reshape(-1).numpy()
            prior = softmax_legal(pol[j], legal)
            N = int((legal > 0.5).sum())
            b = band(t)
            A = agg[b]
            A["N"].append(N)
            A["prior_top1"].append(float(prior.max()))
            A["prior_ent"].append(ent(prior))
            A["visit_top1"].append(float(visit.max()))
            A["visit_ent"].append(ent(visit))
            # prior<->search agreement
            A["agree_top1"].append(1.0 if int(prior.argmax()) == int(visit.argmax()) else 0.0)
            vp = visit[visit > 0]
            pp = prior[visit > 0]
            pp = np.clip(pp, 1e-12, None)
            A["kl_visit_prior"].append(float((vp * np.log(vp / pp)).sum()))
            # nucleus / widening (on raw prior)
            ps = np.sort(prior[prior > 0])[::-1]
            nuc = nucleus_count(ps, WIDEN_MASS, WIDEN_MIN, WIDEN_MAX)
            A["nucleus"].append(nuc)
            A["nuc_at_cap"].append(1.0 if nuc >= WIDEN_MAX else 0.0)
            # root policy temperature effect
            for T in (1.0, 1.1, 1.25):
                q = np.power(prior, 1.0 / T)
                q = q / q.sum()
                A[f"top1_T{T}"].append(float(q.max()))
            # forced playouts: n_forced over the nucleus children (top-nuc priors)
            topp = ps[:nuc]
            nf = np.floor(np.sqrt(FPK * topp * VISITS)).astype(int)
            A["forced_total"].append(int(nf.sum()))
            A["forced_frac_budget"].append(float(min(nf.sum(), VISITS)) / VISITS)
            A["forced_children_ge1"].append(int((nf >= 1).sum()))
            # Dirichlet flattening sweep (avg over seeds). Also, at the canonical
            # (alpha, eps), record how spread the raw noise vector is: effective #
            # moves the Dirichlet draw actually boosts = exp(entropy of the draw).
            pl = prior[legal.numpy() > 0.5]
            for al in (6.0, 10.83, 16.0, 21.0):
                for ep in (0.15, 0.25, 0.35):
                    per = max(al / N, 1e-6)
                    t1s, effs = [], []
                    for s in range(SEEDS):
                        rng = np.random.default_rng(1000 * s + 7)
                        g = rng.standard_gamma(per, size=N)
                        g = g / g.sum()
                        noised = (1 - ep) * pl + ep * g
                        noised = noised / noised.sum()
                        t1s.append(noised.max())
                        if al == ALPHA and ep == EPS:
                            effs.append(math.exp(ent(g)))
                    A[f"noise_top1_a{al}_e{ep}"].append(float(np.mean(t1s)))
                    if al == ALPHA and ep == EPS:
                        A["noise_eff_moves"].append(float(np.mean(effs)))

def stats(v):
    a = np.array(v, dtype=float)
    return dict(mean=round(float(a.mean()), 4), med=round(float(np.median(a)), 4),
               p10=round(float(np.percentile(a, 10)), 4), p90=round(float(np.percentile(a, 90)), 4))

out = {}
for b in ["00-09", "10-19", "20-39", "40-69", "70+"]:
    if b not in agg:
        continue
    A = agg[b]
    row = {"n_positions": len(A["N"])}
    for key in ["N", "prior_top1", "prior_ent", "visit_top1", "visit_ent",
                "agree_top1", "kl_visit_prior", "nucleus", "nuc_at_cap",
                "forced_total", "forced_frac_budget", "forced_children_ge1",
                "top1_T1.0", "top1_T1.1", "top1_T1.25", "noise_eff_moves"]:
        if A.get(key):
            row[key] = stats(A[key])
    # dirichlet sweep: report mean noised-top1 (lower = more flattening)
    sweep = {}
    for al in (6.0, 10.83, 16.0, 21.0):
        for ep in (0.15, 0.25, 0.35):
            k = f"noise_top1_a{al}_e{ep}"
            if A.get(k):
                sweep[f"a{al}_e{ep}"] = round(float(np.mean(A[k])), 4)
    row["dirichlet_sweep_noised_top1"] = sweep
    out[b] = row

print(json.dumps(out, indent=1))
