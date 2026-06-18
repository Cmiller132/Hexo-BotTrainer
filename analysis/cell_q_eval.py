"""Empirical eval of the hexfield v3 cell_q head: does it predict the right move?

READ-ONLY w.r.t. the run dir. CPU only. Reuses the production training data path:
  window.load_packed_shard -> PackedWindow.row_view -> expand_backends._row_view_to_sample
  -> samples.expand_sample -> batching.collate_training -> HexfieldNet.forward.
Decode: predicted scalar Q = softmax(cell_q logits over 65 bins) . linspace(-1,1,65)
(== losses.decode_binned_value). Target scalar Q = ExpandedRow.cell_q (already [-1,1]),
considered only where cell_q_mask==1.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch

from hexfield.batching import PAD_QUANTUM, collate_training
from hexfield.constants import VALUE_BINS
from hexfield.expand_backends import _row_view_to_sample
from hexfield.losses import value_bins
from hexfield.model import HexfieldNet
from hexfield.samples import expand_sample
from hexfield.window import load_packed_shard

RUN = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_3"
CKPT = f"{RUN}/checkpoints/epoch_000033.pt"      # 3 epochs behind live edge (ep36)
SHARD_EPOCH_DIR = f"{RUN}/samples/epoch_000030"  # well behind, never mid-write
TARGET_POSITIONS = 600
RNG_SEED = 12345


def load_net(path: str) -> HexfieldNet:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "model" in payload, "no model state in checkpoint"
    net = HexfieldNet()
    net.load_state_dict(payload["model"], strict=True)
    net.eval()
    meta = payload.get("meta", {})
    return net, meta


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] < 2:
        return float("nan")
    a = a.astype(np.float64); b = b.astype(np.float64)
    a -= a.mean(); b -= b.mean()
    den = np.sqrt((a**2).sum() * (b**2).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


@torch.no_grad()
def main():
    bins = value_bins().numpy().astype(np.float64)  # (65,) linspace(-1,1,65)
    net, meta = load_net(CKPT)
    print(f"checkpoint epoch (meta): {meta.get('epoch')}  path={CKPT}", file=sys.stderr)

    # Gather rows (one PackedRowView -> HexfieldSampleData -> ExpandedRow) from
    # several epoch-30 shards until we have TARGET_POSITIONS expanded rows.
    shard_paths = sorted(glob.glob(f"{SHARD_EPOCH_DIR}/*.npz"))
    rng = np.random.default_rng(RNG_SEED)
    rng.shuffle(shard_paths)

    expanded = []
    n_shards_used = 0
    for sp in shard_paths:
        if len(expanded) >= TARGET_POSITIONS:
            break
        win = load_packed_shard(sp)
        n_shards_used += 1
        for i in range(win.n):
            sample = _row_view_to_sample(win.row_view(i))
            er = expand_sample(sample, symmetry=0)
            expanded.append(er)
            if len(expanded) >= TARGET_POSITIONS:
                break

    # Per-position accumulators
    top1_q_agree = []        # argmax pred Q == argmax target Q (masked cells)
    top1_q_vs_policy = []     # argmax pred Q == argmax policy
    top1_tgt_vs_policy = []   # argmax target Q == argmax policy (ceiling ref)
    top3_contain = []         # target-best in pred top-3
    sp_rhos = []; pe_rhos = []
    legal_counts = []; masked_counts = []
    abs_err_all = []; signed_err_all = []
    best_pred_q = []; pos_value = []   # for best-pred-Q vs game value corr

    # Forward in N-bucketed chunks (mirror head_audit chunking to bound the
    # (rows, S, S) attention transient on CPU).
    ROW_BUDGET_NODES = 20000
    i = 0
    n = len(expanded)
    flat_pred = []  # collect (pred_q_per_cell) aligned to expanded order
    # We must keep cell_q per ROW; process row-by-row decode after each batch.
    results = [None] * n  # each entry: (pred_q (L,), tgt_q (L,), mask (L,), policy (L,))

    while i < n:
        j = i
        max_nodes = 0
        while j < n:
            nodes = expanded[j].support.num_nodes
            cand = max(max_nodes, nodes)
            pad_to = -(-cand // PAD_QUANTUM) * PAD_QUANTUM
            if j > i and (j - i + 1) * pad_to * pad_to > ROW_BUDGET_NODES * PAD_QUANTUM:
                break
            max_nodes = cand
            j += 1
        chunk = expanded[i:j]
        pad_to = -(-max(r.support.num_nodes for r in chunk) // PAD_QUANTUM) * PAD_QUANTUM
        batch = collate_training(chunk, pad_to=pad_to)
        out = net(batch["feats"], batch["nbr"], batch["mask"], batch["coords"])
        cq_logits = out["cell_q"].float()          # (B, Npad, 65)
        pol_logits = out["policy"].float()         # (B, Npad)
        # decode predicted scalar Q per cell
        prob = torch.softmax(cq_logits, dim=-1)
        pred_scalar = (prob * torch.from_numpy(bins).float()).sum(-1).clamp(-1, 1)  # (B,Npad)
        pred_scalar = pred_scalar.numpy()
        pol_np = pol_logits.numpy()
        for b, er in enumerate(chunk):
            L = er.cell_q.shape[0]
            results[i + b] = (
                pred_scalar[b, :L].copy(),
                er.cell_q.astype(np.float64).copy(),
                er.cell_q_mask.astype(bool).copy(),
                pol_np[b, :L].copy(),
                float(er.value),
            )
        i = j

    n_used = 0
    for r in results:
        if r is None:
            continue
        pred_q, tgt_q, mask, pol, value = r
        L = pred_q.shape[0]
        m_idx = np.nonzero(mask)[0]
        if m_idx.shape[0] < 1:
            continue
        n_used += 1
        legal_counts.append(L)
        masked_counts.append(m_idx.shape[0])

        pred_m = pred_q[m_idx]
        tgt_m = tgt_q[m_idx]

        # (a) top-1 pred-Q vs target-Q, restricted to masked cells
        pred_best_local = int(np.argmax(pred_m))
        tgt_best_local = int(np.argmax(tgt_m))
        pred_best_cell = int(m_idx[pred_best_local])
        tgt_best_cell = int(m_idx[tgt_best_local])
        top1_q_agree.append(pred_best_cell == tgt_best_cell)

        # policy argmax over the full legal prefix (policy mass only on legal set)
        pol_best_cell = int(np.argmax(pol))  # over all L legal cells
        # restrict policy argmax to legal prefix (pol is length-L already)
        top1_q_vs_policy.append(pred_best_cell == pol_best_cell)
        top1_tgt_vs_policy.append(tgt_best_cell == pol_best_cell)

        # (c) top-3 containment: target-best within pred top-3 (over masked cells)
        k = min(3, m_idx.shape[0])
        top3_local = np.argsort(-pred_m)[:k]
        top3_cells = set(int(m_idx[t]) for t in top3_local)
        top3_contain.append(tgt_best_cell in top3_cells)

        # (d) rank corr over masked cells
        sp_rhos.append(spearman(pred_m, tgt_m))
        pe_rhos.append(pearson(pred_m, tgt_m))

        # (e) calibration over masked cells
        err = pred_m - tgt_m
        abs_err_all.append(np.abs(err))
        signed_err_all.append(err)

        best_pred_q.append(float(pred_m.max()))
        pos_value.append(value)

    abs_err_cat = np.concatenate(abs_err_all)
    signed_err_cat = np.concatenate(signed_err_all)
    lc = np.array(legal_counts); mc = np.array(masked_counts)
    sp_arr = np.array(sp_rhos); sp_arr = sp_arr[~np.isnan(sp_arr)]
    pe_arr = np.array(pe_rhos); pe_arr = pe_arr[~np.isnan(pe_arr)]
    bpq = np.array(best_pred_q); pv = np.array(pos_value)

    def frac(x): return float(np.mean(x))

    print("==================== cell_q HEAD EVAL ====================")
    print(f"checkpoint            : epoch_000033.pt (meta epoch={meta.get('epoch')})")
    print(f"shard epoch dir       : {SHARD_EPOCH_DIR}")
    print(f"n shards read         : {n_shards_used}")
    print(f"n positions (decisions): {n_used}")
    print(f"n masked cells (total): {int(mc.sum())}")
    print(f"mean legal cells/pos  : {lc.mean():.2f}  (min {lc.min()}, max {lc.max()})")
    print(f"mean masked cells/pos : {mc.mean():.2f}")
    print("--- (a) HEADLINE: top-1 move agreement ---")
    print(f"argmax(pred Q) == argmax(target Q)     : {frac(top1_q_agree):.4f}")
    print(f"  random baseline (1/mean masked)      : {1.0/mc.mean():.4f}")
    print("--- (b) cell_q vs policy (most-visited / played) ---")
    print(f"argmax(pred Q)   == argmax(policy)     : {frac(top1_q_vs_policy):.4f}")
    print(f"argmax(target Q) == argmax(policy) [ceil]: {frac(top1_tgt_vs_policy):.4f}")
    print("--- (c) top-3 containment ---")
    print(f"target-best in pred top-3 Q           : {frac(top3_contain):.4f}")
    print("--- (d) rank correlation (per-pos mean over masked cells) ---")
    print(f"mean Spearman(pred Q, target Q)       : {sp_arr.mean():.4f}  (n={sp_arr.shape[0]})")
    print(f"mean Pearson (pred Q, target Q)       : {pe_arr.mean():.4f}  (n={pe_arr.shape[0]})")
    print("--- (e) calibration / magnitude (over masked cells) ---")
    print(f"MAE(pred Q, target Q)                 : {abs_err_cat.mean():.4f}")
    print(f"signed bias (pred - target)           : {signed_err_cat.mean():+.4f}")
    print(f"target Q  mean/std                    : {np.concatenate([r[1][r[2]] for r in results if r is not None]).mean():.3f} / {np.concatenate([r[1][r[2]] for r in results if r is not None]).std():.3f}")
    print(f"pred   Q  mean/std (masked)           : {np.concatenate([r[0][r[2]] for r in results if r is not None]).mean():.3f} / {np.concatenate([r[0][r[2]] for r in results if r is not None]).std():.3f}")
    print(f"Pearson(best pred Q, position value)  : {pearson(bpq, pv):.4f}")

    # Distribution sense: agreement split by legal-count terciles
    order = np.argsort(mc)
    t = mc.shape[0] // 3
    agree = np.array(top1_q_agree)
    for name, sel in [("small", order[:t]), ("mid", order[t:2*t]), ("large", order[2*t:])]:
        if sel.shape[0]:
            print(f"top-1 agree | {name:5s} masked-count (mean {mc[sel].mean():.1f}): {agree[sel].mean():.4f}")


if __name__ == "__main__":
    main()
