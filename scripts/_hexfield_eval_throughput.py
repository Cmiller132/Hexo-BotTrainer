"""M8 evaluator throughput on a MID-GAME support mix (the honest serve cost).

Builds the §5.2 CSR payload from the Rust featurizer over real mid-game states
and times HexfieldEvaluator.evaluate_payload (featurize is already done; this
isolates pack -> forward -> softmax -> reply), reporting evals/s, nodes/s, and
peak VRAM at production flush sizes. Run on a quiet GPU."""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))

import numpy as np
import torch

from hexfield import _rust
from hexfield.inference import HexfieldEvaluator
from hexfield.model import HexfieldNet
from _hexfield_perf import midgame_states


def build_payload(states, request_ml: bool) -> dict:
    """Assemble the §5.2 CSR payload (size-desc sort) from featurize_states."""
    rows = _rust.featurize_states(states)
    rows = sorted(enumerate(rows), key=lambda ir: (-ir[1]["num_nodes"], ir[0]))
    feats, qr, nbr, offsets, legal = [], [], [], [0], []
    for _, r in rows:
        n = r["num_nodes"]
        feats.append(np.frombuffer(r["feats"], dtype=np.float32).astype(np.float16))
        qr.append(np.frombuffer(r["coords"], dtype=np.int16))
        nb = np.frombuffer(r["nbr"], dtype=np.int32)
        nb = np.where(nb < 0, 0xFFFF, nb).astype(np.uint16)
        nbr.append(nb)
        offsets.append(offsets[-1] + n)
        legal.append(r["legal_count"])
    return {
        "abi": 1,
        "shape": (len(rows), offsets[-1]),
        "node_feats": np.concatenate(feats).tobytes(),
        "node_qr": np.concatenate(qr).tobytes(),
        "nbr": np.concatenate(nbr).tobytes(),
        "node_row_offsets": offsets,
        "legal_counts": np.asarray(legal, dtype=np.int32).tobytes(),
        "request_moves_left": request_ml,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HexfieldNet()
    if len(sys.argv) > 1:
        p = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
        model.load_state_dict(p.get("model", p), strict=True)
    evaluator = HexfieldEvaluator(model, device=device)

    states = midgame_states(512, seed=11)
    flush = 256
    payloads = [build_payload(states[i:i + flush], request_ml=True)
                for i in range(0, len(states), flush)]
    total_nodes = sum(p["shape"][1] for p in payloads)
    total_evals = len(states)

    # warm up
    evaluator.evaluate_payload(payloads[0])
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    reps = 3
    for _ in range(reps):
        for pl in payloads:
            evaluator.evaluate_payload(pl)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / reps
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0

    sizes = [s for pl in payloads for s in
             np.diff(pl["node_row_offsets"])]
    print(f"device: {device}")
    print(f"states: {total_evals}  support mean: {np.mean(sizes):.0f}  "
          f"min/max: {min(sizes)}/{max(sizes)}")
    print(f"flush seconds (mean of {reps}): {dt:.3f}")
    print(f"EVALS/S: {total_evals / dt:.0f}   NODES/S: {total_nodes / dt:.3e}")
    print(f"peak VRAM: {peak:.3f} GiB  (request_moves_left=True)")
    # implied positions/s at production visits (pos/s ~ evals_per_s / visits,
    # the eval-bound regime; ignores tree/CPU select overlap)
    eps = total_evals / dt
    for v in (128, 512):
        print(f"  implied eval-bound pos/s @ {v} visits: {eps / v:.1f}")


if __name__ == "__main__":
    main()
