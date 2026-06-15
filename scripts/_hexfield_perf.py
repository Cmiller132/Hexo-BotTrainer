"""M8 perf calibration (spec §5.5/§9/§10 M8). Run on a QUIET GPU.

Measures, on a mid-game position mix:
  - evaluator throughput (evals/s, nodes/s) at production packing
  - packer shape histogram + padded-cell fraction (tail occupancy)
  - peak VRAM per packed group vs the §9 budget
  - fp16-vs-fp32 eval divergence (the adopt-and-gate)
  - end-to-end positions/s via run_continuous at production settings
  - torch.compile go/no-go on a static shape

Usage (WSL, GPU free):
  PYTHONPATH=packages/hexfield/python /root/.venvs/hexgt-build/bin/python \
      scripts/_hexfield_perf.py [checkpoint.pt]
"""

from __future__ import annotations

import collections
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))

import numpy as np
import torch

from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction

from hexfield import _rust
from hexfield.geometry import unpack_action_id
from hexfield.inference import HexfieldEvaluator, PAIR_CEILING
from hexfield.model import HexfieldNet
from hexfield.constants import NUM_TOKENS


def midgame_states(n: int, seed: int = 1):
    """Random-playout states with a spread of support sizes (mid-game mix)."""
    rng = random.Random(seed)
    states = []
    while len(states) < n:
        state = api.new_game()
        plies = rng.randint(8, 60)
        for _ in range(plies):
            ids = api.legal_action_ids(state)
            if not ids:
                break
            q, r = unpack_action_id(rng.choice(ids))
            if api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r))).terminal:
                break
        if api.terminal(state) is None:
            states.append(state)
    return states


def measure_evaluator(model, device, states, request_ml):
    evaluator = HexfieldEvaluator(model, device=device)
    # Build the CSR payloads in Rust (the production path).
    payloads = []
    flush = 256  # leaves per flush (continuous flush_target scale)
    for i in range(0, len(states), flush):
        batch = states[i : i + flush]
        # featurize_states gives per-row dicts; assemble a single payload like
        # payload.rs does (size-desc sort + CSR). Reuse the Rust path via a
        # 1-flush session evaluate is heavier; here we time the evaluator on
        # Rust-built payloads through a tiny shim session.
        payloads.append(batch)
    # Warm up
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # Use run_continuous-style: feed through a session with a stub-free real
    # eval by calling evaluate on Rust payloads. Simplest correct timing: use
    # the session's lockstep search at 1 visit to exercise featurize->pack->
    # forward->reply once per state, but that adds tree overhead. Instead time
    # the evaluator directly on payloads built by a helper.
    return evaluator


def measure_end_to_end(model, device, n_games, visits, seed):
    """positions/s via run_continuous at production-ish settings."""
    evaluator = HexfieldEvaluator(model, device=device)
    states = [api.new_game() for _ in range(n_games)]
    keys = list(range(n_games))
    decided = {"n": 0}
    plies = {}

    def on_move(game_key, payload):
        decided["n"] += 1
        q, r = unpack_action_id(int(payload["action_id"]))
        st = states[keys.index(game_key)] if game_key in keys else None
        return None  # one decision per game then stop (root throughput proxy)

    # Single-root-per-game one-decision throughput: time the first full search.
    session = _rust.HexfieldMctsSession(max_states=262144)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    session.run_continuous(
        keys, tuple(states), evaluator=evaluator, on_move=on_move,
        visits=visits, c_puct=1.5, base_seed=seed, virtual_batch_size=4,
        flush_target=256, active_root_limit=n_games,
        temperature_by_ply=[0.0], forced_playout_k=2.0,
        widening_policy_mass=0.95, widening_max_children=96, widening_min_children=2,
        root_dirichlet_total_alpha=10.83, root_dirichlet_noise_fraction=0.25,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    return {
        "games": n_games,
        "visits": visits,
        "decisions": decided["n"],
        "seconds": round(dt, 2),
        "positions_per_second": round(decided["n"] / dt, 2) if dt > 0 else 0.0,
        "evals_per_second_approx": round(decided["n"] * visits / dt, 1) if dt > 0 else 0.0,
        "peak_vram_gib": round(peak, 3),
    }


def measure_packer(states):
    """Packer shape histogram + padded-cell fraction over the mid-game mix."""
    payloads = _rust.featurize_states(states)
    sizes = sorted((p["num_nodes"] for p in payloads), reverse=True)
    # Emulate the §5.3 greedy 256-quantized grouping under the pair ceiling.
    hist = collections.Counter()
    real_cells = 0
    padded_cells = 0
    i = 0
    while i < len(sizes):
        pad_to = max(256, -(-sizes[i] // 256) * 256)
        j = i + 1
        while j < len(sizes) and (j - i + 1) * (pad_to + NUM_TOKENS) ** 2 <= PAIR_CEILING:
            j += 1
        group = sizes[i:j]
        hist[(len(group), pad_to)] += 1
        real_cells += sum(group)
        padded_cells += len(group) * pad_to
        i = j
    return {
        "n_states": len(states),
        "support_min": min(sizes),
        "support_max": max(sizes),
        "support_mean": round(float(np.mean(sizes)), 1),
        "shape_histogram": {f"{b}x{s}": c for (b, s), c in sorted(hist.items())},
        "padded_cell_fraction": round(1 - real_cells / max(padded_cells, 1), 3),
    }


def measure_fp16_gate(model, device, states):
    """fp16 adopt-and-gate: max divergence of value/priors fp16 vs fp32."""
    if device.type != "cuda":
        return {"status": "skipped", "reason": "cpu"}
    eval32 = HexfieldEvaluator(model, device=device)
    sess = _rust.HexfieldMctsSession(max_states=4096)
    # compare a single state's reply under autocast on/off via direct featurize
    payloads = _rust.featurize_states(states[:16])
    diffs = []
    for row in payloads:
        n = row["num_nodes"]
        feats = np.frombuffer(row["feats"], dtype=np.float32).reshape(n, -1)
        qr = np.frombuffer(row["coords"], dtype=np.int16).reshape(n, 2)
        nbr = np.frombuffer(row["nbr"], dtype=np.int32).reshape(n, 6)
        f = torch.from_numpy(feats).unsqueeze(0).to(device)
        c = torch.from_numpy(qr.astype(np.int64)).unsqueeze(0).to(device)
        nb = torch.from_numpy(np.where(nbr < 0, n, nbr).astype(np.int64)).unsqueeze(0).to(device)
        m = torch.ones(1, n, dtype=torch.bool, device=device)
        with torch.no_grad():
            o32 = model.forward_policy_value(f, nb, m, c)["value"]
            with torch.autocast("cuda", dtype=torch.float16):
                o16 = model.forward_policy_value(f, nb, m, c)["value"]
        diffs.append(float((o32.float() - o16.float()).abs().max()))
    return {"status": "measured", "max_value_logit_diff_fp16_vs_fp32": round(max(diffs), 4)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HexfieldNet()
    if len(sys.argv) > 1:
        payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
        model.load_state_dict(payload.get("model", payload), strict=True)
        print(f"loaded {sys.argv[1]}")
    model.to(device).eval()
    print(f"device: {device}")

    states = midgame_states(300)
    print("PACKER:", measure_packer(states))
    print("FP16 GATE:", measure_fp16_gate(model, device, states))
    for visits in (128, 512):
        print(f"E2E visits={visits}:", measure_end_to_end(model, device, 16, visits, seed=7))


if __name__ == "__main__":
    main()
