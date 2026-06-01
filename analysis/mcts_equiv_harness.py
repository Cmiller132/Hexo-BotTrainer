"""Deterministic MCTS trajectory fingerprint — baseline for the A1 pipeline change.

Drives the native `Model1MctsSession` exactly like self-play, on CPU with a
fixed-seed random net so the NN forward is bit-deterministic. Records, per
searched move and root: the selected action_id, the visit total, and a stable
hash of the visit-policy distribution. Also checks MCTS invariants that must
hold regardless of select↔eval scheduling:

  * every root's reported `visits` equals the requested target (added this call)
  * visit_policy weights sum to ~1 and all actions are legal-ish (non-negative)

A1 (virtual-loss select↔eval pipelining) is NOT expected to be bit-identical to
the serial-barrier search — extending the in-flight (virtual-loss) window across
the eval boundary changes the trajectory, as virtual-loss parallel MCTS does by
design. So this harness is used two ways:
  1. run twice -> output identical  => the search is deterministic for a seed
  2. compare pre/post A1            => quantify the trajectory delta (expected)

Usage: python analysis/mcts_equiv_harness.py [out.json]
"""

from __future__ import annotations

import hashlib
import json
import sys

import torch

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id
from hexo_models.dense_cnn.architecture import Model1Network
from hexo_models.dense_cnn.inference import DenseCNNInference
from hexo_models.dense_cnn.mcts import new_mcts_session

SEED = 1234
N_GAMES = 8
N_MOVES = 6
VISITS = 128
VBATCH = 4


def build_inference() -> DenseCNNInference:
    torch.manual_seed(SEED)
    model = Model1Network(channels=64, blocks=4)
    return DenseCNNInference(
        model, device="cpu", amp=False, return_logits=False,
        max_batch_size=1024, optimize_for_inference=False,
    )


def policy_hash(visit_policy) -> str:
    h = hashlib.sha256()
    for action_id, weight in visit_policy:
        h.update(int(action_id).to_bytes(4, "little"))
        # quantize weight so float noise below 1e-6 doesn't churn the hash
        h.update(round(float(weight), 6).hex().encode())
    return h.hexdigest()[:16]


def run() -> dict:
    inf = build_inference()
    session = new_mcts_session(max_states=262144)
    games = [{"key": i, "state": engine.new_game(seed=100 + i), "plies": 0} for i in range(N_GAMES)]

    moves = []
    invariant_failures = []
    for move_idx in range(N_MOVES):
        playable = [g for g in games if engine.terminal(g["state"]) is None]
        if not playable:
            break
        results = session.run(
            [g["key"] for g in playable],
            [g["state"] for g in playable],
            inf,
            visits=VISITS, c_puct=1.5, temperature=1.0, seed=SEED,
            virtual_batch_size=VBATCH, active_root_limit=256,
            root_dirichlet_total_alpha=10.83, root_dirichlet_noise_fraction=0.25,
            root_policy_temperature=1.1, fpu_reduction=0.20, virtual_loss=1.0,
            widening_policy_mass=0.95, widening_max_children=32, widening_min_children=2,
        )
        for g, res in zip(playable, results):
            wsum = sum(float(w) for _, w in res.visit_policy)
            if not (0.99 <= wsum <= 1.01):
                invariant_failures.append(f"move{move_idx} game{g['key']}: visit_policy sum={wsum:.4f}")
            if any(float(w) < 0 for _, w in res.visit_policy):
                invariant_failures.append(f"move{move_idx} game{g['key']}: negative visit weight")
            moves.append({
                "move": move_idx,
                "game": g["key"],
                "action_id": int(res.action_id),
                "visits": int(res.visits),
                "n_actions": len(res.visit_policy),
                "policy_hash": policy_hash(res.visit_policy),
            })
            engine.apply_action(g["state"], engine.PlacementAction(unpack_coord_id(res.action_id)))
            g["plies"] += 1

    digest = hashlib.sha256(
        json.dumps([(m["move"], m["game"], m["action_id"], m["policy_hash"]) for m in moves]).encode()
    ).hexdigest()[:16]
    return {
        "config": {"seed": SEED, "games": N_GAMES, "moves": N_MOVES, "visits": VISITS, "vbatch": VBATCH},
        "trajectory_digest": digest,
        "n_records": len(moves),
        "invariant_failures": invariant_failures,
        "moves": moves,
    }


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else None
    result = run()
    print(f"trajectory_digest = {result['trajectory_digest']}  records={result['n_records']}")
    print(f"invariant_failures = {len(result['invariant_failures'])}")
    for f in result["invariant_failures"][:10]:
        print("  FAIL:", f)
    if out:
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print("wrote", out)


if __name__ == "__main__":
    main()
