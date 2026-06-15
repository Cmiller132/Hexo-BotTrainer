"""Diagnose the continuous-parity divergence: capture per-record diagnostics
from both sides and print the first differing record with context."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))
sys.path.insert(0, str(REPO / "packages" / "dense_cnn_restnet" / "python"))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np

from hexfield_testkit import api
from hexo_engine.types import AxialCoord, PlacementAction
from hexfield.geometry import unpack_action_id
from test_hexfield_search_parity import DenseStub, HexfieldStub
from test_hexfield_continuous_parity import _continuous_kwargs, MAX_PLIES

from hexfield import _rust as hexfield_rust
from hexo_models._rust import dense_cnn as dense_rust


class Recorder:
    def __init__(self):
        self.states = {}
        self.records = []

    def add_game(self, key):
        self.states[key] = api.new_game()

    def __call__(self, game_key, payload):
        state = self.states[game_key]
        action_id = payload["action_id"]
        deltas = np.frombuffer(bytes(payload["visit_policy_weights_bytes"]), dtype=np.float32)
        prior_w = np.frombuffer(bytes(payload["root_prior_policy_weights_bytes"]), dtype=np.float32)
        self.records.append(
            dict(
                game=game_key,
                action=action_id,
                pcr_full=bool(payload["pcr_full"]),
                init=bool(payload["policy_init"]),
                visits=int(payload["visits"]),
                ids=bytes(payload["visit_policy_action_ids_bytes"]),
                weights=tuple(round(float(w), 6) for w in deltas),
                prior_ids=bytes(payload["root_prior_policy_action_ids_bytes"]),
                prior_w=tuple(round(float(w), 7) for w in prior_w),
                root_value=round(float(payload["root_value"]), 6),
            )
        )
        q, r = unpack_action_id(action_id)
        result = api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
        ply = sum(1 for rec in self.records if rec["game"] == game_key)
        if result.terminal or ply >= MAX_PLIES:
            return None
        return ("advance", state)


VARIANTS = {
    # pure subtree reuse, greedy, no exploration machinery
    "A": dict(pcr_full_proportion=1.0, policy_init_fraction=0.0,
              root_dirichlet_total_alpha=None, root_dirichlet_noise_fraction=None,
              forced_playout_k=0.0, root_policy_temperature=1.0,
              root_policy_temperature_early=0.0, root_policy_temperature_halflife=0.0,
              temperature_by_ply=[0.0]),
    # reuse + noise + ramp + forced playouts, no PCR/policy-init
    "B": dict(pcr_full_proportion=1.0, policy_init_fraction=0.0),
    # PCR fast/full mix, no noise/ramp/forced
    "D": dict(policy_init_fraction=0.0,
              root_dirichlet_total_alpha=None, root_dirichlet_noise_fraction=None,
              forced_playout_k=0.0, root_policy_temperature=1.0,
              root_policy_temperature_early=0.0, root_policy_temperature_halflife=0.0),
    # everything (the failing baseline)
    "C": dict(),
}

variant = sys.argv[1] if len(sys.argv) > 1 else "C"
kwargs = {**_continuous_kwargs(), **VARIANTS[variant]}
if kwargs.get("root_dirichlet_total_alpha") is None:
    kwargs.pop("root_dirichlet_total_alpha", None)
    kwargs.pop("root_dirichlet_noise_fraction", None)
print(f"variant {variant}")

keys = list(range(700, 706))
hex_rec, dense_rec = Recorder(), Recorder()
for k in keys:
    hex_rec.add_game(k)
    dense_rec.add_game(k)

hexfield_rust.HexfieldMctsSession(max_states=65536).run_continuous(
    keys, tuple(hex_rec.states[k] for k in keys), evaluator=HexfieldStub(),
    on_move=hex_rec, search_parity_mode=True, **kwargs,
)
dense_rust.Model1MctsSession(65536).run_continuous(
    keys, tuple(dense_rec.states[k] for k in keys), evaluator=DenseStub(),
    on_move=dense_rec, **kwargs,
)

print(f"records: hex={len(hex_rec.records)} dense={len(dense_rec.records)}")
for i, (h, d) in enumerate(zip(hex_rec.records, dense_rec.records)):
    if h != d:
        print(f"FIRST DIVERGENCE at record {i}")
        for key in h:
            if h[key] != d[key]:
                print(f"  {key}: hex={h[key]}  dense={d[key]}")
        # context: previous records for the same game
        game = h["game"]
        prev = [(j, r) for j, r in enumerate(hex_rec.records[:i]) if r["game"] == game]
        print(f"  game {game} prior decisions: {[(j, r['action'], r['pcr_full'], r['init'], r['visits']) for j, r in prev]}")
        dprev = [(j, r) for j, r in enumerate(dense_rec.records[:i]) if r["game"] == game]
        print(f"  dense same: {[(j, r['action'], r['pcr_full'], r['init'], r['visits']) for j, r in dprev]}")
        break
else:
    print("no divergence")
