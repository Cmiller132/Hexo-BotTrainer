"""Minimal reuse repro: two consecutive lockstep searches on the same session
(promoted subtree reuse), hexfield vs dense, greedy, no exploration."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np

from hexfield_testkit import api
from hexo_engine.types import AxialCoord, PlacementAction
from hexfield.geometry import unpack_action_id
from test_hexfield_search_parity import DenseStub, HexfieldStub

from hexfield import _rust as hexfield_rust
from hexo_models._rust import dense_cnn as dense_rust


VB = int(sys.argv[1]) if len(sys.argv) > 1 else 8
TSS = (sys.argv[2] != "notss") if len(sys.argv) > 2 else True
FPU = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
VLOSS = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0


def drive(session, stub, label, extra):
    state = api.new_game()
    for q, r in [(0, 0), (1, 1), (-1, 0)]:
        api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
    records = []
    for step in range(6):
        results = session.search(
            [42], (state,), visits=16 if step == 0 else 48, c_puct=1.5,
            temperature=0.0, seed=99 + step, evaluator=stub,
            virtual_batch_size=VB, widening_policy_mass=0.95,
            widening_max_children=96, widening_min_children=2,
            fpu_reduction=FPU, virtual_loss=VLOSS, tss_enabled=TSS, **extra,
        )
        r = results[0]
        ids = np.frombuffer(bytes(r["visit_policy_action_ids_bytes"]), dtype=np.uint32)
        w = np.frombuffer(bytes(r["visit_policy_weights_bytes"]), dtype=np.float32)
        prior_ids = np.frombuffer(bytes(r["root_prior_policy_action_ids_bytes"]), dtype=np.uint32)
        prior_w = np.frombuffer(bytes(r["root_prior_policy_weights_bytes"]), dtype=np.float32)
        records.append(
            (step, int(r["action_id"]), int(r["visits"]), ids.tolist(),
             [round(float(x), 6) for x in w], round(float(r["root_value"]), 6),
             len(prior_ids), round(float(prior_w.max()), 7), round(float(prior_w.min()), 9),
             int(prior_ids[int(prior_w.argmax())]))
        )
        q, rr = unpack_action_id(int(r["action_id"]))
        res = api.apply_action(state, PlacementAction(AxialCoord(q=q, r=rr)))
        if res.terminal:
            break
    return records


hex_records = drive(
    hexfield_rust.HexfieldMctsSession(max_states=65536), HexfieldStub(), "hex",
    dict(search_parity_mode=True),
)
dense_records = drive(dense_rust.Model1MctsSession(65536), DenseStub(), "dense", {})

for h, d in zip(hex_records, dense_records):
    if h != d:
        print(f"DIVERGES at step {h[0]}")
        print(" hex  :", h[1], h[2], "edges:", len(h[3]), h[5])
        print(" dense:", d[1], d[2], "edges:", len(d[3]), d[5])
        print(" hex ids  :", h[3])
        print(" dense ids:", d[3])
        print(" hex w  :", h[4])
        print(" dense w:", d[4])
        print(" hex prior (count, max, min, argmax):", h[6:])
        print(" dense prior (count, max, min, argmax):", d[6:])
        sys.exit(0)
print("no divergence over", len(hex_records), "reused searches")
