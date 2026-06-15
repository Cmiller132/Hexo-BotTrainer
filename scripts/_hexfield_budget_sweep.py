"""Fresh-search budget sweep: discard sessions between calls; compare exports
at every visit budget. Divergence here => fresh-selection bug; clean sweep =>
the reuse/advance path carries the bug."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np

from hexfield_testkit import api
from hexo_engine.types import AxialCoord, PlacementAction
from test_hexfield_search_parity import DenseStub, HexfieldStub

from hexfield import _rust as hexfield_rust
from hexo_models._rust import dense_cnn as dense_rust

state = api.new_game()
for q, r in [(0, 0), (1, 1), (-1, 0)]:
    api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))

KW = dict(c_puct=1.5, temperature=0.0, seed=99, virtual_batch_size=1,
          widening_policy_mass=0.95, widening_max_children=96,
          widening_min_children=2, fpu_reduction=0.2, tss_enabled=False)

for visits in range(2, 80):
    hs = hexfield_rust.HexfieldMctsSession(max_states=65536)
    ds = dense_rust.Model1MctsSession(65536)
    h = hs.search([1], (state,), visits=visits, evaluator=HexfieldStub(),
                  search_parity_mode=True, **KW)[0]
    d = ds.search([1], (state,), visits=visits, evaluator=DenseStub(), **KW)[0]
    same = (
        h["action_id"] == d["action_id"]
        and h["visits"] == d["visits"]
        and bytes(h["visit_policy_action_ids_bytes"]) == bytes(d["visit_policy_action_ids_bytes"])
        and bytes(h["visit_policy_weights_bytes"]) == bytes(d["visit_policy_weights_bytes"])
        and abs(h["root_value"] - d["root_value"]) < 1e-7
    )
    if not same:
        print(f"FRESH DIVERGENCE at visits={visits}")
        print(" hex  :", h["action_id"], h["visits"], round(h["root_value"], 6))
        print(" dense:", d["action_id"], d["visits"], round(d["root_value"], 6))
        hi = np.frombuffer(bytes(h["visit_policy_action_ids_bytes"]), dtype=np.uint32)
        di = np.frombuffer(bytes(d["visit_policy_action_ids_bytes"]), dtype=np.uint32)
        hw = np.frombuffer(bytes(h["visit_policy_weights_bytes"]), dtype=np.float32)
        dw = np.frombuffer(bytes(d["visit_policy_weights_bytes"]), dtype=np.float32)
        print(" hex ids/w :", list(zip(hi.tolist(), np.round(hw, 4).tolist())))
        print(" dense ids/w:", list(zip(di.tolist(), np.round(dw, 4).tolist())))
        sys.exit(0)
print("fresh searches identical for all budgets 2..79 -> bug is in reuse/advance")
