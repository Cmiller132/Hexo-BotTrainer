"""Probe the promoted-root carry: dump hexfield's stored tree after step 0,
and after step 1 compute what carry dense's root_value implies."""

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


def setup():
    state = api.new_game()
    for q, r in [(0, 0), (1, 1), (-1, 0)]:
        api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
    return state


KW = dict(c_puct=1.5, temperature=0.0, virtual_batch_size=1,
          widening_policy_mass=0.95, widening_max_children=96,
          widening_min_children=2, fpu_reduction=0.2, tss_enabled=False)

# --- hexfield with dumps ---
hs = hexfield_rust.HexfieldMctsSession(max_states=65536)
state = setup()
r0 = hs.search([42], (state,), visits=16, seed=99, evaluator=HexfieldStub(),
               search_parity_mode=True, **KW)[0]
dump0 = hs.debug_dump(42)
root0 = dump0["nodes"][0]
print("hex step0: action", r0["action_id"], "root_value", round(r0["root_value"], 6))
print("hex promoted root: visits", root0["visits"], "value_sum", round(root0["value_sum"], 5),
      "completed", dump0["completed"], "edges", len(root0["edges"]))

q, rr = unpack_action_id(int(r0["action_id"]))
api.apply_action(state, PlacementAction(AxialCoord(q=q, r=rr)))
r1 = hs.search([42], (state,), visits=48, seed=100, evaluator=HexfieldStub(),
               search_parity_mode=True, **KW)[0]
dump1 = hs.debug_dump(42)
print("hex step1: action", r1["action_id"], "root_value", round(r1["root_value"], 6))

# --- dense ---
ds = dense_rust.Model1MctsSession(65536)
state = setup()
d0 = ds.search([42], (state,), visits=16, seed=99, evaluator=DenseStub(), **KW)[0]
print("dense step0: action", d0["action_id"], "root_value", round(d0["root_value"], 6))
q, rr = unpack_action_id(int(d0["action_id"]))
api.apply_action(state, PlacementAction(AxialCoord(q=q, r=rr)))
d1 = ds.search([42], (state,), visits=48, seed=100, evaluator=DenseStub(), **KW)[0]
print("dense step1: action", d1["action_id"], "root_value", round(d1["root_value"], 6))

# Implied: root_value1 = (S0 + Sigma_new) / V_final. hex knows S0/V0 exactly;
# if Sigma_new is shared, dense's S0 follows.
V0 = root0["visits"]
S0 = root0["value_sum"]
# hex final root after step1:
rootf = dump1["nodes"][0] if dump1 else None
if rootf:
    print("hex final root: visits", rootf["visits"], "value_sum", round(rootf["value_sum"], 5))
    print("hex step1 visits-field:", r1["visits"], " completed/target:",
          dump1["completed"], dump1["target"])
    print("hex step0 carried edges:", [(e[0], e[1]) for e in root0["edges"]])
    print("hex final  root edges:", [(e[0], e[1]) for e in rootf["edges"]])
    print("dense step1 visits-field:", d1["visits"])