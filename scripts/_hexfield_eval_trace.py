"""Trace-level diff: log every evaluator request's per-row legal counts on
both sides across step0 (fresh) + step1 (reused). First differing entry =
first diverging selection."""

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


class TracingHexStub(HexfieldStub):
    def __init__(self):
        self.trace = []

    def __call__(self, payload):
        ls = np.frombuffer(payload["legal_counts"], dtype=np.int32)
        self.trace.append(tuple(int(x) for x in ls))
        return super().__call__(payload)


class TracingDenseStub(DenseStub):
    def __init__(self):
        self.trace = []

    def __call__(self, payload):
        off = payload["legal_row_offsets"]
        self.trace.append(tuple(int(off[i + 1]) - int(off[i]) for i in range(len(off) - 1)))
        return super().__call__(payload)


def drive(session, stub, parity):
    state = api.new_game()
    for q, r in [(0, 0), (1, 1), (-1, 0)]:
        api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
    marks = []
    for step in range(2):
        kw = dict(visits=16 if step == 0 else 48, c_puct=1.5, temperature=0.0,
                  seed=99 + step, evaluator=stub, virtual_batch_size=1,
                  widening_policy_mass=0.95, widening_max_children=96,
                  widening_min_children=2, fpu_reduction=0.2, tss_enabled=False)
        if parity:
            kw["search_parity_mode"] = True
        r = session.search([42], (state,), **kw)[0]
        marks.append(len(stub.trace))
        q, rr = unpack_action_id(int(r["action_id"]))
        api.apply_action(state, PlacementAction(AxialCoord(q=q, r=rr)))
    return marks


hex_stub = TracingHexStub()
dense_stub = TracingDenseStub()
hex_marks = drive(hexfield_rust.HexfieldMctsSession(max_states=65536), hex_stub, True)
dense_marks = drive(dense_rust.Model1MctsSession(65536), dense_stub, False)

print("flush counts: hex", hex_marks, len(hex_stub.trace), "| dense", dense_marks, len(dense_stub.trace))
n = min(len(hex_stub.trace), len(dense_stub.trace))
for i in range(n):
    if hex_stub.trace[i] != dense_stub.trace[i]:
        step = "step0" if i < hex_marks[0] else "step1"
        print(f"FIRST TRACE DIVERGENCE at flush {i} ({step})")
        print(" hex  :", hex_stub.trace[i])
        print(" dense:", dense_stub.trace[i])
        print(" context hex  :", hex_stub.trace[max(0, i - 3):i])
        print(" context dense:", dense_stub.trace[max(0, i - 3):i])
        sys.exit(0)
if len(hex_stub.trace) != len(dense_stub.trace):
    print(f"trace LENGTH differs: hex {len(hex_stub.trace)} dense {len(dense_stub.trace)}")
    longer, name = (hex_stub.trace, "hex") if len(hex_stub.trace) > n else (dense_stub.trace, "dense")
    print(f" extra {name} flushes:", longer[n:n+5])
else:
    print("eval traces identical")
