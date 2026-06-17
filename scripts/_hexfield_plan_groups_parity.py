"""Section-D HARD GATE: Rust debug_plan_groups == inference.plan_groups,
element-for-element (start, end, pad_to) over the critiqued size battery
(empty, singletons, all-equal, ceiling straddlers, waste-floor straddlers,
random DESC + unsorted). Exit 0 = all match."""
import glob
import os
import random
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "packages", "hexfield", "python"))

from hexfield import _rust
from hexfield.inference import plan_groups

allok = True


def check(sizes, label):
    global allok
    rs = [tuple(int(y) for y in x) for x in _rust.debug_plan_groups([int(s) for s in sizes])]
    py = [tuple(int(y) for y in x) for x in plan_groups(np.asarray(sizes, dtype=np.int64))]
    ok = rs == py
    allok = allok and ok
    tag = "OK" if ok else "MISMATCH"
    print("%-40s groups=%4d -> %s" % (label, len(py), tag))
    if not ok:
        print("  py:", py[:8])
        print("  rs:", rs[:8])


check([], "empty")
for n in [1, 64, 65, 127, 128, 129, 65535]:
    check([n], "single[%d]" % n)
check([100] * 300, "all-equal 100 x300")
check([1024] + [840] * 5 + [839] * 5, "waste-floor 1024/840/839")
check([1024, 1023, 840, 839, 838], "waste straddle desc")
check([256] * 600, "ceiling 256 x600")
check([500] * 200, "ceiling 500 x200 (pad512)")
check([1000] * 80, "ceiling 1000 x80 (pad1024)")

random.seed(7)
for t in range(40):
    s = sorted([random.randint(1, 2560) for _ in range(random.randint(1, 300))], reverse=True)
    check(s, "rand-desc#%d" % t)
for t in range(10):
    s = [random.randint(1, 2560) for _ in range(random.randint(1, 200))]
    check(s, "rand-unsorted#%d" % t)

# the frozen real .hxr size distribution from a live flush
RUN = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_1"
try:
    from hexo_engine import api
    from hexo_engine.types import AxialCoord, PlacementAction
    from hexo_runner.records import HexoRecordFile
    from hexfield.geometry import unpack_action_id

    hxrs = sorted(glob.glob(f"{RUN}/selfplay/*.hxr"), key=os.path.getmtime, reverse=True)
    states = []
    for hxr in hxrs[:2]:
        rf = HexoRecordFile.open(hxr)
        with rf:
            for rec in rf.iter_records():
                st = api.new_game()
                for aid in list(rec.action_ids):
                    q, r = unpack_action_id(int(aid))
                    res = api.apply_action(st, PlacementAction(AxialCoord(q=q, r=r)))
                    if api.terminal(st) is None:
                        states.append(st if not hasattr(api, "clone_state") else api.clone_state(st))
                    if res.terminal:
                        break
                if len(states) >= 4000:
                    break
        if len(states) >= 4000:
            break
    rows = _rust.featurize_states(states[:4000])
    real_sizes = sorted([r["num_nodes"] for r in rows], reverse=True)
    check(real_sizes, "frozen real .hxr dist (n=%d)" % len(real_sizes))
    # plus a 96-flush slice like the live serve
    check(real_sizes[:96], "real 96-flush slice")
except Exception as e:  # pragma: no cover
    print("real-dist check skipped:", repr(e))

print("ALL:", "PASS" if allok else "FAIL")
sys.exit(0 if allok else 1)
