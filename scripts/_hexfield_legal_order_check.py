"""Pin the engine legal-move ordering (the differential stub keys priors by
rank-in-row; dense's payload order must equal hexfield's ascending prefix)."""

import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))

from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction

from hexfield.geometry import unpack_action_id

rng = random.Random(3)
bad = 0
for game in range(8):
    state = api.new_game()
    for ply in range(40):
        ids = api.legal_action_ids(state)
        if not ids:
            break
        if list(ids) != sorted(ids):
            bad += 1
        q, r = unpack_action_id(rng.choice(ids))
        if api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r))).terminal:
            break
print("rows out of ascending order:", bad)
