"""Sanity: the search session imports and mix_seed matches dense's golden vector."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))

from hexfield import _rust

print("session:", _rust.HexfieldMctsSession)
# dense mcts.rs test golden: mix_seed(123_456_789, 987_654_321, 42, 1)
print("mix_seed(123456789, 987654321, 42, 1) =", hex(_rust.mix_seed(123456789, 987654321, 42, 1)))
print("mix_seed(0,0,0,0) =", hex(_rust.mix_seed(0, 0, 0, 0)))
print("mix_seed(1,2,3,4) =", hex(_rust.mix_seed(1, 2, 3, 4)))
session = _rust.HexfieldMctsSession(max_states=1024)
print("session len:", session.len())
