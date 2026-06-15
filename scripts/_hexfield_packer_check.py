"""CPU-only M8 packer check: shape histogram + padded-cell fraction on a
mid-game mix (validates §5.3 packing economics ahead of the GPU run)."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))

from _hexfield_perf import measure_packer, midgame_states  # noqa: E402

states = midgame_states(300, seed=2)
result = measure_packer(states)
for k, v in result.items():
    print(f"{k}: {v}")
