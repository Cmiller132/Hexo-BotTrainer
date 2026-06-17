"""One-off: assert incremental-writer shards == engine-mirror shards (M3)."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))

from hexfield.shards import read_compact_shard

for split in ("train", "val"):
    for p in sorted((REPO / "data" / "hexfield_bootstrap_smoke" / split).glob("*.npz")):
        a = read_compact_shard(p)
        b = read_compact_shard(REPO / "data" / "hexfield_bootstrap_smoke2" / split / p.name)
        assert len(a) == len(b), (p, len(a), len(b))
        for ra, rb in zip(a, b):
            assert ra == rb, (p, ra.turn_index)
print("identical: incremental facts == engine-mirror facts on all smoke rows")
