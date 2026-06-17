"""Print finished-game length stats for one or more epoch .hxr files.
Usage: python _hf_glen.py RUN_DIR EPOCH [EPOCH ...]"""
import sys
from pathlib import Path

sys.path.insert(0, "packages/dense_cnn_restnet/python")
import numpy as np
from hexo_utils.records import HexoRecordFile

run = Path(sys.argv[1])
for ep in sys.argv[2:]:
    p = run / "selfplay" / f"epoch_{int(ep):06d}.hxr"
    if not p.is_file():
        print(f"ep{ep}: (no .hxr)")
        continue
    try:
        with HexoRecordFile.open(p) as rf:
            lens = [len(r.action_ids) for r in rf.iter_records()]
    except Exception as e:
        print(f"ep{ep}: read error {e!r}")
        continue
    if not lens:
        print(f"ep{ep}: 0 finished games")
        continue
    a = np.array(lens)
    print(f"ep{ep}: n={len(a)} len min/med/mean/p90/max = "
          f"{a.min()}/{int(np.median(a))}/{a.mean():.1f}/{int(np.percentile(a,90))}/{a.max()}")
