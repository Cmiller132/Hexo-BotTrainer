"""Read-only smoke: re-fit the LIVE pool through the modified Stage-D path to
confirm the E2/E3/E4 changes do not break against real edges. Plays NO games,
mutates NOTHING (loads a COPY of the pool doc, append=False, write_diagnostics
not used). The live run dir is read-only.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from hexfield.config import parse_hexfield_config
from hexfield.multistage_eval import _stage_d_pool, select_opponents

RUN = Path("/mnt/e/Hexo-BotTrainer/runs/hexfield_main_2")
POOL = RUN / "diagnostics" / "eval_pool.json"


def main():
    assert POOL.is_file(), f"live pool missing: {POOL}"
    cfg = parse_hexfield_config({}).multi_stage_eval
    pool_doc = json.loads(POOL.read_text(encoding="utf-8"))
    pool_copy = copy.deepcopy(pool_doc)

    # Resolve a roster as the live eval would for ep35 (no games, pure path res).
    cand = RUN / "checkpoints" / "epoch_000035.pt"
    roster = select_opponents(RUN, cand, cfg, candidate_epoch=35)
    print("roster opponents:", [o.label for o in roster.opponents])
    print("dropped_anchors:", roster.dropped_anchors)

    stage_d, ratings, verdict, _ = _stage_d_pool(
        cfg, roster, [], RUN, pool_doc=pool_copy, append=False
    )
    print("stage_d.status:", stage_d.status)
    print("anchor:", ratings["fit"].get("anchor"))
    print("ood_opponents:", ratings["fit"].get("ood_opponents"))
    print("verdict.label:", verdict.get("label"))
    print("verdict.degraded:", verdict.get("degraded"))
    print("verdict.ood_opponents:", verdict.get("ood_opponents"))
    # The original pool file must be untouched on disk.
    after = json.loads(POOL.read_text(encoding="utf-8"))
    assert after == pool_doc, "LIVE POOL WAS MUTATED — must be read-only!"
    print("LIVE-READ SMOKE OK (pool unchanged on disk)")


if __name__ == "__main__":
    main()
