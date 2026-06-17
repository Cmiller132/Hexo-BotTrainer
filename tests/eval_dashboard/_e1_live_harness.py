"""E1 LIVE GPU harness -- prove the REAL concurrent eval path writes non-empty .hxr.

Drives the ACTUAL play_multi_checkpoint_match on the GPU with a TINY budget
(2 games per opponent, 16 visits) against the live checkpoints, exactly as the
run does (multistage_eval.py:2391), then DECODES the produced .hxr and asserts
num_records > 0 with sane action_ids. SCRATCH-ONLY OUTPUT.
"""
from __future__ import annotations

import glob
import shutil
import sys
from pathlib import Path

from hexo_runner.records import HexoRecordFile

from hexfield.config import parse_hexfield_config
from hexfield.eval_arena import play_multi_checkpoint_match

RUN = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_2"
CANDIDATE = f"{RUN}/checkpoints/epoch_000040.pt"
OPPONENT = f"{RUN}/checkpoints/epoch_000005.pt"

SCRATCH = Path("/mnt/e/hexgt-evaldash/scratch/e1_live")
DIAG = SCRATCH / "run" / "diagnostics"


def main() -> int:
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    DIAG.mkdir(parents=True, exist_ok=True)

    cfg = parse_hexfield_config({})

    print(f"[harness] candidate={CANDIDATE}")
    print(f"[harness] opponent ={OPPONENT}")
    print("[harness] running play_multi_checkpoint_match (2 games, 16 visits)...")
    matches = play_multi_checkpoint_match(
        CANDIDATE,
        [("ep5", OPPONENT)],
        2,
        config=cfg,
        candidate_label="cand_ep40",
        visits=16,
        virtual_batch_size=4,
        diagnostics_dir=str(DIAG),
    )

    mr = matches.get("ep5")
    assert mr is not None, f"no match result for ep5; got keys={list(matches)}"
    meta = mr.get("meta", {})
    print("[harness] match meta: games_requested=" + str(meta.get("games_requested")) + " score=" + str(mr.get("score")))

    hxr_files = sorted(glob.glob(str(SCRATCH / "run" / "evaluation" / "**" / "*.hxr"),
                                 recursive=True))
    assert hxr_files, f"NO .hxr written under {SCRATCH}/run/evaluation"
    print(f"[harness] found {len(hxr_files)} .hxr file(s):")

    total_records = 0
    total_actions = 0
    for p in hxr_files:
        rf = HexoRecordFile.open(p)
        recs = list(rf.iter_records())
        total_records += len(recs)
        print(f"  {p}: num_records={len(recs)}")
        for rec in recs:
            n_act = len(list(rec.action_ids))
            total_actions += n_act
            print(f"     game_id={rec.game_id} status={rec.status} winner={rec.winner} actions={n_act}")
            assert n_act > 0, f"record {rec.game_id} has 0 actions (empty replay!)"

    assert total_records > 0, "ZERO records across all .hxr -- empty-replay bug NOT fixed"
    print(f"\n[harness] PASS: total_records={total_records} total_actions={total_actions}")
    print(f"E1_LIVE_RECORDS={total_records}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
