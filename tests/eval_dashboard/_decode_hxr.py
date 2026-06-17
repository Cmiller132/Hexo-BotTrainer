import glob
import sys

from hexo_runner.records import HexoRecordFile

paths = sys.argv[1:] or sorted(
    glob.glob("/mnt/e/Hexo-BotTrainer/runs/hexfield_main_2/evaluation/epoch_000035/*.hxr")
)
for p in paths:
    rf = HexoRecordFile.open(p)
    recs = rf.iter_records()
    print(f"{p}: num_records={len(recs)}")
    for rec in recs:
        print(f"   game_id={rec.game_id} status={rec.status} winner={rec.winner} "
              f"actions={len(rec.action_ids)}")
