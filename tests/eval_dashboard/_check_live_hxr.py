"""Read-only: decode each live eval .hxr, report record count + mtime, to see
whether the CURRENT running build produces empty (num_records=0) replays."""
import datetime
import glob
import os

from hexo_runner.records import HexoRecordFile

RUN = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_2"

for ep in sorted(glob.glob(RUN + "/evaluation/epoch_*")):
    hxrs = sorted(glob.glob(ep + "/*.hxr"))
    if not hxrs:
        print(os.path.basename(ep), "(no .hxr)")
    for h in hxrs:
        try:
            rf = HexoRecordFile.open(h)
            with rf:
                n = sum(1 for _ in rf.iter_records())
            n = str(n)
        except Exception as e:  # noqa: BLE001
            n = f"ERR {type(e).__name__}: {e}"
        mt = datetime.datetime.fromtimestamp(os.path.getmtime(h)).isoformat(timespec="seconds")
        print(f"{os.path.basename(ep)}  {os.path.basename(h):42s} records={n:6s} mtime={mt}")
