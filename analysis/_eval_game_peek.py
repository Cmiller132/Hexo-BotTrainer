"""Read-only peek at recorded SealBot eval games (.hxr) to characterize outcomes
and openings. No model inference, no GPU."""
import glob
import os

from hexo_engine.types import unpack_coord_id
from hexo_runner.records import HexoRecordFile

RUN = "/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8"


def model_role(players):
    for p in players:
        pid = str(getattr(p, "player_id", "")).lower()
        label = str(getattr(p, "label", "")).lower()
        if "sealbot" not in pid and "sealbot" not in label:
            return str(getattr(p, "role", ""))
    return None


def peek(epoch, sample=12):
    d = os.path.join(RUN, "evaluation", f"epoch_{epoch:06d}")
    files = sorted(glob.glob(os.path.join(d, "*.hxr")))
    print(f"=== epoch {epoch}: {len(files)} eval game files ===")
    wins = losses = other = 0
    shown = 0
    printed_players = False
    for f in files[:sample]:
        try:
            with HexoRecordFile.open(f) as rf:
                players = list(rf.players)
                recs = list(rf.iter_records())
        except Exception as e:  # noqa: BLE001
            print("  read fail", os.path.basename(f), repr(e))
            continue
        mrole = model_role(players)
        if not printed_players:
            print("  players:", [(getattr(p, "role", ""), getattr(p, "player_id", ""), getattr(p, "label", "")) for p in players], "-> model_role=", mrole)
            printed_players = True
        for rec in recs:
            w = rec.winner
            if w == mrole:
                wins += 1
            elif w is not None:
                losses += 1
            else:
                other += 1
            if shown < 4:
                acts = [int(a) for a in rec.action_ids]
                opening = [(unpack_coord_id(a).q, unpack_coord_id(a).r) for a in acts[:5]]
                print(f"  {os.path.basename(f)}: winner={w} model_won={w == mrole} len={len(acts)} status={rec.status} opening5={opening}")
                shown += 1
    print(f"  -> sampled(first {sample}): model_wins={wins} losses={losses} draw/none={other}")


for ep in (17, 10):
    peek(ep)
