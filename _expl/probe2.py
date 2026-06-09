import sys
sys.path.insert(0, "/mnt/e/Hexo-BotTrainer-hexgt/_expl")
import common as C
from hexo_utils.records import HexoRecordFile
from hexo_models.dense_cnn.compact_io import read_compact_shard
from pathlib import Path

hxr = f"{C.SP_DIR}/epoch_000041.hxr"
recs = list(HexoRecordFile.open(hxr).iter_records())
print("n records", len(recs))
r = recs[0]
print("attrs", [a for a in dir(r) if not a.startswith("_")])
print("game_id", r.game_id, "n_actions", len(list(r.action_ids)))
for attr in ("seed", "winner", "completed", "truncated", "num_actions"):
    print("  ", attr, getattr(r, attr, "<none>"))

idx = C.shard_index_from_game_id(r.game_id)
shp = Path(f"{C.SP_DIR}/epoch_000041_game_{idx:06d}.npz")
rows = read_compact_shard(shp)
print("shard rows", len(rows), "turns", sorted({int(x.turn_index) for x in rows})[:10], "...")
print("row0 fields: turn", rows[0].turn_index, "player", rows[0].current_player,
      "value", round(float(rows[0].value),3), "npolicy", len(list(rows[0].policy)))
