"""Phase 3b: denser value-calibration check over late epochs (is the ep41 dip real?)."""
import sys, glob, math
from pathlib import Path
sys.path.insert(0, "/mnt/e/Hexo-BotTrainer-hexgt/_expl")
import common as C
from hexo_utils.records import HexoRecordFile
from hexo_models.dense_cnn.compact_io import read_compact_shard

EPOCHS = [25, 29, 33, 35, 37, 39, 41]
GAMES = 70
def mean(xs): return (sum(xs)/len(xs)) if xs else float("nan")
def corr(pairs):
    xs=[a for a,b in pairs if b==b]; ys=[b for a,b in pairs if b==b]
    if len(xs)<3: return float("nan")
    mx,my=mean(xs),mean(ys); dx=math.sqrt(sum((a-mx)**2 for a in xs)); dy=math.sqrt(sum((b-my)**2 for b in ys))
    return sum((a-mx)*(b-my) for a,b in zip(xs,ys))/(dx*dy) if dx>0 and dy>0 else float("nan")

print(f"epoch | corr(v,z) |   n  | mean|v| | late-game(ply>=40) corr | n_late | len_mean(games)")
for ep in EPOCHS:
    if not Path(f"{C.CKPT_DIR}/hexgnn_rl_epoch{ep:06d}.pt").exists(): continue
    net,_=C.load_model(ep); ev=C.Evaluator(net,n=4)
    recs=list(HexoRecordFile.open(f"{C.SP_DIR}/epoch_{ep:06d}.hxr").iter_records())[:GAMES]
    vz=[]; vz_late=[]; lens=[]
    for rec in recs:
        idx=C.shard_index_from_game_id(rec.game_id)
        shp=glob.glob(f"{C.SP_DIR}/epoch_{ep:06d}_game_{idx:06d}.npz")
        if not shp: continue
        z_by={int(r.turn_index):float(r.value) for r in read_compact_shard(Path(shp[0]))}
        actions=[int(a) for a in rec.action_ids]; lens.append(len(actions))
        s=C.new_game(seed=int(rec.seed))
        for t,a in enumerate(actions):
            if C.engine.terminal(s) is not None: break
            if t in z_by:
                v=ev.value(s); vz.append((v,z_by[t]))
                if t>=40: vz_late.append((v,z_by[t]))
            C.apply(s,a)
    print(f"{ep:5d} |  {corr(vz):+.3f}   | {len(vz):4d} |  {mean([abs(v) for v,z in vz]):.2f}   |"
          f"        {corr(vz_late):+.3f}          | {len(vz_late):5d}  |  {mean(lens):.1f}")
