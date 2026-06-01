"""Throughput vs latency for the training step. Real `_optimizer_step` forces a
per-step sync via `.cpu().item()`. Measure: (a) per-step-synced LATENCY, (b) fully
pipelined THROUGHPUT (no per-step sync, no .item(), one sync at the end). The gap
is what a prefetch + deferred-loss-logging loop could recover. Read-only."""
from __future__ import annotations
import glob, os, json, time
import numpy as np, torch
from hexo_models.dense_cnn.architecture import Model1Network
from hexo_models.dense_cnn.losses import model1_loss
from hexo_models.dense_cnn.trainer import _batch_from_npz
from hexo_models.dense_cnn.replay import INPUT_KEY

H=(1,4,8); BS=256; DEV=torch.device("cuda")
base=r"E:\Hexo-BotTrainer\runs\dense_cnn_model1_scratch_64\shuffleddata"
g=sorted(d for d in glob.glob(os.path.join(base,"*-epoch_*")) if not d.endswith(".tmp"))[-1]
f0=sorted(glob.glob(os.path.join(g,"train","*.npz")))[0]
torch.backends.cudnn.benchmark=True
m=Model1Network(channels=64,blocks=4,short_term_value_horizons=H).to(DEV,memory_format=torch.channels_last); m.train()
opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4); scaler=torch.amp.GradScaler("cuda",enabled=True)
with np.load(f0) as d:
    arrays={k:d[k] for k in d.files}
class Mem:
    def __getitem__(self,k): return arrays[k]
mem=Mem(); rows=arrays[INPUT_KEY].shape[0]
batches=[]
for off in range(0,min(rows,BS*32),BS):
    cb=_batch_from_npz(mem,off,min(off+BS,rows),H)
    batches.append({k:(v.to(DEV,non_blocking=True,memory_format=torch.channels_last) if k=="input" else v.to(DEV,non_blocking=True)) for k,v in cb.items()})
nb=len(batches)

def fwdbwd(b):
    bb=dict(b); inp=bb.pop("input"); opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda",enabled=True):
        out=m(inp); loss,_=model1_loss(out,bb,policy_weight=1.0,value_weight=1.0,opp_policy_weight=0.25,short_term_value_weight=0.25)
    scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
    scaler.step(opt); scaler.update(); return loss.detach()

for _ in range(20): fwdbwd(batches[0])
torch.cuda.synchronize()
# (a) latency: sync + .item() every step
t0=time.perf_counter()
for i in range(120):
    l=fwdbwd(batches[i%nb]); float(l.cpu().item()); torch.cuda.synchronize()
lat=(time.perf_counter()-t0)/120*1000
# (b) throughput: no per-step sync/.item(); one sync at end
torch.cuda.synchronize(); t0=time.perf_counter()
acc=None
for i in range(120):
    l=fwdbwd(batches[i%nb]); acc=l if acc is None else acc+l
torch.cuda.synchronize(); thr=(time.perf_counter()-t0)/120*1000
out={"latency_per_step_synced_ms":lat,"throughput_per_step_pipelined_ms":thr,"recoverable_gap_ms":lat-thr,"recoverable_per_epoch_s_391":(lat-thr)*391/1000}
print(json.dumps(out,indent=2)); json.dump(out,open(os.path.join(os.path.dirname(__file__),"train_step_pipeline_summary.json"),"w"),indent=2)
