"""CPU: is the policy still LEARNING despite flat CE? Measure each checkpoint's raw
policy top-1 agreement with its OWN-epoch search visit target (same-strength search,
no cross-epoch distribution shift). Rising agreement => prior catching up to search.
Also reports policy CE and the visit-target entropy floor on the same positions."""
import glob, importlib.util as u, math
import torch
spec = u.spec_from_file_location("exp", "_lambda0_experiment.py")
m = u.module_from_spec(spec); spec.loader.exec_module(m)
from hexo_models.dense_cnn.compact_io import read_compact_shard
from hexo_models.hexgt.expand import build_training_batch
from pathlib import Path
torch.set_grad_enabled(False)

CASES = [("epoch4", 4), ("epoch6", 6), ("epoch8", 8), ("epoch10", 10)]
N_SHARDS = 40

def topk_eval(ckpt_path, epoch):
    net, _ck, _a, _g = m.load_grafted(ckpt_path, "cpu"); net.eval()
    shards = sorted(glob.glob(f"{m.SP}/epoch_{epoch:06d}_game_*.npz"))[:N_SHARDS]
    rows = []
    for s in shards:
        rows.extend(read_compact_shard(Path(s)))
    agree = ntot = 0
    ce_sum = ent_sum = 0.0
    nseg_total = 0
    for i in range(0, len(rows), 128):
        chunk = rows[i:i+128]
        batch, targets = build_training_batch(chunk, n=3, horizons=(4,12,24), prune_max_dropped_mass=1.0)
        if batch is None:
            continue
        out = net(batch)
        logits = out["policy"].float()
        tgt = targets["policy"].float()
        seg = batch["candidate_graph"].long()
        ng = int(batch["num_graphs"])
        # per-graph top-1 agreement + CE + target entropy
        for g in range(ng):
            mask = seg == g
            if not bool(mask.any()):
                continue
            lt = logits[mask]; tt = tgt[mask]
            if float(tt.sum()) <= 0:
                continue
            p = torch.softmax(lt, dim=0)
            tnorm = tt / tt.sum()
            if int(torch.argmax(lt)) == int(torch.argmax(tt)):
                agree += 1
            ntot += 1
            ce_sum += float(-(tnorm * torch.log(p.clamp_min(1e-9))).sum())
            ent_sum += float(-(tnorm * torch.log(tnorm.clamp_min(1e-9))).sum())
            nseg_total += 1
    return (agree/ntot if ntot else float('nan'), ce_sum/nseg_total if nseg_total else float('nan'),
            ent_sum/nseg_total if nseg_total else float('nan'), ntot)

print(f"{'checkpoint':<10} | top1-agree w/ search | policy CE | target-entropy floor | gap(CE-floor) | n")
print("-"*92)
for name, ep in CASES:
    pth = f"{m.CK}/hexgt_rl_epoch{ep:06d}.pt"
    if not glob.glob(pth):
        print(f"{name}: checkpoint missing"); continue
    a, ce, ent, n = topk_eval(pth, ep)
    print(f"{name:<10} | {a:6.1%}              | {ce:.3f}    | {ent:.3f}                | {ce-ent:+.3f}        | {n}")
print("\nRising top1-agree across epochs => policy IS learning (prior matches its own search better).")
print("CE flat while target-entropy floor ~flat => not a rising-floor artifact; judge by top1 + gap trend.")
