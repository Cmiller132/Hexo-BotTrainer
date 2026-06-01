"""Decompose dense_cnn loss into components across checkpoints on a FIXED batch.

Answers: is the rising epoch-loss genuine model divergence, or target-drift /
surprise-resampling artifact? Evaluates several checkpoints on the SAME held-out
rows, unweighted, and reports policy CE / value CE / aux + policy entropy.
"""
import sys, glob, json
import numpy as np
import torch

RD = "/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x6"
sys.path.insert(0, "/mnt/e/Hexo-BotTrainer/packages/hexo_models/dense_cnn/python")

from hexo_models.dense_cnn.architecture import Model1Network
from hexo_models.dense_cnn.losses import model1_loss, decode_binned_value, value_bins
from hexo_models.dense_cnn.trainer import _materialize_npz, _batch_from_npz

HORIZONS = (1, 4, 8)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixed eval batch: first 2048 rows of a fixed recent shuffled train shard.
shards = sorted(glob.glob(f"{RD}/shuffleddata/*/train/data00000.npz"))
fixed_shard = shards[-1]
print(f"FIXED eval batch from: {fixed_shard}", flush=True)
with np.load(fixed_shard) as data:
    arrays = _materialize_npz(data)
N = min(2048, arrays[list(arrays.keys())[0]].shape[0])
batch_cpu = _batch_from_npz(arrays, 0, N, HORIZONS)

# Target policy entropy (data side, model-independent)
tp = batch_cpu["policy"].numpy()
tp = tp / np.clip(tp.sum(1, keepdims=True), 1e-9, None)
tgt_entropy = float((-(tp * np.log(np.clip(tp, 1e-12, None))).sum(1)).mean())
tgt_support = float((tp > 1e-6).sum(1).mean())  # avg #moves with mass
# value target: scalar in [-1,1] (model1_loss bins it internally)
vt = batch_cpu["value"].numpy()
v_scalar_tgt = vt.reshape(-1) if vt.ndim == 1 else (vt / np.clip(vt.sum(1, keepdims=True), 1e-9, None) * np.linspace(-1, 1, vt.shape[1])).sum(1)
print(f"N={N}  target_policy_entropy={tgt_entropy:.3f} nats  avg_support_moves={tgt_support:.1f}", flush=True)
print(f"target value scalar: mean={v_scalar_tgt.mean():+.3f} std={v_scalar_tgt.std():.3f} "
      f"|frac>0.9|={np.mean(np.abs(v_scalar_tgt)>0.9):.3f}", flush=True)

def load_model(ckpt):
    payload = torch.load(ckpt, map_location="cpu")
    ms = payload.get("model_state", payload)
    net = Model1Network(in_channels=13, channels=96, blocks=6, dropout=0.0,
                        short_term_value_horizons=HORIZONS)
    net.load_state_dict(ms)
    return net.to(dev).eval()

batch = {k: v.to(dev) for k, v in batch_cpu.items()}
inp = batch.pop("input")
if inp.ndim == 4:
    inp = inp.to(memory_format=torch.channels_last)

epochs = [int(e) for e in (sys.argv[1:] or ["4", "6", "8", "10", "11"])]
print(f"\n{'ep':>3} {'total':>7} {'policy':>7} {'value':>7} {'opp':>7} {'stv1':>6} {'stv4':>6} {'stv8':>6} {'m_polH':>7} {'m_valStd':>8}", flush=True)
rows = []
for e in epochs:
    ck = f"{RD}/checkpoints/epoch_{e:06d}.pt"
    if not glob.glob(ck):
        continue
    net = load_model(ck)
    with torch.no_grad():
        out = net(inp)
        _, comp = model1_loss(out, batch, policy_weight=1.0, value_weight=1.0,
                              opp_policy_weight=0.25, short_term_value_weight=0.25)
        # model policy entropy
        pl = torch.log_softmax(out["policy"], dim=-1)
        p = pl.exp()
        m_polH = float((-(p * pl).sum(-1)).mean())
        v_scalar = decode_binned_value(out["value"]).float().cpu().numpy()
    g = lambda k: float(comp[k]) if k in comp else float("nan")
    print(f"{e:>3} {g('total'):>7.3f} {g('policy'):>7.3f} {g('value'):>7.3f} {g('opp_policy'):>7.3f} "
          f"{g('stvalue_1'):>6.3f} {g('stvalue_4'):>6.3f} {g('stvalue_8'):>6.3f} "
          f"{m_polH:>7.3f} {v_scalar.std():>8.3f}", flush=True)
    rows.append((e, g('policy'), g('value'), m_polH, float(v_scalar.mean()), float(v_scalar.std())))
    del net
    torch.cuda.empty_cache() if dev.type == "cuda" else None

print("\nNotes: target_policy_entropy is the data ceiling for policy CE.", flush=True)
print("If model policy CE on this FIXED batch RISES e4->e11 => genuine divergence.", flush=True)
print("If it falls/flat while epoch-loss rises => rising epoch-loss is target-drift/resample artifact.", flush=True)
