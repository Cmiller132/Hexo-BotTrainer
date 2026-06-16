"""Phase 3: how well does the raw policy head match its own MCTS visit target?

Compares, per move bucket (epoch 21):
  - visit-target sharpness (top1, effective moves)  [the label]
  - raw-policy sharpness (top1, effective moves)     [the prediction]
  - policy cross-entropy H(target, pred) and KL(target||pred)
  - rank assigned by raw policy to the target's argmax move (is the best move
    even near the top of the prior MCTS can widen to?)
A large sharp-target / diffuse-pred gap with high KL == the policy head is
underfitting sharp labels -> policy-pathway capacity/architecture limit, not
diffuse labels.
"""
import sys, os, glob
import numpy as np, torch
sys.path.insert(0, "E:/Hexo-BotTrainer/packages/hexo_models/dense_cnn/python")
from hexo_models.dense_cnn.architecture import Model1Network

RUN = "E:/Hexo-BotTrainer/runs/dense_cnn_model1_scratch_64/selfplay"

def load_net(ep):
    payload = torch.load(f"E:/Hexo-BotTrainer/analysis/_ckpt_epoch{ep:02d}.pt", map_location="cpu")
    net = Model1Network(in_channels=13, channels=64, blocks=4, short_term_value_horizons=(1, 4, 8))
    net.load_state_dict(payload["model_state"]); net.eval()
    return net

@torch.no_grad()
def run(ep, max_games=120):
    net = load_net(ep)
    rows = []
    for f in sorted(glob.glob(os.path.join(RUN, f"epoch_{ep:06d}_game_*.npz")))[:max_games]:
        d = np.load(f); md = d["metadataInputNC"]
        if md.shape[0] == 0:
            continue
        N = md.shape[0]
        inp = torch.from_numpy(d["inputNCHW"]).float()
        legal = torch.from_numpy(d["legalMaskNCHW"].reshape(N, -1).astype(bool))
        tgt = torch.from_numpy(d["policyTargetsNCHW"].reshape(N, -1)).double()
        tgt = tgt / tgt.sum(1, keepdim=True).clamp_min(1e-12)
        logits = []
        for i in range(0, N, 64):
            logits.append(net.forward_policy_value(inp[i:i+64])["policy"])
        logits = torch.cat(logits, 0).reshape(N, -1).double()
        neg = torch.finfo(torch.float64).min / 4
        masked = torch.where(legal, logits, torch.full_like(logits, neg))
        pred = torch.softmax(masked, 1)
        pred = torch.where(legal, pred, torch.zeros_like(pred))
        pred = pred / pred.sum(1, keepdim=True).clamp_min(1e-12)
        # metrics
        lp = torch.where(pred > 0, pred.log(), torch.zeros_like(pred))
        lt = torch.where(tgt > 0, tgt.log(), torch.zeros_like(tgt))
        ce = -(tgt * lp).sum(1)                      # cross-entropy H(target,pred)
        kl = (tgt * (lt - lp)).sum(1)                # KL(target||pred)
        tgt_top1 = tgt.max(1).values
        pred_top1 = pred.max(1).values
        tgt_eff = (-(tgt * lt).sum(1)).exp()
        pred_eff = (-(pred * lp).sum(1)).exp()
        tgt_arg = tgt.argmax(1)
        pred_at_tgtarg = pred.gather(1, tgt_arg[:, None]).squeeze(1)
        # rank of target argmax within pred (1 = pred also ranks it first)
        order = pred.argsort(1, descending=True)
        rank = (order == tgt_arg[:, None]).float().argmax(1) + 1
        turn = torch.from_numpy(md[:, 0]).double()
        for i in range(N):
            rows.append((turn[i].item(), tgt_top1[i].item(), pred_top1[i].item(),
                         tgt_eff[i].item(), pred_eff[i].item(), ce[i].item(), kl[i].item(),
                         pred_at_tgtarg[i].item(), rank[i].item()))
    return np.array(rows)

C = ["turn", "tgt_top1", "pred_top1", "tgt_eff", "pred_eff", "ce", "kl", "pred@tgtarg", "rank"]
BINS = [0, 10, 20, 40, 60, 100, 200]

if __name__ == "__main__":
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    a = run(ep)
    print(f"EPOCH {ep}  positions={a.shape[0]}\n")
    hdr = f"{'move#':>10} {'n':>6} | {'tgtTop1':>7} {'predTop1':>8} | {'tgtEff':>7} {'predEff':>7} | {'polCE':>6} {'KL':>6} | {'pred@best':>9} {'bestRank':>8}"
    print(hdr); print("-"*len(hdr))
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (a[:, 0] >= lo) & (a[:, 0] < hi)
        if m.sum() < 8: continue
        s = a[m]
        def mean(c): return s[:, C.index(c)].mean()
        print(f"{f'[{lo},{hi})':>10} {int(m.sum()):>6} | {mean('tgt_top1'):>7.3f} {mean('pred_top1'):>8.3f} | "
              f"{mean('tgt_eff'):>7.2f} {mean('pred_eff'):>7.2f} | {mean('ce'):>6.2f} {mean('kl'):>6.2f} | "
              f"{mean('pred@tgtarg'):>9.3f} {np.median(s[:,C.index('rank')]):>8.0f}")
