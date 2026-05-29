"""Phase 2: RAW network policy/value head behavior via light CPU inference.

Feeds the already-encoded inputNCHW planes from logged selfplay games through a
copied checkpoint (CPU, eval). NO Dirichlet noise, NO root_policy_temperature,
NO MCTS -- this is the network's own policy/value head.

Quantifies, binned by move number and legal-move count:
  RAW POLICY: effective #moves, top1, normalized entropy over legal set, and
              top-32 cumulative prior mass (32 == widening_max_children cap, so
              this is the mass MCTS can even consider at the root).
  VALUE HEAD: predicted scalar value vs game outcome z -> sign accuracy,
              correlation, and mean |pred| (confidence) by move bucket. Tests
              whether the value head loses discrimination late-game.
"""
import sys, os, glob, json
import numpy as np
import torch

sys.path.insert(0, "E:/Hexo-BotTrainer/packages/hexo_models/dense_cnn/python")
from hexo_models.dense_cnn.architecture import Model1Network  # noqa: E402

BOARD_AREA = 41 * 41
VALUE_BINS = 65
RUN = "E:/Hexo-BotTrainer/runs/dense_cnn_model1_scratch_64/selfplay"
BINS_BIN = torch.linspace(-1.0, 1.0, VALUE_BINS)

def load_net(ckpt_path):
    payload = torch.load(ckpt_path, map_location="cpu")
    state = payload["model_state"] if "model_state" in payload else payload
    net = Model1Network(in_channels=13, channels=64, blocks=4,
                        short_term_value_horizons=(1, 4, 8))
    net.load_state_dict(state)
    net.eval()
    return net

@torch.no_grad()
def forward_all(net, inputs, bs=64):
    pol, val = [], []
    for i in range(0, inputs.shape[0], bs):
        x = torch.from_numpy(inputs[i:i + bs]).float()
        out = net.forward_policy_value(x)
        pol.append(out["policy"])
        val.append(out["value"])
    return torch.cat(pol, 0), torch.cat(val, 0)

def raw_policy_stats(policy_logits, legalNCHW):
    """logits (N,1681) -> masked softmax over legal cells; per-row stats."""
    N = policy_logits.shape[0]
    logits = policy_logits.reshape(N, -1).double()
    legal = torch.from_numpy(legalNCHW.reshape(N, -1).astype(bool))
    neg = torch.finfo(torch.float64).min / 4
    masked = torch.where(legal, logits, torch.full_like(logits, neg))
    p = torch.softmax(masked, dim=1)
    p = torch.where(legal, p, torch.zeros_like(p))
    p = p / p.sum(1, keepdim=True).clamp_min(1e-12)
    logp = torch.where(p > 0, p.log(), torch.zeros_like(p))
    H = -(p * logp).sum(1)
    top1 = p.max(1).values
    eff = H.exp()
    legal_n = legal.sum(1).double()
    Hnorm = H / legal_n.clamp_min(2).log()
    # top-32 cumulative mass (widening_max_children cap)
    sorted_p, _ = torch.sort(p, dim=1, descending=True)
    top32_mass = sorted_p[:, :32].sum(1)
    top8_mass = sorted_p[:, :8].sum(1)
    return (H.numpy(), top1.numpy(), eff.numpy(), Hnorm.numpy(),
            legal_n.numpy(), top32_mass.numpy(), top8_mass.numpy())

def value_pred(value_logits):
    p = torch.softmax(value_logits.double(), dim=1)
    return (p * BINS_BIN.double()).sum(1).numpy()

def collect(net, epoch, max_games=120):
    files = sorted(glob.glob(os.path.join(RUN, f"epoch_{epoch:06d}_game_*.npz")))[:max_games]
    rows = []
    for f in files:
        d = np.load(f)
        md = d["metadataInputNC"]
        if md.shape[0] == 0:
            continue
        inputs = d["inputNCHW"]
        legal = d["legalMaskNCHW"]
        val_target = d["valueTargetsN"]
        turn = md[:, 0]
        pl, vl = forward_all(net, inputs)
        H, t1, eff, Hn, ln, m32, m8 = raw_policy_stats(pl, legal)
        vpred = value_pred(vl)
        for i in range(md.shape[0]):
            rows.append((turn[i], ln[i], eff[i], t1[i], Hn[i], m32[i], m8[i],
                         vpred[i], val_target[i]))
    return np.array(rows, dtype=np.float64)

COLS = ["turn", "legal_n", "eff", "top1", "Hnorm", "m32", "m8", "vpred", "vtgt"]

def binned(arr, xcol, bins):
    xi = COLS.index(xcol)
    x = arr[:, xi]
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() < 8:
            continue
        sub = arr[m]
        vp = sub[:, COLS.index("vpred")]
        vt = sub[:, COLS.index("vtgt")]
        sign_acc = float(np.mean(np.sign(vp) == np.sign(vt)))
        corr = float(np.corrcoef(vp, vt)[0, 1]) if np.std(vp) > 1e-9 else float("nan")
        out.append({
            "bin": f"[{lo:.0f},{hi:.0f})", "n": int(m.sum()),
            "legal_n": float(sub[:, COLS.index("legal_n")].mean()),
            "eff": float(sub[:, COLS.index("eff")].mean()),
            "top1": float(sub[:, COLS.index("top1")].mean()),
            "Hnorm": float(sub[:, COLS.index("Hnorm")].mean()),
            "m32": float(sub[:, COLS.index("m32")].mean()),
            "m8": float(sub[:, COLS.index("m8")].mean()),
            "vpred_abs": float(np.mean(np.abs(vp))),
            "v_sign_acc": sign_acc, "v_corr": corr,
        })
    return out

def fmt(rows, xlabel):
    hdr = (f"{xlabel:>12} {'n':>6} {'legalN':>8} | {'effRAW':>7} {'top1':>6} {'Hnorm':>6} "
           f"{'top8m':>6} {'top32m':>7} | {'|vpred|':>7} {'vSignAcc':>8} {'vCorr':>6}")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(f"{r['bin']:>12} {r['n']:>6} {r['legal_n']:>8.1f} | "
                     f"{r['eff']:>7.2f} {r['top1']:>6.3f} {r['Hnorm']:>6.3f} "
                     f"{r['m8']:>6.3f} {r['m32']:>7.3f} | "
                     f"{r['vpred_abs']:>7.3f} {r['v_sign_acc']:>8.3f} {r['v_corr']:>6.3f}")
    return "\n".join(lines)

if __name__ == "__main__":
    move_bins = [0, 10, 20, 30, 40, 60, 80, 100, 130, 200]
    legal_bins = [0, 50, 100, 200, 300, 400, 600, 900, 1200, 1700]
    out = {}
    for ep in [int(x) for x in sys.argv[1:]] or [21]:
        ckpt = f"E:/Hexo-BotTrainer/analysis/_ckpt_epoch{ep:02d}.pt"
        net = load_net(ckpt)
        arr = collect(net, ep)
        print(f"\n############ EPOCH {ep} RAW HEADS  (positions={arr.shape[0]}) ############")
        print("\n=== by MOVE NUMBER ===")
        bm = binned(arr, "turn", move_bins)
        print(fmt(bm, "move#"))
        print("\n=== by LEGAL-MOVE COUNT ===")
        bl = binned(arr, "legal_n", legal_bins)
        print(fmt(bl, "legalN"))
        out[ep] = {"by_move": bm, "by_legal": bl}
    with open("E:/Hexo-BotTrainer/analysis/phase2_summary.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote analysis/phase2_summary.json")
