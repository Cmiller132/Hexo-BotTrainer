"""Dirichlet-noise decomposition (the 'is the noise doing anything' test).

On a sample of root positions: forward the BC seed (CPU) to get the raw policy
PRIOR over candidates, then mix in Dirichlet noise exactly as the MCTS does
(alpha_i = total_alpha/num_candidates, eps fraction) and measure how much the
noise changes the ROOT prior: KL(noised||prior), top-1-move-change rate, and the
effective-move-count change. Small KL + rare top-1 change => noise too weak to
shape the search. By phase. CPU-only, ~40 positions, does not touch the live GPU run.
"""
from __future__ import annotations
import glob, math, random, sys
from collections import defaultdict
import numpy as np
import torch

from hexo_models.dense_cnn.compact_io import read_compact_shard
from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.expand import reconstruct_state
from hexo_models.hexgt.graph_build import batch_from_states
from hexo_models.hexgt.losses import segment_log_softmax

BC = "/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_bc/hexgt_bc_step006009.pt"
RUN = "/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main"
TOTAL_ALPHA = 6.6
EPS = 0.25
N_POS = 45
DRAWS = 32


def phase(ply):
    return "opening" if ply < 15 else ("mid" if ply < 60 else "end")


def main():
    ck = torch.load(BC, map_location="cpu", weights_only=False)
    a = ck["arch"]
    model = HexgtNetwork(token_dim=a["token_dim"], gnn_layers=a["gnn_layers"], ctx_layers=a["ctx_layers"],
                         ffn_dim=a["ffn_dim"], attention_heads=a["attention_heads"],
                         short_term_value_horizons=tuple(a["short_term_value_horizons"])).cpu().eval()
    model.load_state_dict(ck["model"])

    # sample positions across phases
    shards = sorted(glob.glob(f"{RUN}/selfplay/epoch_*_game_*.npz"))
    rng = random.Random(1); rng.shuffle(shards)
    picks = []  # (state, ply)
    want = {"opening": N_POS // 3, "mid": N_POS // 3, "end": N_POS - 2 * (N_POS // 3)}
    got = defaultdict(int)
    for sp in shards:
        if sum(got.values()) >= N_POS:
            break
        try:
            rows = read_compact_shard(sp)
        except Exception:
            continue
        rows = [r for r in rows if r.policy]
        rng.shuffle(rows)
        for r in rows:
            ph = phase(int(r.turn_index))
            if got[ph] >= want[ph]:
                continue
            try:
                picks.append((reconstruct_state(r.placement_history), int(r.turn_index)))
                got[ph] += 1
            except Exception:
                pass
            if got[ph] >= want[ph]:
                break

    rng_np = np.random.default_rng(0)
    res = defaultdict(list)
    with torch.no_grad():
        for state, ply in picks:
            batch = batch_from_states([state], 3)
            out = model.forward_policy_value(batch)
            cg = batch["candidate_graph"]
            logp = segment_log_softmax(out["policy"].float(), cg, 1)
            P = logp.exp().numpy().astype(np.float64)
            P = P / P.sum()
            K = P.size
            if K < 2:
                continue
            alpha = TOTAL_ALPHA / K
            top_raw = int(P.argmax())
            effP = math.exp(float(-(P * np.log(P + 1e-12)).sum()))
            kls, top_changes, effs = [], 0, []
            for _ in range(DRAWS):
                D = rng_np.dirichlet([alpha] * K)
                Pn = (1 - EPS) * P + EPS * D
                Pn = Pn / Pn.sum()
                kls.append(float((Pn * (np.log(Pn + 1e-12) - np.log(P + 1e-12))).sum()))
                if int(Pn.argmax()) != top_raw:
                    top_changes += 1
                effs.append(math.exp(float(-(Pn * np.log(Pn + 1e-12)).sum())))
            ph = phase(ply)
            res[ph].append((K, effP, np.mean(kls), top_changes / DRAWS, np.mean(effs)))

    print(f"### Dirichlet-noise decomposition (BC seed prior, total_alpha={TOTAL_ALPHA}, eps={EPS})\n")
    print(f"{'phase':8} {'n':>3} {'cands':>6} {'alpha_i':>8} | {'priorEffN':>9} {'noisedEffN':>10} | "
          f"{'KL(noised||prior)':>17} {'P(top-move changes)':>20}")
    for ph in ("opening", "mid", "end"):
        rows = res[ph]
        if not rows:
            continue
        arr = np.array(rows, float)
        K = arr[:, 0].mean()
        print(f"{ph:8} {len(rows):3d} {K:6.0f} {TOTAL_ALPHA/K:8.4f} | "
              f"{arr[:,1].mean():9.1f} {arr[:,4].mean():10.1f} | {arr[:,2].mean():17.3f} {arr[:,3].mean():20.1%}")
    print("\nReading: KL ~0 and top-change ~0% => noise irrelevant (too weak). "
          "Large KL / frequent top-change => noise meaningfully reshapes the root prior.")


if __name__ == "__main__":
    main()
