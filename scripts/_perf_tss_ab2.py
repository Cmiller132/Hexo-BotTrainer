"""Move-bounded A/B self-play throughput for the TSS no-scan short-circuit.

Real parallel regime: 64 concurrent games, visits=512, vbatch=64, widening=96,
candidate_radius=3, the pretrained model3 seed, torch.compile on. Advances a fixed
number of moves over real epoch-0 shard positions (bounded so the slow "before"
finishes quickly) and reports searched positions/sec. The ONLY difference between
the two invocations is HEXGT_TSS_NOSCAN_FASTPATH (0=legacy full scan, 1=O(1)
has_threats short-circuit) -> a clean controlled measurement of the scan cost.
"""
from __future__ import annotations
import argparse, os, random, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hexo_engine.api as eng
from hexo_engine.types import unpack_coord_id
from hexo_models.hexgt.inference import HexgtInference
from hexo_models.hexgt.mcts import new_mcts_session
import _rl_train as drv


def _states_from_shards(n_states, shard_dir, seed=1):
    from hexo_models.dense_cnn.compact_io import read_compact_shard
    from hexo_models.hexgt.expand import reconstruct_state
    rng = random.Random(seed)
    npzs = sorted(Path(shard_dir).glob("*_game_*.npz"))
    rng.shuffle(npzs)
    states = []
    for npz in npzs:
        try:
            rows = read_compact_shard(npz)
        except Exception:
            continue
        for row in rows:
            s = reconstruct_state(row.placement_history)
            if eng.terminal(s) is None:
                states.append(s)
            if len(states) >= n_states:
                return states
    return states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--moves", type=int, default=10)
    ap.add_argument("--visits", type=int, default=512)
    ap.add_argument("--vbatch", type=int, default=64)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--shards", default=str(ROOT / "runs/hexgt_rl_main3/selfplay"))
    ap.add_argument("--seed-ckpt", default=str(ROOT / "runs/hexgt_rl_main3/pretrain/hexgt_model3_pretrain.pt"))
    ap.add_argument("--tag", default="RUN")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fastpath = os.environ.get("HEXGT_TSS_NOSCAN_FASTPATH", "1") != "0"

    ck = torch.load(args.seed_ckpt, map_location=device, weights_only=False)
    arch_meta = dict(ck["arch"]); arch_meta["short_term_value_horizons"] = list(drv.STV_HORIZONS)
    model = drv.build_model(arch_meta, device)
    drv.expand_value_readout_columns(model, ck["model"])
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    model.forward_policy_value = torch.compile(model.forward_policy_value, dynamic=True)
    inference = HexgtInference(model, device=device, fp16=(device.type == "cuda"))

    SR = dict(visits=args.visits, c_puct=1.5, virtual_batch_size=args.vbatch,
              widening_policy_mass=0.95, widening_max_children=96, widening_min_children=2,
              fpu_reduction=0.20, virtual_loss=1.0, root_dirichlet_total_alpha=6.6,
              root_dirichlet_noise_fraction=0.25, root_policy_temperature=1.0, temperature=1.0)

    pool = _states_from_shards(args.games, args.shards, seed=1)
    games = {i: s for i, s in enumerate(pool)}
    sess = new_mcts_session(max_states=1 << 21, n=args.n)

    # warm one move (compile + cache) before timing
    wk = list(games.keys())
    sess.run(wk, [games[k] for k in wk], inference, seed=99, **SR)
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter(); searched = 0
    enc_s = ev_s = parse_s = 0.0; uniq = req = hits = 0
    for move in range(args.moves):
        active = [(k, s) for k, s in games.items() if eng.terminal(s) is None]
        if not active:
            break
        res = sess.run([k for k, _ in active], [s for _, s in active], inference, seed=1000 + move, **SR)
        searched += len(active)
        d = dict(dict(res[0].diagnostics).get("batch", {}).get("evaluation", {}))
        enc_s += float(d.get("encoding_seconds", 0.0))
        ev_s += float(d.get("evaluator_seconds", 0.0))
        parse_s += float(d.get("parse_seconds", 0.0))
        uniq += int(d.get("unique_states", 0)); req += int(d.get("requested_states", 0))
        hits += int(d.get("cache_hits", 0))
        for (k, s), r in zip(active, res):
            eng.apply_action(s, eng.PlacementAction(unpack_coord_id(r.action_id)))
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    print(f"[{args.tag}] fastpath={'ON' if fastpath else 'OFF'} games={len(pool)} moves={args.moves} "
          f"visits={args.visits} | searched_pos={searched} wall={dt:.1f}s "
          f"| pos/s={searched/dt:.2f} | peak_alloc={peak_gb*1000:.0f}MB")
    print(f"  EVAL split (sum over moves): rust_encode={enc_s:.1f}s  py_evaluator(fwd)={ev_s:.1f}s  "
          f"parse={parse_s:.1f}s | fwd/pos={uniq/max(1,searched):.1f} cache_hit={hits/max(1,req):.0%} "
          f"uniq={uniq} req={req}")
    print("PERF_AB2_DONE")


if __name__ == "__main__":
    main()
