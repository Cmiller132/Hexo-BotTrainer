"""SealBot best-50ms STRENGTH A/B: TRT-backed vs torch-backed dense_cnn search.

Move-agreement is not proof of harm; this measures actual play strength. Runs N
paired games of dense_cnn (512-sim MCTS) vs SealBot best (50 ms/move) with the
TRT FP16 evaluator, and N with the torch FP16 evaluator, using the SAME per-game
seeds/openings for both arms (paired) so the only difference is the evaluator.
Reports win/loss/draw and win-rate per arm + the delta. Verdict: if TRT win-rate
is within noise of torch, TRT holds strength.

Run in WSL (TRT + the WSL-built SealBot minimax_cpp):
  SEALBOT_PATH=/mnt/e/SealBot PYTHONPATH=/mnt/e/SealBot:... python tv4_sealbot_ab.py --games 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
for p in ("hexo_engine", "hexo_utils", "hexo_runner", "hexo_train", "hexo_models", "hexo_frontend"):
    sys.path.insert(0, str(REPO / "packages" / p / "python"))

import torch

CONFIG = REPO / "configs" / "dense_cnn_model1_target_96x6.toml"
CKPT = REPO / "runs" / "dense_cnn_model1_target_96x6" / "checkpoints" / "bootstrap_sealbot_prefit.pt"
RESULT = Path(__file__).resolve().parent / "_tv4_sealbot_ab.json"


def build():
    import tomllib
    from hexo_models.dense_cnn.config import parse_model1_config
    from hexo_models.dense_cnn.plugin import DenseCNNPlugin
    section = tomllib.load(open(CONFIG, "rb"))["model"]["config"]
    parsed = parse_model1_config(section)
    model = DenseCNNPlugin().build_model(game_spec={}, config=section)
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model_state"], strict=True)
    model.eval().to("cuda", memory_format=torch.channels_last)
    return model, parsed


def run_arm(name, shared_inf, model, parsed, n_games, tmp, seed_base):
    from hexo_models.dense_cnn.player import DenseCNNPlayer
    from hexo_runner.adapters.sealbot import SealBotConfig, SealBotPlayer
    from hexo_runner.modes.match import run_match
    from hexo_runner.session import GameSpec

    trainer = SimpleNamespace(config=parsed, device=torch.device("cuda"),
                              inference_batch_size=1024, selfplay_batch_size=256,
                              mcts_virtual_batch_size=4)
    sb_cfg = SealBotConfig(path=os.environ.get("SEALBOT_PATH"), variant="best", time_limit=0.05)
    wins = losses = draws = completed = 0
    for g in range(n_games):
        dense_is_p0 = (g % 2 == 0)
        seed = seed_base + g
        dense = DenseCNNPlayer(identity_id="dense-eval", model=model, trainer=trainer,
                               record_samples=False, eval_seed=seed,
                               opening_temperature=parsed.evaluation.opening_temperature,
                               opening_moves=parsed.evaluation.opening_moves)
        dense.inference = shared_inf  # inject shared evaluator (no per-game TRT rebuild)
        sb = SealBotPlayer(sb_cfg, player_id="sealbot-best-50ms")
        players = (dense, sb) if dense_is_p0 else (sb, dense)
        res = run_match(GameSpec(game_id=f"{name}-{g:03d}", seed=seed, is_evaluation=True,
                                 max_actions=parsed.evaluation.max_actions), players, tmp)
        if str(res.status) == "completed":
            completed += 1
        dense_role = "player0" if dense_is_p0 else "player1"
        if res.winner == dense_role:
            wins += 1
        elif res.winner is not None:
            losses += 1
        else:
            draws += 1
        print(f"  [{name}] game {g+1}/{n_games} winner={res.winner} "
              f"(dense={dense_role}) -> W{wins} L{losses} D{draws}", flush=True)
    return {"arm": name, "games": n_games, "completed": completed,
            "wins": wins, "losses": losses, "draws": draws,
            "win_rate": wins / max(n_games, 1)}


def main():
    import tempfile
    ap = argparse.ArgumentParser(); ap.add_argument("--games", type=int, default=20); args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from hexo_models.dense_cnn.inference import DenseCNNInference
    from hexo_models.dense_cnn import trt_backend

    model, parsed = build()
    print("[tv4] building shared torch + TRT(fp16) evaluators...", flush=True)
    inf_torch = DenseCNNInference(model, device="cuda", amp=True, max_batch_size=1024, use_trt=False)
    inf_trt = DenseCNNInference(model, device="cuda", amp=True, max_batch_size=1024, use_trt=False)
    fwd, gate = trt_backend.build_trt_forward(inf_trt.model, max_batch=1024, device="cuda",
                                              precision="fp16", argmax_match_min=0.0, value_tol=1.0e9)
    inf_trt._trt_forward = fwd
    print(f"[tv4] TRT adopted={fwd is not None} gate={ {k:gate.get(k) for k in ('policy_argmax_match','value_max_abs_err')} }", flush=True)

    tmp = Path(tempfile.mkdtemp(prefix="tv4_ab_"))
    seed_base = 770000
    print(f"[tv4] === TORCH arm ({args.games} games, paired seeds) ===", flush=True)
    torch_res = run_arm("torch", inf_torch, model, parsed, args.games, tmp, seed_base)
    print(f"[tv4] === TRT arm ({args.games} games, paired seeds) ===", flush=True)
    trt_res = run_arm("trt", inf_trt, model, parsed, args.games, tmp, seed_base)

    out = {"games_per_arm": args.games, "sealbot": "best-50ms", "search_visits": parsed.selfplay.search_visits,
           "torch": torch_res, "trt": trt_res,
           "win_rate_delta_trt_minus_torch": trt_res["win_rate"] - torch_res["win_rate"]}
    print(json.dumps(out, indent=2), flush=True)
    RESULT.write_text(json.dumps(out, indent=2))
    import shutil; shutil.rmtree(tmp, ignore_errors=True)
    print(f"[tv4] wrote {RESULT.name}", flush=True)


if __name__ == "__main__":
    main()
