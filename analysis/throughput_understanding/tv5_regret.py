"""Low-variance STRENGTH gate: paired per-decision VALUE regret, TRT vs torch.

For N positions (paired seeds), run a 512-sim search with the torch evaluator and
with the TRT evaluator and take each one's chosen move. Where they differ, score
each move's quality by a 1-ply lookahead under a STRONG reference (a high-sim
TORCH search of the resulting position): quality(m) = -ref_root_value(apply(s,m))
(value to the mover; root_value of the child is from the opponent's view).

  regret_i = quality(m_torch) - quality(m_trt)   (win-prob units, root_value∈[-1,1])
           = 0 when the moves agree.
  >0  => torch's move is better under the reference => TRT degrades that decision.

Most positions agree (regret 0) so the mean over all N has a tight CI from modest
N — the point of a per-decision metric vs a noisy win-rate A/B. Verdict: mean
regret ≈ 0 within a tight CI => TRT's flips are between equally-good moves =>
strength-equivalent => safe to enable. Systematically positive => keep off.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in ("hexo_engine", "hexo_utils", "hexo_runner", "hexo_train", "hexo_models", "hexo_frontend"):
    sys.path.insert(0, str(REPO / "packages" / p / "python"))

import torch

CONFIG = REPO / "configs" / "dense_cnn_model1_target_96x6.toml"
CKPT = REPO / "runs" / "dense_cnn_model1_target_96x6" / "checkpoints" / "bootstrap_sealbot_prefit.pt"
RESULT = Path(__file__).resolve().parent / "_tv5_regret.json"


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


def gen_positions(n):
    import random
    import hexo_engine as engine
    from hexo_engine.types import unpack_coord_id
    rng = random.Random(31337)
    states = []
    gi = 0
    while len(states) < n:
        st = engine.new_game(seed=800_000 + gi); gi += 1
        for _ in range(rng.randint(4, 130)):
            if engine.terminal(st) is not None:
                break
            a = engine.legal_action_ids(st)
            if not a:
                break
            engine.apply_action(st, engine.PlacementAction(unpack_coord_id(rng.choice(a))))
        if engine.terminal(st) is None:
            states.append(engine.clone_state(st))
    return states


def search(session, state, inf, parsed, sims, seed):
    res = session.run(
        [0], [state], inf, visits=sims, c_puct=parsed.selfplay.c_puct,
        temperature=0.0, seed=seed, virtual_batch_size=4,
        active_root_limit=parsed.selfplay.mcts_active_root_limit,
        root_dirichlet_total_alpha=None, root_dirichlet_noise_fraction=None,
        root_policy_temperature=parsed.selfplay.root_policy_temperature,
        fpu_reduction=parsed.selfplay.fpu_reduction, virtual_loss=parsed.selfplay.virtual_loss,
        widening_policy_mass=parsed.selfplay.widening_policy_mass,
        widening_max_children=parsed.selfplay.widening_max_children,
        widening_min_children=parsed.selfplay.widening_min_children,
    )[0]
    return int(res.action_id), float(res.root_value)


def quality_of_move(state, action_id, ref_inf, parsed, ref_sims, mk_session):
    """Value to the mover after playing `action_id`, via a high-sim torch ref."""
    import hexo_engine as engine
    from hexo_engine.types import unpack_coord_id
    child = engine.clone_state(state)
    engine.apply_action(child, engine.PlacementAction(unpack_coord_id(action_id)))
    term = engine.terminal(child)
    if term is not None:
        mover = engine.current_player(state)
        if term.winner is None:
            return 0.0
        return 1.0 if str(term.winner) == str(mover) else -1.0
    sess = mk_session()
    _, child_root_value = search(sess, child, ref_inf, parsed, ref_sims, seed=424242)
    return -child_root_value  # child root_value is from the opponent's perspective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--ref-sims", type=int, default=1536)
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from hexo_models.dense_cnn.inference import DenseCNNInference
    from hexo_models.dense_cnn.mcts import new_mcts_session
    from hexo_models.dense_cnn import trt_backend
    import hexo_engine as engine

    model, parsed = build()
    sims = parsed.selfplay.search_visits
    print(f"[tv5] sims={sims} ref_sims={args.ref_sims} n={args.n}; building evaluators...", flush=True)
    inf_torch = DenseCNNInference(model, device="cuda", amp=True, max_batch_size=1024, use_trt=False)
    inf_trt = DenseCNNInference(model, device="cuda", amp=True, max_batch_size=1024, use_trt=False)
    fwd, gate = trt_backend.build_trt_forward(inf_trt.model, max_batch=1024, device="cuda",
                                              precision="fp16", argmax_match_min=0.0, value_tol=1.0e9)
    inf_trt._trt_forward = fwd
    print(f"[tv5] TRT adopted={fwd is not None} argmax={gate.get('policy_argmax_match')} "
          f"val_err={gate.get('value_max_abs_err')}", flush=True)

    mk = lambda: new_mcts_session(max_states=parsed.selfplay.mcts_session_cache_max_states)
    states = gen_positions(args.n)
    regrets = []
    flips = 0
    flip_regret_detail = []
    for i, st in enumerate(states):
        m_torch, _ = search(mk(), engine.clone_state(st), inf_torch, parsed, sims, seed=5000 + i)
        m_trt, _ = search(mk(), engine.clone_state(st), inf_trt, parsed, sims, seed=5000 + i)
        if m_torch == m_trt:
            regrets.append(0.0)
        else:
            flips += 1
            q_torch = quality_of_move(st, m_torch, inf_torch, parsed, args.ref_sims, mk)
            q_trt = quality_of_move(st, m_trt, inf_torch, parsed, args.ref_sims, mk)
            r = q_torch - q_trt
            regrets.append(r)
            flip_regret_detail.append(round(r, 4))
        if (i + 1) % 50 == 0:
            arr = np.array(regrets)
            print(f"  [{i+1}/{len(states)}] flips={flips} mean_regret={arr.mean():+.5f}", flush=True)

    arr = np.array(regrets)
    n = len(arr)
    mean = float(arr.mean()); std = float(arr.std(ddof=1)) if n > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(n) if n else 0.0
    flip_arr = np.array([r for r in flip_regret_detail]) if flip_regret_detail else np.array([0.0])
    out = {
        "n_positions": n, "search_visits": sims, "ref_sims": args.ref_sims,
        "flip_rate": flips / max(n, 1), "flips": flips,
        "mean_regret": mean, "regret_std": std, "regret_95ci": ci95,
        "regret_ci_low": mean - ci95, "regret_ci_high": mean + ci95,
        "mean_regret_on_flips_only": float(flip_arr.mean()),
        "flips_torch_better": int((flip_arr > 0.01).sum()),
        "flips_trt_better": int((flip_arr < -0.01).sum()),
        "flips_tied": int((np.abs(flip_arr) <= 0.01).sum()),
        "max_single_regret": float(flip_arr.max()) if flip_regret_detail else 0.0,
        "note": "regret = quality(torch_move) - quality(trt_move) in win-prob units; >0 => TRT worse",
    }
    print(json.dumps(out, indent=2), flush=True)
    RESULT.write_text(json.dumps(out, indent=2))
    print(f"[tv5] DONE: mean_regret={mean:+.5f} +/- {ci95:.5f} (95% CI) over {n} decisions, "
          f"flip_rate={flips/max(n,1):.3f}", flush=True)


if __name__ == "__main__":
    main()
