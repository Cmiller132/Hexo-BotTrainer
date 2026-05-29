"""TRT FP16 SEARCH-OUTCOME equivalence: does TRT change the move chosen by a full
512-sim MCTS search vs the torch FP16 evaluator?

Per-forward error (policy 0.05 logits / value 4.6e-5 decoded) is not the gate —
what matters is whether 512 sequential sims accumulate into a DIFFERENT chosen
move. For N representative positions we run the SAME search (512 visits, noise
OFF, temperature 0 -> deterministic given the evaluator, same seed) twice: once
with the torch evaluator, once with the TRT evaluator, and compare the selected
action. Reports move-agreement %, the per-position visit-distribution L1, and the
top-move-visit-share gap. This is the search-outcome-equivalence gate.

(A full SealBot best-50ms win-rate A/B is the ideal final gate but needs the
Windows SealBot exe; this move-agreement test is the WSL-runnable proxy and is
the tighter numeric check.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in ("hexo_engine", "hexo_utils", "hexo_runner", "hexo_train", "hexo_models", "hexo_frontend"):
    sys.path.insert(0, str(REPO / "packages" / p / "python"))

import torch

CONFIG = REPO / "configs" / "dense_cnn_model1_target_96x6.toml"
CKPT = REPO / "runs" / "dense_cnn_model1_target_96x6" / "checkpoints" / "bootstrap_sealbot_prefit.pt"
RESULT = Path(__file__).resolve().parent / "_tv1_trt_move_agreement.json"


def build_model_cfg():
    import tomllib
    from hexo_models.dense_cnn.config import parse_model1_config
    from hexo_models.dense_cnn.plugin import DenseCNNPlugin
    section = tomllib.load(open(CONFIG, "rb"))["model"]["config"]
    parsed = parse_model1_config(section)
    model = DenseCNNPlugin().build_model(game_spec={}, config=section)
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model_state"], strict=True)
    model.eval()
    return model, parsed


def gen_positions(n):
    import random
    import hexo_engine as engine
    from hexo_engine.types import unpack_coord_id
    rng = random.Random(2024)
    states = []
    gi = 0
    while len(states) < n:
        st = engine.new_game(seed=400_000 + gi); gi += 1
        for _ in range(rng.randint(4, 130)):
            if engine.terminal(st) is not None:
                break
            aids = engine.legal_action_ids(st)
            if not aids:
                break
            engine.apply_action(st, engine.PlacementAction(unpack_coord_id(rng.choice(aids))))
        if engine.terminal(st) is None:
            states.append(engine.clone_state(st))
    return states


def search_move(session, state, inf, parsed, seed):
    # Deterministic search: noise OFF, temperature 0 -> argmax over visits.
    res = session.run(
        [0], [state], inf,
        visits=parsed.selfplay.search_visits, c_puct=parsed.selfplay.c_puct,
        temperature=0.0, seed=seed, virtual_batch_size=4,
        active_root_limit=parsed.selfplay.mcts_active_root_limit,
        root_dirichlet_total_alpha=None, root_dirichlet_noise_fraction=None,
        root_policy_temperature=parsed.selfplay.root_policy_temperature,
        fpu_reduction=parsed.selfplay.fpu_reduction, virtual_loss=parsed.selfplay.virtual_loss,
        widening_policy_mass=parsed.selfplay.widening_policy_mass,
        widening_max_children=parsed.selfplay.widening_max_children,
        widening_min_children=parsed.selfplay.widening_min_children,
    )
    s = res[0]
    return int(s.action_id), dict(s.visit_policy)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--precision", default="fp16", choices=["fp16", "bf16"])
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from hexo_models.dense_cnn.inference import DenseCNNInference
    from hexo_models.dense_cnn.mcts import new_mcts_session
    import hexo_engine as engine

    from hexo_models.dense_cnn import trt_backend
    model, parsed = build_model_cfg()
    print(f"[tv1] precision={args.precision}; building torch + TRT evaluators...", flush=True)
    inf_torch = DenseCNNInference(model, device="cuda", amp=True, max_batch_size=1024, use_trt=False)
    # FORCE TRT on for the strength test (bypass the per-forward build gate) so we
    # measure the true SEARCH-OUTCOME effect, not the conservative fallback.
    inf_trt = DenseCNNInference(model, device="cuda", amp=True, max_batch_size=1024, use_trt=False)
    fwd, gate = trt_backend.build_trt_forward(
        inf_trt.model, max_batch=1024, device="cuda",
        precision=args.precision, argmax_match_min=0.0, value_tol=1.0e9,  # always adopt for the test
    )
    inf_trt._trt_forward = fwd
    inf_trt.trt_info = gate
    trt_adopted = fwd is not None
    print(f"[tv1] TRT({args.precision}) force-adopted={trt_adopted} per-forward gate={gate}", flush=True)

    # Forward speed: torch FP16 vs TRT, at the production buckets, CUDA-event timed.
    def time_fwd(forward_fn, bs):
        xx = torch.zeros((bs, 13, 41, 41), device="cuda").to(memory_format=torch.channels_last)
        nz = (torch.rand((bs, 13, 41, 41), device="cuda") > 0.7).float().to(memory_format=torch.channels_last)
        for _ in range(40):
            forward_fn(nz)
        torch.cuda.synchronize()
        s = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
        e = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
        for k in range(100):
            s[k].record(); forward_fn(nz); e[k].record()
        torch.cuda.synchronize()
        ms = float(np.mean([a.elapsed_time(b) for a, b in zip(s, e)]))
        return bs / (ms / 1000.0)
    def torch_fwd(z):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            return inf_torch.model.forward_policy_value(z)
    speed = {}
    for bs in (128, 256):
        speed[f"torch_fp16_bs{bs}"] = time_fwd(torch_fwd, bs)
        speed[f"trt_{args.precision}_bs{bs}"] = time_fwd(fwd, bs)
    print(f"[tv1] forward fwd/s: {json.dumps({k: round(v) for k, v in speed.items()})}", flush=True)

    states = gen_positions(args.n)
    print(f"[tv1] {len(states)} positions; 512-sim searches torch vs TRT({args.precision})...", flush=True)
    agree = 0; compared = 0; nan_aborts = 0
    torch_self_agree = 0  # torch-vs-torch nondeterminism floor (same eval, fresh session)
    l1s = []; flips = []
    for i, st in enumerate(states):
        sess_t = new_mcts_session(max_states=parsed.selfplay.mcts_session_cache_max_states)
        sess_t2 = new_mcts_session(max_states=parsed.selfplay.mcts_session_cache_max_states)
        sess_r = new_mcts_session(max_states=parsed.selfplay.mcts_session_cache_max_states)
        mt, vt = search_move(sess_t, engine.clone_state(st), inf_torch, parsed, seed=1000 + i)
        mt2, _ = search_move(sess_t2, engine.clone_state(st), inf_torch, parsed, seed=1000 + i)
        torch_self_agree += int(mt == mt2)
        try:
            mr, vr = search_move(sess_r, engine.clone_state(st), inf_trt, parsed, seed=1000 + i)
        except Exception as ex:
            if "finite" in str(ex).lower() or "nan" in str(ex).lower():
                nan_aborts += 1
                continue
            raise
        compared += 1
        same = (mt == mr); agree += int(same)
        keys = set(vt) | set(vr)
        st_sum = sum(vt.values()) or 1.0; sr_sum = sum(vr.values()) or 1.0
        l1 = sum(abs(vt.get(k, 0) / st_sum - vr.get(k, 0) / sr_sum) for k in keys)
        l1s.append(l1)
        if not same:
            flips.append({"pos": i, "torch_move": mt, "trt_move": mr, "visit_l1": round(l1, 4)})

    out = {
        "precision": args.precision, "n_positions": len(states),
        "trt_adopted": trt_adopted, "trt_gate": inf_trt.trt_info, "forward_fwd_per_s": speed,
        "nan_aborts": nan_aborts, "compared": compared,
        "torch_vs_torch_agreement": torch_self_agree / max(len(states), 1),
        "move_agreement": agree / max(compared, 1),
        "move_flip_rate": 1.0 - agree / max(compared, 1),
        "trt_excess_flip_vs_torch_nondeterminism": (torch_self_agree / max(len(states), 1)) - (agree / max(compared, 1)),
        "visit_l1_mean": float(np.mean(l1s)) if l1s else None,
        "visit_l1_p95": float(np.percentile(l1s, 95)) if l1s else None,
        "flips": flips[:20],
    }
    print(json.dumps({k: v for k, v in out.items() if k != "flips"}, indent=2), flush=True)
    Path(str(RESULT).replace(".json", f"_{args.precision}.json")).write_text(json.dumps(out, indent=2))
    print(f"[tv1] done precision={args.precision} nan_aborts={nan_aborts} "
          f"move_agreement={out['move_agreement']:.4f}", flush=True)


if __name__ == "__main__":
    main()
