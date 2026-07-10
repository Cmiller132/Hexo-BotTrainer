#!/usr/bin/env python
"""Wave-1 serve throughput (pos/s) for one arm (SPEC_RAYTAP_CONV.md §6.3).

Runs the real continuous scheduler over a few ply-capped games with a
VISIT-based search budget through the FULL serve stack (the production serve
env profile: fp16 half serve, Triton kernels, CUDA graphs — whatever the
sourced env enables) and reports decisions/s next to the fused/reference path
label for equipped convs. Source the ARM's env file (arch env) + the serve
profile BEFORE running; the checkpoint's arch meta must match the env.

  set -a; source scripts/prefit_env/hexfield_eq_raytap_a2.env; set +a
  python scripts/raytap_serve_throughput.py <ckpt> [visits] [games] [ply_cap]

RUN ONLY ON AN IDLE GPU — never against the live soak.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for pkg in ("hexfield_eq", "hexo_engine", "hexo_utils"):
    p = REPO / "packages" / pkg / "python"
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch  # noqa: E402

from hexo_engine import api  # noqa: E402
from hexo_engine.types import AxialCoord, PlacementAction  # noqa: E402

from hexfield_eq import _rust  # noqa: E402
from hexfield_eq import constants as C  # noqa: E402
from hexfield_eq.geometry import unpack_action_id  # noqa: E402
from hexfield_eq.inference import HexfieldEvaluator  # noqa: E402
from hexfield_eq.model import HexfieldNet, infer_net_kwargs_from_state_dict  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else None
VISITS = int(sys.argv[2]) if len(sys.argv) > 2 else 512
GAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 16
PLY_CAP = int(sys.argv[4]) if len(sys.argv) > 4 else 20


def main() -> int:
    device = torch.device("cuda")
    if CKPT:
        payload = torch.load(CKPT, map_location="cpu", weights_only=False)
        sd = payload.get("model", payload)
        meta = payload.get("meta") or {}
        kwargs = infer_net_kwargs_from_state_dict(sd, meta)
        model = HexfieldNet(**kwargs)
        model.load_state_dict(sd, strict=True)
    else:
        model = HexfieldNet()
    raytap = getattr(model, "_raytap", "0")
    evaluator = HexfieldEvaluator(model, device=device)

    # Fused-vs-reference label for equipped convs (spec §2.4: pre-K1 numbers
    # must be labeled reference-path).
    from hexfield_eq import model as M
    if raytap == "0":
        path_label = "baseline (no equipped convs)"
    elif getattr(M, "_hex_conv_ln_raytap_fused", None) is not None:
        path_label = "fused-K1 (falls back per-shape on compile failure)"
    else:
        path_label = "reference-path (K1 kernel not enabled/available)"

    states = {k: api.new_game() for k in range(GAMES)}
    plies = {k: 0 for k in range(GAMES)}
    decisions = {"n": 0}

    def on_move(game_key, payload):
        decisions["n"] += 1
        st = states[game_key]
        q, r = unpack_action_id(int(payload["action_id"]))
        res = api.apply_action(st, PlacementAction(AxialCoord(q=q, r=r)))
        plies[game_key] += 1
        if res.terminal or plies[game_key] >= PLY_CAP:
            return None
        return ("advance", st)

    session = _rust.HexfieldMctsSession(max_states=262144)
    torch.cuda.synchronize()
    t0 = time.time()
    session.run_continuous(
        list(range(GAMES)), tuple(states.values()), evaluator=evaluator,
        on_move=on_move, visits=VISITS, c_puct=1.5, base_seed=7,
        virtual_batch_size=96, flush_target=256, active_root_limit=GAMES,
        temperature_by_ply=[1.0] * 8 + [0.3] * 200,
        forced_playout_k=2.0, widening_policy_mass=0.95,
        widening_max_children=96, widening_min_children=2,
        root_dirichlet_total_alpha=10.83, root_dirichlet_noise_fraction=0.25,
        pcr_full_proportion=0.33, pcr_fast_visits=128,
        policy_init_fraction=0.25, policy_init_avg_plies=4.0,
        policy_init_max_plies=8, policy_init_temperature=1.4,
    )
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(json.dumps({
        "ckpt": CKPT,
        "arch": {"trunk": model._trunk_layout, "raytap": raytap,
                 "feature_version": C.FEATURE_VERSION, "channels": C.CHANNELS},
        "serve_path": path_label,
        "visits": VISITS, "games": GAMES, "ply_cap": PLY_CAP,
        "decisions": decisions["n"], "seconds": round(dt, 1),
        "pos_per_s": round(decisions["n"] / dt, 2),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
