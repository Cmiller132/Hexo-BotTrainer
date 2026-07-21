"""V2 — fixed-budget h2h at ep90 (PLAN_TSS_MCTS_INTEGRATION.md §9 V2):
the horizon-shape decision (§10.3), played out under FULL production
consumption.

Both sides are the SAME net (ep90) and the SAME production TSS consumption
config (main_3 `_resume_config.toml`: tss_enabled + interior guard + mode 3 +
root guard + async 8 threads / inline16 4 + park 150 ms + node cap 500 +
zone off), differing ONLY in the solver horizon:

  Arm A (challenger) : tss_solver_horizon = 0   (unbounded + node cap)
  Arm B (incumbent)  : tss_solver_horizon = 16  (flat h16, engine default)

Paired CRN openings, pentanomial pairs, fixed visit budget
(cfg.selfplay.search_visits — the production 256).

Deviations from the plan's V2 sentence (pre-registered here, argued in
V2_H2H_REPORT.md):
- No narrow-width opponent arm: the ported engine is wholesale-wide
  (`TssSolverSlot::default()` is always leaf-configured); width was closed by
  the owner's §10.1 wholesale ruling + V1's paired dominance (narrow_only=0).
- No `tss_zone=true` arm: the plan expected the ladder to make zones live;
  V1 measured `zone_nodes = 0` under ladder AND unbounded at cap 500, and
  production itself retired zones at ep35 for the same reason.

Usage:
    python run_v2_h2h.py <n_games> <out_json> [visits]
"""

from __future__ import annotations

import arch_env  # noqa: F401  MUST precede any hexfield_eq import

import json
import sys
import time
import tomllib
from pathlib import Path

RUN_DIR = Path("/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main_3")
CKPT = RUN_DIR / "checkpoints" / "epoch_000090.pt"


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("raws/v2_h2h_result.json")
    visits = int(sys.argv[3]) if len(sys.argv) > 3 else None

    from hexfield_eq.serve_env import prime_serve_env

    prime_serve_env()
    from hexfield_eq import eval_arena
    from hexfield_eq.config import build_divergence_overrides, parse_hexfield_config

    raw = tomllib.load(open(RUN_DIR / "_resume_config.toml", "rb"))
    cfg = parse_hexfield_config(raw["model"]["config"])
    sp = cfg.selfplay

    base = build_divergence_overrides(sp)
    # Guard: this must be the PRODUCTION consumption profile, or the A/B is
    # not measuring what §9 V2 asks for.
    # tss_enabled travels via build_eval_search_kwargs(sp), not the overrides.
    assert sp.tss_enabled, "tss_enabled must be on in the run config"
    assert int(base.get("tss_solver_mode", 0)) == 3, base.get("tss_solver_mode")
    assert base.get("tss_solver_root_guard"), "root guard must be on"
    assert base.get("tss_solver_async") and base.get("tss_solver_park"), "async+park"
    assert not base.get("tss_zone", False), "zone must be off (ep35 production)"
    assert int(base.get("tss_solver_node_cap", 0)) == 500, base.get("tss_solver_node_cap")

    ov_a = dict(base)
    ov_a["tss_solver_horizon"] = 0
    ov_a["tss_solver_horizon_ladder"] = False
    ov_b = dict(base)
    ov_b["tss_solver_horizon"] = 16
    ov_b["tss_solver_horizon_ladder"] = False

    eff_visits = visits if visits is not None else sp.search_visits
    print(
        f"V2 h2h: {n_games} games ({n_games // 2} pairs), visits={eff_visits}, "
        f"A=unbounded+cap vs B=h16-flat, mode=3 async+park", flush=True
    )
    t0 = time.time()
    res = eval_arena.play_checkpoint_match(
        CKPT,
        CKPT,
        n_games,
        config=cfg,
        label_a="ep90_tss_unbounded",
        label_b="ep90_tss_h16flat",
        visits=visits,
        divergence_overrides_a=ov_a,
        divergence_overrides_b=ov_b,
        diagnostics_dir="raws/_v2_diag",
        game_seed_base=20260720,
    )
    wall = time.time() - t0
    res.setdefault("meta", {})["v2_wall_seconds"] = wall

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2, default=str))

    score = res.get("score", {})
    pent = res.get("pentanomial") or {}
    headline = {k: v for k, v in pent.items() if k != "pairs"}
    print(f"V2 DONE in {wall:.1f}s: score={json.dumps(score)}", flush=True)
    print(f"V2 PENTANOMIAL: {json.dumps(headline, default=str)}", flush=True)


if __name__ == "__main__":
    main()
