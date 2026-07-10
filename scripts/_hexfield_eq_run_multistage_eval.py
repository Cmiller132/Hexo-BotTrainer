"""Standalone runner for the multi-stage hexfield_eq strength eval.

The hexfield_eq (D6-equivariant, 25-plane) twin of
``scripts/_hexfield_run_multistage_eval.py``. PURELY EVAL + STANDALONE: it
plays GPU eval games against the eq re-anchor roster, pools every edge into the
persisted Bradley-Terry rating (diagnostics/eval_pool.json), and prints the
pooled rating table + the single PRIMARY verdict LABEL. It NEVER gates,
promotes, halts, or mutates the training run.

EQ-LINEAGE ANCHOR POLICY (decision D4, docs/DEPLOYMENT_CHECKLIST_HEXFIELD_EQ.md
§4): ``permanent_anchors = ()`` — the old hexfield-lineage BC-prefit/ep5
anchors are 15-plane nets and can NEVER strict-load into this 25-plane
lineage. The pool is anchored on **Strix** (``strix_enabled=true`` +
``strix_ckpt``, full BT weight, the pinned 0-Elo zero-point) + **SealBot**
(``$SEALBOT_PATH``, 0.5 over-dispersion) + the run's own sliding self-anchor
bracket once epochs accumulate. Configure Strix/SealBot per run in the toml's
``[model.config.multi_stage_eval.opponents]`` table; this script adds
``--strix-ckpt`` / ``--no-strix`` overrides.

ENV (checklist B5): every eq entry point MUST pin
``HEXFIELD_EQ_SUPPORT_RADIUS`` to the radius the candidate was trained at
(main_11-corpus arms: 4). The serve payload / checkpoint-meta guards turn a
mismatch into a loud error, but only the env makes it correct. Arch knobs
(``HEXFIELD_EQ_TRUNK`` / ``HEXFIELD_EQ_REG_LANE`` / ...) are meta-first from
the checkpoint, so they need not be exported for foreign-arch candidates.

Usage:
  HEXFIELD_EQ_SUPPORT_RADIUS=4 \
  python scripts/_hexfield_eq_run_multistage_eval.py RUN_DIR CANDIDATE_CKPT \
      [--epoch N] [--label NAME] [--config CONFIG.toml] [--budget N] \
      [--visits N] [--full-visits N] [--no-sprt] [--no-sealbot] [--no-strix] \
      [--strix-ckpt PATH] [--dry-run] [--device cuda|cpu] \
      [--parts] [--opponent LABEL] [--aggregate-only] [--resume|--no-resume]

  RUN_DIR          the RL run dir (<run>/checkpoints/epoch_*.pt, <run>/diagnostics/).
  CANDIDATE_CKPT   the checkpoint under test (player L). If it lives in RUN_DIR's
                   checkpoints/ the epoch is inferred from the filename.
  --config         optional run toml whose [model.config] drives search/opening
                   knobs + the [model.config.multi_stage_eval] section (e.g.
                   configs/hexfield_eq_main_1.toml). Defaults to the built-in
                   defaults (Strix/SealBot both need the toml/env to resolve).
  --budget         override multi_stage_eval.games_budget (Stage C paired games).
  --visits         override eval visits.
  --no-sprt        skip Stage B.
  --no-sealbot     drop the SealBot zero-point.
  --no-strix       drop the Strix zero-point (pool anchors on SealBot, then self).
  --strix-ckpt     override opponents.strix_ckpt (the relocated anchor lives at
                   /mnt/e/Hexo-BotTrainer/anchors/strix/checkpoint_00237000.pt).
  --dry-run        resolve the roster + allocation and print them; play NO games
                   and write NOTHING (pure plumbing check, safe to run anytime).

RUN-IN-PARTS (resumable per-opponent chunks — a long eval becomes a sequence of
short ones, and an interrupted eval keeps every completed part):
  --parts          run as a SEQUENCE of per-opponent parts (strix + sealbot +
                   each checkpoint opponent), each appending its edge to the
                   pool and persisting immediately, then the aggregate BT fit +
                   verdict.
  --opponent LABEL run a SINGLE part for one opponent (e.g. ``strix``,
                   ``sealbot``, ``ep5``, or the champion ``epN``) and exit.
  --aggregate-only fit the persisted pool + print the rating table / verdict,
                   playing NO games (the final pass after the chunks).
  --resume/--no-resume  with --parts/--opponent, SKIP a part whose edge for this
                   candidate epoch is already pooled (default: resume ON).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# The eq package plus its shared dependencies (the hexfield twin only inserted
# its own package and relied on ambient PYTHONPATH for the rest; insert all
# four so the script runs from a bare venv).
for _pkg in ("hexfield_eq", "hexo_engine", "hexo_utils", "hexo_train"):
    _root = REPO / "packages" / _pkg / "python"
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from hexfield_eq.config import HexfieldConfig, parse_hexfield_config  # noqa: E402
from hexfield_eq import multistage_eval as mse  # noqa: E402


def _load_config(config_path: str | None) -> HexfieldConfig:
    """Parse a run toml's [model.config] into a HexfieldConfig, or defaults."""
    if not config_path:
        return parse_hexfield_config({})
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover - older interpreters
        import tomli as tomllib  # type: ignore
    raw = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    model_cfg = (raw.get("model") or {}).get("config") or {}
    return parse_hexfield_config(model_cfg)


def _apply_overrides(cfg: HexfieldConfig, args: argparse.Namespace) -> HexfieldConfig:
    """Apply CLI overrides by re-parsing a merged config dict.

    HexfieldConfig is frozen, so we round-trip through a plain dict + the
    parser. Unlike the hexfield twin (which hand-listed every multi_stage_eval
    field and twice silently DROPPED newly-added toml keys on the round-trip),
    the whole section is carried via ``dataclasses.asdict`` and only the
    CLI-exposed knobs are overlaid — a new config field rides through with no
    edit here."""
    section = asdict(cfg.multi_stage_eval)  # recurses into opponents/sprt
    if args.budget is not None:
        section["games_budget"] = int(args.budget)
    if args.visits is not None:
        section["eval_visits"] = int(args.visits)
    if args.full_visits is not None:
        section["full_search_visits"] = int(args.full_visits)
    if args.no_sealbot:
        section["opponents"]["sealbot_enabled"] = False
    if args.no_strix:
        section["opponents"]["strix_enabled"] = False
    if args.strix_ckpt is not None:
        section["opponents"]["strix_ckpt"] = str(args.strix_ckpt)
        section["opponents"]["strix_enabled"] = True
    if args.no_sprt:
        section["sprt"]["enabled"] = False
    merged = {
        "device": args.device or cfg.device,
        "selfplay": asdict(cfg.selfplay),
        "training": asdict(cfg.training),
        "multi_stage_eval": section,
    }
    return parse_hexfield_config(merged)


def _print_report(report: dict) -> None:
    """Pretty-print the rating table + verdict to stdout."""
    meta = report["meta"]
    print("=" * 78)
    print(f"hexfield_eq multi-stage eval  candidate={meta['candidate_label']} "
          f"(epoch {meta['candidate_epoch']})")
    print(f"run_dir={meta['run_dir']}")
    print(f"anchor={meta['anchor']}  pure_eval={meta['pure_eval']}  "
          f"gating={meta['gating_enabled']}  promotion={meta['promotion_enabled']}")
    print("=" * 78)

    # Stage summary.
    for s in report["stages"]:
        line = f"  [{s['stage']:<8}] {s['status']}"
        if s["stage"] == "B_sprt" and s.get("sprt_verdict"):
            line += (f"  llr={s['llr']} bounds={s['bounds']} "
                     f"-> {s['sprt_verdict']} ({s['triage']})  "
                     f"{s['wins_cand']}-{s['wins_champion']} vs {s['vs']}")
        elif s["stage"] == "C_deep":
            line += f"  edges={s.get('n_edges')} alloc={s.get('allocation')}"
        elif s["stage"] == "D_pool":
            line += (f"  pool_edges={s.get('pool_edges_total')} "
                     f"bt_edges={s.get('bt_edges_aggregated')} anchor={s.get('anchor')} "
                     f"converged={s.get('converged')}")
        print(line)
    print("-" * 78)

    # Rating table (pooled BT, the pinned anchor at 0).
    ratings = report["ratings"]
    fit = ratings.get("fit") or {}
    print(f"POOLED Bradley-Terry ratings  (anchor={ratings.get('anchor')} pinned 0 Elo)")
    if fit.get("converged"):
        print(f"  fit: max|grad|={fit.get('max_grad'):.2e} iters={fit.get('iterations')} "
              f"players={fit.get('n_players')} edges={fit.get('n_edges')}")
        print(f"  {'player':>16} {'elo':>8} {'95% CI':>18} {'SE':>7}")
        for row in ratings["players"]:
            lo, hi = row["elo_ci95"]
            tag = "  <-- anchor" if row["is_anchor"] else ""
            print(f"  {row['label']:>16} {row['elo']:8.1f} "
                  f"[{lo:7.1f},{hi:7.1f}] {row['se_elo']:7.1f}{tag}")
    else:
        print(f"  [fit unavailable] {fit.get('error')}")
    print("-" * 78)

    # Descriptive edges this epoch.
    print("This-epoch edges (DESCRIPTIVE except the PRIMARY):")
    for e in report["edges"]:
        prim = "PRIMARY" if e.get("primary") else "descr. "
        wr = e.get("winrate")
        ci = e.get("winrate_ci95")
        print(f"  [{prim}] vs {e['opponent']:>14} ({e['role']:<9}) "
              f"winrate={wr} ci95={ci} decided={e.get('decided')}")
    print("-" * 78)

    # The single PRIMARY verdict.
    verdict = report["verdict"]
    print(f"VERDICT (reported only, gates nothing): {verdict['label']}")
    primary = verdict.get("primary")
    if primary:
        print(f"  primary: {primary['candidate']} vs {primary['champion']}  "
              f"elo_diff={primary['elo_diff']} ci95={primary['elo_diff_ci95']} "
              f"se={primary['se_elo']}")
        print(f"  thresholds: promote>{primary['promote_threshold_elo']}  "
              f"regress<{primary['regress_threshold_elo']}  alpha={primary['alpha']}")
    if verdict.get("note"):
        print(f"  note: {verdict['note']}")
    print(f"  sealbot winrate ci95: {report.get('sealbot_winrate_ci95')}")
    print("=" * 78)


def _print_part(result: dict, *, indent: str = "") -> None:
    """One-line summary of a single part's outcome (run_eval_part result)."""
    status = result.get("status")
    part = result.get("part")
    line = f"{indent}[part {part:>14}] {status}"
    if status == "unknown_part":
        line += f"  available={result.get('available_parts')}"
    elif status in {"skipped", "empty", "unavailable"}:
        line += f"  ({result.get('reason')})"
    elif status == "played":
        edge = result.get("edge") or {}
        line += (f"  role={result.get('role')} winrate={edge.get('winrate')} "
                 f"ci95={edge.get('winrate_ci95')} decided={edge.get('decided')} "
                 f"pool_edges={result.get('pool_edges_total')}")
    print(line)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone multi-stage hexfield_eq eval (PURE EVAL)."
    )
    parser.add_argument("run_dir")
    parser.add_argument("candidate_ckpt")
    parser.add_argument("--epoch", type=int, default=None, help="override candidate epoch")
    parser.add_argument("--label", default=None, help="override candidate label")
    parser.add_argument("--config", default=None,
                        help="run toml for search/eval knobs (configs/hexfield_eq_main_1.toml)")
    parser.add_argument("--budget", type=int, default=None, help="override Stage C paired games")
    parser.add_argument("--visits", type=int, default=None,
                        help="override the reduced eval_visits knob (legacy/cheap screen)")
    parser.add_argument("--full-visits", type=int, default=None,
                        help="override the FULL-sims eval budget (default: selfplay.search_visits=512); "
                             "this is what Stage B + Stage C actually play at")
    parser.add_argument("--device", default=None, help="cuda|cpu (default: config/cuda)")
    parser.add_argument("--no-sprt", action="store_true", help="skip Stage B SPRT screen")
    parser.add_argument("--no-sealbot", action="store_true", help="drop the SealBot zero-point")
    parser.add_argument("--no-strix", action="store_true",
                        help="drop the Strix zero-point (D4 preferred anchor)")
    parser.add_argument("--strix-ckpt", default=None,
                        help="override opponents.strix_ckpt (implies strix_enabled)")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve roster + allocation only; play no games, write nothing")
    parser.add_argument("--json", action="store_true", help="print the full report JSON")
    # ----- Run-in-parts (resumable per-opponent chunks). -----
    parser.add_argument("--parts", action="store_true",
                        help="run as a sequence of per-opponent parts (durable + resumable), "
                             "then the aggregate fit + verdict")
    parser.add_argument("--opponent", default=None,
                        help="run a SINGLE part for one opponent label (strix|sealbot|ep5|...) "
                             "and exit; appends that one edge to the pool")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="fit the persisted pool + print the verdict, playing NO games")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                        help="skip parts already pooled for this epoch (default ON)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="force parts to replay even if already pooled (appends a fresh row)")
    args = parser.parse_args()

    # Checklist B5: the support radius is a process-global read; an unset env
    # builds radius-8 and every main_11-corpus arm is radius-4. The load-time
    # guards will refuse the checkpoint, but say why up front.
    if os.environ.get("HEXFIELD_EQ_SUPPORT_RADIUS") is None:
        print(
            "[WARN] HEXFIELD_EQ_SUPPORT_RADIUS is not set - this process builds "
            "the radius-8 default. main_11-corpus arm checkpoints are radius-4 "
            "and their meta guard will refuse to load. Set it explicitly "
            "(checklist B5).",
            file=sys.stderr,
        )

    run_dir = Path(args.run_dir)
    candidate_ckpt = Path(args.candidate_ckpt)
    if not candidate_ckpt.is_file():
        print(f"[FATAL] candidate checkpoint not found: {candidate_ckpt}", file=sys.stderr)
        return 2

    cfg = _apply_overrides(_load_config(args.config), args)
    section = cfg.multi_stage_eval

    if args.dry_run:
        roster = mse.select_opponents(
            run_dir, candidate_ckpt, section,
            candidate_epoch=args.epoch, candidate_label=args.label,
        )
        alloc = mse.allocate_budget(
            section.games_budget,
            n_checkpoint_opponents=len(roster.opponents),
            has_sealbot=roster.sealbot is not None,
            sealbot_share=section.sealbot_share,
            has_strix=roster.strix is not None,
            strix_share=section.strix_share,
        )
        print(json.dumps(
            {
                "dry_run": True,
                "roster": mse._roster_summary(roster),
                "allocation": alloc,
                "config": mse._config_summary(section),
            },
            indent=2,
        ))
        return 0

    # ----- Single PART: play one opponent, append its edge, exit. -----
    if args.opponent is not None:
        result = mse.run_eval_part(
            run_dir,
            candidate_ckpt,
            args.opponent,
            cfg,
            candidate_epoch=args.epoch,
            candidate_label=args.label,
            write_pool=True,
            resume=args.resume,
        )
        # ``pool_doc`` is internal plumbing for the in-parts orchestrator; never
        # print the whole pool from the single-part CLI.
        result.pop("pool_doc", None)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            _print_part(result)
        # An unknown part label is a usage error; everything else is success
        # (a fail-open SealBot/Strix drop is a recorded status, not a failure).
        return 2 if result.get("status") == "unknown_part" else 0

    # ----- AGGREGATE-ONLY: fit the persisted pool + verdict, play no games. -----
    if args.aggregate_only:
        report = mse.aggregate_pool(
            run_dir,
            candidate_ckpt,
            cfg,
            candidate_epoch=args.epoch,
            candidate_label=args.label,
            write_diagnostics=True,
        )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_report(report)
        return 0

    # ----- Full eval: monolithic (default) or run-in-parts (--parts). -----
    if args.parts:
        report = mse.run_multistage_eval_in_parts(
            run_dir,
            candidate_ckpt,
            cfg,
            candidate_epoch=args.epoch,
            candidate_label=args.label,
            write_diagnostics=True,
            resume=args.resume,
        )
    else:
        report = mse.run_multistage_eval(
            run_dir,
            candidate_ckpt,
            cfg,
            candidate_epoch=args.epoch,
            candidate_label=args.label,
            write_diagnostics=True,
        )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        if report.get("parts"):
            print("PARTS (resumable per-opponent chunks):")
            for p in report["parts"]:
                _print_part(p, indent="  ")
            print("-" * 78)
        _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
