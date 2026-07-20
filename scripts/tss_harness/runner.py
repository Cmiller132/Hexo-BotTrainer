"""Harness runner — `make a change, run it`. PLAN §4/§5, BENCH_GAPS wiring.

Usage (harness-dev venv, worktree root):
    python scripts/tss_harness/runner.py run \
        --label mychange --tier quick|standard|full \
        [--config-json '{"horizon":0,...}'] [--baseline <run-dir>] \
        [--adoption]           # include holdout splits (adoption gates only)
    python scripts/tss_harness/runner.py compare <run-dir-a> <run-dir-b>

Tiers:
    quick    canaries + manifest + determinism shard + ~15% stratified sample
    standard full dev-split sweep over every frozen set + the 5-min bench
    full     standard + second determinism pass + warmth/goal-both arms

Exit nonzero on any HARD gate failure (the run directory records why).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tss_harness import canaries  # noqa: F401  populate registry
from tss_harness.adapters.tss_batch import TssBatchAdapter, declared_features
from tss_harness.archive import RunArchive, load_records
from tss_harness.contract import ArmSpec, LOSS, WIN
from tss_harness.diff import compare, unknown_wall_share
from tss_harness.gates import (
    GateReport,
    GateResult,
    canary_for,
    gate_determinism,
    gate_features_have_canaries,
    gate_ground_truth,
    gate_manifest,
    gate_soundness,
)
from tss_harness.sets import SETS_DIR, load_set

BENCH_PYTHON = "/root/.venvs/hexo-bottrainer-wsl/bin/python"
COVERAGE_SETS = ("selfplay_v1", "human_v1", "puzzle_v3")   # frozen-if-present
# puzzle_v1 is superseded (its must_solve required dedicated-loss verdicts a
# Both-goal arm structurally cannot produce — see SOLVER_NOTES P4); pinned
# file kept immutable per the no-in-place-edit rule.

U32_MAX = 4294967295


def _bench_overlay(effective: dict) -> dict:
    """Translate the adapter's effective config into the bench's tss_* toml
    vocabulary. The bench builds its config from the production toml, whose
    defaults are NOT the arm's defaults (caught live 2026-07-20: config {}
    benched engine-default h16 while the coverage sweep ran unbounded) — so
    the FULL effective arm config is always passed, never just the user's
    delta. Raises on arm fields the bench cannot express rather than
    silently benching something else."""
    if not effective.get("wide", True):
        raise RuntimeError("bench cannot express the narrow profile")
    if effective.get("shared_fragments"):
        # TSS_SHARED_FRAGMENTS would need to travel via subprocess env AND
        # be verified engine-side; not wired yet — refuse, don't fake.
        raise RuntimeError("bench cannot yet carry shared_fragments arms")
    return {
        "tss_solver_node_cap": int(effective["node_cap"]),
        "tss_solver_horizon": int(effective["horizon"]),
        "tss_solver_horizon_ladder": bool(effective["ladder"]),
        "tss_zone": bool(effective["zone"]),
        "tss_solver_dual_pass": bool(effective.get("dual_pass", False)),
    }


def gate_bench_identity(manifest: dict, scorecard: dict) -> GateResult:
    """The bench must have measured the SAME solver arm the coverage sweep
    solved with: compare the bench's resolved tss fields against the arm
    manifest (both are echoes, not intents)."""
    eff = scorecard.get("effective_tss") or {}
    problems = []
    if not eff:
        problems.append("scorecard carries no effective_tss echo")
    else:
        if int(eff.get("tss_solver_node_cap", -1)) != int(manifest["node_cap"]):
            problems.append(
                f"node_cap {eff.get('tss_solver_node_cap')} != {manifest['node_cap']}")
        if bool(eff.get("tss_solver_dual_pass", False)) != bool(
                manifest.get("dual_pass", False)):
            problems.append(
                f"dual_pass mismatch: bench {eff.get('tss_solver_dual_pass')} "
                f"vs manifest {manifest.get('dual_pass')}")
        bench_unbounded = int(eff.get("tss_solver_horizon", -1)) == 0
        arm_unbounded = int(manifest["semantic_horizon"]) == U32_MAX
        if bench_unbounded != arm_unbounded:
            problems.append(
                f"horizon mismatch: bench tss_solver_horizon="
                f"{eff.get('tss_solver_horizon')} vs manifest semantic_horizon="
                f"{manifest['semantic_horizon']}")
        if not bool(eff.get("tss_enabled", False)):
            problems.append("bench ran with tss_enabled=false")
    return GateResult(
        gate="bench_identity", passed=not problems, hard=True,
        detail="; ".join(problems) or "bench arm matches coverage arm",
    )


def _available_sets() -> list[str]:
    return [n for n in COVERAGE_SETS if (SETS_DIR / f"{n}.jsonl").exists()]


def _sample(positions, fraction: float, salt: str):
    keep = []
    for p in positions:
        h = int(hashlib.sha256(f"{salt}:{p.pos_id}".encode()).hexdigest()[:8], 16)
        if h / 0xFFFFFFFF < fraction:
            keep.append(p)
    return keep


def _coverage(records) -> dict:
    return {
        "n": len(records),
        "win": sum(1 for r in records if r.status == WIN and r.verified),
        "loss": sum(1 for r in records if r.status == LOSS and r.verified),
    }


def run(args) -> int:
    config = json.loads(args.config_json) if args.config_json else {}
    arm = ArmSpec(
        name=args.label,
        adapter="tss_batch",
        config=config,
        declared=json.loads(args.declared_json) if args.declared_json else {},
        features=declared_features({**TssBatchAdapter.DEFAULTS, **config}),
    )
    adapter = TssBatchAdapter(config)
    archive = RunArchive(args.label)
    gates = GateReport()
    print(f"run dir: {archive.dir}")

    # 1. features must be checkable
    gates.add(gate_features_have_canaries(arm))
    # 2. manifest echo vs declared intent
    echoed = adapter.manifest()
    archive.save_manifest(arm.name, echoed)
    gates.add(gate_manifest(arm, echoed))
    # 3. canaries, both directions, before any expensive work
    canary_status = {}
    if not gates.fatal:
        make = lambda cfg: TssBatchAdapter({**config, **cfg})  # noqa: E731
        for feat in arm.features:
            fn = canary_for(feat)
            if fn is None:
                continue  # gate 1 already failed the run
            fired, detail = fn(make)
            canary_status[feat] = {"fired": fired, "detail": detail}
            if not fired:
                gates.add(GateResult(
                    gate=f"canary:{feat}", passed=False, hard=True, detail=detail
                ))
        print(f"canaries: {json.dumps(canary_status)}")

    report = {"arm": arm.name, "config": config, "tier": args.tier,
              "canaries": canary_status, "sets": {}}

    if not gates.fatal:
        frac = 0.15 if args.tier == "quick" else 1.0
        for set_name in _available_sets():
            split = "all" if args.adoption else "dev"
            positions = load_set(set_name, split)
            if frac < 1.0:
                positions = _sample(positions, frac, f"tier:{set_name}")
            records = adapter.solve_sequence(positions)
            archive.save_records(arm.name, set_name, records)
            gates.add(gate_soundness(records))
            if set_name.startswith("puzzle"):
                gates.add(gate_ground_truth(positions, records))
            node_cap = int(echoed.get("node_cap", 500))
            report["sets"][set_name] = {
                "split": split,
                "coverage": _coverage(records),
                "economics": unknown_wall_share(records, node_cap),
            }
            print(f"{set_name}: {json.dumps(report['sets'][set_name]['coverage'])}")

        # determinism shard: 5% re-solved, bit-identical verdict+cost
        det_positions = _sample(load_set("selfplay_v1", "dev"), 0.05, "determinism")
        first = adapter.solve_sequence(det_positions)
        second = adapter.solve_sequence(det_positions)
        gates.add(gate_determinism(first, second))

    # bench: every standard/full run (owner ruling), report-only adoption
    if args.tier in ("standard", "full") and not gates.fatal and not args.no_bench:
        out = archive.dir / "scorecard.json"
        overlay = _bench_overlay({**TssBatchAdapter.DEFAULTS, **config})
        cmd = [BENCH_PYTHON, "scripts/tss_harness/bench.py", "--full",
               "--config-json", json.dumps(overlay), "--out", str(out)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        report["bench_exit"] = proc.returncode
        if proc.returncode != 0:
            print(f"bench failed/refused:\n{proc.stdout[-2000:]}{proc.stderr[-2000:]}")
            gates.add(GateResult(gate="bench_ran", passed=False, hard=True,
                                 detail=f"bench exit {proc.returncode}"))
        else:
            scorecard = json.loads(out.read_text())
            gates.add(gate_bench_identity(echoed, scorecard))
            print(f"bench: {scorecard['moves_per_min']} moves/min, "
                  f"{scorecard['decisions']} decisions, "
                  f"vf={scorecard['verify_failed_total']}")

    archive.save_gates(gates.to_json())
    archive.save_report(report)

    if args.baseline:
        base_dir = Path(args.baseline)
        for set_name in report["sets"]:
            try:
                base = load_records(base_dir, _arm_of(base_dir), set_name)
            except FileNotFoundError:
                continue
            mine = load_records(archive.dir, arm.name, set_name)
            d = compare(base, mine, label_a="baseline", label_b=arm.name)
            (archive.dir / f"diff_{set_name}.json").write_text(
                json.dumps(d, indent=2), newline="\n")
            print(f"diff {set_name}: coverage {d['coverage_a']} -> "
                  f"{d['coverage_b']} (p={d['mcnemar_p']:.3g})"
                  + ("  !! VERIFIED CONTRADICTIONS" if d["verified_contradictions"] else ""))

    print("GATES:", "FATAL" if gates.fatal else "ALL PASS")
    for r in gates.results:
        if not r.passed:
            print(f"  FAIL {r.gate}: {r.detail}")
    return 1 if gates.fatal else 0


def _arm_of(run_dir: Path) -> str:
    m = list(run_dir.glob("manifest_*.json"))
    return m[0].stem[len("manifest_"):] if m else "unknown"


def cmd_compare(args) -> int:
    a, b = Path(args.a), Path(args.b)
    for set_name in COVERAGE_SETS:
        try:
            ra = load_records(a, _arm_of(a), set_name)
            rb = load_records(b, _arm_of(b), set_name)
        except FileNotFoundError:
            continue
        d = compare(ra, rb, label_a=a.name, label_b=b.name)
        print(json.dumps(d, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--label", required=True)
    r.add_argument("--tier", choices=("quick", "standard", "full"), default="quick")
    r.add_argument("--config-json", default="")
    r.add_argument("--declared-json", default="")
    r.add_argument("--baseline", default="")
    r.add_argument("--adoption", action="store_true")
    r.add_argument("--no-bench", action="store_true",
                   help="skip the bench (dev machines without the GPU venv)")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    return run(args) if args.cmd == "run" else cmd_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
