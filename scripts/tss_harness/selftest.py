"""Harness self-test — every gate is deliberately violated and MUST catch
it (PLAN §3.7: a gate that cannot fail is not a gate). Pure-python mocks,
no engine required.

Usage: python scripts/tss_harness/selftest.py   (exit 0 = harness honest)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tss_harness.contract import ArmSpec, Position, SolveRecord
from tss_harness.diff import compare, mcnemar_p
from tss_harness.gates import (
    gate_determinism,
    gate_features_have_canaries,
    gate_ground_truth,
    gate_manifest,
    gate_soundness,
)

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, caught: bool, why: str = "") -> None:
    CHECKS.append((name, caught, why))
    print(f"{'CAUGHT' if caught else 'MISSED'}  {name}" + (f" ({why})" if why else ""))


def rec(pid="p1", status="win", verified=True, vf=0, cost=10) -> SolveRecord:
    return SolveRecord(pos_id=pid, status=status, verified=verified,
                       verify_failed=vf, wall_nanos=1000, cost=cost)


def main() -> int:
    # 1. Manifest lie: declared value differs from the echo.
    arm = ArmSpec(name="x", adapter="mock",
                  declared={"shared_fragments_enabled": True})
    g = gate_manifest(arm, {"shared_fragments_enabled": False})
    check("manifest: declared-but-not-effective", g.fatal, g.detail)
    g = gate_manifest(arm, {})
    check("manifest: declared key absent", g.fatal, g.detail)

    # 2. Feature without canary is unclaimable.
    arm = ArmSpec(name="x", adapter="mock", features=("nonexistent_feature",))
    g = gate_features_have_canaries(arm)
    check("canary rule: undeclared-checkable feature", g.fatal, g.detail)

    # 3. Soundness: verify counter, vocabulary, decided-but-unverified.
    check("soundness: verify_failed>0",
          gate_soundness([rec(vf=1)]).fatal)
    check("soundness: bad status vocabulary",
          gate_soundness([rec(status="draw")]).fatal)
    check("soundness: decided but unverified",
          gate_soundness([rec(verified=False)]).fatal)
    check("soundness: clean records pass",
          not gate_soundness([rec()]).fatal)

    # 4. Ground truth: contradiction and lost must-solve.
    pos = [Position(pos_id="p1", source="fixture", moves=(),
                    labels={"verdict": "loss"})]
    check("ground truth: verified contradiction",
          gate_ground_truth(pos, [rec(status="win")]).fatal)
    pos_must = [Position(pos_id="p1", source="fixture", moves=(),
                         labels={"verdict": "win", "must_solve": True})]
    check("ground truth: lost must-solve",
          gate_ground_truth(pos_must, [rec(status="unknown", verified=False)]).fatal)
    check("ground truth: matching verdict passes",
          not gate_ground_truth(pos_must, [rec(status="win")]).fatal)

    # 5. Determinism: cost or verdict drift between identical re-solves.
    check("determinism: cost drift",
          gate_determinism([rec(cost=10)], [rec(cost=11)]).fatal)
    check("determinism: verdict drift",
          gate_determinism([rec(status="win")], [rec(status="unknown", verified=False)]).fatal)
    check("determinism: identical passes",
          not gate_determinism([rec()], [rec()]).fatal)

    # 6. Diff: verified contradictions surface as a soundness alarm; churn
    #    is never netted; McNemar sanity.
    a = [rec("p1", "win"), rec("p2", "unknown", verified=False)]
    b = [rec("p1", "loss"), rec("p2", "unknown", verified=False)]
    d = compare(a, b)
    check("diff: verified WIN-vs-LOSS contradiction alarms",
          bool(d["verified_contradictions"]))
    a = [rec("p1", "win"), rec("p2", "unknown", verified=False)]
    b = [rec("p1", "unknown", verified=False), rec("p2", "win")]
    d = compare(a, b)
    check("diff: churn not netted",
          len(d["upgrades_b_over_a"]) == 1 and len(d["downgrades_b_under_a"]) == 1)
    check("diff: mcnemar p sane",
          abs(mcnemar_p(0, 0) - 1.0) < 1e-9 and mcnemar_p(0, 15) < 1e-3)

    missed = [n for n, caught, _ in CHECKS if not caught]
    print(f"\n{len(CHECKS) - len(missed)}/{len(CHECKS)} violations caught")
    if missed:
        print("HARNESS DISHONEST — missed:", missed)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
