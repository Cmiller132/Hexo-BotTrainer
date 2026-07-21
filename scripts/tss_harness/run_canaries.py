"""Run every registered canary against the tss_batch adapter.

Usage (harness-dev venv, from the worktree root):
    python scripts/tss_harness/run_canaries.py

Exit 0 = all canaries fired correctly in both directions; nonzero = at
least one failed (run-invalidating in a real harness run).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tss_harness import canaries  # noqa: F401  populates the registry
from tss_harness.adapters.tss_batch import TssBatchAdapter
from tss_harness.gates import _CANARIES, canary_for


def main() -> int:
    make = lambda cfg: TssBatchAdapter(cfg)  # noqa: E731
    failed = 0
    for feat in sorted(_CANARIES):
        fired, detail = canary_for(feat)(make)
        tag = "PASS" if fired else "FAIL"
        print(f"{feat}: {tag} - {detail}")
        failed += 0 if fired else 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
