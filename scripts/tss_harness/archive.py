"""Run archive — every harness run is a self-describing directory. PLAN §5.

harness_runs/<utc>_<label>/
    fingerprint.json   environment + set hashes + schema + git rev
    manifest_<arm>.json  the adapter's effective-config echo (verbatim)
    records_<arm>_<set>.jsonl  SolveRecords
    gates.json         GateReport
    report.json        coverage/economics summary
    scorecard.json     bench scorecard (when the run included the bench)

Cross-build comparison is first-class: diff.py consumes any two archives.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from .contract import SCHEMA_VERSION, SolveRecord

RUNS_DIR = Path(__file__).resolve().parent / "harness_runs"


def _git_rev(cwd: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
            text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _load_fingerprint() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    sets_dir = Path(__file__).resolve().parent / "sets"
    set_hashes = {
        p.stem: p.read_text().split()[0]
        for p in sorted(sets_dir.glob("*.sha256"))
    }
    return {
        "schema": SCHEMA_VERSION,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_rev": _git_rev(root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "set_hashes": set_hashes,
        "tss_env": {
            k: v for k, v in __import__("os").environ.items()
            if k.startswith(("TSS_", "HEXFIELD_EQ_"))
        },
    }


class RunArchive:
    def __init__(self, label: str):
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        self.dir = RUNS_DIR / f"{stamp}_{label}"
        self.dir.mkdir(parents=True, exist_ok=False)
        self._write_json("fingerprint.json", _load_fingerprint())

    def _write_json(self, name: str, obj: Any) -> None:
        with open(self.dir / name, "w", newline="\n") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True, default=str)

    def save_manifest(self, arm: str, manifest: dict[str, Any]) -> None:
        self._write_json(f"manifest_{arm}.json", manifest)

    def save_records(
        self, arm: str, set_name: str, records: list[SolveRecord]
    ) -> None:
        path = self.dir / f"records_{arm}_{set_name}.jsonl"
        with open(path, "w", newline="\n") as fh:
            for r in records:
                fh.write(json.dumps(r.to_json(), sort_keys=True) + "\n")

    def save_gates(self, gates_json: list[dict[str, Any]]) -> None:
        self._write_json("gates.json", gates_json)

    def save_report(self, report: dict[str, Any]) -> None:
        self._write_json("report.json", report)

    def save_scorecard(self, scorecard: dict[str, Any]) -> None:
        self._write_json("scorecard.json", scorecard)


def load_records(run_dir: Path, arm: str, set_name: str) -> list[SolveRecord]:
    out = []
    for line in open(run_dir / f"records_{arm}_{set_name}.jsonl"):
        row = json.loads(line)
        out.append(SolveRecord(
            pos_id=row["pos_id"], status=row["status"],
            verified=row["verified"], verify_failed=row["verify_failed"],
            wall_nanos=row["wall_nanos"], cost=row["cost"],
            counters=row.get("counters", {}),
        ))
    return out


def list_runs() -> list[Path]:
    return sorted(RUNS_DIR.glob("*_*")) if RUNS_DIR.exists() else []
