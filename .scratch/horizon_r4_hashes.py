"""Write the final SHA-256 manifest for Horizon R4 deliverables/evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / ".scratch"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    explicit = {
        ROOT / "docs" / "REPORT_HORIZON_R4.md",
        SCRATCH / "HORIZON_R4_STATE.md",
        SCRATCH / "horizon_native" / "Cargo.toml",
        SCRATCH / "horizon_native" / "Cargo.lock",
        SCRATCH / "horizon_native" / "README.md",
        SCRATCH / "horizon_native" / "driver.py",
        SCRATCH / "horizon_native" / "src" / "lib.rs",
        SCRATCH / "horizon_native" / "src" / "main.rs",
        SCRATCH / "horizon_native" / ".target" / "release" / "horizon_native.exe",
        Path(__file__),
    }
    generated = {
        path
        for pattern in ("horizon_r4_*.py", "horizon_r4_*.json", "horizon_r4_*_rows.jsonl")
        for path in SCRATCH.glob(pattern)
        if path.name != "horizon_r4_hashes.json"
    }
    files = sorted(explicit | generated, key=lambda path: str(path).lower())
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    records = {
        str(path.relative_to(ROOT)).replace("\\", "/"): {
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
        for path in files
    }
    payload = {
        "metadata": {
            "schema": 1,
            "algorithm": "SHA-256 over exact file bytes",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "tracked_head": "43cbdffb77d412b8b6800a239c2af9a67006623c",
            "branch": "claude/deadline-ladder",
            "manifest_self_excluded": True,
        },
        "file_count": len(records),
        "files": records,
    }
    out = SCRATCH / "horizon_r4_hashes.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"HORIZON_R4_HASHES_OK files={len(records)} out={out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
