"""Runner-specific replay file maintenance."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from models_common.replay import iter_jsonl, list_replay_files

from .config import HexoConfig, resolve_path


def update_replay_from_cycle(config: HexoConfig, cycle_dir: str | Path) -> Path:
    """Refresh the configured latest replay file from a self-play cycle."""

    latest = resolve_path(config, config.paths.replay_latest)
    latest.parent.mkdir(parents=True, exist_ok=True)
    files = list_replay_files(cycle_dir)
    if not files:
        latest.touch()
        return latest

    with latest.open("wb") as output:
        for file_path in files:
            if file_path.suffix == ".zst":
                for sample in iter_jsonl(file_path):
                    output.write(json.dumps(sample, separators=(",", ":")).encode("utf-8") + b"\n")
            else:
                with file_path.open("rb") as source:
                    shutil.copyfileobj(source, output)
    return latest
