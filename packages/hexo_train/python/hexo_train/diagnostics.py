"""Diagnostics and run-output helpers for training orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import json
import time


@dataclass(slots=True)
class StageDiagnostic:
    """Small serializable record for one pipeline stage."""

    stage: str
    status: str
    elapsed_seconds: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


class DiagnosticsWriter:
    """Writes stage and run diagnostics to the run output directory."""

    def __init__(self, diagnostics_dir: Path) -> None:
        self.diagnostics_dir = diagnostics_dir
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)

    def start_stage(self, stage: str) -> float:
        self.write_event("stage_started", {"stage": stage})
        return time.perf_counter()

    def finish_stage(
        self,
        *,
        stage: str,
        started_at: float,
        status: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> StageDiagnostic:
        diagnostic = StageDiagnostic(
            stage=stage,
            status=status,
            elapsed_seconds=time.perf_counter() - started_at,
            metadata=dict(metadata or {}),
        )
        self.write_json(f"{stage}.json", diagnostic)
        self.write_event("stage_finished", diagnostic)
        return diagnostic

    def write_event(self, name: str, payload: Any) -> None:
        event_path = self.diagnostics_dir / "events.jsonl"
        with event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": name, "payload": _jsonable(payload)}))
            handle.write("\n")

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.diagnostics_dir / name
        path.write_text(
            json.dumps(_jsonable(payload), indent=2, default=str),
            encoding="utf-8",
        )
        return path


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {
            field_name: _jsonable(getattr(value, field_name))
            for field_name in value.__dataclass_fields__
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
