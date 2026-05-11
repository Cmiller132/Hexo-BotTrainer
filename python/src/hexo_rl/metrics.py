"""Small metrics helpers for command-line runs."""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Mapping


@dataclass(frozen=True)
class MetricEvent:
    name: str
    value: float
    step: int | None = None
    tags: Mapping[str, Any] | None = None
    timestamp: float = 0.0

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "step": self.step,
            "tags": dict(self.tags or {}),
            "timestamp": self.timestamp or time.time(),
        }


class MetricLogger:
    def __init__(self, path: str | Path | None = None, *, window: int = 100) -> None:
        self.path = Path(path) if path else None
        self.window = window
        self.history: dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window))
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        name: str,
        value: float,
        *,
        step: int | None = None,
        tags: Mapping[str, Any] | None = None,
    ) -> None:
        self.history[name].append(float(value))
        if self.path:
            event = MetricEvent(name=name, value=float(value), step=step, tags=tags)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.payload(), separators=(",", ":")) + "\n")

    def averages(self) -> dict[str, float]:
        return {
            name: sum(values) / len(values)
            for name, values in self.history.items()
            if values
        }

