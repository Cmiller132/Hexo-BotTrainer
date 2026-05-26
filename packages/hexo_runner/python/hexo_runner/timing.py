"""Small timing helpers for runner records."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True, slots=True)
class Timer:
    started: float

    @classmethod
    def start(cls) -> "Timer":
        return cls(started=perf_counter())

    def elapsed_ms(self) -> float:
        return round((perf_counter() - self.started) * 1000.0, 3)
