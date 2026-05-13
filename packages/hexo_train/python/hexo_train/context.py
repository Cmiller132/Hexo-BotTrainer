"""Run-scoped state for one training invocation.

`RunContext` is deliberately separate from model components. It owns facts
about this run: config, directories, diagnostics, stage outputs, and shared
artifact locations. Model packages receive the context, but they should not
turn it into a home for tensor semantics or model-specific training logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .config import TrainingConfig
from .diagnostics import DiagnosticsWriter


@dataclass(slots=True)
class RunContext:
    """Mutable state shared by the ordered training stages."""

    config: TrainingConfig
    output_dir: Path
    checkpoint_dir: Path
    diagnostics_dir: Path
    samples_dir: Path
    diagnostics: DiagnosticsWriter
    stage_outputs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: TrainingConfig) -> "RunContext":
        """Create run directories and diagnostics from normalized config."""

        output_dir = config.run.output_dir
        checkpoint_dir = output_dir / "checkpoints"
        diagnostics_dir = output_dir / "diagnostics"
        samples_dir = output_dir / "samples"
        for directory in (output_dir, checkpoint_dir, diagnostics_dir, samples_dir):
            directory.mkdir(parents=True, exist_ok=True)

        return cls(
            config=config,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            diagnostics_dir=diagnostics_dir,
            samples_dir=samples_dir,
            diagnostics=DiagnosticsWriter(diagnostics_dir),
        )

    def section(self, name: str) -> Mapping[str, Any]:
        """Return a top-level config section as a mapping if it exists."""

        value = self.config.raw.get(name, {})
        if isinstance(value, Mapping):
            return value
        return {}

    def remember(self, stage: str, result: Any) -> Any:
        """Store a stage result for later stages and diagnostics."""

        self.stage_outputs[stage] = result
        return result

