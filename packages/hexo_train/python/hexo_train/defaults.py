"""Default training components provided by `hexo_train`.

Defaults are intentionally small. They are useful for common policy/value
models, directory layout, and diagnostics, but they do not define what a
model's tensors mean. A plugin may accept a default or replace it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
import json

from .components import DefaultTrainingComponents, SharedComponents
from .context import RunContext


@dataclass(frozen=True, slots=True)
class ScalarValueTargetHelper:
    """Default scalar value target for simple win/loss/draw models."""

    win_value: float = 1.0
    loss_value: float = -1.0
    draw_value: float = 0.0

    def from_terminal_result(
        self,
        *,
        winner: Any | None,
        perspective: Any,
        is_draw: bool = False,
    ) -> float:
        """Return the value target from the sample player's perspective."""

        if is_draw or winner is None:
            return self.draw_value
        if winner == perspective:
            return self.win_value
        return self.loss_value


@dataclass(frozen=True, slots=True)
class LegalPolicyTargetHelper:
    """Default target helper for weights over engine-provided legal actions."""

    def normalize(self, weights: Mapping[Any, float]) -> Mapping[Any, float]:
        """Normalize action weights into a probability distribution."""

        total = sum(max(0.0, float(weight)) for weight in weights.values())
        if total <= 0.0:
            return {action: 0.0 for action in weights}
        return {
            action: max(0.0, float(weight)) / total
            for action, weight in weights.items()
        }


@dataclass(frozen=True, slots=True)
class CheckpointStore:
    """Run-local checkpoint path and placeholder metadata helper."""

    checkpoint_dir: Path

    def path_for(self, name: str) -> Path:
        return self.checkpoint_dir / f"{name}.ckpt"

    def write_placeholder(self, name: str, metadata: Mapping[str, Any]) -> Path:
        """Write a tiny metadata file until model checkpoint IO is implemented."""

        path = self.path_for(name)
        path.write_text(json.dumps(dict(metadata), indent=2), encoding="utf-8")
        return path


def build_shared_components(ctx: RunContext) -> SharedComponents:
    """Build model-neutral handles for one training run."""

    checkpoint_store = CheckpointStore(ctx.checkpoint_dir)
    defaults = DefaultTrainingComponents(
        scalar_value_target=ScalarValueTargetHelper(),
        legal_policy_target=LegalPolicyTargetHelper(),
        checkpoint_store=checkpoint_store,
        diagnostics=ctx.diagnostics,
    )
    shared = SharedComponents(
        defaults=defaults,
        game_spec=_build_game_spec(ctx.section("shared")),
    )
    _write_run_manifest(ctx)
    return shared


def _build_game_spec(shared_config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Describe engine/game dimensions needed by model construction."""

    return dict(shared_config.get("game", {}))


def _write_run_manifest(ctx: RunContext) -> None:
    """Write run metadata before any long-running stage starts."""

    manifest = {
        "run": asdict(ctx.config.run),
        "model": asdict(ctx.config.model),
        "stages": list(ctx.config.stages),
        "output_dir": str(ctx.output_dir),
    }
    (ctx.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )
