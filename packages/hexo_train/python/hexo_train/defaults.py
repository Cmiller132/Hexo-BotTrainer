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

from hexo_utils.samples import LegalPolicyTargetHelper, ScalarValueTargetHelper

from .components import DefaultTrainingComponents, SharedComponents
from .context import RunContext
from .symmetry import D6SymmetrySelector


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
        symmetry_selector=D6SymmetrySelector(),
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
