"""Config-free helpers for the optional Rust models-common extension."""

from __future__ import annotations

from typing import Any


def import_rust_module() -> Any | None:
    """Return the optional PyO3 extension module when it is installed."""

    try:
        from models_common import _rust  # type: ignore[attr-defined]
    except ImportError:
        return None
    return _rust


def rust_available() -> bool:
    """True when the optional Rust extension can be imported."""

    return import_rust_module() is not None


def run_uniform_selfplay(
    *,
    max_placements: int,
    crop_size: int,
    visits: int,
    c_puct: float,
    temperature: float,
) -> dict[str, Any] | None:
    """Run the Rust uniform-evaluator self-play smoke path if available."""

    module = import_rust_module()
    if module is None or not hasattr(module, "run_uniform_selfplay"):
        return None

    game_config = module.PySelfplayConfig(
        max_placements=max_placements,
        crop_size=crop_size,
    )
    mcts_config = module.PyMctsConfig(
        visits=visits,
        c_puct=c_puct,
        crop_size=crop_size,
        temperature=temperature,
    )
    summary = module.run_uniform_selfplay(game_config, mcts_config)
    return {
        "samples": int(summary.samples),
        "placements_made": int(summary.placements_made),
        "terminal": bool(summary.terminal),
    }
