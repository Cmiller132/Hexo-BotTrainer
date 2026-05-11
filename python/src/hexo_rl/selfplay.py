"""Python side of the Rust self-play bridge."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import HexoConfig, resolve_path


@dataclass(frozen=True)
class SelfplayResult:
    cycle: int
    output_dir: Path
    games: int
    samples: int
    bridge: str
    status: str


class RustSelfplayBridge:
    """Adapter for future PyO3 bindings or a Rust self-play binary."""

    def __init__(self, config: HexoConfig) -> None:
        self.config = config

    def _rust_module(self) -> Any | None:
        try:
            import hexo_rl_rust  # type: ignore[import-not-found]
        except ImportError:
            return None
        return hexo_rl_rust

    def test_engine(self) -> dict[str, Any]:
        module = self._rust_module()
        if module and hasattr(module, "test_engine"):
            return dict(module.test_engine())
        if module and hasattr(module, "run_uniform_selfplay"):
            game_config = module.PySelfplayConfig(
                max_placements=min(self.config.game.max_placements, 32),
                crop_size=self.config.game.crop_size,
            )
            mcts_config = module.PyMctsConfig(
                visits=4,
                c_puct=self.config.mcts.c_puct,
                crop_size=self.config.game.crop_size,
                temperature=0.0,
            )
            summary = module.run_uniform_selfplay(game_config, mcts_config)
            return {
                "status": "ok",
                "engine": "hexo_rl_rust",
                "samples": int(summary.samples),
                "placements_made": int(summary.placements_made),
                "terminal": bool(summary.terminal),
            }
        return {
            "status": "unavailable",
            "reason": "hexo_rl_rust bindings are not installed yet",
        }

    def random_game(self) -> dict[str, Any]:
        module = self._rust_module()
        if module and hasattr(module, "random_game"):
            return dict(module.random_game())
        if module and hasattr(module, "run_uniform_selfplay"):
            game_config = module.PySelfplayConfig(
                max_placements=self.config.game.max_placements,
                crop_size=self.config.game.crop_size,
            )
            mcts_config = module.PyMctsConfig(
                visits=max(1, min(self.config.selfplay.mcts_visits, 16)),
                c_puct=self.config.mcts.c_puct,
                crop_size=self.config.game.crop_size,
                temperature=1.0,
            )
            summary = module.run_uniform_selfplay(game_config, mcts_config)
            return {
                "status": "completed",
                "engine": "hexo_rl_rust",
                "samples": int(summary.samples),
                "placements_made": int(summary.placements_made),
                "terminal": bool(summary.terminal),
            }
        return {
            "status": "unavailable",
            "reason": "random game generation lives in the Rust engine",
        }

    def run_cycle(self, cycle: int, output_dir: Path) -> SelfplayResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.selfplay.rust_binary:
            command = [
                self.config.selfplay.rust_binary,
                "selfplay",
                "--cycle",
                str(cycle),
                "--output",
                str(output_dir),
                "--games",
                str(self.config.selfplay.games_per_cycle),
            ]
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            metadata = {
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
            (output_dir / "bridge.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            return SelfplayResult(
                cycle=cycle,
                output_dir=output_dir,
                games=self.config.selfplay.games_per_cycle,
                samples=0,
                bridge="subprocess",
                status="completed",
            )

        module = self._rust_module()
        if module and hasattr(module, "generate_selfplay"):
            result = module.generate_selfplay(self.config.to_dict(), cycle, str(output_dir))
            return SelfplayResult(
                cycle=cycle,
                output_dir=output_dir,
                games=int(result.get("games", 0)),
                samples=int(result.get("samples", 0)),
                bridge="pyo3",
                status=str(result.get("status", "completed")),
            )

        manifest = {
            "status": "placeholder",
            "cycle": cycle,
            "message": "Rust self-play bridge is not wired yet.",
        }
        (output_dir / "bridge.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return SelfplayResult(
            cycle=cycle,
            output_dir=output_dir,
            games=0,
            samples=0,
            bridge="placeholder",
            status="placeholder",
        )


def cycle_output_dir(config: HexoConfig, cycle: int) -> Path:
    root = resolve_path(config, config.paths.selfplay_root)
    return root / f"cycle_{cycle:06d}"


def run_selfplay_cycle(config: HexoConfig, cycle: int) -> SelfplayResult:
    bridge = RustSelfplayBridge(config)
    return bridge.run_cycle(cycle, cycle_output_dir(config, cycle))
