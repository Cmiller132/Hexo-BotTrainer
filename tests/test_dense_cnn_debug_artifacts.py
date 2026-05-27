from __future__ import annotations

import importlib
import struct
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_engine", "hexo_train", "hexo_runner"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _modules() -> tuple[Any, Any]:
    pytest.importorskip("torch")
    return (
        importlib.import_module("hexo_models.dense_cnn.debug_artifacts"),
        importlib.import_module("hexo_models.dense_cnn.d6"),
    )


def _action_ids(d6: Any, coords: list[tuple[int, int]]) -> tuple[int, ...]:
    return tuple(d6.pack_coord_id(d6.Axial(q, r)) for q, r in coords)


def _png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert data[12:16] == b"IHDR"
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def test_render_preview_game_actions_writes_png_pages_and_metadata(tmp_path: Path) -> None:
    debug_artifacts, d6 = _modules()
    action_ids = _action_ids(d6, [(0, 0), (1, 0), (1, 1), (2, 1)])

    metadata = debug_artifacts.render_preview_game_actions(
        action_ids,
        tmp_path,
        game_id="unit game",
        actions_per_image=2,
        max_images=3,
    )

    assert metadata["status"] == "rendered"
    assert metadata["game_id"] == "unit game"
    assert metadata["action_count"] == 4
    assert metadata["rendered_actions"] == 4
    assert metadata["omitted_actions"] == 0
    assert metadata["crop_size"] == 41
    assert metadata["center"] == {"q": 1, "r": 0}
    assert metadata["renderer"] in {"matplotlib", "pillow", "builtin_png"}
    assert len(metadata["files"]) == 2
    assert len(metadata["actions"]) == 4
    assert metadata["actions"][0]["q"] == 0
    assert metadata["actions"][0]["r"] == 0
    assert metadata["actions"][-1]["player"] == "player1"
    assert all(action["in_crop"] for action in metadata["actions"])

    for file_metadata in metadata["files"]:
        path = Path(file_metadata["path"])
        assert path.exists()
        assert path.name.startswith("unit-game.")
        assert path.stat().st_size > 0
        assert file_metadata["bytes"] == path.stat().st_size
        assert file_metadata["renderer"] in {"matplotlib", "pillow", "builtin_png"}
        width, height = _png_dimensions(path)
        assert width > 0
        assert height > 0


def test_render_preview_game_actions_respects_action_and_image_bounds(tmp_path: Path) -> None:
    debug_artifacts, d6 = _modules()
    action_ids = _action_ids(d6, [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])

    metadata = debug_artifacts.render_preview_game_actions(
        action_ids,
        tmp_path,
        game_id="bounded",
        max_actions=3,
        max_images=1,
        actions_per_image=2,
    )

    assert metadata["status"] == "rendered"
    assert metadata["action_count"] == 5
    assert metadata["rendered_actions"] == 2
    assert metadata["omitted_actions"] == 3
    assert len(metadata["actions"]) == 2
    assert len(metadata["files"]) == 1
    assert metadata["files"][0]["action_end"] == 2
    assert Path(metadata["files"][0]["path"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
