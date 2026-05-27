"""Bounded PNG previews for dense CNN debug game histories."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, Sequence
import re
import struct
import zlib

from .constants import BOARD_SIZE
from .d6 import Axial, unpack_coord_id
from .geometry import coord_to_row_col, crop_center

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True, slots=True)
class _Placement:
    index: int
    action_id: int
    coord: Axial
    row_col: tuple[int, int] | None

    @property
    def player(self) -> str:
        return "player0" if self.index % 2 == 0 else "player1"


def render_preview_game_actions(
    action_ids: Sequence[int],
    output_dir: str | Path,
    *,
    game_id: str = "preview-game",
    file_prefix: str | None = None,
    max_actions: int = 128,
    max_images: int = 4,
    actions_per_image: int = 64,
    crop_size: int = BOARD_SIZE,
    cell_pixels: int = 18,
) -> dict[str, Any]:
    """Render a bounded dense CNN crop preview for a game action list.

    The helper intentionally consumes only packed action IDs. Coordinates are
    decoded with the model package's axial helpers and projected into the same
    square crop used by dense CNN input planes.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    resolved_max_actions = max(0, int(max_actions))
    resolved_max_images = max(0, int(max_images))
    resolved_actions_per_image = max(1, int(actions_per_image))
    page_capacity = resolved_max_images * resolved_actions_per_image
    rendered_limit = min(len(action_ids), resolved_max_actions, page_capacity)
    rendered_ids = tuple(int(action_id) for action_id in action_ids[:rendered_limit])
    coords = tuple(unpack_coord_id(action_id) for action_id in rendered_ids)
    center = crop_center(coords)
    resolved_crop_size = max(1, int(crop_size))
    placements = tuple(
        _Placement(
            index=index,
            action_id=action_id,
            coord=coord,
            row_col=coord_to_row_col(coord, center=center, size=resolved_crop_size),
        )
        for index, (action_id, coord) in enumerate(zip(rendered_ids, coords))
    )

    metadata: dict[str, Any] = {
        "status": "rendered" if placements and resolved_max_images > 0 else "skipped",
        "game_id": str(game_id),
        "action_count": len(action_ids),
        "rendered_actions": len(placements),
        "omitted_actions": max(0, len(action_ids) - len(placements)),
        "crop_size": resolved_crop_size,
        "center": {"q": int(center.q), "r": int(center.r)},
        "renderer": None,
        "files": [],
        "actions": [_placement_metadata(placement) for placement in placements],
    }
    if not placements or resolved_max_images <= 0:
        return metadata

    renderer = _select_renderer()
    metadata["renderer"] = renderer
    prefix = _safe_filename(file_prefix or game_id or "preview-game")
    page_count = (len(placements) + resolved_actions_per_image - 1) // resolved_actions_per_image
    page_count = min(page_count, resolved_max_images)
    for page_index in range(page_count):
        end = min(len(placements), (page_index + 1) * resolved_actions_per_image)
        page_path = output_path / f"{prefix}.{page_index + 1:02d}.png"
        actual_renderer = _render_png(
            page_path,
            placements[:end],
            center=center,
            crop_size=resolved_crop_size,
            cell_pixels=max(4, int(cell_pixels)),
            renderer=renderer,
        )
        if page_index == 0:
            metadata["renderer"] = actual_renderer
        metadata["files"].append(
            {
                "path": str(page_path),
                "filename": page_path.name,
                "action_start": 0,
                "action_end": end,
                "action_count": end,
                "bytes": page_path.stat().st_size,
                "renderer": actual_renderer,
            }
        )
    return metadata


def _placement_metadata(placement: _Placement) -> dict[str, Any]:
    row_col = placement.row_col
    return {
        "index": placement.index,
        "action_id": placement.action_id,
        "q": int(placement.coord.q),
        "r": int(placement.coord.r),
        "row": None if row_col is None else row_col[0],
        "col": None if row_col is None else row_col[1],
        "player": placement.player,
        "in_crop": row_col is not None,
    }


def _select_renderer() -> str:
    if importlib_util.find_spec("matplotlib") is not None:
        return "matplotlib"
    if importlib_util.find_spec("PIL") is not None:
        return "pillow"
    return "builtin_png"


def _render_png(
    path: Path,
    placements: Sequence[_Placement],
    *,
    center: Axial,
    crop_size: int,
    cell_pixels: int,
    renderer: str,
) -> str:
    if renderer == "matplotlib":
        try:
            _render_matplotlib(path, placements, crop_size=crop_size)
            return "matplotlib"
        except Exception:
            renderer = "pillow" if importlib_util.find_spec("PIL") is not None else "builtin_png"
    if renderer == "pillow":
        try:
            _render_pillow(path, placements, crop_size=crop_size, cell_pixels=cell_pixels)
            return "pillow"
        except Exception:
            pass
    # Dependency-free fallback: write a simple RGB PNG with zlib-compressed
    # scanlines. This keeps debug preview generation available in minimal envs.
    _render_builtin_png(path, placements, center=center, crop_size=crop_size, cell_pixels=cell_pixels)
    return "builtin_png"


def _render_matplotlib(path: Path, placements: Sequence[_Placement], *, crop_size: int) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot
    from matplotlib.colors import ListedColormap

    values = [[0 for _col in range(crop_size)] for _row in range(crop_size)]
    for placement in placements:
        if placement.row_col is None:
            continue
        row, col = placement.row_col
        values[row][col] = 1 if placement.player == "player0" else 2
    if placements and placements[-1].row_col is not None:
        row, col = placements[-1].row_col
        values[row][col] = 3

    figure, axes = pyplot.subplots(figsize=(6.0, 6.0), dpi=120)
    axes.imshow(
        values,
        cmap=ListedColormap(("#f8fafc", "#2563eb", "#dc2626", "#111827")),
        vmin=0,
        vmax=3,
        interpolation="nearest",
    )
    axes.set_xticks([item - 0.5 for item in range(crop_size + 1)], minor=True)
    axes.set_yticks([item - 0.5 for item in range(crop_size + 1)], minor=True)
    axes.grid(which="minor", color="#cbd5e1", linewidth=0.35)
    axes.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    for spine in axes.spines.values():
        spine.set_visible(False)
    figure.tight_layout(pad=0.05)
    figure.savefig(path, format="png")
    pyplot.close(figure)


def _render_pillow(
    path: Path,
    placements: Sequence[_Placement],
    *,
    crop_size: int,
    cell_pixels: int,
) -> None:
    from PIL import Image, ImageDraw

    width = crop_size * cell_pixels + 1
    height = crop_size * cell_pixels + 1
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    _draw_cells(draw.rectangle, placements, crop_size=crop_size, cell_pixels=cell_pixels)
    grid_color = (203, 213, 225)
    for index in range(crop_size + 1):
        pos = index * cell_pixels
        draw.line([(pos, 0), (pos, height - 1)], fill=grid_color)
        draw.line([(0, pos), (width - 1, pos)], fill=grid_color)
    image.save(path, format="PNG")


def _render_builtin_png(
    path: Path,
    placements: Sequence[_Placement],
    *,
    center: Axial,
    crop_size: int,
    cell_pixels: int,
) -> None:
    _ = center
    width = crop_size * cell_pixels + 1
    height = crop_size * cell_pixels + 1
    pixels = bytearray((248, 250, 252) * width * height)

    def rectangle(box: tuple[int, int, int, int], fill: tuple[int, int, int]) -> None:
        left, top, right, bottom = box
        left = max(0, left)
        top = max(0, top)
        right = min(width - 1, right)
        bottom = min(height - 1, bottom)
        for y in range(top, bottom + 1):
            offset = (y * width + left) * 3
            for _x in range(left, right + 1):
                pixels[offset : offset + 3] = bytes(fill)
                offset += 3

    _draw_cells(rectangle, placements, crop_size=crop_size, cell_pixels=cell_pixels)
    grid_color = (203, 213, 225)
    for index in range(crop_size + 1):
        pos = index * cell_pixels
        rectangle((pos, 0, pos, height - 1), grid_color)
        rectangle((0, pos, width - 1, pos), grid_color)
    _write_rgb_png(path, width, height, bytes(pixels))


def _draw_cells(
    rectangle: Any,
    placements: Sequence[_Placement],
    *,
    crop_size: int,
    cell_pixels: int,
) -> None:
    fill_by_player = {
        "player0": (37, 99, 235),
        "player1": (220, 38, 38),
    }
    for placement in placements:
        if placement.row_col is None:
            continue
        row, col = placement.row_col
        if not 0 <= row < crop_size or not 0 <= col < crop_size:
            continue
        left = col * cell_pixels + 1
        top = row * cell_pixels + 1
        right = (col + 1) * cell_pixels - 1
        bottom = (row + 1) * cell_pixels - 1
        rectangle((left, top, right, bottom), fill=fill_by_player[placement.player])
    if placements and placements[-1].row_col is not None:
        row, col = placements[-1].row_col
        left = col * cell_pixels + 2
        top = row * cell_pixels + 2
        right = (col + 1) * cell_pixels - 2
        bottom = (row + 1) * cell_pixels - 2
        rectangle((left, top, right, bottom), fill=(17, 24, 39))


def _write_rgb_png(path: Path, width: int, height: int, pixels: bytes) -> None:
    if len(pixels) != width * height * 3:
        raise ValueError("RGB pixel buffer has an unexpected size")
    rows = []
    stride = width * 3
    for y in range(height):
        rows.append(b"\x00" + pixels[y * stride : (y + 1) * stride])
    raw = b"".join(rows)
    chunks = [
        _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
        _png_chunk(b"IDAT", zlib.compress(raw, level=6)),
        _png_chunk(b"IEND", b""),
    ]
    path.write_bytes(_PNG_SIGNATURE + b"".join(chunks))


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return cleaned or "preview-game"


__all__ = ["render_preview_game_actions"]
