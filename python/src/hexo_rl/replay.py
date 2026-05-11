"""Replay sample JSONL helpers."""

from __future__ import annotations

import json
import random
import shutil
from io import TextIOWrapper
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, TextIO

import torch

from .config import HexoConfig, resolve_path
from .inference import _as_tensor, encoded_state_tensor, legal_mask_from_state


@dataclass(frozen=True)
class ReplaySample:
    game: str
    rules_version: int
    state: Mapping[str, Any]
    current_player: int
    phase: str
    legal_actions: Sequence[Mapping[str, int] | Sequence[int]]
    policy_target: Sequence[Mapping[str, Any] | Sequence[Any]]
    value_target: float
    placements_made: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _open_text(path: Path, mode: str) -> TextIO:
    if path.suffix == ".zst":
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError("Reading .zst replay files requires zstandard") from exc
        binary_mode = mode.replace("t", "").replace("b", "") + "b"
        raw = path.open(binary_mode)
        if "r" in mode:
            stream = zstd.ZstdDecompressor().stream_reader(raw)
        else:
            stream = zstd.ZstdCompressor().stream_writer(raw)
        return TextIOWrapper(stream, encoding="utf-8")  # type: ignore[arg-type]
    return path.open(mode, encoding="utf-8")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    replay_path = Path(path)
    with _open_text(replay_path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, samples: Iterable[ReplaySample | Mapping[str, Any]]) -> int:
    replay_path = Path(path)
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with _open_text(replay_path, "wt") as handle:
        for sample in samples:
            payload = sample.to_json() if isinstance(sample, ReplaySample) else dict(sample)
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
            count += 1
    return count


def list_replay_files(path: str | Path) -> list[Path]:
    replay_path = Path(path)
    if replay_path.is_file():
        return [replay_path]
    if not replay_path.exists():
        return []
    return sorted(
        item
        for item in replay_path.rglob("*")
        if item.suffix in {".jsonl", ".zst"} or item.name.endswith(".jsonl.zst")
    )


def inspect_replay(path: str | Path) -> dict[str, Any]:
    files = list_replay_files(path)
    phases: Counter[str] = Counter()
    samples = 0
    for file_path in files:
        for sample in iter_jsonl(file_path):
            samples += 1
            phases[str(sample.get("phase", "unknown"))] += 1
    return {
        "path": str(path),
        "files": len(files),
        "samples": samples,
        "phases": dict(phases),
    }


def _coord_key(coord: Mapping[str, Any] | Sequence[Any]) -> str:
    if isinstance(coord, Mapping):
        return f"{coord['q']},{coord['r']}"
    return f"{coord[0]},{coord[1]}"


def _policy_target_tensor(
    sample: Mapping[str, Any],
    state: Mapping[str, Any],
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    dense = sample.get("policy_tensor", state.get("policy_target"))
    if dense is not None:
        tensor = _as_tensor(dense, dtype=torch.float32)
        if tensor.shape == legal_mask.shape:
            return tensor

    target = torch.zeros_like(legal_mask)
    sparse = sample.get("policy_target", [])
    coord_to_index = state.get("coord_to_index", {})
    origin = state.get("origin", {})
    for item in sparse:
        if isinstance(item, Mapping):
            weight = float(item.get("prob", item.get("weight", item.get("value", 0.0))))
            if "row" in item and "col" in item:
                target[int(item["row"]), int(item["col"])] = weight
                continue
            coord = item.get("coord", item)
        else:
            coord = item[0]
            weight = float(item[1])

        index = coord_to_index.get(_coord_key(coord)) if isinstance(coord_to_index, Mapping) else None
        if index is None and isinstance(coord, Mapping) and isinstance(origin, Mapping):
            row = int(coord["r"]) - int(origin["r"])
            col = int(coord["q"]) - int(origin["q"])
            if 0 <= row < target.shape[0] and 0 <= col < target.shape[1]:
                target[row, col] = weight
            continue
        if index is None:
            continue
        row, col = index if not isinstance(index, Mapping) else (index["row"], index["col"])
        target[int(row), int(col)] = weight
    return target


def _sample_to_tensors(sample: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    state = sample.get("state", {})
    if not isinstance(state, Mapping):
        raise TypeError("Replay sample state must be a mapping")
    state_tensor = encoded_state_tensor(state)
    legal_mask = legal_mask_from_state(state, state_tensor)
    policy_target = _policy_target_tensor(sample, state, legal_mask)
    return {
        "state_tensor": state_tensor,
        "legal_mask": legal_mask,
        "policy_target": policy_target.reshape_as(legal_mask),
        "value_target": torch.tensor(float(sample["value_target"]), dtype=torch.float32),
    }


def batch_from_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    items = [_sample_to_tensors(sample) for sample in samples]
    return {
        "state_tensor": torch.stack([item["state_tensor"] for item in items], dim=0),
        "legal_mask": torch.stack([item["legal_mask"] for item in items], dim=0),
        "policy_target": torch.stack([item["policy_target"] for item in items], dim=0),
        "value_target": torch.stack([item["value_target"] for item in items], dim=0),
    }


class ReplayBuffer:
    def __init__(self, samples: Sequence[Mapping[str, Any]]) -> None:
        if not samples:
            raise ValueError("ReplayBuffer requires at least one sample")
        self.samples = list(samples)

    @classmethod
    def from_path(cls, path: str | Path, *, limit: int | None = None) -> "ReplayBuffer":
        loaded: list[Mapping[str, Any]] = []
        for file_path in list_replay_files(path):
            loaded.extend(iter_jsonl(file_path))
            if limit is not None and len(loaded) >= limit:
                loaded = loaded[-limit:]
        return cls(loaded[-limit:] if limit is not None else loaded)

    def sample_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        return batch_from_samples(random.choices(self.samples, k=batch_size))


def update_replay_from_cycle(config: HexoConfig, cycle_dir: str | Path) -> Path:
    """Refresh the latest replay file from a cycle directory.

    The Rust self-play implementation will eventually append many shards. For the
    skeleton, this concatenates JSONL shards into the configured latest replay path.
    """

    latest = resolve_path(config, config.paths.replay_latest)
    latest.parent.mkdir(parents=True, exist_ok=True)
    files = list_replay_files(cycle_dir)
    if not files:
        latest.touch()
        return latest
    with latest.open("wb") as output:
        for file_path in files:
            if file_path.suffix == ".zst":
                for sample in iter_jsonl(file_path):
                    output.write(json.dumps(sample, separators=(",", ":")).encode("utf-8") + b"\n")
            else:
                with file_path.open("rb") as source:
                    shutil.copyfileobj(source, output)
    return latest
