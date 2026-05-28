from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models",):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _samples_module() -> Any:
    return importlib.import_module("hexo_models.dense_cnn.samples")


def _sample(sample_id: str, *, target_schema_version: int | None) -> Any:
    samples = _samples_module()
    metadata: dict[str, Any] = {"sample_id": sample_id}
    if target_schema_version is not None:
        metadata["target_schema_version"] = target_schema_version
    return samples.Model1SampleData(
        game_id=sample_id,
        turn_index=0,
        current_player="player0",
        phase="Opening",
        center=(0, 0),
        stones=(),
        legal_action_ids=(1,),
        policy=((1, 1.0),),
        value=0.25,
        metadata=metadata,
    )


def _compressed_state_entry(sample: Any) -> dict[str, Any]:
    samples = _samples_module()
    compressed = samples.CompressedSample.from_data(sample, compression_level=1)
    return {
        "payload": compressed.payload,
        "uncompressed_bytes": compressed.uncompressed_bytes,
        "compression": compressed.compression,
        "compressed": compressed.compressed,
    }


def _buffer_state(*entries: dict[str, Any]) -> dict[str, Any]:
    return {
        "capacity": 200_000,
        "recency_halflife": 50_000.0,
        "compression_level": 1,
        "total_appended": len(entries),
        "draw_count": 0,
        "samples": list(entries),
    }


def test_sample_buffer_load_state_dict_rejects_legacy_schema_samples() -> None:
    samples = _samples_module()
    current_entry = _compressed_state_entry(
        _sample("current", target_schema_version=samples.CURRENT_TARGET_SCHEMA_VERSION)
    )
    legacy_entry = _compressed_state_entry(_sample("legacy", target_schema_version=1))

    buffer = samples.SampleBuffer(capacity=200_000, compression_level=1)

    with pytest.raises(ValueError, match="target_schema_version"):
        buffer.load_state_dict(_buffer_state(legacy_entry, current_entry))
    assert buffer.sample_count == 0


def test_checkpoint_loader_rejects_incompatible_replay_schema(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    samples = _samples_module()
    checkpoints = importlib.import_module("hexo_models.dense_cnn.checkpoints")

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    buffer = samples.SampleBuffer(capacity=200_000, compression_level=1)
    components = SimpleNamespace(
        model=SimpleNamespace(
            model=model,
            optimizer=optimizer,
            trainer=SimpleNamespace(buffer=buffer),
        )
    )
    current_entry = _compressed_state_entry(
        _sample("current", target_schema_version=samples.CURRENT_TARGET_SCHEMA_VERSION)
    )
    legacy_entry = _compressed_state_entry(_sample("legacy", target_schema_version=None))
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "sample_buffer": _buffer_state(current_entry, legacy_entry),
            "epoch": 12,
            "metadata": {"run": "unit"},
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="target_schema_version"):
        checkpoints.DenseCNNCheckpointLoader().load(checkpoint_path, ctx=None, components=components)
    assert buffer.sample_count == 0
