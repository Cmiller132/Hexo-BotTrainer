from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_utils", "hexo_engine"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def test_sample_store_writes_manifest_index_window_and_reads_records(tmp_path: Path) -> None:
    from hexo_utils.samples import (
        ModelSamplePayload,
        PolicyOutputRecord,
        SampleRequest,
        TrainingSampleRecord,
        append_samples,
        build_sample_window,
        load_sample_manifest,
        open_sample_store,
        refresh_sample_index,
        sample_training_samples,
    )

    dense_payload = ModelSamplePayload(
        game_id="g",
        turn_index=1,
        model_id="m",
        namespace="dense_cnn",
        schema_version=3,
        payload={"planes_ref": "chunk-a", "values": [1, 2, 3]},
    )
    dense_record = TrainingSampleRecord(
        game_id="g",
        turn_index=1,
        legal_action_ids=(10, 20),
        policy=PolicyOutputRecord(
            game_id="g",
            turn_index=1,
            model_id="m",
            selected_action_id=20,
            logits=(0.1, 0.9),
            value=0.5,
        ),
        model_payloads=(dense_payload,),
        metadata={"split": "train"},
    )
    other_record = TrainingSampleRecord(
        game_id="g",
        turn_index=2,
        legal_action_ids=(30,),
        model_payloads=(
            ModelSamplePayload(
                game_id="g",
                turn_index=2,
                model_id="m",
                namespace="other_model",
                schema_version=1,
            ),
        ),
    )

    store = open_sample_store(tmp_path / "samples", metadata={"run": "unit"})
    result = append_samples(store, (dense_record, other_record), metadata={"epoch": 1})

    assert result.count == 2
    assert result.chunks[0].path.exists()
    assert store.manifest_path.exists()

    chunk_payload = json.loads(zlib.decompress(result.chunks[0].path.read_bytes()).decode("utf-8"))
    assert chunk_payload["records"][0]["game_id"] == "g"

    manifest = load_sample_manifest(store)
    assert manifest.sample_count == 2
    assert manifest.schema.extensions["dense_cnn"] == 3

    index = refresh_sample_index(store)
    assert index.sample_count == 2
    assert len(index.entries) == 2

    window = build_sample_window(index, window_size=1, seed=7)
    assert window.window_size == 1
    assert window.sample_count == 1

    batch = sample_training_samples(
        index,
        SampleRequest(count=10, required_extensions=("dense_cnn",), filters={"metadata.split": "train"}),
    )

    assert batch.metadata["returned_count"] == 1
    [record] = batch.records
    assert record == dense_record


def test_empty_sample_window_keeps_requested_size_for_pipeline_compatibility(tmp_path: Path) -> None:
    from hexo_utils.samples import build_sample_window, open_sample_store, refresh_sample_index

    store = open_sample_store(tmp_path / "empty")
    index = refresh_sample_index(store)
    window = build_sample_window(index, window_size=123, seed=1)

    assert window.window_size == 123
    assert window.sample_count == 0


def test_sample_store_can_write_plain_json_chunks(tmp_path: Path) -> None:
    from hexo_utils.samples import ModelSamplePayload, TrainingSampleRecord, append_samples, open_sample_store, read_sample_records

    record = TrainingSampleRecord(
        game_id="g",
        turn_index=1,
        legal_action_ids=(1,),
        model_payloads=(
            ModelSamplePayload(
                game_id="g",
                turn_index=1,
                model_id="m",
                namespace="hexformer_ar",
                schema_version=1,
                payload={"x": 1},
            ),
        ),
    )
    store = open_sample_store(tmp_path / "samples", metadata={"compression": "json"})
    result = append_samples(store, (record,))

    assert result.metadata["compression"] == "json"
    assert result.chunks[0].path.suffix == ".json"
    assert read_sample_records(store)[0] == record


def test_sample_store_uses_manifest_compression_when_suffix_is_custom(tmp_path: Path) -> None:
    from hexo_utils.samples import ModelSamplePayload, TrainingSampleRecord, append_samples, open_sample_store, read_sample_records

    record = TrainingSampleRecord(
        game_id="g",
        turn_index=1,
        legal_action_ids=(1,),
        model_payloads=(
            ModelSamplePayload(
                game_id="g",
                turn_index=1,
                model_id="m",
                namespace="hexformer_ar",
                schema_version=1,
                payload={"x": 1},
            ),
        ),
    )
    store = open_sample_store(tmp_path / "samples", metadata={"compression": "json"})
    result = append_samples(store, (record,))
    custom_path = result.chunks[0].path.with_suffix(".chunk")
    result.chunks[0].path.rename(custom_path)
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    manifest["chunks"][0]["path"] = f"chunks/{custom_path.name}"
    manifest["chunks"][0]["metadata"]["compression"] = "json"
    store.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert read_sample_records(store)[0] == record
