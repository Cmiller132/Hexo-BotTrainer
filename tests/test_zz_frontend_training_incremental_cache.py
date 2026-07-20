from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_frontend", "hexo_runner", "hexo_engine", "hexo_utils"):
    package_path = ROOT / "packages" / package / "python"
    if str(package_path) not in sys.path:
        sys.path.insert(0, str(package_path))


def _install_windows_record_stub() -> None:
    """Let pure dashboard-cache tests import web.py when the Rust codec is absent."""

    try:
        import hexo_utils._rust  # type: ignore[import-not-found]  # noqa: F401
        return
    except ImportError:
        pass
    rust = types.ModuleType("hexo_utils._rust")

    class UnavailableRecordFile:
        @classmethod
        def open(cls, _path: object) -> object:
            raise OSError("Rust record codec unavailable in cache unit test")

    for name in ("AbortRecord", "HexoRecord", "HexoRecordGameWriter", "HexoRecordPlayer"):
        setattr(rust, name, type(name, (), {}))
    rust.HexoRecordFile = UnavailableRecordFile
    rust.HEXO_RECORD_MAGIC = b"HXR"
    rust.HEXO_RECORD_SCHEMA_VERSION = 1
    sys.modules["hexo_utils._rust"] = rust


_install_windows_record_stub()
from hexo_frontend import web  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_training_caches() -> None:
    caches = (
        web._training_run_cache,
        web._training_runs_cache,
        web._training_live_cache,
        web._training_epochs_cache,
        web._hxr_history_cache,
        web._hxr_count_cache,
        web._seed_sidecar_cache,
        web._json_file_cache,
        web._jsonl_file_cache,
        web._file_tail_cache,
        web._frozen_json_file_cache,
        web._frozen_file_stat_cache,
        web._live_artifact_stat_cache,
        web._history_artifact_scan_cache,
        web._training_derived_cache,
        web._epoch_inventory_cache,
        web._training_build_locks,
    )
    for cache in caches:
        cache.clear()
    yield
    for cache in caches:
        cache.clear()


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(data)


def test_json_file_cache_hits_and_invalidates_on_mtime_or_size(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    path = tmp_path / "epoch_000001.json"
    _write_bytes(path, b'{"value":1}')
    real_open = Path.open
    reads = 0

    def counted_open(self: Path, *args: object, **kwargs: object) -> object:
        nonlocal reads
        if self == path and args and str(args[0]).startswith("r"):
            reads += 1
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)
    assert web._read_json_file(path) == {"value": 1}
    assert web._read_json_file(path) == {"value": 1}
    assert reads == 1

    before = path.stat()
    with real_open(path, "wb") as handle:
        handle.write(b'{"value":2}')  # same size, explicitly different mtime
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    assert web._read_json_file(path) == {"value": 2}
    assert reads == 2

    with real_open(path, "wb") as handle:
        handle.write(b'{"value":300}')  # size invalidation
    assert web._read_json_file(path) == {"value": 300}
    assert reads == 3


def test_jsonl_append_parses_only_tail_and_truncation_reloads(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    path = tmp_path / "events.jsonl"
    first = b'{"event":"one"}\n'
    appended = b'{"event":"two"}\n'
    replacement = b'{"event":"replacement"}\n'
    _write_bytes(path, first)

    parsed_lengths: list[int] = []
    real_parse = web._parse_jsonl_bytes

    def counted_parse(data: bytes, records: list[object]) -> tuple[list[object], bytes]:
        parsed_lengths.append(len(data))
        return real_parse(data, records)

    monkeypatch.setattr(web, "_parse_jsonl_bytes", counted_parse)
    assert web._iter_jsonl(path) == [{"event": "one"}]
    assert parsed_lengths == [len(first)]

    with path.open("ab") as handle:
        handle.write(appended)
    assert web._iter_jsonl(path) == [{"event": "one"}, {"event": "two"}]
    assert parsed_lengths[-1] == len(appended)

    _write_bytes(path, replacement)
    assert web._iter_jsonl(path) == [{"event": "replacement"}]
    assert parsed_lengths[-1] == len(replacement)

    # Growth alone is insufficient for append resumption: replacing the head
    # forces a full parse and must not retain records from the former file.
    replaced_growth = b'{"event":"new-head"}\n{"event":"new-tail"}\n'
    _write_bytes(path, replaced_growth)
    assert web._iter_jsonl(path) == [{"event": "new-head"}, {"event": "new-tail"}]
    assert parsed_lengths[-1] == len(replaced_growth)


def test_derived_cache_rebuilds_only_for_changed_file_stats(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "derived"
    path = run_dir / "diagnostics" / "epoch_000001.json"
    _write_bytes(path, b'{"epoch":1}')
    builds = 0

    def build() -> object:
        nonlocal builds
        builds += 1
        return {"build": builds}

    fingerprint = web._file_fingerprint([path])
    first = web._cached_training_derived("unit", run_dir, fingerprint, build)
    second = web._cached_training_derived("unit", run_dir, fingerprint, build)
    assert second is first
    assert builds == 1

    _write_bytes(path, b'{"epoch":100}')
    changed = web._file_fingerprint([path])
    third = web._cached_training_derived("unit", run_dir, changed, build)
    assert third == {"build": 2}
    assert builds == 2


def test_frozen_sidecars_use_one_scandir_and_no_path_stats(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    samples_dir = tmp_path / "samples" / "epoch_000001"
    for game in range(256):
        payload = {"seeded": True, "seed_ply": game} if game % 17 == 0 else {}
        _write_bytes(samples_dir / f"game_{game}.json", json.dumps(payload).encode())

    path_stats = 0
    real_stat = Path.stat

    def counted_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal path_stats
        path_stats += 1
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", counted_stat)
    first = web._seed_provenance_by_game_key(samples_dir, frozen=True)
    second = web._seed_provenance_by_game_key(samples_dir, frozen=True)
    assert second is first
    assert first["0"] == {"seeded": True, "seed_ply": 0}
    assert path_stats == 0


def test_live_sidecars_are_scandir_throttled(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    samples_dir = tmp_path / "samples" / "epoch_000002"
    for game in range(32):
        _write_bytes(samples_dir / f"game_{game}.json", b'{"seeded":true,"seed_ply":4}')

    real_scandir = os.scandir
    scans = 0

    def counted_scandir(path: object) -> object:
        nonlocal scans
        if Path(path) == samples_dir:
            scans += 1
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", counted_scandir)
    first = web._seed_provenance_by_game_key(samples_dir, frozen=False)
    second = web._seed_provenance_by_game_key(samples_dir, frozen=False)
    assert second is first
    assert scans == 1


def test_seed_provenance_batch_scans_each_epoch_once(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = tmp_path / "runs" / "batch_sidecars"
    samples_dir = run_dir / "samples" / "epoch_000007"
    _write_bytes(samples_dir / "game_1.json", b'{"seeded":true,"seed_ply":4}')
    game_ids = [f"epoch-000007-game-{game}" for game in range(256)]

    real_scandir = os.scandir
    scans = 0

    def counted_scandir(path: object) -> object:
        nonlocal scans
        if Path(path) == samples_dir:
            scans += 1
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", counted_scandir)
    joined = web._seed_provenance_for_game_ids(run_dir, game_ids)
    assert joined == {"epoch-000007-game-1": {"seeded": True, "seed_ply": 4}}
    assert scans == 1


def test_steady_run_rebuild_stats_no_completed_epoch_artifacts(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = tmp_path / "runs" / "stat_freeze"
    _write_bytes(run_dir / "manifest.json", b'{"model":{"name":"hexfield"}}')
    _write_bytes(
        run_dir / "diagnostics" / "epoch_000001.json",
        b'{"status":"completed","metadata":{"result":{"epoch":1}}}',
    )
    _write_bytes(
        run_dir / "diagnostics" / "hexfield.selfplay.epoch_000001.json",
        b'{"epoch":1,"status":"completed"}',
    )
    # Epoch 2 exists but has no merged epoch_000002.json, so its HXR remains
    # live/re-stattable while epoch 1 is permanently frozen.
    _write_bytes(
        run_dir / "diagnostics" / "hexfield.selfplay.epoch_000002.json",
        b'{"epoch":2,"status":"completed"}',
    )
    _write_bytes(
        run_dir / "diagnostics" / "hexfield.selfplay.live.json",
        b'{"epoch":2,"status":"running"}',
    )
    _write_bytes(run_dir / "selfplay" / "epoch_000001.hxr", b"old-hxr")
    _write_bytes(run_dir / "selfplay" / "epoch_000002.hxr", b"live-hxr")
    _write_bytes(run_dir / "checkpoints" / "epoch_000001.pt", b"checkpoint")
    _write_bytes(
        run_dir / "samples" / "epoch_000001" / "game_1.json",
        b'{"seeded":true,"seed_ply":3}',
    )

    real_scandir = os.scandir
    real_path_stat = Path.stat
    completed_stats = 0
    live_stats = 0

    def classify(path: object) -> str | None:
        text = os.fspath(path).replace("\\", "/").lower()
        if "epoch_000001" in text or "epoch-000001" in text:
            return "completed"
        if "epoch_000002" in text or "epoch-000002" in text or text.endswith(".live.json"):
            return "live"
        return None

    class CountedEntry:
        def __init__(self, entry: os.DirEntry[str]) -> None:
            self._entry = entry

        def __getattr__(self, name: str) -> object:
            return getattr(self._entry, name)

        def stat(self, *args: object, **kwargs: object) -> os.stat_result:
            nonlocal completed_stats, live_stats
            kind = classify(self._entry.path)
            if kind == "completed":
                completed_stats += 1
            elif kind == "live":
                live_stats += 1
            return self._entry.stat(*args, **kwargs)

    class CountedScandir:
        def __init__(self, path: object) -> None:
            self._inner = real_scandir(path)

        def __iter__(self) -> "CountedScandir":
            return self

        def __next__(self) -> CountedEntry:
            return CountedEntry(next(self._inner))

        def __enter__(self) -> "CountedScandir":
            self._inner.__enter__()
            return self

        def __exit__(self, *args: object) -> object:
            return self._inner.__exit__(*args)

        def close(self) -> None:
            self._inner.close()

    def counted_path_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal completed_stats, live_stats
        kind = classify(self)
        if kind == "completed":
            completed_stats += 1
        elif kind == "live":
            live_stats += 1
        return real_path_stat(self, *args, **kwargs)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(os, "scandir", CountedScandir)
    monkeypatch.setattr(Path, "stat", counted_path_stat)

    web._training_run(run_dir.name)
    assert completed_stats > 0  # cold sight takes one exact metadata snapshot

    completed_stats = 0
    live_stats = 0
    web._epoch_inventory_cache[str(run_dir)] = (0.0, web._epoch_inventory_cache[str(run_dir)][1])
    for key, (_timestamp, files) in list(web._history_artifact_scan_cache.items()):
        web._history_artifact_scan_cache[key] = (0.0, files)
    for key, (_timestamp, stat) in list(web._live_artifact_stat_cache.items()):
        web._live_artifact_stat_cache[key] = (0.0, stat)

    web._training_run(run_dir.name)
    assert completed_stats == 0
    assert live_stats > 0  # instrumentation still sees the allowed live checks


def test_epoch_history_hxr_backfill_skips_seed_sidecars(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    calls: list[bool] = []

    def base_rows(
        _path: Path,
        _run_dir: Path,
        _known_stat: os.stat_result | None = None,
        *,
        include_seed_provenance: bool = True,
    ) -> list[dict[str, object]]:
        calls.append(include_seed_provenance)
        return [{"status": "completed", "length": 10, "winner": "player0"}]

    monkeypatch.setattr(web, "_hxr_base_rows", base_rows)
    stats = web._selfplay_game_stats_from_records(tmp_path, 7)
    assert stats["win_p0_fraction"] == 1.0
    assert calls == [False]


def test_aggregate_cache_check_has_constant_path_stat_cost(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = tmp_path / "runs" / "many_epochs"
    _write_bytes(run_dir / "manifest.json", b'{"model":{"name":"hexfield"}}')
    for epoch in range(1, 101):
        _write_bytes(
            run_dir / "diagnostics" / f"epoch_{epoch:06d}.json",
            json.dumps({"status": "completed", "metadata": {"result": {"epoch": epoch}}}).encode(),
        )
        _write_bytes(
            run_dir / "diagnostics" / f"hexfield.evaluation.epoch_{epoch:06d}.json",
            json.dumps({"epoch": epoch, "status": "completed", "games": 2}).encode(),
        )

    web._epoch_history(run_dir)
    inventory_key = str(run_dir)
    cached_inventory = web._epoch_inventory_cache[inventory_key][1]
    web._epoch_inventory_cache[inventory_key] = (0.0, cached_inventory)

    path_stats = 0
    real_stat = Path.stat

    def counted_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal path_stats
        path_stats += 1
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", counted_stat)
    web._epoch_history(run_dir)
    # Manifest and lineage are a small constant; historical epoch count does
    # not affect cache-check stat cost. DirEntry.stat is intentionally excluded.
    assert path_stats <= 3


def test_training_epochs_cache_check_has_constant_path_stat_cost(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = tmp_path / "runs" / "many_segments"
    _write_bytes(run_dir / "manifest.json", b'{"model":{"name":"hexfield"}}')
    for epoch in range(1, 101):
        for kind in ("selfplay", "select", "training"):
            _write_bytes(
                run_dir / "diagnostics" / f"hexfield.{kind}.epoch_{epoch:06d}.json",
                json.dumps({"epoch": epoch, "status": "completed"}).encode(),
            )

    web._training_epochs(run_dir)
    inventory_key = str(run_dir)
    cached_inventory = web._epoch_inventory_cache[inventory_key][1]
    web._epoch_inventory_cache[inventory_key] = (0.0, cached_inventory)

    path_stats = 0
    real_stat = Path.stat

    def counted_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal path_stats
        path_stats += 1
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", counted_stat)
    web._training_epochs(run_dir)
    # Only manifest + events are single-file stat checks in this fixture.
    # Historical segment count changes neither fingerprint nor parse cost.
    assert path_stats <= 2


def test_appended_epoch_invalidates_name_set_after_inventory_ttl(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "append_visibility"
    _write_bytes(run_dir / "manifest.json", b'{"model":{"name":"hexfield"}}')
    first_path = run_dir / "diagnostics" / "hexfield.selfplay.epoch_000001.json"
    _write_bytes(first_path, b'{"epoch":1,"status":"completed"}')

    first = web._training_epochs(run_dir)
    assert [row["epoch"] for row in first["epochs"]] == [1]

    second_path = run_dir / "diagnostics" / "hexfield.selfplay.epoch_000002.json"
    _write_bytes(second_path, b'{"epoch":2,"status":"completed"}')
    cached_inventory = web._epoch_inventory_cache[str(run_dir)][1]
    web._epoch_inventory_cache[str(run_dir)] = (0.0, cached_inventory)

    second = web._training_epochs(run_dir)
    assert [row["epoch"] for row in second["epochs"]] == [1, 2]


def test_run_cache_serializes_concurrent_rebuilds(monkeypatch: Any) -> None:
    workers = 8
    barrier = Barrier(workers)
    count_lock = Lock()
    builds = 0

    def build(_name: str) -> dict[str, object]:
        nonlocal builds
        with count_lock:
            builds += 1
        return {"ok": True}

    monkeypatch.setattr(web, "_training_run", build)

    def request() -> dict[str, object]:
        barrier.wait()
        return web._training_run_cached("same-run")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda _index: request(), range(workers)))
    assert builds == 1
    assert all(result is results[0] for result in results)


_FIXED_WALL_CLOCK = 1_800_000_000.0
_GOLDEN_HASHES = {
    # Captured from the pre-refactor web.py (git HEAD at task start) on this
    # deterministic fixture. Canonicalization changes only the absolute temp
    # run path; the fixed wall clock means no payload fields are excluded.
    "run": "a73ba0aafbb10cf6967c423165e73e8c8a8b6db0ab3bdf60e38aed093b478d48",
    "epoch": "a01fb35cad59589f6ec0c5014101ef73eedff4a57f7a4539a8ba9c8cb6c62fd0",
    "epochs": "c103659ab283aa2a06acf5cd3768ed2ebed928330acc2314f2b7e080b15d316f",
    "live": "494b6303ed4acb8c5464c5e5f1db31736c364fe2d63a610c81d71c6e20cebf47",
}


def _golden_run_fixture(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "golden_hexfield"
    next_mtime = 1_700_000_000_000_000_000

    def write(relative: str, payload: object) -> None:
        nonlocal next_mtime
        path = run_dir / relative
        if isinstance(payload, bytes):
            data = payload
        elif relative.endswith(".jsonl"):
            data = b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in payload)  # type: ignore[union-attr]
        else:
            data = json.dumps(payload, sort_keys=True).encode()
        _write_bytes(path, data)
        os.utime(path, ns=(next_mtime, next_mtime))
        next_mtime += 1_000_000_000

    write(
        "manifest.json",
        {
            "model": {
                "name": "hexfield",
                "config": {
                    "architecture": {"channels": 32},
                    "selfplay": {"search_visits": 128},
                    "training": {"batch_size": 64},
                },
            }
        },
    )
    write(
        "diagnostics/events.jsonl",
        [
            {"event": "stage_started", "payload": {"stage": "run_epochs"}},
            {"event": "stage_started", "payload": {"stage": "epoch_000002"}},
        ],
    )
    write(
        "diagnostics/epoch_000001.json",
        {
            "status": "completed",
            "elapsed_seconds": 12.0,
            "metadata": {
                "result": {
                    "epoch": 1,
                    "selfplay": {"status": "completed", "games_finished": 2, "rows_written": 40},
                    "training": {"status": "completed", "loss": 1.25, "steps": 4},
                    "checkpoint": {"checkpoint_path": "checkpoints/epoch_000001.pt"},
                }
            },
        },
    )
    write(
        "diagnostics/hexfield.selfplay.epoch_000001.json",
        {"epoch": 1, "status": "completed", "games_finished": 2, "rows_written": 40},
    )
    write(
        "diagnostics/hexfield.select.epoch_000001.json",
        {"epoch": 1, "status": "completed", "selected_samples": 32},
    )
    write(
        "diagnostics/hexfield.training.epoch_000001.json",
        {"epoch": 1, "status": "completed", "loss": 1.25, "train_seconds": 8.0},
    )
    write(
        "diagnostics/hexfield.evaluation.epoch_000001.json",
        {"epoch": 1, "status": "completed", "games": 2, "wins": 1, "losses": 1, "mean_game_length": 80.0},
    )
    write(
        "diagnostics/hexfield.multistage_eval.epoch_000001.json",
        {
            "meta": {"candidate_epoch": 1, "candidate_label": "ep1", "config": {}},
            "ratings": {"anchor": "sealbot", "players": [], "fit": {}},
            "verdict": {"label": "hold"},
            "edges": [],
            "roster": {},
            "stages": [],
        },
    )
    write(
        "diagnostics/eval_pool.json",
        {"format": "bt", "version": 1, "anchor": "sealbot", "edges": []},
    )
    write(
        "diagnostics/hexfield.selfplay.live.json",
        {
            "epoch": 2,
            "status": "running",
            "timestamp": _FIXED_WALL_CLOCK - 5.0,
            "games_finished": 1,
            "requested_games": 2,
        },
    )
    write("checkpoints/epoch_000001.pt", b"checkpoint")
    return run_dir


def _canonical_payload_hash(payload: object, run_dir: Path) -> str:
    run_path = str(run_dir)

    def normalize(value: object) -> object:
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, str):
            return value.replace(run_path, "<RUN_DIR>")
        return value

    canonical = json.dumps(normalize(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def test_golden_training_payloads(tmp_path: Path, monkeypatch: Any) -> None:
    run_dir = _golden_run_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(web, "wall_clock", lambda: _FIXED_WALL_CLOCK)
    outputs = {
        "run": web._training_run(run_dir.name),
        "epoch": web._training_epoch(run_dir.name, 1),
        "epochs": web._training_epochs(run_dir),
        "live": web._training_live_cached(run_dir.name),
    }
    hashes = {name: _canonical_payload_hash(payload, run_dir) for name, payload in outputs.items()}
    assert hashes == _GOLDEN_HASHES
