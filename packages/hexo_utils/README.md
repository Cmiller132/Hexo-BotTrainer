# hexo_utils

Shared-utility layer between the engine, runner, trainer, and model packages.
A Rust crate plus a thin Python package, distributed via maturin as `hexo_utils`
with a private PyO3 extension `hexo_utils._rust` (built with `features=["python"]`,
see `pyproject.toml`).

## Subsystems

| Subsystem | Role |
| --- | --- |
| `.hxr` record codec (`rust/src/records.rs` + `pybridge.rs` + `python/hexo_utils/records.py`) | The repo's cross-package game-record format. Every self-play / evaluation / match path writes `.hxr` through it; the dashboard and health scripts read it. |
| `state_hash` (`rust/src/state_hash.rs`) | Rust-only. Cache key for every model lineage's MCTS evaluator (dense_cnn, hexgt, hexgnn `mcts_eval.rs` / `mcts_tree.rs`). |
| `samples/` JSON-chunk sample store + `encoding/` D6 contracts | Generic shared-store layer for the `hexo_train` pipeline. The current model plugins set `uses_shared_sample_store=False` and own their own NPZ replay storage, so this layer is exercised through `hexo_train`'s generic path and the test suite. |

## The .hxr record format

Owned and specified here (codec core: `rust/src/records.rs`). Wire format, schema v1:
magic `HEXOREC1`, a varint-encoded header (engine metadata, players), then varint/zigzag
per-game payloads (game_id, seed, status, action ids as u32 LE, winner, placements,
optional abort record). Readers reject other schema versions; `HEXO_RECORD_SCHEMA_VERSION`
is bumped on any wire change. API: `HexoRecordFile` reader/writer (`iter_records()` on a
write-mode handle reopens the path for reading) and `HexoRecordGameWriter`, an append-only
per-game writer.

## Module table

| File | Role |
| --- | --- |
| `rust/src/records.rs` | `.hxr` binary codec core (~1000 lines): the wire format above, plus round-trip/corruption unit tests. |
| `rust/src/state_hash.rs` | `hash_state(HexoState) -> u64`: deterministic, placement-order-sensitive state identity (splitmix64-style mixing over placement history + player/phase/terminal) for neural-eval caches. |
| `rust/src/pybridge.rs` | PyO3 bridge behind the `python` feature: `PyHexoRecordFile` / `PyHexoRecordGameWriter` / `PyHexoRecord` / `PyAbortRecord` / `PyHexoRecordPlayer`; duck-typed parsers (players via `.identity`, action ids from int or `.coord.q/.r`); `PyHexoRecord.replay()` re-runs action ids through the `hexo_engine` Python module. Defines the `_rust` pymodule. |
| `rust/src/lib.rs` | Crate root: re-exports records + state_hash; pybridge gated behind the `python` feature. |
| `python/hexo_utils/records.py` | Python facade re-exporting the codec classes from `hexo_utils._rust`. Production callers reach it through `hexo_runner.records`, which wraps this module. |
| `python/hexo_utils/samples/buffer.py` | Generic sample store (~840 lines): compressed-JSON chunks under `<store>/chunks` + `manifest.json`; open/append/index/window/sample API; zlib codec by default, with json/zstd/lz4 options. |
| `python/hexo_utils/samples/records.py` | Data-only sample schema shapes: `SampleSchema` (v1), `TrainingSampleRecord`, `PolicyOutputRecord`, `ModelSamplePayload` (opaque model-owned payloads). |
| `python/hexo_utils/samples/targets.py` | Shared policy/value target builders (`ScalarValueTargetHelper`, `LegalPolicyTargetHelper`, `build_legal_policy_value_target` with D6 action-id remap), wired into `hexo_train.defaults`. |
| `python/hexo_utils/encoding/symmetry.py` | D6 symmetry transport contract: `D6_SIZE=12`, frozen `D6Symmetry`, `ActionSymmetryMapper` Protocol, `transform_action_ids`. Consumed by `hexo_train.symmetry` and `samples/targets.py`. |
| `Cargo.toml` / `pyproject.toml` | Workspace crate (rlib + cdylib, depends on `hexo_engine`, pyo3 optional) / maturin config (`module-name = "hexo_utils._rust"`, `python-source = "python"`). |

## Connections to other packages

Inbound (who uses hexo_utils):

- `hexo_runner.records.record` imports `HexoRecordFile` / `HexoRecordGameWriter` / `HexoRecord` / `HexoRecordPlayer` / `AbortRecord` / magic + schema constants from `hexo_utils.records` and re-exports them. All production `.hxr` IO flows through that path: `dense_cnn_restnet`, `hexo_models/dense_cnn`, `hexo_models/hexgt`, and `hexgnn` selfplay/evaluation write records; `hexo_frontend/web.py` reads them for the dashboard.
- Rust: `hexo_models` (dense_cnn + hexgt subcrates) and `hexgnn` depend on the `hexo_utils` workspace crate for `use hexo_utils::{hash_state, StateHash}` in their `mcts_eval.rs` / `mcts_tree.rs` (evaluator cache keys). The active `dense_cnn_restnet` lineage reaches this indirectly through `hexo_models._rust.dense_cnn`.
- `hexo_train`: `defaults.py` builds the target helpers, `symmetry.py` imports `D6_SIZE` / `D6Symmetry`, `epoch/samples.py` lazily imports the sample-store API on the shared-store path.
- Scripts/analysis: `scripts/_wf_r4_health.py` and `analysis/exploration_diversity.py` import `hexo_utils.records.HexoRecordFile` directly to audit run records.

Outbound (what hexo_utils depends on):

- Rust crate depends on the `hexo_engine` crate (`HexoState`, `HexCoord`, `pack_coord`, `Player`, `TurnPhase`, outcome types) for both `state_hash` and record replay tests.
- `pybridge.rs` `PyHexoRecord.replay()` and the duck-typed parsers form a runtime contract with the `hexo_engine` Python package (`new_game` / `PlacementAction` / `apply_action` / `terminal`, `unpack_coord_id`) and with `hexo_runner` player objects (`.identity.player_id` / `.label`) and AbortRecord-shaped objects (`.stage` / `.exception_type` / `.message`).

Formats owned here: the `.hxr` game-record contract (specified above) and the sample-store
on-disk layout (`manifest.json` + `chunks/*.json` compressed), spoken by `buffer.py` and
`hexo_train/epoch/samples.py`.

## Entry points / how it gets exercised

- No CLI. Pure library, imported transitively by nearly every Python entry point in the repo via `hexo_runner.records`.
- Build: the PyO3 extension is built into the source tree (`python/hexo_utils/_rust.cpython-312-*.so`, untracked) by maturin in the WSL `hexgt-build` venv (editable install); Rust changes take effect after re-running maturin. `scripts/_rebuild_hexo_models_hexgt.sh` rebuilds the hexo_models crate (which links hexo_utils as a workspace dependency); the `hexo_utils._rust` extension itself rebuilds via maturin against this package's `pyproject.toml`. On Windows Python the extension is absent; importers lazy-guard the import.
- Tests: `tests/test_hexo_utils_sample_store.py` exercises the samples subpackage directly; many other tests gate on `pytest.importorskip("hexo_utils._rust")`; `cargo test -p hexo_utils` runs the Rust codec/hash unit suites. Per project convention, Python tests are authoritative only in the WSL venv.
- Scripts: `scripts/_wf_r4_health.py` (run-health audit over `.hxr`); `scripts/goal_benchmark.py` and `tests/test_hexo_runner_match_mode.py` call `record.replay()`.
