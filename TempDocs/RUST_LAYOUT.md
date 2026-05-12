# Rust Layout

## Purpose

This document is the canonical reference for how the Rust side of Hexo-RL is
organized to support the package families described in `HEXO_ENGINE.md`,
`HEXO_UTILS.md`, `HEXO_MODEL.md`, and `HEXO_RUNNER.md`. The Python-side split
is meaningful only if the underlying Rust crates and PyO3 wheels enforce the
same boundaries; this file specifies the Rust contract that the
component-level docs depend on.

The high-level rule is that Rust code follows the same authority model as the
Python packages: rules, state, and tactics live in one authority crate;
mechanisms (encoder default, search, MCTS) live in a utilities crate that
depends on it; model-specific representation and search live in
model-package crates that depend on the authority crate (and optionally on
utilities).

## Rust Crates

There are two foundational Rust crates plus zero-or-more model-package Rust
crates. Each model package decides whether it needs Rust at all; bringing a
Rust crate is a normal option, not an exception.

### `hexgame-engine`

Authority crate. Owns rules, state identity, tactics, and terminal
evaluation.

Contents (migrated from today's `crates/hexgame-core/src/`):

- `board.rs` — `HexGameState`, `MoveRecord`, `GameError`
- `core.rs` — `Hex`, `Turn`, hex math, `HEX_DIRECTIONS`, `PLACEMENT_RADIUS`,
  `WIN_LENGTH`, `WindowKey`
- `threats.rs` — tactical analysis derived from rules
- `eval/` — terminal eval helpers and rules-derived scoring

This crate is dependency-light. It does not depend on any other Hexo crate.
It contains no encoder code, no search code, no MCTS, and no
model-specific types.

### `hexgame-utils`

Mechanism crate. Depends on `hexgame-engine`.

Contents (migrated from today's `crates/hexgame-core/src/`):

- `encoder.rs` — the default 33×33×13 board encoder. This is the canonical
  default for the dense/ResNet/attention/graph model family; see "Encoder
  Pluggability" below for how models opt into different encodings.
- `search.rs` — classical alpha-beta search. Gains the `WIN_SCORE` constant,
  relocated from `encoder.rs` where it was misfiled (it is a search/eval
  constant, not an encoder one).
- `mcts.rs` — neural MCTS engine. Calls the default encoder directly inside
  the leaf-batch hot path. This coupling is intentional and correct for the
  shared model family.

### `hexgame-model-v1` (and other `hexgame-model-*` crates)

Model-package crate. Depends on `hexgame-engine` and may depend on
`hexgame-utils`. Brought by a model package when its architecture benefits
from Rust-level work — FFI volume, branchy state mutation, representation
coupling, or custom search.

Contents for V1 specifically (migrated from today's
`crates/hexgame-core/src/`):

- `v1.rs` — V1 row-identity, pair-row, and tactical payload contracts
- `v1_pair_search.rs` — V1 pair-native search

Other model packages may add their own Rust crate (`hexgame-model-resnet-rs`,
`hexgame-model-foo-rs`) when justified, or stay Python-only and import the
`hexgame_utils` PyO3 wheel directly. The decision belongs to the model
package.

## PyO3 Wheels

Each Rust crate has its own PyO3 wheel:

| Wheel | Rust crate | Primary Python consumer |
|-------|------------|-------------------------|
| `hexgame_engine` | `hexgame-engine` | `hexo-engine`, all downstream |
| `hexgame_utils` | `hexgame-utils` | `hexo-utils`, model packages |
| `hexgame_model_v1` | `hexgame-model-v1` | `hexo-model-v1` |
| `hexgame_model_*` | `hexgame-model-*` | corresponding `hexo-model-*` |

PyO3 types defined in one wheel are passed across wheel boundaries as Python
objects, not via direct Rust dependencies between FFI crates. Engine APIs
never return model-specific PyO3 types; model-specific types stay inside their
owning wheel.

## Encoder Pluggability

Pluggability between encoders happens at the existing Python adapter layer,
not at the Rust level. The current code already supports this:

- `ArchitectureSpec` carries `input_contract_id`, `training_adapter_id`, and
  `inference_adapter_id`. Each adapter implementation chooses its own
  encoding path.
- The `crop:v1` / `crop_13x33x33:v1` adapter calls into the Rust default
  encoder via `hexgame_utils`.
- Graph adapters compute their own graph features Python-side and call the
  Rust encoder only for tactical masks.
- Future adapters may bypass `hexgame_utils` entirely and use a
  model-package-owned Rust encoder, or stay pure Python.

The Rust MCTS in `hexgame-utils` calls the Rust default encoder directly.
This is intentional: the dense/ResNet/attention/graph family all use the
33×33×13 crop tensor as input fan-in, with graph models computing
supplementary representations on top of the same state. Models that need
fundamentally different MCTS leaf encoding (V1) bring their own search and
do not reuse `mcts.rs`.

No new Rust encoder trait is required.

## File Migration Map

| Today | Tomorrow |
|-------|----------|
| `crates/hexgame-core/src/board.rs` | `crates/hexgame-engine/src/board.rs` |
| `crates/hexgame-core/src/core.rs` | `crates/hexgame-engine/src/core.rs` |
| `crates/hexgame-core/src/threats.rs` | `crates/hexgame-engine/src/threats.rs` |
| `crates/hexgame-core/src/eval/` | `crates/hexgame-engine/src/eval/` |
| `crates/hexgame-core/src/encoder.rs` (sans `WIN_SCORE`) | `crates/hexgame-utils/src/encoder.rs` |
| `crates/hexgame-core/src/search.rs` (gains `WIN_SCORE`) | `crates/hexgame-utils/src/search.rs` |
| `crates/hexgame-core/src/mcts.rs` | `crates/hexgame-utils/src/mcts.rs` |
| `crates/hexgame-core/src/v1.rs` | `crates/hexgame-model-v1/src/v1.rs` |
| `crates/hexgame-core/src/v1_pair_search.rs` | `crates/hexgame-model-v1/src/v1_pair_search.rs` |
| `crates/hexgame-py/src/engine.rs` | split across `hexgame-engine-py`, `hexgame-utils-py`, `hexgame-model-v1-py` by ownership |
| `crates/hexgame-py/src/encode.rs` | `crates/hexgame-utils-py/src/encode.rs` |
| `crates/hexgame-py/src/protocol.rs` | engine-owned encoders to `hexgame-engine-py`; V1-owned encoders to `hexgame-model-v1-py` |

The PyO3 split is the most invasive part of this migration. Today
`engine.rs` is one 2359-line file that imports from every subsystem of
`hexgame-core`; the split must redistribute its `#[pymethods]` blocks across
three FFI crates and ensure that types crossing FFI boundaries continue to
round-trip through Python objects rather than becoming direct Rust deps
between FFI crates.

## Test Ownership

Each Rust crate keeps its own `tests/` directory. Specifically:

- The brute-force oracle (today: `crates/hexgame-core/tests/oracle/`) stays
  with `hexgame-engine` because it validates rules.
- Encoder and MCTS tests (today inside `crates/hexgame-core/tests/`) follow
  their files to `hexgame-utils`.
- V1 tests follow to `hexgame-model-v1`.

Cross-crate integration tests can live in a small dedicated crate or in the
PyO3-wheel level, where multiple crates already meet.

## Dependency Graph

```text
hexgame-engine
    ^
    |
hexgame-utils ───────► depends on hexgame-engine
    ^
    |
hexgame-model-v1 ────► depends on hexgame-engine; may depend on hexgame-utils
hexgame-model-*  ────► Python-only, or own Rust crate following the model-v1 template
```

PyO3 wheels mirror this graph exactly.

## Consequences For Other Docs

- `HEXO_ENGINE.md` removes mentions of search-support utilities being
  inside the engine crate.
- `HEXO_UTILS.md` documents the default encoder and the neural MCTS as
  shipped utilities, with pluggability handled at the Python adapter layer.
- `HEXO_MODEL.md` treats Rust-bearing model packages as a normal pattern,
  not an exception. V1 is the established example, not a special case.
- `up_next/PACKAGING_DEPENDENCIES.md` enumerates the Rust crate and PyO3
  wheel dependency direction.
- `PERFORMANCE_CONSIDERATIONS.md` references `hexgame-utils` for shared
  search and encoder hot paths.
