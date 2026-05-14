# HEXO_UTILS

## Purpose

`hexo-utils` holds a small set of reusable mechanisms shared by model packages,
runner modes, and training pipelines.

It reduces duplicated infrastructure without becoming a second engine, a model
policy layer, or a catch-all package.

## Owns

- Shared encoding helpers when they are not model-specific.
- Shared MCTS search machinery.
- Training sample buffer and serialization helpers.
- Schema and version helpers.

## Does Not Own

- Game legality.
- Terminal state authority.
- Model architecture.
- Model-specific target meaning.
- Core game recording or runner lifecycle.
- Policy decisions.
- Reward interpretation.

## Package Layout

```text
packages/hexo_utils/
  pyproject.toml
  Cargo.toml                 # when Rust helpers are needed
  python/
    hexo_utils/
      __init__.py
      py.typed
      rust_bridge.py
      encoding/
        __init__.py
        crop.py
        masks.py
        symmetry.py
      search/
        __init__.py
        mcts.py
      samples/
        __init__.py
        buffer.py
        records.py
        targets.py
  rust/
    src/
      lib.rs
      encoder.rs
      mcts.rs
      position.rs
      pybridge.rs
      samples.rs
      mcts/
        evaluator.rs
        search.rs
        tree.rs
```

Any Rust utility code lives inside `packages/hexo_utils` with the Python
utility package. Utility Rust may depend on engine contracts, but it must not
duplicate engine rules.

Current status: this package intentionally mixes real Rust prototypes with
Python contract scaffolding. The Python modules define typed surfaces and
exports; several operations still raise `NotImplementedError` until the Rust
bridge and storage paths are wired.

## File Responsibilities

| File | Role |
| --- | --- |
| `pyproject.toml` | Python/Rust package metadata and maturin bridge settings. |
| `Cargo.toml` | Rust crate metadata for utility code. |
| `python/hexo_utils/__init__.py` | Package status/version marker for the Python utility surface. |
| `python/hexo_utils/rust_bridge.py` | Placeholder loader boundary for the compiled `_rust` extension. |
| `python/hexo_utils/py.typed` | Marker that the package ships type information. |
| `encoding/__init__.py` | Public exports for crop, mask, and symmetry helpers. |
| `encoding/crop.py` | Python crop-request/window contract; implementation is placeholder-backed. |
| `encoding/masks.py` | Python action-mask contract and simple id-mask helpers. |
| `encoding/symmetry.py` | Shared D6 symmetry labels and action-transform protocol. |
| `search/__init__.py` | Public exports for Python search contracts. |
| `search/mcts.py` | Python MCTS request/result contracts; search execution is placeholder-backed. |
| `samples/__init__.py` | Public sample-buffer, record, and target helper exports. |
| `samples/buffer.py` | Sample store, append result, index, window, request, and batch scaffolding. |
| `samples/records.py` | Shared sample schema metadata and neutral training record shapes. |
| `samples/targets.py` | Common legal-action policy/value target helpers. |
| `rust/src/lib.rs` | Rust crate root and public utility exports. |
| `rust/src/encoder.rs` | Fixed square crop encoder with current 12-plane layout. |
| `rust/src/mcts.rs` | Public Rust MCTS module declaration. |
| `rust/src/position.rs` | Search position wrapper around cloned engine state. |
| `rust/src/pybridge.rs` | Minimal importable PyO3 bridge scaffold. |
| `rust/src/samples.rs` | Draft Rust sample manifest marker. |
| `rust/src/mcts/evaluator.rs` | Evaluator-facing policy/value contracts for search. |
| `rust/src/mcts/search.rs` | Autoregressive single-stone PUCT search prototype. |
| `rust/src/mcts/tree.rs` | In-memory search tree structures. |

## Utility Rule

A helper belongs in `hexo-utils` when:

- it has more than one plausible consumer,
- it has a stable contract,
- it does not make model-specific policy choices,
- it does not reinterpret game legality or terminal state.

Otherwise it belongs in the engine, runner, or a model package.

## Core Areas

`encoding`: optional shared board/crop/mask helpers. Crops may be square,
circular, multi-window, or bypassed entirely by model packages that consume the
whole board or a custom representation. Masks translate engine legal actions
into model-facing shapes; threat masks may only strip engine legal actions
using engine tactical facts. Symmetry helpers define D6 transport types and
action-id transform protocols; `hexo_train` decides when training samples
receive D6 selections, and model packages decide how those transforms apply to
their tensors.

`search`: reusable MCTS machinery and supporting search statistics. In Rust,
`mcts.rs` is the public search module, `mcts/search.rs` owns the autoregressive
single-stone PUCT search prototype, `mcts/evaluator.rs` owns evaluator-facing
policy/value contracts that adapt encoded crop logits to legal-action priors,
and `mcts/tree.rs` owns the in-memory node/edge tree used by a single MCTS run.
`position.rs` wraps a cloned `hexo_engine::HexoState` for search rollouts so
MCTS can simulate through engine-authoritative transitions without mutating the
caller's root state.

`samples`: training sample schemas and record shapes, consolidated sample
buffer mechanics in `buffer.py`, common policy logits over legal actions when
the model opts into that default shape, default legal-action policy/value target
helpers, and validation tools.

The Python samples package stays intentionally compact:

- `buffer.py`: sample store handles, append results, index refresh, training
  windows, and future deterministic sample requests;
- `records.py`: schema version metadata and neutral training sample records;
- `targets.py`: reusable legal-action policy/value target builders.

The samples layer should not own the authoritative position trail. That belongs
to `hexo_runner.records`. It exists so model packages can write trainable
samples during self-play, finalize result-dependent targets when a game ends,
shuffle and sample those buffers, and attach custom payloads without teaching
shared utilities about model-specific heads or logic.

A first real sample buffer should support:

- appending finalized samples;
- flushing compact chunk files;
- refreshing a compact index;
- selecting deterministic training windows;
- carrying symmetry selections from `hexo_train`;
- leaving tensor decoding to the model package.

The default target path is intentionally narrow: legal-action policy logits and
an optional scalar value. The default builder reads one action order from
`TrainingSampleRecord.legal_action_ids` and can apply a D6 symmetry supplied by
`hexo_train` through an engine/model-provided mapper. Pair policies, auxiliary
heads, search traces, and architecture-specific labels remain model-owned
extensions.

Shared sample helpers do not make datasets interchangeable. Models own the
rules for writing their own self-play samples and interpreting their policy
outputs.

## Contract Flow

```text
engine supplies canonical state and legal actions
utils supplies reusable mechanisms
model supplies policy interpretation and training meaning
runner supplies orchestration
```

Utilities should remain opt-in. A model package can use shared helpers when
the semantics match, or keep custom code when its representation needs it.
