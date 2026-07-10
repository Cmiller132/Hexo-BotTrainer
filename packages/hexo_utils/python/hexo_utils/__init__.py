"""Shared Python utility package for Hexo training and replay.

Search, model encoding, and sample generation live inside the model packages.
This package keeps stable cross-model utilities such as records, sample-store
helpers, and symmetry contracts.

Subsystem status (see README.md for the full map):

- `records.py` (ACTIVE): Python facade over the Rust `.hxr` codec in
  `rust/src/records.rs` + `rust/src/pybridge.rs`. Production callers reach it
  through `packages/hexo_runner/python/hexo_runner/records/record.py`, which
  re-exports these classes.
- `samples/` + `encoding/` (LEGACY scaffolding): built for hexo_train's
  generic shared-sample-store path. Still imported by
  `packages/hexo_train/python/hexo_train/{defaults,symmetry,epoch/samples}.py`
  and covered by tests, but bypassed at runtime because every model plugin
  sets `uses_shared_sample_store=False` and owns its own NPZ replay storage.

The Rust crate (`rust/src`) also exports `hash_state` (state_hash.rs), the
MCTS evaluator-cache key used by the hexo_models crate; it has no
Python surface.
"""

__version__ = "0.1.0"
