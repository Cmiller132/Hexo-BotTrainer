# hexfield_search_core

Single-source home for the MCTS search core shared by the `hexfield` and
`hexfield_eq` rust crates: `src/search.rs`, `src/tree.rs`, `src/cache.rs`,
`src/state.rs`.

This is **not** a Cargo crate. Each lineage includes these files directly via
`#[path]` module declarations in its own `rust/src/lib.rs` (same pattern as
`threats_shared.rs` from `hexo_models`), e.g.:

```rust
#[path = "../../../hexfield_search_core/src/search.rs"]
mod search;
```

## Inclusion contract

These files compile *inside each including crate*. They may only reference
`crate::` items whose interfaces exist identically in both lineages:
`payload`, `threats_shared`, and each other (`cache`, `search`, `state`,
`tree`). The lineages' `payload.rs` implementations deliberately differ, but
the items this core imports from them must keep matching signatures in both.

Keeping the files crate-local (rather than a shared dependency crate)
preserves the deliberate cdylib/rebuild isolation between lineages.

## Rule

**Any change here affects BOTH production lineages** (`hexfield` and
`hexfield_eq`). Build and test both after editing.
