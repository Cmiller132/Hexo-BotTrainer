//! hexgnn (Model 2) Rust accelerator package.
//!
//! The Rust half of hexgnn owns the engine-grounded native path shared by
//! sample-gen and live MCTS:
//!
//! 1. Enumerate the dynamic candidate set + active-window tokens from a live
//!    `hexo_engine` state (`candidates`).
//! 2. Build the bounded typed graph (`candidates::build_graph`). SPARSE REWRITE:
//!    the window-hub routing was removed — only adjacency + recency edges are
//!    emitted, with window counts folded into per-node features (see
//!    candidates.rs / features.rs).
//! 3. Featurize + collate leaf batches (`features`, rayon, zero-copy buffers)
//!    and run the batched tree search (`mcts` / `mcts_tree` / `mcts_eval`).
//!
//! Like dense_cnn, this crate is NOT a workspace member; it is `#[path]`-included
//! into `hexo_models::_rust` (see `packages/hexo_models/rust/src/lib.rs`).
//! `lib.rs` only registers pieces into the Python submodule and publishes
//! capability metadata; it contains no model logic.

mod candidates;
mod constants;
mod features;
mod mcts;
mod mcts_eval;
mod mcts_tree;
mod state;
mod threats;
mod vcf;

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Capability metadata for the hexgnn accelerator, surfaced to Python via
/// `rust_bridge.capabilities()`:
/// `status`/`model_family` identify the build, `state_source`/`coordinate_encoding`
/// the engine-state intake, `candidate_set`/`edge_construction`/`policy` the graph
/// + head contract, and `search`/`eval_transport` the MCTS + byte-payload path.
///
/// UNUSED(2026-06-12): no caller found in packages/tests/scripts — the
/// `rust_bridge.capabilities()` wrapper is itself uncalled (the tests that probe
/// capabilities use the hexgt/dense_cnn bridges). Kept as an API mirror of the
/// sibling lineages. NOTE the `edge_construction` value string predates the
/// sparse rewrite (window hubs removed); it is a frozen identifier, not a
/// description of the current graph.
#[pyfunction]
pub fn capabilities(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("status", "mcts")?;
    dict.set_item("model_family", "hexgnn")?;
    dict.set_item("state_source", "direct_engine_state")?;
    dict.set_item("coordinate_encoding", "u32_i16_pair")?;
    dict.set_item("candidate_set", "active_windows_union_n_radius")?;
    dict.set_item("edge_construction", "window_hub_bounded_no_cliques")?;
    dict.set_item("policy", "dynamic_per_candidate")?;
    dict.set_item("search", "batched_puct_nucleus_widening")?;
    dict.set_item("eval_transport", "graph_facts_candidate_csr")?;
    Ok(dict.into_any().unbind())
}

pub fn register_pybridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(capabilities, module)?)?;
    candidates::register_pybridge(module)?;
    features::register_pybridge(module)?;
    mcts::register_pybridge(module)?;
    threats::register_pybridge(module)?;
    vcf::register_pybridge(module)?;
    Ok(())
}
