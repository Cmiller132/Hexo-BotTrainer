//! hexgt (Model 2) Rust accelerator package.
//!
//! The Rust half of hexgt owns the engine-grounded native path shared by
//! sample-gen and live MCTS:
//!
//! 1. Enumerate the dynamic candidate set + active-window tokens from a live
//!    `hexo_engine` state (Phase 1, `candidates`).
//! 2. Build the bounded, window-hub-routed typed graph (no same-axis cliques)
//!    (Phase 1, `graph`).
//! 3. Encode the packed-graph payload for the Python/Torch evaluator and run
//!    batched tree search (Phase 5).
//!
//! Like dense_cnn, this crate is NOT a workspace member; it is `#[path]`-included
//! into `hexo_models::_rust` (see `packages/hexo_models/rust/src/lib.rs`).
//! `lib.rs` only registers pieces into the Python submodule and publishes
//! capability metadata; it contains no model logic.

use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
pub fn capabilities(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("status", "scaffold")?;
    dict.set_item("model_family", "hexgt")?;
    dict.set_item("state_source", "direct_engine_state")?;
    dict.set_item("coordinate_encoding", "u32_i16_pair")?;
    dict.set_item("candidate_set", "active_windows_union_n_radius")?;
    dict.set_item("edge_construction", "window_hub_bounded_no_cliques")?;
    dict.set_item("policy", "dynamic_per_candidate")?;
    Ok(dict.into_any().unbind())
}

pub fn register_pybridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(capabilities, module)?)?;
    Ok(())
}
