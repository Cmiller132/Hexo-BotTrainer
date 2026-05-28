//! Dense CNN Rust accelerator package.
//!
//! `lib.rs` intentionally only wires together Python exports. The model logic is
//! split by responsibility: direct engine-state intake, tensor encoding, neural
//! evaluation payloads, MCTS, and compact sample generation.

mod constants;
mod encoding;
mod mcts;
mod mcts_eval;
mod mcts_tree;
mod sample_gen;
mod state;

use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
pub fn capabilities(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("status", "ready")?;
    dict.set_item("model_family", "dense_cnn")?;
    dict.set_item("state_source", "direct_engine_state")?;
    dict.set_item("coordinate_encoding", "u32_i16_pair")?;
    dict.set_item("model1_batch_inputs", true)?;
    dict.set_item("model1_batched_mcts", true)?;
    dict.set_item("model1_mcts_progressive_widening", true)?;
    dict.set_item(
        "model1_mcts_progressive_widening_reference",
        "Chaslot_2008_progressive_unpruning",
    )?;
    dict.set_item("model1_mcts_evaluation_cache", true)?;
    dict.set_item("model1_mcts_tree_reuse_session", true)?;
    dict.set_item(
        "model1_mcts_tree_reuse_reference",
        "KataGo_Search_makeMove_promote_child",
    )?;
    dict.set_item("model1_mcts_lazy_staged_edges", true)?;
    dict.set_item(
        "model1_mcts_lazy_staged_edges_reference",
        "KataGo_SearchNode_children0_1_2",
    )?;
    dict.set_item("model1_sample_from_state", true)?;
    dict.set_item("model1_finalize_game_samples", true)?;
    Ok(dict.into_any().unbind())
}

pub fn register_pybridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(capabilities, module)?)?;
    encoding::register_pybridge(module)?;
    mcts::register_pybridge(module)?;
    sample_gen::register_pybridge(module)?;
    Ok(())
}
