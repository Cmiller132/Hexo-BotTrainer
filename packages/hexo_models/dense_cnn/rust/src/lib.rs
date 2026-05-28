//! Dense CNN Rust accelerator package.
//!
//! The Rust half of dense_cnn owns the model-specific native path:
//!
//! 1. Clone live `hexo_engine.HexoState` objects through `state`.
//! 2. Encode those states into Model 1 tensors through `encoding`.
//! 3. Run batched tree search through `mcts` and `mcts_tree`.
//! 4. Call the Python/Torch evaluator through the strict byte contract in
//!    `mcts_eval`.
//! 5. Generate and finalize compact self-play samples through `sample_gen`.
//!
//! `lib.rs` only registers those pieces into the Python extension module and
//! publishes capability metadata. It deliberately contains no model logic.

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
    dict.set_item("model1_mcts_all_legal_candidates", true)?;
    dict.set_item("model1_mcts_tree_reuse_session", true)?;
    dict.set_item("model1_mcts_session_search", true)?;
    dict.set_item(
        "model1_mcts_tree_reuse_reference",
        "KataGo_Search_makeMove_promote_child",
    )?;
    dict.set_item("model1_mcts_root_dirichlet_noise", true)?;
    dict.set_item("model1_mcts_root_policy_temperature", true)?;
    dict.set_item("model1_mcts_first_play_urgency", true)?;
    dict.set_item("model1_mcts_virtual_loss", true)?;
    dict.set_item("model1_sample_from_state", true)?;
    Ok(dict.into_any().unbind())
}

pub fn register_pybridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(capabilities, module)?)?;
    encoding::register_pybridge(module)?;
    mcts::register_pybridge(module)?;
    sample_gen::register_pybridge(module)?;
    Ok(())
}
