//! Hexformer AR Rust accelerator package.
//!
//! `lib.rs` stays as the PyO3 export map. Implementation details are split by
//! responsibility so MCTS, evaluator parsing, tree mechanics, state intake,
//! and sample generation can evolve independently.

mod constants;
mod state;
mod mcts_eval;
mod mcts_tree;
mod mcts;
mod sample_gen;

use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
pub fn capabilities(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("status", "ready")?;
    dict.set_item("model_family", "hexformer_ar")?;
    dict.set_item("state_source", "direct_engine_state")?;
    dict.set_item("coordinate_encoding", "u32_i16_pair")?;
    dict.set_item("hexformer_ar_mcts", true)?;
    dict.set_item("hexformer_ar_batched_mcts", true)?;
    dict.set_item("sparse_input_payload", true)?;
    dict.set_item("sparse_input_payloads", true)?;
    dict.set_item("selfplay_sample_payloads", true)?;
    Ok(dict.into_any().unbind())
}

pub fn register_pybridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(capabilities, module)?)?;
    mcts::register_pybridge(module)?;
    sample_gen::register_pybridge(module)?;
    Ok(())
}
