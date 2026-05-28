//! Hexformer AR Rust accelerator package.
//!
//! `lib.rs` stays as the PyO3 export map. Implementation details are split by
//! responsibility so MCTS, evaluator parsing, tree mechanics, state intake,
//! and sample generation can evolve independently.

mod constants;
mod engine_state;
mod mcts;
mod mcts_eval;
mod mcts_tree;
mod sample_gen;

use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
pub fn capabilities(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("status", "ready")?;
    dict.set_item("model_family", "hexformer_ar")?;
    dict.set_item("state_source", "engine_state_clone")?;
    dict.set_item("engine_state_clone", true)?;
    dict.set_item("state_api_version", engine_state::STATE_API_VERSION)?;
    dict.set_item("coordinate_encoding", "u32_i16_pair")?;
    dict.set_item(
        "coordinate_range_note",
        "The game model uses sparse infinite-board logic within the current engine coordinate range. The current ActionId transport is u32_i16_pair and is bounded to i16 coordinate components.",
    )?;
    dict.set_item("hexformer_ar_mcts", true)?;
    dict.set_item("hexformer_ar_batched_mcts", true)?;
    dict.set_item("sparse_input_payload_from_state", true)?;
    dict.set_item("sparse_input_payloads_from_states", true)?;
    dict.set_item("selfplay_sample_payloads_from_states", true)?;
    Ok(dict.into_any().unbind())
}

pub fn register_pybridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(capabilities, module)?)?;
    mcts::register_pybridge(module)?;
    sample_gen::register_pybridge(module)?;
    Ok(())
}
