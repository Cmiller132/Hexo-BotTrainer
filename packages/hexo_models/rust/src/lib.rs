//! Model-owned Rust accelerators for Hexo model families.

#[cfg(feature = "python")]
#[path = "../../dense_cnn/rust/src/lib.rs"]
mod dense_cnn;

#[cfg(feature = "python")]
#[path = "../../hexgt/rust/src/lib.rs"]
mod hexgt;

// hexgnn: a forked-from-hexgt crate for the sparse-graph/perf rewrite. Compiled
// into the SAME native module as a separate submodule `hexo_models._rust.hexgnn`
// so the hexgnn lineage can sparsify/optimize the featurizer independently of the
// (permanently-halted) hexgt run. See docs/analysis/HEXGNN_SPARSE_SPEED_SPEC.md.
#[cfg(feature = "python")]
#[path = "../../../hexgnn/rust/src/lib.rs"]
mod hexgnn;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
pub fn _rust(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    let dense_cnn_module = PyModule::new(py, "dense_cnn")?;
    dense_cnn::register_pybridge(&dense_cnn_module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("hexo_models._rust.dense_cnn", &dense_cnn_module)?;
    module.add_submodule(&dense_cnn_module)?;

    let hexgt_module = PyModule::new(py, "hexgt")?;
    hexgt::register_pybridge(&hexgt_module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("hexo_models._rust.hexgt", &hexgt_module)?;
    module.add_submodule(&hexgt_module)?;

    let hexgnn_module = PyModule::new(py, "hexgnn")?;
    hexgnn::register_pybridge(&hexgnn_module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("hexo_models._rust.hexgnn", &hexgnn_module)?;
    module.add_submodule(&hexgnn_module)?;
    Ok(())
}
