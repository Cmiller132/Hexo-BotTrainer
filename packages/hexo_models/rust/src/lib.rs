//! Model-owned Rust accelerators for Hexo model families.

#[cfg(feature = "python")]
#[path = "../../dense_cnn/rust/src/lib.rs"]
mod dense_cnn;

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
    Ok(())
}
