//! Minimal PyO3 bridge for the Python engine package.
//!
//! The bridge should stay narrow: expose opaque state handles and typed calls
//! that forward to the Rust rules authority. Python owns ergonomics and package
//! shape; Rust owns legality, state transitions, identity, and snapshots.

use pyo3::prelude::*;

/// Return a tiny capabilities object until the full API is bound.
#[pyfunction]
pub fn capabilities(py: Python<'_>) -> PyResult<PyObject> {
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("status", "placeholder")?;
    dict.set_item("engine_api", false)?;
    dict.set_item("message", "hexo_engine PyO3 bridge is awaiting API binding")?;
    Ok(dict.into_py(py))
}

/// Private Python extension module entry point.
#[pymodule]
pub fn _rust(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(capabilities, module)?)?;
    Ok(())
}
