//! Minimal PyO3 bridge while the Python layer is redesigned.
//!
//! Keep the extension importable, but do not expose self-play, replay writing,
//! or evaluator callbacks yet. Those contracts should be rebuilt deliberately
//! after the new Python top layer is specified.

use pyo3::prelude::*;

/// Return a tiny capabilities object for smoke tests and packaging checks.
#[pyfunction]
pub fn capabilities(py: Python<'_>) -> PyResult<PyObject> {
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("status", "placeholder")?;
    dict.set_item("selfplay", false)?;
    dict.set_item("replay", false)?;
    dict.set_item("message", "models_common PyO3 bridge is awaiting redesign")?;
    Ok(dict.into_py(py))
}

/// Register Python-visible functions on a module.
pub fn register_pybridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(capabilities, module)?)?;
    Ok(())
}

/// Private Python extension module entry point.
#[pymodule]
pub fn _rust(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    register_pybridge(module)
}
