//! Engine-state intake for Hexformer.
//!
//! Production Hexformer paths receive live Python `hexo_engine.HexoState`
//! objects. This module is the narrow model-side boundary that clones those
//! objects through the engine's private C-ABI capsule, immediately copies the
//! cloned Rust state into Hexformer's ownership, and frees the temporary
//! capsule allocation. No borrowed engine pointer escapes this file.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::ffi::c_void;
use std::ptr;

use hexo_engine::{HexoState as RustHexoState, MoveError};

const STATE_API_CAPSULE_NAME: &str = "hexo_engine._rust.state_api";
pub(crate) const STATE_API_VERSION: u32 = 2;

#[repr(C)]
struct EngineStateApi {
    version: u32,
    clone_state: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> i32,
    free_state: unsafe extern "C" fn(*mut c_void),
}

pub(crate) fn clone_py_engine_state(
    py: Python<'_>,
    state_obj: &Bound<'_, PyAny>,
) -> PyResult<RustHexoState> {
    clone_py_engine_state_with_api(load_state_api(py)?, state_obj)
}

pub(crate) fn clone_py_engine_states(
    py: Python<'_>,
    states_obj: &Bound<'_, PyAny>,
) -> PyResult<Vec<RustHexoState>> {
    let api = load_state_api(py)?;
    let mut states = Vec::new();
    for item in states_obj.try_iter()? {
        states.push(clone_py_engine_state_with_api(api, &item?)?);
    }
    Ok(states)
}

fn load_state_api(py: Python<'_>) -> PyResult<&'static EngineStateApi> {
    let module = py.import("hexo_engine._rust")?;
    let capsule = module.call_method0("state_api_capsule")?;
    let name = pyo3::ffi::c_str!("hexo_engine._rust.state_api");
    debug_assert_eq!(name.to_string_lossy(), STATE_API_CAPSULE_NAME);
    let pointer = unsafe { pyo3::ffi::PyCapsule_GetPointer(capsule.as_ptr(), name.as_ptr()) };
    if pointer.is_null() {
        return Err(PyErr::fetch(py));
    }
    let api = unsafe { &*(pointer as *const EngineStateApi) };
    if api.version != STATE_API_VERSION {
        return Err(PyRuntimeError::new_err(format!(
            "unsupported hexo_engine state API version {}; expected {}",
            api.version, STATE_API_VERSION
        )));
    }
    Ok(api)
}

fn clone_py_engine_state_with_api(
    api: &EngineStateApi,
    state_obj: &Bound<'_, PyAny>,
) -> PyResult<RustHexoState> {
    let mut raw: *mut c_void = ptr::null_mut();
    let code = unsafe {
        (api.clone_state)(
            state_obj.as_ptr() as *mut c_void,
            &mut raw as *mut *mut c_void,
        )
    };
    if code != 0 || raw.is_null() {
        return Err(PyValueError::new_err(format!(
            "failed to clone HexoState through state API: code={code}"
        )));
    }
    let owned = unsafe { (&*raw.cast::<RustHexoState>()).clone() };
    unsafe {
        (api.free_state)(raw);
    }
    Ok(owned)
}

pub(crate) fn move_error(error: MoveError) -> PyErr {
    PyValueError::new_err(error.to_string())
}
