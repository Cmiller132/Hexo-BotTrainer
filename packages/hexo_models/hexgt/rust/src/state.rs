//! hexgt state intake from live engine objects.
//!
//! Identical in spirit to `dense_cnn/rust/src/state.rs`: Python passes live
//! `hexo_engine.HexoState` objects, the engine capsule clones each state into an
//! owned Rust handle, and hexgt copies that handle into its local graph/search
//! code. This is the narrow cooperation point between the generic engine and the
//! model-specific accelerator; if the capsule version changes, hexgt fails at
//! use time rather than reading an incompatible state layout.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::ffi::c_void;
use std::ptr;

use hexo_engine::HexoState as RustHexoState;

const STATE_API_CAPSULE_NAME: &str = "hexo_engine._rust.state_api";
const STATE_API_VERSION: u32 = 2;

#[repr(C)]
struct HexoStateApi {
    version: u32,
    clone_state: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> i32,
    free_state: unsafe extern "C" fn(*mut c_void),
}

pub(crate) fn state_from_py_state(
    py: Python<'_>,
    state: &Bound<'_, PyAny>,
) -> PyResult<RustHexoState> {
    state_from_py_state_with_api(engine_state_api(py)?, state)
}

fn engine_state_api(py: Python<'_>) -> PyResult<&'static HexoStateApi> {
    let module = py.import("hexo_engine._rust")?;
    let capsule = module.call_method0("state_api_capsule")?;
    let name = pyo3::ffi::c_str!("hexo_engine._rust.state_api");
    debug_assert_eq!(name.to_string_lossy(), STATE_API_CAPSULE_NAME);
    let pointer = unsafe { pyo3::ffi::PyCapsule_GetPointer(capsule.as_ptr(), name.as_ptr()) };
    if pointer.is_null() {
        return Err(PyErr::fetch(py));
    }
    let api = unsafe { &*(pointer as *const HexoStateApi) };
    if api.version != STATE_API_VERSION {
        return Err(PyRuntimeError::new_err(format!(
            "unsupported hexo_engine state API version {}; expected {}",
            api.version, STATE_API_VERSION
        )));
    }
    Ok(api)
}

fn state_from_py_state_with_api(
    api: &HexoStateApi,
    state: &Bound<'_, PyAny>,
) -> PyResult<RustHexoState> {
    let mut handle: *mut c_void = ptr::null_mut();
    let code = unsafe {
        (api.clone_state)(
            state.as_ptr() as *mut c_void,
            &mut handle as *mut *mut c_void,
        )
    };
    if code != 0 {
        return Err(PyValueError::new_err(format!(
            "hexo_engine could not clone state through capsule; code={code}"
        )));
    }
    if handle.is_null() {
        return Err(PyRuntimeError::new_err(
            "hexo_engine returned an empty state handle",
        ));
    }
    let cloned = unsafe { (&*handle.cast::<RustHexoState>()).clone() };
    unsafe {
        (api.free_state)(handle);
    }
    Ok(cloned)
}
