//! Dense CNN state intake from live engine objects.
//!
//! The only production path is direct state handoff. Python passes live
//! `hexo_engine.HexoState` objects, the engine capsule clones each state into
//! an owned Rust handle, and dense_cnn immediately copies that handle into its
//! local search/encoding code. No move-history transport, Python mirror state,
//! or model-specific engine API is involved.
//!
//! This module is the narrow cooperation point between the generic engine and a
//! model-specific accelerator. If the capsule version changes, dense_cnn should
//! fail at import/use time rather than reading an incompatible state layout.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::ffi::c_void;
use std::ptr;

use hexo_engine::{HexoState as RustHexoState, MoveError};

const STATE_API_CAPSULE_NAME: &str = "hexo_engine._rust.state_api";
const STATE_API_VERSION: u32 = 2;

#[repr(C)]
struct HexoStateApi {
    version: u32,
    clone_state: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> i32,
    free_state: unsafe extern "C" fn(*mut c_void),
}

pub(crate) fn states_from_py_states(
    py: Python<'_>,
    states: &Bound<'_, PyAny>,
) -> PyResult<Vec<RustHexoState>> {
    // Resolve the capsule once per batch. Each item is independently cloned so
    // MCTS can mutate search-local states without touching the live Python game.
    let api = engine_state_api(py)?;
    let mut roots = Vec::new();
    for item in states.try_iter()? {
        let item = item?;
        roots.push(state_from_py_state_with_api(api, &item)?);
    }
    Ok(roots)
}

pub(crate) fn state_from_py_state(
    py: Python<'_>,
    state: &Bound<'_, PyAny>,
) -> PyResult<RustHexoState> {
    state_from_py_state_with_api(engine_state_api(py)?, state)
}

fn engine_state_api(py: Python<'_>) -> PyResult<&'static HexoStateApi> {
    // `hexo_engine._rust` owns the capsule. Dense_cnn reads only the stable C
    // ABI pointers exposed by that capsule, keeping engine internals generic.
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
    // The capsule returns an opaque heap handle. We clone the Rust state value
    // out of it immediately, then ask the engine API to free the handle.
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

pub(crate) fn move_error(error: MoveError) -> PyErr {
    PyValueError::new_err(error.to_string())
}
