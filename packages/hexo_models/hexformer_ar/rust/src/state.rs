//! Shared Hexformer state reconstruction from packed histories.
//!
//! Both MCTS and sparse sample generation operate from the same minimal input:
//! a row of u32 packed coordinates. Keeping this here prevents either path from
//! depending on model-specific helpers in `hexo_engine`'s Python bridge.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use hexo_engine::{
    apply_placement, unpack_coord, HexoState as RustHexoState, MoveError, PackedCoord,
    Placement,
};

pub(crate) fn states_from_history_rows(history_rows: &Bound<'_, PyAny>) -> PyResult<Vec<RustHexoState>> {
    let mut states = Vec::new();
    for row in history_rows.try_iter()? {
        let row = row?;
        states.push(state_from_history_row(&row)?);
    }
    Ok(states)
}

pub(crate) fn state_from_history_row(row: &Bound<'_, PyAny>) -> PyResult<RustHexoState> {
    let mut state = RustHexoState::new();
    for item in row.try_iter()? {
        let action_id = item?.extract::<PackedCoord>()?;
        apply_placement(
            &mut state,
            Placement {
                coord: unpack_coord(action_id),
            },
        )
        .map_err(move_error)?;
    }
    Ok(state)
}

pub(crate) fn move_error(error: MoveError) -> PyErr {
    PyValueError::new_err(error.to_string())
}