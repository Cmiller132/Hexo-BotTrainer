//! Shared dense CNN state reconstruction from packed histories.
//!
//! Python sends model accelerators only u32 coordinate IDs. This module is the
//! small, shared reconstruction layer used by dense CNN MCTS, encoding, and
//! sample generation without adding model hooks to `hexo_engine`.

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
