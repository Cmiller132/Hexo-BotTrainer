//! threats.rs (hexgt) — thin shim over the shared TSS core.
//!
//! The threat/hitting-set/verdict logic that used to live here now lives in
//! `crate::threats_shared` (the single definition shared by every lineage's
//! native MCTS, so dense_cnn and hexgt can never drift). hexgt's MCTS still
//! refers to `super::threats::{analyze, tactical_cells, ThreatAnalysis, ...}`;
//! those names are re-exported below, so the search code is unchanged. This file
//! keeps only hexgt's PyO3 diagnostic hook (a different pyfunction name per
//! lineage), which now funnels through the shared dict builder.

pub(crate) use crate::threats_shared::*;

use pyo3::prelude::*;

/// Diagnostic: run the phase-aware threat analysis on a live engine state and
/// return its facts (drives the TEST C/G/H/I regression fixtures from Python and
/// lets self-play instrument how often the override/injection fire).
#[pyfunction]
fn hexgt_threat_analysis(py: Python<'_>, state: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let s = super::state::state_from_py_state(py, state)?;
    Ok(crate::threats_shared::analysis_pydict(py, &s)?
        .into_any()
        .unbind())
}

pub fn register_pybridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(hexgt_threat_analysis, module)?)?;
    Ok(())
}
