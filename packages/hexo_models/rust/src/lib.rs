//! Model-owned Rust accelerators for Hexo model families.
//!
//! This small crate root is the ONE Cargo crate / ONE maturin cdylib that
//! physically hosts every lineage's native accelerator: the dense_cnn and hexgt
//! source trees (under this package) and the hexgnn tree (in the sibling
//! `packages/hexgnn`) are `#[path]`-included below and registered as the Python
//! submodules `hexo_models._rust.{dense_cnn,hexgt,hexgnn}`. Consumers reach
//! them through each lineage's Python `rust_bridge.py` -- including
//! `packages/dense_cnn_restnet/python/dense_cnn_restnet/rust_bridge.py`, the
//! ACTIVE training lineage, which ships no Rust of its own and drives
//! `_rust.dense_cnn` read-only. Rebuilding this crate therefore changes search
//! semantics for every lineage at once (intended, per HANDOFF.md).
//!
//! Build: maturin via `packages/hexo_models/pyproject.toml` (module-name
//! `hexo_models._rust`, feature `python`); canonical rebuild script is
//! `scripts/_rebuild_hexo_models_hexgt.sh` (WSL hexgt-build venv, --release).
//! Rust edits are inert until that script runs. `cargo test -p hexo_models`
//! exercises the threats_shared unit tests without the `python` feature.

// Threat-Space Search core, shared by the dense_cnn and hexgt native MCTS.
// Pure board geometry over the engine `WindowStore`; no graph/network coupling,
// so both #[path]-included lineages reach it via `crate::threats_shared` and
// share one definition of threat/win-now/forced-loss semantics. NOTE: the
// hexgnn crate included below carries its own forked copy
// (packages/hexgnn/rust/src/threats.rs) and does NOT use this module.
mod threats_shared;

#[cfg(feature = "python")]
#[path = "../../dense_cnn/rust/src/lib.rs"]
mod dense_cnn;

#[cfg(feature = "python")]
#[path = "../../hexgt/rust/src/lib.rs"]
mod hexgt;

// hexgnn: a forked-from-hexgt crate for the sparse-graph/perf rewrite. Compiled
// into the SAME native module as a separate submodule `hexo_models._rust.hexgnn`
// so the hexgnn lineage could sparsify/optimize the featurizer independently of
// the (permanently-halted) hexgt run. The lineage is now PARKED (explored and
// set aside per HANDOFF.md; see packages/hexgnn/README.md), but it still
// compiles into every build of this crate. CAUTION: this #[path] reaches
// outside the package directory -- the build breaks if packages/hexgnn moves,
// and the sdist must bundle ../hexgnn/rust (special-cased in pyproject.toml).
#[cfg(feature = "python")]
#[path = "../../../hexgnn/rust/src/lib.rs"]
mod hexgnn;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Extension-module entry point: registers each lineage's pybridge into a
/// child PyModule and exposes it both as an attribute (`add_submodule`) and as
/// a real entry in `sys.modules`. The `sys.modules` patch is load-bearing:
/// `add_submodule` alone only sets an attribute on `_rust`, so statements like
/// `import hexo_models._rust.dense_cnn` (used by every lineage's
/// rust_bridge.py) would fail without it.
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
