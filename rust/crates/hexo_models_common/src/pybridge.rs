//! Minimal PyO3 bridge.
//!
//! The main Python training loop is still a skeleton, but this module exposes a
//! small uniform-evaluator self-play smoke path. It proves the Rust engine,
//! MCTS, and Python packaging can connect without committing to the final
//! neural inference protocol yet.

use pyo3::prelude::*;

use crate::mcts::{MctsConfig, UniformEvaluator};
use crate::selfplay::{play_selfplay_game, SelfplayConfig};

/// Python-facing MCTS config.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyMctsConfig {
    /// Number of simulations per placement.
    #[pyo3(get, set)]
    pub visits: u32,
    /// PUCT exploration constant.
    #[pyo3(get, set)]
    pub c_puct: f32,
    /// Encoder crop size.
    #[pyo3(get, set)]
    pub crop_size: usize,
    /// Root sampling temperature.
    #[pyo3(get, set)]
    pub temperature: f32,
}

#[pymethods]
impl PyMctsConfig {
    #[new]
    #[pyo3(signature = (visits = 64, c_puct = 1.5, crop_size = 31, temperature = 0.0))]
    pub fn new(visits: u32, c_puct: f32, crop_size: usize, temperature: f32) -> Self {
        Self {
            visits,
            c_puct,
            crop_size,
            temperature,
        }
    }
}

/// Convert Python config wrapper into the internal Rust config.
impl From<&PyMctsConfig> for MctsConfig {
    fn from(value: &PyMctsConfig) -> Self {
        Self {
            visits: value.visits,
            c_puct: value.c_puct,
            crop_size: value.crop_size,
            temperature: value.temperature,
        }
    }
}

/// Python-facing game self-play config.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PySelfplayConfig {
    /// Placement cap for a game.
    #[pyo3(get, set)]
    pub max_placements: u32,
    /// Replay encoder crop size.
    #[pyo3(get, set)]
    pub crop_size: usize,
}

#[pymethods]
impl PySelfplayConfig {
    #[new]
    #[pyo3(signature = (max_placements = 300, crop_size = 31))]
    pub fn new(max_placements: u32, crop_size: usize) -> Self {
        Self {
            max_placements,
            crop_size,
        }
    }
}

/// Convert Python config wrapper into the internal Rust config.
impl From<&PySelfplayConfig> for SelfplayConfig {
    fn from(value: &PySelfplayConfig) -> Self {
        Self {
            max_placements: value.max_placements,
            crop_size: value.crop_size,
        }
    }
}

/// Small summary object returned to Python.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PySelfplaySummary {
    /// Number of placement-level samples generated.
    #[pyo3(get)]
    pub samples: usize,
    /// Number of stones placed in the game.
    #[pyo3(get)]
    pub placements_made: u32,
    /// True if the game ended with a winner before the placement cap.
    #[pyo3(get)]
    pub terminal: bool,
}

/// Run one uniform-evaluator self-play game.
///
/// This is intentionally small: it is a smoke/integration function, not the
/// final high-throughput actor API.
#[pyfunction]
pub fn run_uniform_selfplay(
    game_config: &PySelfplayConfig,
    mcts_config: &PyMctsConfig,
) -> PyResult<PySelfplaySummary> {
    let mut evaluator = UniformEvaluator;
    let game_config = SelfplayConfig::from(game_config);
    let mcts_config = MctsConfig::from(mcts_config);
    let game = play_selfplay_game(&game_config, &mcts_config, &mut evaluator)
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(format!("{error:?}")))?;

    Ok(PySelfplaySummary {
        samples: game.samples.len(),
        placements_made: game.placements_made,
        terminal: game.outcome.is_some(),
    })
}

/// Register all Python classes/functions on a module.
pub fn register_pybridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMctsConfig>()?;
    module.add_class::<PySelfplayConfig>()?;
    module.add_class::<PySelfplaySummary>()?;
    module.add_function(wrap_pyfunction!(run_uniform_selfplay, module)?)?;
    Ok(())
}

/// Python extension module entry point.
#[pymodule]
pub fn models_common_rust(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    register_pybridge(module)
}
