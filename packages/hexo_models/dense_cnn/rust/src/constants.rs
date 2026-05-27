//! Dense CNN Model1 tensor dimensions and plane indices.
//!
//! These values define the Python-visible tensor contract for Model1. Keep them
//! model-local: the engine only deals in board state and packed coordinates.

use std::sync::OnceLock;

pub(crate) const MODEL1_BOARD_SIZE: usize = 41;
pub(crate) const MODEL1_BOARD_AREA: usize = MODEL1_BOARD_SIZE * MODEL1_BOARD_SIZE;
pub(crate) const MODEL1_INPUT_CHANNELS: usize = 13;
pub(crate) const MODEL1_PLANE_OWN_STONES: usize = 0;
pub(crate) const MODEL1_PLANE_OPPONENT_STONES: usize = 1;
pub(crate) const MODEL1_PLANE_EMPTY: usize = 2;
pub(crate) const MODEL1_PLANE_LEGAL: usize = 3;
pub(crate) const MODEL1_PLANE_SECOND_PLACEMENT: usize = 4;
pub(crate) const MODEL1_PLANE_FIRST_STONE: usize = 5;
pub(crate) const MODEL1_PLANE_PLAYER_COLOUR: usize = 6;
pub(crate) const MODEL1_PLANE_OWN_RECENCY: usize = 7;
pub(crate) const MODEL1_PLANE_OPPONENT_RECENCY: usize = 8;
pub(crate) const MODEL1_PLANE_OPPONENT_HOT: usize = 9;
pub(crate) const MODEL1_PLANE_OWN_HOT: usize = 10;
pub(crate) const MODEL1_PLANE_CENTER_DISTANCE: usize = 11;
pub(crate) const MODEL1_PLANE_OPPONENT_LAST_TURN: usize = 12;
pub(crate) static MODEL1_BASE_PLANES: OnceLock<Vec<f32>> = OnceLock::new();
