//! Hexformer sparse input dimensions, defaults, and candidate tags.
//!
//! These constants are the Rust side of the Hexformer sample payload contract.
//! They stay model-local so the engine only exposes generic board state.

pub(crate) const DEFAULT_CANDIDATE_FEATURE_DIM: usize = 24;
pub(crate) const DEFAULT_STONE_FEATURE_DIM: usize = 18;
pub(crate) const DEFAULT_WINDOW_FEATURE_DIM: usize = 24;
pub(crate) const DEFAULT_GLOBAL_FEATURE_DIM: usize = 16;
pub(crate) const DEFAULT_LOCAL_INPUT_CHANNELS: usize = 13;
pub(crate) const DEFAULT_LOCAL_CROP_SIZE: usize = 41;
pub(crate) const DEFAULT_MAX_LOCAL_WINDOWS: usize = 3;
pub(crate) const DEFAULT_MAX_CANDIDATES: usize = 768;
pub(crate) const DEFAULT_MAX_STONES: usize = 512;
pub(crate) const DEFAULT_MAX_WINDOWS: usize = 768;
pub(crate) const DEFAULT_MAX_REL_EDGES: usize = 4096;
pub(crate) const DEFAULT_REL_EDGE_FEATURE_DIM: usize = 12;
pub(crate) const DEFAULT_LOOKAHEAD_HORIZONS: [i32; 4] = [1, 2, 4, 8];

pub(crate) const DEFAULT_TACTICAL_RADIUS: i16 = 2;
pub(crate) const DEFAULT_RECENT_RADIUS: i16 = 3;
pub(crate) const DEFAULT_FRONTIER_RADIUS: i16 = 8;
pub(crate) const DEFAULT_INCLUDE_ALL_LEGAL_BELOW: usize = DEFAULT_MAX_CANDIDATES;
pub(crate) const DEFAULT_REQUIRE_TACTICAL_CANDIDATES: bool = true;

// Candidate tags are bit flags. They let Python inspect why a move survived
// pruning without needing to duplicate frontier construction.
pub(crate) const TAG_LEGAL: u32 = 1 << 0;
pub(crate) const TAG_TACTICAL: u32 = 1 << 1;
pub(crate) const TAG_RECENT: u32 = 1 << 2;
pub(crate) const TAG_FRONTIER: u32 = 1 << 3;
pub(crate) const TAG_IMMEDIATE_WIN: u32 = 1 << 4;
pub(crate) const TAG_MUST_BLOCK: u32 = 1 << 5;
