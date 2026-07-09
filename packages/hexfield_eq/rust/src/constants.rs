//! hexfield constants. Corresponding Python values are in python/hexfield/constants.py.

pub const LEGAL_RADIUS: i32 = 8;
pub const HALO_DIST: i32 = 9;

/// Fixed direction order D: the rotate60 orbit of (1, 0).
pub const DIRECTIONS: [(i16, i16); 6] = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)];

// Node features (F = NUM_FEATURES = 25). Planes 0-10 are the 11 kept scalars;
// planes 11-22 are the 12 graded per-(cell, axis) window planes (4 quantities x
// 3 axes Q/R/QR, contiguous so a D6 axis-permutation acts on 3-slot blocks);
// planes 23-24 are the 2 scalar fork planes. The binary hot/win planes of the
// hexfield lineage are retired (see docs/PLAN_D6_EQUIVARIANT_REWRITE.md §3).
pub const NUM_FEATURES: usize = 25;
pub const F_OWN_STONE: usize = 0;
pub const F_OPP_STONE: usize = 1;
pub const F_EMPTY: usize = 2;
pub const F_LEGAL: usize = 3;
pub const F_PHASE_SECOND: usize = 4;
pub const F_FIRST_STONE: usize = 5;
pub const F_PLAYER_COLOUR: usize = 6;
pub const F_OWN_RECENCY: usize = 7;
pub const F_OPP_RECENCY: usize = 8;
pub const F_DIST_TO_STONE: usize = 9;
pub const F_OPP_LAST_TURN: usize = 10;
// Graded per-axis window planes. Each quantity spans 3 contiguous slots ordered
// by Axis::ALL == [Q, R, QR], so `BASE + Axis::index()` selects the axis plane.
pub const F_OWN_LINE_Q: usize = 11;
pub const F_OWN_LINE_R: usize = 12;
pub const F_OWN_LINE_QR: usize = 13;
pub const F_OPP_LINE_Q: usize = 14;
pub const F_OPP_LINE_R: usize = 15;
pub const F_OPP_LINE_QR: usize = 16;
pub const F_OWN_LIVE_Q: usize = 17;
pub const F_OWN_LIVE_R: usize = 18;
pub const F_OWN_LIVE_QR: usize = 19;
pub const F_OPP_LIVE_Q: usize = 20;
pub const F_OPP_LIVE_R: usize = 21;
pub const F_OPP_LIVE_QR: usize = 22;
pub const F_OWN_FORK: usize = 23;
pub const F_OPP_FORK: usize = 24;

pub const WINDOW_LEN: usize = 6;

// Side-relative ray lengths (docs/PLAN_REGISTER_LANE_RAY_ATTENTION.md Phase L0):
// per cell u8[RAYLEN_SLOTS], flat index side*6 + axis*2 + dir with side in
// {own=0, opp=1}, axis in Axis::ALL order [Q, R, QR], dir in {+=0, -=1}.
// Values 0..=RAY_REACH; the reach is the window-6 geometry made exact (a
// length-6 window through x extends at most 5 cells along the axis), not a knob.
pub const RAYLEN_SLOTS: usize = 12;
pub const RAY_REACH: usize = WINDOW_LEN - 1;

// Graded-feature normalizers (match python/hexfield_eq/constants.py):
//   line count / 5 (a clean window holds at most 5 own stones in a decision
//   state — 6 is a played win), live window count / 6 (6 windows per cell per
//   axis), fork axis count / 3 (3 axes). A raw per-axis line count >=
//   FORK_LINE_THRESHOLD marks that axis as forking.
pub const LINE_NORM: f32 = 5.0;
pub const LIVE_NORM: f32 = 6.0;
pub const FORK_NORM: f32 = 3.0;
pub const FORK_LINE_THRESHOLD: u32 = 3;

pub const DIST_SCALE: f32 = LEGAL_RADIUS as f32;
