//! Hexformer sparse sample generation and payload assembly.
//!
//! This module owns the model-specific sample facts used by self-play and
//! inference: tactical windows, candidate frontiers, local crops, relation
//! edges, and target tensors. It reconstructs core states from packed histories
//! through `state.rs` and does not rely on engine-side model accelerators.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use hexo_engine::coord::coords_within_radius;
use hexo_engine::{
    hex_distance, pack_coord, unpack_coord, Axis, HexCoord, HexoState as RustHexoState,
    PackedCoord, Player, TurnPhase,
};

use crate::state::{state_from_history_row, states_from_history_rows};

use crate::constants::*;

#[derive(Clone, Debug)]
struct ArchitectureConfig {
    candidate_feature_dim: usize,
    stone_feature_dim: usize,
    window_feature_dim: usize,
    global_feature_dim: usize,
    local_input_channels: usize,
    local_crop_size: usize,
    max_local_windows: usize,
    max_candidates: usize,
    max_stones: usize,
    max_windows: usize,
    max_rel_edges: usize,
    rel_edge_feature_dim: usize,
    lookahead_horizons: Vec<i32>,
}

impl Default for ArchitectureConfig {
    fn default() -> Self {
        Self {
            candidate_feature_dim: DEFAULT_CANDIDATE_FEATURE_DIM,
            stone_feature_dim: DEFAULT_STONE_FEATURE_DIM,
            window_feature_dim: DEFAULT_WINDOW_FEATURE_DIM,
            global_feature_dim: DEFAULT_GLOBAL_FEATURE_DIM,
            local_input_channels: DEFAULT_LOCAL_INPUT_CHANNELS,
            local_crop_size: DEFAULT_LOCAL_CROP_SIZE,
            max_local_windows: DEFAULT_MAX_LOCAL_WINDOWS,
            max_candidates: DEFAULT_MAX_CANDIDATES,
            max_stones: DEFAULT_MAX_STONES,
            max_windows: DEFAULT_MAX_WINDOWS,
            max_rel_edges: DEFAULT_MAX_REL_EDGES,
            rel_edge_feature_dim: DEFAULT_REL_EDGE_FEATURE_DIM,
            lookahead_horizons: DEFAULT_LOOKAHEAD_HORIZONS.to_vec(),
        }
    }
}

impl ArchitectureConfig {
    fn from_py(raw: &Bound<'_, PyAny>) -> PyResult<Self> {
        let default = Self::default();
        Ok(Self {
            candidate_feature_dim: get_usize(raw, "candidate_feature_dim", default.candidate_feature_dim)?,
            stone_feature_dim: get_usize(raw, "stone_feature_dim", default.stone_feature_dim)?,
            window_feature_dim: get_usize(raw, "window_feature_dim", default.window_feature_dim)?,
            global_feature_dim: get_usize(raw, "global_feature_dim", default.global_feature_dim)?,
            local_input_channels: get_usize(raw, "local_input_channels", default.local_input_channels)?,
            local_crop_size: get_usize(raw, "local_crop_size", default.local_crop_size)?,
            max_local_windows: get_usize(raw, "max_local_windows", default.max_local_windows)?,
            max_candidates: get_usize(raw, "max_candidates", default.max_candidates)?,
            max_stones: get_usize(raw, "max_stones", default.max_stones)?,
            max_windows: get_usize(raw, "max_windows", default.max_windows)?,
            max_rel_edges: get_usize(raw, "max_rel_edges", default.max_rel_edges)?,
            rel_edge_feature_dim: get_usize(raw, "rel_edge_feature_dim", default.rel_edge_feature_dim)?,
            lookahead_horizons: get_i32_vec(raw, "lookahead_horizons", default.lookahead_horizons)?,
        })
    }
}

#[derive(Clone, Debug)]
struct CandidateConfig {
    max_candidates: usize,
    tactical_radius: i16,
    recent_radius: i16,
    frontier_radius: i16,
    include_all_legal_below: usize,
    require_tactical_candidates: bool,
}

impl Default for CandidateConfig {
    fn default() -> Self {
        Self {
            max_candidates: DEFAULT_MAX_CANDIDATES,
            tactical_radius: DEFAULT_TACTICAL_RADIUS,
            recent_radius: DEFAULT_RECENT_RADIUS,
            frontier_radius: DEFAULT_FRONTIER_RADIUS,
            include_all_legal_below: DEFAULT_INCLUDE_ALL_LEGAL_BELOW,
            require_tactical_candidates: DEFAULT_REQUIRE_TACTICAL_CANDIDATES,
        }
    }
}

impl CandidateConfig {
    fn from_py(raw: &Bound<'_, PyAny>) -> PyResult<Self> {
        let default = Self::default();
        Ok(Self {
            max_candidates: get_usize(raw, "max_candidates", default.max_candidates)?,
            tactical_radius: get_i16(raw, "tactical_radius", default.tactical_radius)?,
            recent_radius: get_i16(raw, "recent_radius", default.recent_radius)?,
            frontier_radius: get_i16(raw, "frontier_radius", default.frontier_radius)?,
            include_all_legal_below: get_usize(
                raw,
                "include_all_legal_below",
                default.include_all_legal_below,
            )?,
            require_tactical_candidates: get_bool(
                raw,
                "require_tactical_candidates",
                default.require_tactical_candidates,
            )?,
        })
    }
}

#[derive(Clone, Debug, Default)]
struct SparseTargets {
    policy: Vec<(PackedCoord, f32)>,
    opp_policy: Vec<(PackedCoord, f32)>,
    value: Option<f32>,
    distance: Option<f32>,
    lookahead: Vec<(i32, f32)>,
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    action_id: PackedCoord,
    coord: HexCoord,
    tags: u32,
    priority: f32,
}

#[derive(Clone, Debug)]
struct CandidateMetadata {
    legal_count: usize,
    candidate_count: usize,
    truncated: bool,
    immediate_win_count: usize,
    must_block_count: usize,
    tactical_radius: i16,
    frontier_radius: i16,
    require_tactical_candidates: bool,
}

#[derive(Clone, Debug)]
struct CandidateSet {
    candidates: Vec<Candidate>,
    metadata: CandidateMetadata,
}

#[derive(Clone, Debug)]
struct TacticalWindow {
    start: HexCoord,
    axis: Axis,
    cells: [HexCoord; 6],
    counts: [u8; 2],
    empty_cells: Vec<HexCoord>,
    immediate_win_action_ids: Vec<PackedCoord>,
    must_block_action_ids: Vec<PackedCoord>,
    threat_player: Option<Player>,
}

#[derive(Clone, Debug)]
struct TacticalMetadata {
    window_count: usize,
    tactical_action_count: usize,
    immediate_win_count: usize,
    must_block_count: usize,
}

#[derive(Clone, Debug)]
struct TacticalSummary {
    windows: Vec<TacticalWindow>,
    tactical_action_ids: Vec<PackedCoord>,
    immediate_win_action_ids: Vec<PackedCoord>,
    must_block_action_ids: Vec<PackedCoord>,
    metadata: TacticalMetadata,
}

#[derive(Clone, Debug)]
struct TensorF32 {
    shape: Vec<usize>,
    data: Vec<f32>,
}

#[derive(Clone, Debug)]
struct TensorI8 {
    shape: Vec<usize>,
    data: Vec<i8>,
}

#[derive(Clone, Debug)]
struct TensorI64 {
    shape: Vec<usize>,
    data: Vec<i64>,
}

#[derive(Clone, Debug)]
struct SparsePayload {
    candidate_action_ids: Vec<PackedCoord>,
    candidate_features: TensorF32,
    candidate_coords: TensorF32,
    candidate_mask: TensorI8,
    stone_features: TensorF32,
    stone_coords: TensorF32,
    stone_mask: TensorI8,
    window_features: TensorF32,
    window_coords: TensorF32,
    window_mask: TensorI8,
    local_input: TensorF32,
    local_inputs: TensorF32,
    local_window_coords: TensorF32,
    local_window_mask: TensorI8,
    rel_edge_index: TensorI64,
    rel_edge_features: TensorF32,
    rel_edge_mask: TensorI8,
    global_features: TensorF32,
    policy_target: Option<TensorF32>,
    opp_policy_target: Option<TensorF32>,
    wdl_target: Option<TensorF32>,
    distance_target: Option<TensorF32>,
    threat_target: TensorI64,
    relevance_target: TensorF32,
    lookahead_targets: Vec<(i32, TensorF32)>,
    anchor: HexCoord,
    candidate_metadata: CandidateMetadata,
    tactical_metadata: TacticalMetadata,
}

#[pyfunction]
pub fn sparse_input_payload(
    py: Python<'_>,
    history_row: &Bound<'_, PyAny>,
    architecture: &Bound<'_, PyAny>,
    candidates: &Bound<'_, PyAny>,
    policy: &Bound<'_, PyAny>,
    opp_policy: &Bound<'_, PyAny>,
    value: Option<f32>,
    distance: Option<f32>,
    lookahead: &Bound<'_, PyAny>,
    metadata: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let arch = ArchitectureConfig::from_py(architecture)?;
    let candidate_cfg = CandidateConfig::from_py(candidates)?;
    let targets = SparseTargets {
        policy: parse_policy_items(policy)?,
        opp_policy: parse_policy_items(opp_policy)?,
        value,
        distance,
        lookahead: parse_lookahead_items(lookahead)?,
    };
    let state = state_from_history_row(history_row)?;
    let payload = build_sparse_payload(&state, &arch, &candidate_cfg, &targets);
    let payload_obj = sparse_payload_to_py(py, &payload, metadata)?;
    Ok(payload_obj)
}

#[pyfunction]
pub fn sparse_input_payloads(
    py: Python<'_>,
    history_rows: &Bound<'_, PyAny>,
    architecture: &Bound<'_, PyAny>,
    candidates: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let arch = ArchitectureConfig::from_py(architecture)?;
    let candidate_cfg = CandidateConfig::from_py(candidates)?;
    let states = states_from_history_rows(history_rows)?;
    let results = PyList::empty(py);
    let empty = PyDict::new(py);
    let targets = SparseTargets::default();
    for state in &states {
        let payload = build_sparse_payload(state, &arch, &candidate_cfg, &targets);
        results.append(sparse_payload_to_py(py, &payload, empty.as_any())?)?;
    }
    Ok(results.into_any().unbind())
}

#[pyfunction]
pub fn selfplay_sample_payloads(
    py: Python<'_>,
    game_id: String,
    history_rows: &Bound<'_, PyAny>,
    players: &Bound<'_, PyAny>,
    turn_indices: &Bound<'_, PyAny>,
    visit_policies: &Bound<'_, PyAny>,
    root_values: &Bound<'_, PyAny>,
    search_visits: &Bound<'_, PyAny>,
    selected_action_ids: &Bound<'_, PyAny>,
    winner: Option<String>,
    architecture: &Bound<'_, PyAny>,
    candidates: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let arch = ArchitectureConfig::from_py(architecture)?;
    let candidate_cfg = CandidateConfig::from_py(candidates)?;
    let states = states_from_history_rows(history_rows)?;
    let players = parse_string_sequence(players)?;
    let turn_indices = parse_i64_sequence(turn_indices)?;
    let policies = parse_policy_rows(visit_policies)?;
    let root_values = parse_f32_sequence(root_values)?;
    let search_visits = parse_i64_sequence(search_visits)?;
    let selected_action_ids = parse_packed_coord_sequence(selected_action_ids)?;
    let row_count = states.len();
    validate_len("players", players.len(), row_count)?;
    validate_len("turn_indices", turn_indices.len(), row_count)?;
    validate_len("visit_policies", policies.len(), row_count)?;
    validate_len("root_values", root_values.len(), row_count)?;
    validate_len("search_visits", search_visits.len(), row_count)?;
    validate_len("selected_action_ids", selected_action_ids.len(), row_count)?;

    let samples = PyList::empty(py);
    for index in 0..row_count {
        let player = &players[index];
        let value = match winner.as_deref() {
            None => 0.0,
            Some(winning_player) if winning_player == player => 1.0,
            Some(_) => -1.0,
        };
        let mut opp_policy = Vec::new();
        for future in (index + 1)..row_count {
            if players[future] != *player {
                opp_policy = policies[future].clone();
                break;
            }
        }
        let lookahead = arch
            .lookahead_horizons
            .iter()
            .copied()
            .map(|horizon| {
                let target = (index + horizon.max(0) as usize).min(row_count.saturating_sub(1));
                let value = if target <= index || winner.is_none() {
                    0.0
                } else if winner.as_deref() == Some(player.as_str()) {
                    1.0
                } else {
                    -1.0
                };
                (horizon, value)
            })
            .collect();
        let distance = if winner.is_none() {
            0.0
        } else {
            ((row_count - index) as f32 / 128.0).min(1.0)
        };
        let targets = SparseTargets {
            policy: policies[index].clone(),
            opp_policy,
            value: Some(value),
            distance: Some(distance),
            lookahead,
        };
        let base_metadata = PyDict::new(py);
        base_metadata.set_item("game_id", &game_id)?;
        base_metadata.set_item("turn_index", turn_indices[index])?;
        base_metadata.set_item("root_value", root_values[index])?;
        base_metadata.set_item("search_visits", search_visits[index])?;
        base_metadata.set_item("selected_action_id", selected_action_ids[index])?;
        base_metadata.set_item("model_family", "hexformer_ar")?;

        let sparse = build_sparse_payload(&states[index], &arch, &candidate_cfg, &targets);
        let input_payload = sparse_payload_to_py(py, &sparse, base_metadata.as_any())?;
        let input_bound = input_payload.bind(py);
        let metadata = input_bound.get_item("metadata")?;
        let sample = PyDict::new(py);
        sample.set_item("game_id", &game_id)?;
        sample.set_item("turn_index", turn_indices[index])?;
        sample.set_item("input_payload", input_bound)?;
        sample.set_item("metadata", metadata)?;
        samples.append(sample)?;
    }
    Ok(samples.into_any().unbind())
}

pub fn register_pybridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(sparse_input_payload, module)?)?;
    module.add_function(wrap_pyfunction!(sparse_input_payloads, module)?)?;
    module.add_function(wrap_pyfunction!(selfplay_sample_payloads, module)?)?;
    Ok(())
}

fn build_sparse_payload(
    state: &RustHexoState,
    architecture: &ArchitectureConfig,
    candidate_cfg: &CandidateConfig,
    targets: &SparseTargets,
) -> SparsePayload {
    // Main assembly pipeline: tactical scan -> candidate frontier -> token
    // features -> local crops/relation edges -> supervised targets.
    let mut legal_action_ids = Vec::with_capacity(state.legal_move_count());
    state.write_legal_action_ids(&mut legal_action_ids);
    let tactical = build_tactical_summary(state, &legal_action_ids);
    let candidate_set = build_candidate_frontier(state, &legal_action_ids, &tactical, candidate_cfg);
    let opening = matches!(state.phase(), TurnPhase::Opening);
    let anchor = choose_anchor(state.board().occupied_cells(), opening);
    let candidates = candidate_set
        .candidates
        .iter()
        .copied()
        .take(architecture.max_candidates)
        .collect::<Vec<_>>();
    let candidate_ids = candidates
        .iter()
        .map(|candidate| candidate.action_id)
        .collect::<Vec<_>>();

    let mut candidate_features =
        vec![0.0; candidates.len() * architecture.candidate_feature_dim];
    let mut candidate_coords = vec![0.0; candidates.len() * 5];
    let immediate_set = tactical
        .immediate_win_action_ids
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let block_set = tactical
        .must_block_action_ids
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    for (index, candidate) in candidates.iter().enumerate() {
        let rel = relative(candidate.coord, anchor);
        set_row(&mut candidate_coords, 5, index, 0, rel[0]);
        set_row(&mut candidate_coords, 5, index, 1, rel[1]);
        set_row(&mut candidate_coords, 5, index, 2, rel[2]);
        set_row(&mut candidate_coords, 5, index, 3, rel[3]);
        set_row(&mut candidate_coords, 5, index, 4, rel[4]);
        set_row(
            &mut candidate_features,
            architecture.candidate_feature_dim,
            index,
            0,
            1.0,
        );
        set_row(
            &mut candidate_features,
            architecture.candidate_feature_dim,
            index,
            1,
            candidate.tags as f32,
        );
        set_row(
            &mut candidate_features,
            architecture.candidate_feature_dim,
            index,
            2,
            candidate.priority,
        );
        set_row(
            &mut candidate_features,
            architecture.candidate_feature_dim,
            index,
            3,
            immediate_set.contains(&candidate.action_id) as u8 as f32,
        );
        set_row(
            &mut candidate_features,
            architecture.candidate_feature_dim,
            index,
            4,
            block_set.contains(&candidate.action_id) as u8 as f32,
        );
        copy_row_slice(
            &mut candidate_features,
            architecture.candidate_feature_dim,
            index,
            5,
            &rel,
        );
    }

    let current = state.current_player();
    let stones = sorted_stones(state)
        .into_iter()
        .take(architecture.max_stones)
        .collect::<Vec<_>>();
    let mut stone_features = vec![0.0; stones.len() * architecture.stone_feature_dim];
    let mut stone_coords = vec![0.0; stones.len() * 5];
    for (index, (coord, player)) in stones.iter().enumerate() {
        let rel = relative(*coord, anchor);
        copy_row_slice(&mut stone_coords, 5, index, 0, &rel);
        set_row(
            &mut stone_features,
            architecture.stone_feature_dim,
            index,
            0,
            (*player == current) as u8 as f32,
        );
        set_row(
            &mut stone_features,
            architecture.stone_feature_dim,
            index,
            1,
            (*player != current) as u8 as f32,
        );
        copy_row_slice(
            &mut stone_features,
            architecture.stone_feature_dim,
            index,
            2,
            &rel,
        );
    }

    let windows = tactical
        .windows
        .iter()
        .cloned()
        .take(architecture.max_windows)
        .collect::<Vec<_>>();
    let mut window_features = vec![0.0; windows.len() * architecture.window_feature_dim];
    let mut window_coords = vec![0.0; windows.len() * 5];
    for (index, window) in windows.iter().enumerate() {
        let rel = relative(window.start, anchor);
        copy_row_slice(&mut window_coords, 5, index, 0, &rel);
        set_row(
            &mut window_features,
            architecture.window_feature_dim,
            index,
            axis_feature_index(window.axis),
            1.0,
        );
        set_row(
            &mut window_features,
            architecture.window_feature_dim,
            index,
            3,
            window.counts[0] as f32,
        );
        set_row(
            &mut window_features,
            architecture.window_feature_dim,
            index,
            4,
            window.counts[1] as f32,
        );
        set_row(
            &mut window_features,
            architecture.window_feature_dim,
            index,
            5,
            window.empty_cells.len() as f32,
        );
        set_row(
            &mut window_features,
            architecture.window_feature_dim,
            index,
            6,
            (window.threat_player == Some(current)) as u8 as f32,
        );
        set_row(
            &mut window_features,
            architecture.window_feature_dim,
            index,
            7,
            (window.threat_player.is_some() && window.threat_player != Some(current)) as u8 as f32,
        );
        set_row(
            &mut window_features,
            architecture.window_feature_dim,
            index,
            8,
            window.immediate_win_action_ids.len() as f32,
        );
        set_row(
            &mut window_features,
            architecture.window_feature_dim,
            index,
            9,
            window.must_block_action_ids.len() as f32,
        );
        copy_row_slice(
            &mut window_features,
            architecture.window_feature_dim,
            index,
            10,
            &rel,
        );
    }

    let (local_inputs, local_window_coords, local_window_mask) =
        build_local_windows(state, &candidate_ids, &tactical, anchor, architecture);
    let crop_area =
        architecture.local_input_channels * architecture.local_crop_size * architecture.local_crop_size;
    let local_input = if local_inputs.data.is_empty() {
        TensorF32 {
            shape: vec![
                architecture.local_input_channels,
                architecture.local_crop_size,
                architecture.local_crop_size,
            ],
            data: vec![0.0; crop_area],
        }
    } else {
        TensorF32 {
            shape: vec![
                architecture.local_input_channels,
                architecture.local_crop_size,
                architecture.local_crop_size,
            ],
            data: local_inputs.data[0..crop_area].to_vec(),
        }
    };

    let (rel_edge_index, rel_edge_features, rel_edge_mask) =
        build_rel_edges(&candidates, &stones, &windows, local_inputs.shape[0], architecture);
    let global_features =
        build_global_features(state, anchor, candidate_ids.len(), architecture.global_feature_dim);

    SparsePayload {
        candidate_action_ids: candidate_ids.clone(),
        candidate_features: TensorF32 {
            shape: vec![candidates.len(), architecture.candidate_feature_dim],
            data: candidate_features,
        },
        candidate_coords: TensorF32 {
            shape: vec![candidates.len(), 5],
            data: candidate_coords,
        },
        candidate_mask: TensorI8 {
            shape: vec![candidates.len()],
            data: vec![1; candidates.len()],
        },
        stone_features: TensorF32 {
            shape: vec![stones.len(), architecture.stone_feature_dim],
            data: stone_features,
        },
        stone_coords: TensorF32 {
            shape: vec![stones.len(), 5],
            data: stone_coords,
        },
        stone_mask: TensorI8 {
            shape: vec![stones.len()],
            data: vec![1; stones.len()],
        },
        window_features: TensorF32 {
            shape: vec![windows.len(), architecture.window_feature_dim],
            data: window_features,
        },
        window_coords: TensorF32 {
            shape: vec![windows.len(), 5],
            data: window_coords,
        },
        window_mask: TensorI8 {
            shape: vec![windows.len()],
            data: vec![1; windows.len()],
        },
        local_input,
        local_inputs,
        local_window_coords,
        local_window_mask,
        rel_edge_index,
        rel_edge_features,
        rel_edge_mask,
        global_features: TensorF32 {
            shape: vec![architecture.global_feature_dim],
            data: global_features,
        },
        policy_target: policy_vector(&candidate_ids, &targets.policy),
        opp_policy_target: policy_vector(&candidate_ids, &targets.opp_policy),
        wdl_target: wdl_target(targets.value),
        distance_target: targets.distance.map(|value| TensorF32 {
            shape: vec![],
            data: vec![value],
        }),
        threat_target: threat_targets(&candidate_ids, &tactical),
        relevance_target: relevance_targets(&candidate_ids, &tactical),
        lookahead_targets: targets
            .lookahead
            .iter()
            .filter_map(|(horizon, value)| wdl_target(Some(*value)).map(|target| (*horizon, target)))
            .collect(),
        anchor,
        candidate_metadata: candidate_set.metadata,
        tactical_metadata: tactical.metadata,
    }
}

fn build_tactical_summary(state: &RustHexoState, legal_action_ids: &[PackedCoord]) -> TacticalSummary {
    // Windows are the tactical backbone for Hexformer: immediate wins, forced
    // blocks, and nearby candidate expansion all start from this scan.
    let legal_set = legal_action_ids.iter().copied().collect::<HashSet<_>>();
    let current = state.current_player();
    let opponent = current.other();
    let mut windows = Vec::new();
    let mut tactical_ids = HashSet::new();
    let mut win_ids = HashSet::new();
    let mut block_ids = HashSet::new();

    for entry in state.board().windows().entries() {
        let key = entry.key();
        let cells = key.cells();
        let masks = [entry.mask(Player::Player0), entry.mask(Player::Player1)];
        let occupied_mask = masks[0] | masks[1];
        let mut empty_cells = Vec::new();
        let mut empty_ids = Vec::new();
        for (index, cell) in cells.iter().copied().enumerate() {
            if occupied_mask & (1u8 << index) == 0 {
                empty_cells.push(cell);
                let action_id = pack_coord(cell);
                if legal_set.contains(&action_id) {
                    empty_ids.push(action_id);
                }
            }
        }
        let counts = [masks[0].count_ones() as u8, masks[1].count_ones() as u8];
        let threat_player = if counts[0] > 0 && counts[1] == 0 && counts[0] >= 4 {
            Some(Player::Player0)
        } else if counts[1] > 0 && counts[0] == 0 && counts[1] >= 4 {
            Some(Player::Player1)
        } else {
            None
        };
        let mut immediate = Vec::new();
        let mut must_block = Vec::new();
        if let Some(threat) = threat_player {
            tactical_ids.extend(empty_ids.iter().copied());
            if threat == current && counts[current.index()] >= 5 {
                immediate = empty_ids.clone();
                win_ids.extend(immediate.iter().copied());
            }
            if threat == opponent && counts[opponent.index()] >= 5 {
                must_block = empty_ids.clone();
                block_ids.extend(must_block.iter().copied());
            }
        }
        windows.push(TacticalWindow {
            start: key.start,
            axis: key.axis,
            cells,
            counts,
            empty_cells,
            immediate_win_action_ids: immediate,
            must_block_action_ids: must_block,
            threat_player,
        });
    }
    windows.sort_by(|left, right| compare_window(left, right));
    let mut tactical_action_ids = tactical_ids.into_iter().collect::<Vec<_>>();
    tactical_action_ids.sort_unstable();
    let mut immediate_win_action_ids = win_ids.into_iter().collect::<Vec<_>>();
    immediate_win_action_ids.sort_unstable();
    let mut must_block_action_ids = block_ids.into_iter().collect::<Vec<_>>();
    must_block_action_ids.sort_unstable();
    TacticalSummary {
        metadata: TacticalMetadata {
            window_count: windows.len(),
            tactical_action_count: tactical_action_ids.len(),
            immediate_win_count: immediate_win_action_ids.len(),
            must_block_count: must_block_action_ids.len(),
        },
        windows,
        tactical_action_ids,
        immediate_win_action_ids,
        must_block_action_ids,
    }
}

fn build_candidate_frontier(
    state: &RustHexoState,
    legal_action_ids: &[PackedCoord],
    tactical: &TacticalSummary,
    cfg: &CandidateConfig,
) -> CandidateSet {
    // Frontier construction is deliberately deterministic. Priority decides the
    // first ordering key, while coordinate/action ordering gives stable ties.
    let legal_set = legal_action_ids.iter().copied().collect::<HashSet<_>>();
    let mut by_id = HashMap::<PackedCoord, Candidate>::new();
    for action_id in &tactical.immediate_win_action_ids {
        add_candidate(
            &mut by_id,
            &legal_set,
            *action_id,
            TAG_TACTICAL | TAG_IMMEDIATE_WIN,
            100.0,
        );
    }
    for action_id in &tactical.must_block_action_ids {
        add_candidate(
            &mut by_id,
            &legal_set,
            *action_id,
            TAG_TACTICAL | TAG_MUST_BLOCK,
            90.0,
        );
    }
    for action_id in &tactical.tactical_action_ids {
        add_candidate(&mut by_id, &legal_set, *action_id, TAG_TACTICAL, 75.0);
    }
    for action_id in tactical
        .immediate_win_action_ids
        .iter()
        .chain(tactical.must_block_action_ids.iter())
        .chain(tactical.tactical_action_ids.iter())
    {
        let center = unpack_coord(*action_id);
        for coord in coords_within_radius(center, cfg.tactical_radius.max(0)) {
            let distance = hex_distance(coord, center);
            add_candidate(
                &mut by_id,
                &legal_set,
                pack_coord(coord),
                TAG_TACTICAL,
                70.0 - distance as f32,
            );
        }
    }
    let history = state.placement_history();
    let start = history.len().saturating_sub(8);
    for record in &history[start..] {
        for coord in coords_within_radius(record.coord, cfg.recent_radius.max(0)) {
            add_candidate(
                &mut by_id,
                &legal_set,
                pack_coord(coord),
                TAG_RECENT,
                50.0 - hex_distance(coord, record.coord) as f32,
            );
        }
    }
    let occupied = state.board().occupied_cells();
    let start = occupied.len().saturating_sub(64);
    for center in &occupied[start..] {
        for coord in coords_within_radius(*center, cfg.frontier_radius.max(0)) {
            add_candidate(
                &mut by_id,
                &legal_set,
                pack_coord(coord),
                TAG_FRONTIER,
                30.0 - hex_distance(coord, *center) as f32,
            );
        }
    }
    if legal_action_ids.len() <= cfg.include_all_legal_below {
        for action_id in legal_action_ids {
            add_candidate(&mut by_id, &legal_set, *action_id, TAG_FRONTIER, 10.0);
        }
    }
    if cfg.require_tactical_candidates
        && (!tactical.immediate_win_action_ids.is_empty() || !tactical.must_block_action_ids.is_empty())
    {
        for action_id in tactical
            .immediate_win_action_ids
            .iter()
            .chain(tactical.must_block_action_ids.iter())
        {
            if !by_id.contains_key(action_id) {
                add_candidate(&mut by_id, &legal_set, *action_id, TAG_TACTICAL, 95.0);
            }
        }
    }
    if by_id.is_empty() {
        for action_id in legal_action_ids.iter().take(cfg.max_candidates) {
            add_candidate(&mut by_id, &legal_set, *action_id, TAG_FRONTIER, 1.0);
        }
    }
    let mut ordered = by_id.into_values().collect::<Vec<_>>();
    ordered.sort_by(compare_candidate);
    let truncated = ordered.len() > cfg.max_candidates;
    let mut limited = ordered
        .iter()
        .copied()
        .take(cfg.max_candidates)
        .collect::<Vec<_>>();
    if limited.is_empty() && !legal_action_ids.is_empty() {
        let action_id = legal_action_ids[0];
        limited.push(Candidate {
            action_id,
            coord: unpack_coord(action_id),
            tags: TAG_LEGAL,
            priority: 0.0,
        });
    }
    CandidateSet {
        metadata: CandidateMetadata {
            legal_count: legal_action_ids.len(),
            candidate_count: limited.len(),
            truncated,
            immediate_win_count: tactical.immediate_win_action_ids.len(),
            must_block_count: tactical.must_block_action_ids.len(),
            tactical_radius: cfg.tactical_radius,
            frontier_radius: cfg.frontier_radius,
            require_tactical_candidates: cfg.require_tactical_candidates,
        },
        candidates: limited,
    }
}

fn add_candidate(
    by_id: &mut HashMap<PackedCoord, Candidate>,
    legal_set: &HashSet<PackedCoord>,
    action_id: PackedCoord,
    tags: u32,
    priority: f32,
) {
    if !legal_set.contains(&action_id) {
        return;
    }
    let coord = unpack_coord(action_id);
    by_id
        .entry(action_id)
        .and_modify(|candidate| {
            candidate.tags |= tags;
            candidate.priority = candidate.priority.max(priority);
        })
        .or_insert(Candidate {
            action_id,
            coord,
            tags: TAG_LEGAL | tags,
            priority,
        });
}

fn build_local_windows(
    state: &RustHexoState,
    candidate_ids: &[PackedCoord],
    tactical: &TacticalSummary,
    anchor: HexCoord,
    architecture: &ArchitectureConfig,
) -> (TensorF32, TensorF32, TensorI8) {
    // Local crops give the transformer dense views around the global anchor,
    // the latest move, and the first active tactical window.
    let mut anchors = vec![anchor];
    if let Some(record) = state.placement_history().last() {
        anchors.push(record.coord);
    }
    if let Some(window) = tactical.windows.iter().find(|window| {
        !window.immediate_win_action_ids.is_empty()
            || !window.must_block_action_ids.is_empty()
            || window.threat_player.is_some()
    }) {
        anchors.push(window.start);
    }
    let mut unique = Vec::<HexCoord>::new();
    for item in anchors {
        if !unique.contains(&item) {
            unique.push(item);
        }
        if unique.len() >= architecture.max_local_windows.max(1) {
            break;
        }
    }
    if unique.is_empty() {
        unique.push(anchor);
    }

    let crop_area =
        architecture.local_input_channels * architecture.local_crop_size * architecture.local_crop_size;
    let mut local_inputs = vec![0.0; unique.len() * crop_area];
    let mut local_window_coords = vec![0.0; unique.len() * 5];
    for (index, local_anchor) in unique.iter().copied().enumerate() {
        fill_local_crop(
            &mut local_inputs,
            index,
            state,
            candidate_ids,
            local_anchor,
            architecture,
        );
        let rel = relative(local_anchor, anchor);
        copy_row_slice(&mut local_window_coords, 5, index, 0, &rel);
    }
    (
        TensorF32 {
            shape: vec![
                unique.len(),
                architecture.local_input_channels,
                architecture.local_crop_size,
                architecture.local_crop_size,
            ],
            data: local_inputs,
        },
        TensorF32 {
            shape: vec![unique.len(), 5],
            data: local_window_coords,
        },
        TensorI8 {
            shape: vec![unique.len()],
            data: vec![1; unique.len()],
        },
    )
}

fn fill_local_crop(
    local_inputs: &mut [f32],
    local_index: usize,
    state: &RustHexoState,
    candidate_ids: &[PackedCoord],
    anchor: HexCoord,
    architecture: &ArchitectureConfig,
) {
    let size = architecture.local_crop_size;
    let channels = architecture.local_input_channels;
    let half = (size / 2) as i32;
    if channels > 2 {
        for row in 0..size {
            for col in 0..size {
                set_local(local_inputs, local_index, 2, row, col, architecture, 1.0);
            }
        }
    }
    for (coord, player) in sorted_stones(state) {
        let q = coord.q as i32 - anchor.q as i32 + half;
        let r = coord.r as i32 - anchor.r as i32 + half;
        if q >= 0 && r >= 0 && q < size as i32 && r < size as i32 {
            let plane = if player == state.current_player() { 0 } else { 1 };
            set_local(
                local_inputs,
                local_index,
                plane,
                r as usize,
                q as usize,
                architecture,
                1.0,
            );
            set_local(
                local_inputs,
                local_index,
                2,
                r as usize,
                q as usize,
                architecture,
                0.0,
            );
        }
    }
    for action_id in candidate_ids {
        let coord = unpack_coord(*action_id);
        let q = coord.q as i32 - anchor.q as i32 + half;
        let r = coord.r as i32 - anchor.r as i32 + half;
        if q >= 0 && r >= 0 && q < size as i32 && r < size as i32 {
            set_local(
                local_inputs,
                local_index,
                3,
                r as usize,
                q as usize,
                architecture,
                1.0,
            );
        }
    }
    if matches!(state.phase(), TurnPhase::SecondStone { .. }) && channels > 4 {
        for row in 0..size {
            for col in 0..size {
                set_local(local_inputs, local_index, 4, row, col, architecture, 1.0);
            }
        }
    }
}

fn build_rel_edges(
    candidates: &[Candidate],
    stones: &[(HexCoord, Player)],
    windows: &[TacticalWindow],
    local_count: usize,
    architecture: &ArchitectureConfig,
) -> (TensorI64, TensorF32, TensorI8) {
    // Relation edges connect sparse tokens that are close enough to matter
    // tactically. The model consumes these as fixed-width typed edge features.
    let candidate_offset = 1 + local_count;
    let stone_offset = candidate_offset + candidates.len();
    let window_offset = stone_offset + stones.len();
    let mut edge_index = Vec::<i64>::new();
    let mut edge_features = Vec::<f32>::new();

    let mut add = |src: usize, dst: usize, left: HexCoord, right: HexCoord, relation_type: usize| {
        if edge_index.len() / 2 >= architecture.max_rel_edges {
            return;
        }
        let dq = right.q as i32 - left.q as i32;
        let dr = right.r as i32 - left.r as i32;
        let ds = -dq - dr;
        let dist = dq.abs().max(dr.abs()).max(ds.abs());
        let base = [
            dq as f32,
            dr as f32,
            ds as f32,
            dist as f32,
            (dq == 0) as u8 as f32,
            (dr == 0) as u8 as f32,
            (ds == 0) as u8 as f32,
            (relation_type == 0) as u8 as f32,
            (relation_type == 1) as u8 as f32,
            (relation_type == 2) as u8 as f32,
            (relation_type == 3) as u8 as f32,
            1.0,
        ];
        edge_index.push(src as i64);
        edge_index.push(dst as i64);
        for index in 0..architecture.rel_edge_feature_dim {
            edge_features.push(base.get(index).copied().unwrap_or(0.0));
        }
    };

    for (ci, candidate) in candidates.iter().enumerate() {
        let ctoken = candidate_offset + ci;
        for (si, (stone_coord, _player)) in stones.iter().enumerate() {
            if hex_distance(candidate.coord, *stone_coord) <= 4 {
                let stoken = stone_offset + si;
                add(ctoken, stoken, candidate.coord, *stone_coord, 0);
                add(stoken, ctoken, *stone_coord, candidate.coord, 0);
            }
        }
        for (wi, window) in windows.iter().enumerate() {
            if window.empty_cells.contains(&candidate.coord)
                || window.cells.contains(&candidate.coord)
                || hex_distance(candidate.coord, window.start) <= 3
            {
                let wtoken = window_offset + wi;
                add(ctoken, wtoken, candidate.coord, window.start, 1);
                add(wtoken, ctoken, window.start, candidate.coord, 1);
            }
        }
    }
    for (wi, window) in windows.iter().enumerate() {
        let wtoken = window_offset + wi;
        for (si, (stone_coord, _player)) in stones.iter().enumerate() {
            if hex_distance(window.start, *stone_coord) <= 6 {
                let stoken = stone_offset + si;
                add(wtoken, stoken, window.start, *stone_coord, 2);
                add(stoken, wtoken, *stone_coord, window.start, 2);
            }
        }
    }
    for left in 0..candidates.len() {
        for right in (left + 1)..candidates.len().min(left + 32) {
            if hex_distance(candidates[left].coord, candidates[right].coord) <= 2 {
                add(
                    candidate_offset + left,
                    candidate_offset + right,
                    candidates[left].coord,
                    candidates[right].coord,
                    3,
                );
                add(
                    candidate_offset + right,
                    candidate_offset + left,
                    candidates[right].coord,
                    candidates[left].coord,
                    3,
                );
            }
        }
    }
    let edges = edge_index.len() / 2;
    (
        TensorI64 {
            shape: vec![edges, 2],
            data: edge_index,
        },
        TensorF32 {
            shape: vec![edges, architecture.rel_edge_feature_dim],
            data: edge_features,
        },
        TensorI8 {
            shape: vec![edges],
            data: vec![1; edges],
        },
    )
}

fn build_global_features(
    state: &RustHexoState,
    anchor: HexCoord,
    candidate_count: usize,
    feature_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; feature_dim];
    match state.phase() {
        TurnPhase::Opening => set_1d(&mut out, 0, 1.0),
        TurnPhase::FirstStone => set_1d(&mut out, 1, 1.0),
        TurnPhase::SecondStone { .. } => set_1d(&mut out, 2, 1.0),
    }
    set_1d(
        &mut out,
        3,
        (state.current_player() == Player::Player0) as u8 as f32,
    );
    set_1d(&mut out, 4, state.placements_made() as f32);
    set_1d(&mut out, 5, candidate_count as f32);
    set_1d(&mut out, 6, state.board().occupied_cells().len() as f32);
    let max_distance = state
        .board()
        .occupied_cells()
        .iter()
        .map(|coord| relative(*coord, anchor)[3])
        .fold(0.0_f32, f32::max);
    set_1d(&mut out, 7, max_distance);
    if let TurnPhase::SecondStone { first } = state.phase() {
        let rel = relative(first, anchor);
        for (offset, value) in rel.into_iter().enumerate() {
            set_1d(&mut out, 8 + offset, value);
        }
    }
    out
}

fn policy_vector(action_ids: &[PackedCoord], weights: &[(PackedCoord, f32)]) -> Option<TensorF32> {
    if weights.is_empty() {
        return None;
    }
    let weight_map = weights.iter().copied().collect::<HashMap<_, _>>();
    let mut data = action_ids
        .iter()
        .map(|action_id| weight_map.get(action_id).copied().unwrap_or(0.0).max(0.0))
        .collect::<Vec<_>>();
    let total: f32 = data.iter().sum();
    if total <= 0.0 {
        return None;
    }
    for value in &mut data {
        *value /= total;
    }
    Some(TensorF32 {
        shape: vec![action_ids.len()],
        data,
    })
}

fn wdl_target(value: Option<f32>) -> Option<TensorF32> {
    let value = value?;
    let v = value.clamp(-1.0, 1.0);
    let win = v.max(0.0);
    let loss = (-v).max(0.0);
    let draw = (1.0 - win - loss).max(0.0);
    let total = (loss + draw + win).max(1.0e-8);
    Some(TensorF32 {
        shape: vec![3],
        data: vec![loss / total, draw / total, win / total],
    })
}

fn threat_targets(action_ids: &[PackedCoord], tactical: &TacticalSummary) -> TensorI64 {
    let win_ids = tactical
        .immediate_win_action_ids
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let block_ids = tactical
        .must_block_action_ids
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let tactical_ids = tactical
        .tactical_action_ids
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let data = action_ids
        .iter()
        .map(|action_id| {
            if win_ids.contains(action_id) {
                1
            } else if block_ids.contains(action_id) {
                2
            } else if tactical_ids.contains(action_id) {
                3
            } else {
                0
            }
        })
        .collect::<Vec<_>>();
    TensorI64 {
        shape: vec![action_ids.len()],
        data,
    }
}

fn relevance_targets(action_ids: &[PackedCoord], tactical: &TacticalSummary) -> TensorF32 {
    let tactical_ids = tactical
        .tactical_action_ids
        .iter()
        .chain(tactical.immediate_win_action_ids.iter())
        .chain(tactical.must_block_action_ids.iter())
        .copied()
        .collect::<HashSet<_>>();
    TensorF32 {
        shape: vec![action_ids.len()],
        data: action_ids
            .iter()
            .map(|action_id| tactical_ids.contains(action_id) as u8 as f32)
            .collect(),
    }
}

fn choose_anchor(stones: &[HexCoord], opening: bool) -> HexCoord {
    if opening || stones.is_empty() {
        return HexCoord::ZERO;
    }
    let q_sum: i64 = stones.iter().map(|coord| coord.q as i64).sum();
    let r_sum: i64 = stones.iter().map(|coord| coord.r as i64).sum();
    HexCoord {
        q: round_half_away(q_sum as f32 / stones.len() as f32) as i16,
        r: round_half_away(r_sum as f32 / stones.len() as f32) as i16,
    }
}

fn relative(coord: HexCoord, anchor: HexCoord) -> [f32; 5] {
    let dq = coord.q as i32 - anchor.q as i32;
    let dr = coord.r as i32 - anchor.r as i32;
    let ds = -dq - dr;
    let distance = dq.abs().max(dr.abs()).max(ds.abs());
    [
        dq as f32,
        dr as f32,
        ds as f32,
        distance as f32,
        distance as f32,
    ]
}

fn sorted_stones(state: &RustHexoState) -> Vec<(HexCoord, Player)> {
    let mut coords = state.board().occupied_cells().to_vec();
    coords.sort_by_key(|coord| (coord.q, coord.r));
    coords
        .into_iter()
        .filter_map(|coord| state.board().get(coord).map(|player| (coord, player)))
        .collect()
}

fn sparse_payload_to_py(
    py: Python<'_>,
    payload: &SparsePayload,
    base_metadata: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    // Keep the Python payload schema tensor-like but framework-neutral:
    // `{shape, dtype, data}` lets the Python side convert to torch tensors.
    let dict = PyDict::new(py);
    dict.set_item("candidate_action_ids", payload.candidate_action_ids.clone())?;
    dict.set_item(
        "candidate_features",
        tensor_payload_f32(py, &payload.candidate_features)?,
    )?;
    dict.set_item(
        "candidate_coords",
        tensor_payload_f32(py, &payload.candidate_coords)?,
    )?;
    dict.set_item("candidate_mask", tensor_payload_i8(py, &payload.candidate_mask)?)?;
    dict.set_item(
        "stone_features",
        tensor_payload_f32(py, &payload.stone_features)?,
    )?;
    dict.set_item("stone_coords", tensor_payload_f32(py, &payload.stone_coords)?)?;
    dict.set_item("stone_mask", tensor_payload_i8(py, &payload.stone_mask)?)?;
    dict.set_item(
        "window_features",
        tensor_payload_f32(py, &payload.window_features)?,
    )?;
    dict.set_item(
        "window_coords",
        tensor_payload_f32(py, &payload.window_coords)?,
    )?;
    dict.set_item("window_mask", tensor_payload_i8(py, &payload.window_mask)?)?;
    dict.set_item("local_input", tensor_payload_f32(py, &payload.local_input)?)?;
    dict.set_item("local_inputs", tensor_payload_f32(py, &payload.local_inputs)?)?;
    dict.set_item(
        "local_window_coords",
        tensor_payload_f32(py, &payload.local_window_coords)?,
    )?;
    dict.set_item(
        "local_window_mask",
        tensor_payload_i8(py, &payload.local_window_mask)?,
    )?;
    dict.set_item("rel_edge_index", tensor_payload_i64(py, &payload.rel_edge_index)?)?;
    dict.set_item(
        "rel_edge_features",
        tensor_payload_f32(py, &payload.rel_edge_features)?,
    )?;
    dict.set_item("rel_edge_mask", tensor_payload_i8(py, &payload.rel_edge_mask)?)?;
    dict.set_item(
        "global_features",
        tensor_payload_f32(py, &payload.global_features)?,
    )?;
    if let Some(target) = &payload.policy_target {
        dict.set_item("policy_target", tensor_payload_f32(py, target)?)?;
    }
    if let Some(target) = &payload.opp_policy_target {
        dict.set_item("opp_policy_target", tensor_payload_f32(py, target)?)?;
    }
    if let Some(target) = &payload.wdl_target {
        dict.set_item("wdl_target", tensor_payload_f32(py, target)?)?;
    }
    if let Some(target) = &payload.distance_target {
        dict.set_item("distance_target", tensor_payload_f32(py, target)?)?;
    }
    dict.set_item("threat_target", tensor_payload_i64(py, &payload.threat_target)?)?;
    dict.set_item(
        "relevance_target",
        tensor_payload_f32(py, &payload.relevance_target)?,
    )?;
    if !payload.lookahead_targets.is_empty() {
        let lookahead = PyDict::new(py);
        for (horizon, target) in &payload.lookahead_targets {
            lookahead.set_item(horizon.to_string(), tensor_payload_f32(py, target)?)?;
        }
        dict.set_item("lookahead_targets", lookahead)?;
    }
    dict.set_item("metadata", metadata_to_py(py, payload, base_metadata)?)?;
    Ok(dict.into_any().unbind())
}

fn tensor_payload_f32<'py>(py: Python<'py>, tensor: &TensorF32) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("shape", tensor.shape.clone())?;
    dict.set_item("dtype", "float32")?;
    dict.set_item("data", PyList::new(py, tensor.data.iter().copied())?)?;
    Ok(dict)
}

fn tensor_payload_i8<'py>(py: Python<'py>, tensor: &TensorI8) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("shape", tensor.shape.clone())?;
    dict.set_item("dtype", "int8")?;
    dict.set_item("data", PyList::new(py, tensor.data.iter().copied())?)?;
    Ok(dict)
}

fn tensor_payload_i64<'py>(py: Python<'py>, tensor: &TensorI64) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("shape", tensor.shape.clone())?;
    dict.set_item("dtype", "int64")?;
    dict.set_item("data", PyList::new(py, tensor.data.iter().copied())?)?;
    Ok(dict)
}

fn metadata_to_py<'py>(
    py: Python<'py>,
    payload: &SparsePayload,
    base_metadata: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let metadata = copy_py_mapping(py, base_metadata)?;
    metadata.set_item("anchor", (payload.anchor.q, payload.anchor.r))?;
    let candidate = PyDict::new(py);
    candidate.set_item("legal_count", payload.candidate_metadata.legal_count)?;
    candidate.set_item("candidate_count", payload.candidate_metadata.candidate_count)?;
    candidate.set_item("truncated", payload.candidate_metadata.truncated)?;
    candidate.set_item(
        "immediate_win_count",
        payload.candidate_metadata.immediate_win_count,
    )?;
    candidate.set_item("must_block_count", payload.candidate_metadata.must_block_count)?;
    candidate.set_item("tactical_radius", payload.candidate_metadata.tactical_radius)?;
    candidate.set_item("frontier_radius", payload.candidate_metadata.frontier_radius)?;
    candidate.set_item(
        "require_tactical_candidates",
        payload.candidate_metadata.require_tactical_candidates,
    )?;
    metadata.set_item("candidate", candidate)?;

    let tactical = PyDict::new(py);
    tactical.set_item("window_count", payload.tactical_metadata.window_count)?;
    tactical.set_item(
        "tactical_action_count",
        payload.tactical_metadata.tactical_action_count,
    )?;
    tactical.set_item(
        "immediate_win_count",
        payload.tactical_metadata.immediate_win_count,
    )?;
    tactical.set_item("must_block_count", payload.tactical_metadata.must_block_count)?;
    metadata.set_item("tactical", tactical)?;
    Ok(metadata)
}


fn parse_policy_rows(rows: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<(PackedCoord, f32)>>> {
    // Boundary parsing stays at the edge of the module; downstream code works
    // with plain Rust vectors and validated lengths.
    let mut out = Vec::new();
    for row in rows.try_iter()? {
        out.push(parse_policy_items(&row?)?);
    }
    Ok(out)
}

fn parse_policy_items(raw: &Bound<'_, PyAny>) -> PyResult<Vec<(PackedCoord, f32)>> {
    if raw.is_none() {
        return Ok(Vec::new());
    }
    let iterable = match raw.call_method0("items") {
        Ok(items) => items,
        Err(_) => raw.clone(),
    };
    let mut out = Vec::new();
    for item in iterable.try_iter()? {
        let (action_id, weight) = item?.extract::<(PackedCoord, f32)>()?;
        out.push((action_id, weight));
    }
    Ok(out)
}

fn parse_lookahead_items(raw: &Bound<'_, PyAny>) -> PyResult<Vec<(i32, f32)>> {
    if raw.is_none() {
        return Ok(Vec::new());
    }
    let iterable = match raw.call_method0("items") {
        Ok(items) => items,
        Err(_) => raw.clone(),
    };
    let mut out = Vec::new();
    for item in iterable.try_iter()? {
        let (horizon, value) = item?.extract::<(i32, f32)>()?;
        out.push((horizon, value));
    }
    Ok(out)
}

fn parse_string_sequence(raw: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let mut out = Vec::new();
    for item in raw.try_iter()? {
        out.push(item?.extract::<String>()?);
    }
    Ok(out)
}

fn parse_f32_sequence(raw: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    let mut out = Vec::new();
    for item in raw.try_iter()? {
        out.push(item?.extract::<f32>()?);
    }
    Ok(out)
}

fn parse_i64_sequence(raw: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    let mut out = Vec::new();
    for item in raw.try_iter()? {
        out.push(item?.extract::<i64>()?);
    }
    Ok(out)
}

fn parse_packed_coord_sequence(raw: &Bound<'_, PyAny>) -> PyResult<Vec<PackedCoord>> {
    let mut out = Vec::new();
    for item in raw.try_iter()? {
        out.push(item?.extract::<PackedCoord>()?);
    }
    Ok(out)
}

fn validate_len(name: &str, actual: usize, expected: usize) -> PyResult<()> {
    if actual != expected {
        return Err(PyValueError::new_err(format!(
            "{name} length {actual} does not match history_rows length {expected}"
        )));
    }
    Ok(())
}

fn copy_py_mapping<'py>(
    py: Python<'py>,
    raw: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    if raw.is_none() {
        return Ok(out);
    }
    let Ok(items) = raw.call_method0("items") else {
        return Ok(out);
    };
    for item in items.try_iter()? {
        let item = item?;
        let key = item.get_item(0)?;
        let value = item.get_item(1)?;
        out.set_item(key, value)?;
    }
    Ok(out)
}

fn py_get<'py>(raw: &Bound<'py, PyAny>, key: &str) -> Option<Bound<'py, PyAny>> {
    if raw.is_none() {
        return None;
    }
    raw.get_item(key).ok().or_else(|| raw.getattr(key).ok())
}

fn get_usize(raw: &Bound<'_, PyAny>, key: &str, default: usize) -> PyResult<usize> {
    match py_get(raw, key) {
        Some(value) if !value.is_none() => value.extract::<usize>(),
        _ => Ok(default),
    }
}

fn get_i16(raw: &Bound<'_, PyAny>, key: &str, default: i16) -> PyResult<i16> {
    match py_get(raw, key) {
        Some(value) if !value.is_none() => value.extract::<i16>(),
        _ => Ok(default),
    }
}

fn get_bool(raw: &Bound<'_, PyAny>, key: &str, default: bool) -> PyResult<bool> {
    match py_get(raw, key) {
        Some(value) if !value.is_none() => value.extract::<bool>(),
        _ => Ok(default),
    }
}

fn get_i32_vec(raw: &Bound<'_, PyAny>, key: &str, default: Vec<i32>) -> PyResult<Vec<i32>> {
    let Some(value) = py_get(raw, key) else {
        return Ok(default);
    };
    if value.is_none() {
        return Ok(default);
    }
    let mut out = Vec::new();
    for item in value.try_iter()? {
        out.push(item?.extract::<i32>()?);
    }
    Ok(out)
}

fn compare_candidate(left: &Candidate, right: &Candidate) -> Ordering {
    right
        .priority
        .partial_cmp(&left.priority)
        .unwrap_or(Ordering::Equal)
        .then_with(|| left.coord.q.cmp(&right.coord.q))
        .then_with(|| left.coord.r.cmp(&right.coord.r))
        .then_with(|| left.action_id.cmp(&right.action_id))
}

fn compare_window(left: &TacticalWindow, right: &TacticalWindow) -> Ordering {
    left.start
        .q
        .cmp(&right.start.q)
        .then_with(|| left.start.r.cmp(&right.start.r))
        .then_with(|| axis_label(left.axis).cmp(axis_label(right.axis)))
}

fn axis_label(axis: Axis) -> &'static str {
    match axis {
        Axis::Q => "Q",
        Axis::R => "R",
        Axis::QR => "QR",
    }
}

fn axis_feature_index(axis: Axis) -> usize {
    match axis {
        Axis::Q => 0,
        Axis::R => 1,
        Axis::QR => 2,
    }
}

fn round_half_away(value: f32) -> i32 {
    if value >= 0.0 {
        (value + 0.5) as i32
    } else {
        (value - 0.5) as i32
    }
}

fn set_1d(data: &mut [f32], index: usize, value: f32) {
    if let Some(slot) = data.get_mut(index) {
        *slot = value;
    }
}

fn set_row(data: &mut [f32], dim: usize, row: usize, col: usize, value: f32) {
    if col >= dim {
        return;
    }
    if let Some(slot) = data.get_mut(row * dim + col) {
        *slot = value;
    }
}

fn copy_row_slice(data: &mut [f32], dim: usize, row: usize, start: usize, values: &[f32]) {
    for (offset, value) in values.iter().copied().enumerate() {
        set_row(data, dim, row, start + offset, value);
    }
}

fn set_local(
    data: &mut [f32],
    local_index: usize,
    plane: usize,
    row: usize,
    col: usize,
    architecture: &ArchitectureConfig,
    value: f32,
) {
    if plane >= architecture.local_input_channels
        || row >= architecture.local_crop_size
        || col >= architecture.local_crop_size
    {
        return;
    }
    let size = architecture.local_crop_size;
    let channels = architecture.local_input_channels;
    let index = (((local_index * channels + plane) * size + row) * size) + col;
    if let Some(slot) = data.get_mut(index) {
        *slot = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::{apply_placement, Placement};

    fn sample_state() -> RustHexoState {
        let mut state = RustHexoState::new();
        for coord in [
            HexCoord::ZERO,
            HexCoord::new(1, 0),
            HexCoord::new(2, 0),
            HexCoord::new(0, 1),
        ] {
            apply_placement(&mut state, Placement { coord }).unwrap();
        }
        state
    }

    #[test]
    fn sparse_payload_has_existing_trainer_shapes() {
        let state = sample_state();
        let arch = ArchitectureConfig {
            local_crop_size: 9,
            max_candidates: 16,
            max_stones: 8,
            max_windows: 32,
            max_rel_edges: 128,
            lookahead_horizons: vec![1],
            ..ArchitectureConfig::default()
        };
        let cfg = CandidateConfig {
            max_candidates: 16,
            ..CandidateConfig::default()
        };
        let mut legal = Vec::new();
        state.write_legal_action_ids(&mut legal);
        let targets = SparseTargets {
            policy: vec![(legal[0], 3.0)],
            value: Some(1.0),
            distance: Some(0.25),
            lookahead: vec![(1, 0.0)],
            ..SparseTargets::default()
        };

        let payload = build_sparse_payload(&state, &arch, &cfg, &targets);

        assert!(!payload.candidate_action_ids.is_empty());
        assert_eq!(
            payload.candidate_features.shape,
            vec![payload.candidate_action_ids.len(), arch.candidate_feature_dim]
        );
        assert_eq!(payload.local_input.shape, vec![13, 9, 9]);
        assert_eq!(payload.rel_edge_index.shape[1], 2);
        let policy = payload.policy_target.as_ref().unwrap();
        assert!((policy.data.iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
        assert_eq!(payload.wdl_target.unwrap().data, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn packed_history_reconstruction_matches_engine_state() {
        let state = sample_state();
        let history = state
            .placement_history()
            .iter()
            .map(|record| pack_coord(record.coord))
            .collect::<Vec<_>>();
        let mut decoded = RustHexoState::new();
        for action_id in history {
            apply_placement(
                &mut decoded,
                Placement {
                    coord: unpack_coord(action_id),
                },
            )
            .unwrap();
        }

        assert_eq!(decoded.current_player(), state.current_player());
        assert_eq!(decoded.phase(), state.phase());
        assert_eq!(decoded.placements_made(), state.placements_made());
        assert_eq!(decoded.board().occupied_cells(), state.board().occupied_cells());
    }
}
