//! Support-set construction — the one geometric law (spec §1.1), serve-time
//! truth. Mirrors python/hexfield/support.py exactly (parity fixtures pin
//! node order, distances, neighbour tables, and counts).
//!
//! Ground truth is the engine: `legal` comes from `write_legal_moves`, never
//! re-derived. One multi-source BFS of depth 9 from the stones yields the
//! support, the halo, and the dist_to_stone feature in one pass. Node order:
//! segments [ legal | stones | halo ], each ascending by packed action id
//! (== ascending signed (q, r)).

use std::collections::{HashMap, VecDeque};
use std::sync::OnceLock;

use hexo_engine::{HexCoord, HexoState as RustHexoState};

use crate::constants::{DIRECTIONS, HALO_DIST, LEGAL_RADIUS};

/// Model-side legal-move radius (NOT the game engine's). HEXFIELD_SUPPORT_RADIUS
/// restricts the support to legal cells within hex-dist <= R of a stone (default
/// LEGAL_RADIUS = 8 == the engine's legality, i.e. UNCHANGED). R<8 shrinks the
/// support -> a cheaper O(support^2) forward, while the engine still allows
/// radius-8; the BIAS table + DIST_SCALE stay at 8 so the network is unchanged.
/// Read once (must match python/hexfield/support.py's HEXFIELD_SUPPORT_RADIUS).
fn support_radius() -> i32 {
    static R: OnceLock<i32> = OnceLock::new();
    *R.get_or_init(|| {
        std::env::var("HEXFIELD_SUPPORT_RADIUS")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .filter(|&r| (1..=HALO_DIST).contains(&r))
            .unwrap_or(LEGAL_RADIUS)
    })
}

pub struct Support {
    /// [legal | stones | halo], each segment ascending by (q, r).
    pub coords: Vec<HexCoord>,
    pub legal_count: usize,
    pub stone_count: usize,
    pub halo_count: usize,
    /// Raw hex distance to the nearest stone (0 everywhere on ply 0).
    pub dist: Vec<i32>,
    /// Row-local neighbour index per DIRECTIONS; -1 when absent.
    pub nbr: Vec<[i32; 6]>,
    pub index: HashMap<(i16, i16), usize>,
}

impl Support {
    pub fn num_nodes(&self) -> usize {
        self.coords.len()
    }

    pub fn row(&self, coord: HexCoord) -> Option<usize> {
        self.index.get(&(coord.q, coord.r)).copied()
    }
}

pub fn build_support(state: &RustHexoState) -> Support {
    if state.placements_made() == 0 {
        // Ply 0: support = origin + its 6 halo neighbours (7 nodes, 1 legal);
        // dist_to_stone := 0 everywhere on this one state.
        let mut halo: Vec<HexCoord> = DIRECTIONS
            .iter()
            .map(|&(dq, dr)| HexCoord { q: dq, r: dr })
            .collect();
        halo.sort_by_key(|c| (c.q, c.r));
        let mut coords = vec![HexCoord { q: 0, r: 0 }];
        coords.extend(halo);
        return finish(coords, 1, 0, 6, vec![0; 7]);
    }

    let radius = support_radius();
    let halo_dist = radius + 1; // == HALO_DIST when radius == LEGAL_RADIUS (default)

    let mut legal: Vec<HexCoord> = Vec::with_capacity(state.legal_move_count());
    state.write_legal_moves(&mut legal);
    legal.sort_by_key(|c| (c.q, c.r));

    // Multi-source BFS depth (radius+1) from the stones.
    let history = state.placement_history();
    let mut stones: Vec<HexCoord> = history.iter().map(|r| r.coord).collect();
    stones.sort_by_key(|c| (c.q, c.r));

    let mut dist_map: HashMap<(i16, i16), i32> = HashMap::with_capacity(stones.len() * 300);
    let mut frontier: VecDeque<HexCoord> = VecDeque::with_capacity(stones.len() * 64);
    for &stone in &stones {
        dist_map.insert((stone.q, stone.r), 0);
        frontier.push_back(stone);
    }
    while let Some(cell) = frontier.pop_front() {
        let d = dist_map[&(cell.q, cell.r)];
        if d == halo_dist {
            continue;
        }
        for &(dq, dr) in &DIRECTIONS {
            let next = (cell.q + dq, cell.r + dr);
            if !dist_map.contains_key(&next) {
                dist_map.insert(next, d + 1);
                frontier.push_back(HexCoord {
                    q: next.0,
                    r: next.1,
                });
            }
        }
    }

    // Model-side radius filter: the engine's legal set is radius-8; keep only the
    // moves within hex-dist <= radius (== keep-all when radius == LEGAL_RADIUS, the
    // BFS only reaches radius+1 so farther legal cells are absent from dist_map).
    legal.retain(|c| dist_map.get(&(c.q, c.r)).is_some_and(|&d| d <= radius));

    let mut halo: Vec<HexCoord> = dist_map
        .iter()
        .filter(|&(_, &d)| d == halo_dist)
        .map(|(&(q, r), _)| HexCoord { q, r })
        .collect();
    halo.sort_by_key(|c| (c.q, c.r));

    let legal_count = legal.len();
    let stone_count = stones.len();
    let halo_count = halo.len();
    let mut coords = legal;
    coords.extend(stones);
    coords.extend(halo);
    let dist: Vec<i32> = coords
        .iter()
        .map(|c| *dist_map.get(&(c.q, c.r)).expect("support cell missing from BFS"))
        .collect();
    finish(coords, legal_count, stone_count, halo_count, dist)
}

fn finish(
    coords: Vec<HexCoord>,
    legal_count: usize,
    stone_count: usize,
    halo_count: usize,
    dist: Vec<i32>,
) -> Support {
    let index: HashMap<(i16, i16), usize> = coords
        .iter()
        .enumerate()
        .map(|(i, c)| ((c.q, c.r), i))
        .collect();
    let nbr: Vec<[i32; 6]> = coords
        .iter()
        .map(|c| {
            let mut row = [-1i32; 6];
            for (k, &(dq, dr)) in DIRECTIONS.iter().enumerate() {
                if let Some(&j) = index.get(&(c.q + dq, c.r + dr)) {
                    row[k] = j as i32;
                }
            }
            row
        })
        .collect();
    Support {
        coords,
        legal_count,
        stone_count,
        halo_count,
        dist,
        nbr,
        index,
    }
}
