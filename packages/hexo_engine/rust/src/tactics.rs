//! Incremental six-cell window tracking.
//!
//! A placement touches exactly 18 length-6 windows: 3 axes times 6 possible
//! offsets inside the window. The store keeps those windows incrementally so
//! wins and threats can be read from compact six-bit masks.

use super::coord::{hex_distance, HexCoord};
use super::board::Board;
use super::state::Player;
use ahash::AHashMap;
use serde::{Deserialize, Serialize};
use std::{fmt, ops::Deref};

/// Number of cells in a win/threat window.
pub const WINDOW_LEN: i16 = 6;

/// Number of six-cell windows affected by one placement.
pub const WINDOWS_PER_PLACEMENT: usize = 3 * WINDOW_LEN as usize;

const WINDOW_MASK: u8 = 0b0011_1111;

/// One of the three unique straight-line axes on the hex grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Axis {
    /// Increasing q: `(1, 0)`.
    Q,
    /// Increasing r: `(0, 1)`.
    R,
    /// Increasing q while decreasing r: `(1, -1)`.
    QR,
}

impl Axis {
    /// All unique axes. Opposite directions are represented by different starts.
    pub const ALL: [Self; 3] = [Self::Q, Self::R, Self::QR];

    /// Stable order for sorting/debug output.
    pub const fn index(self) -> u8 {
        match self {
            Self::Q => 0,
            Self::R => 1,
            Self::QR => 2,
        }
    }

    /// Direction vector for walking this axis.
    pub const fn vector(self) -> HexCoord {
        match self {
            Self::Q => HexCoord { q: 1, r: 0 },
            Self::R => HexCoord { q: 0, r: 1 },
            Self::QR => HexCoord { q: 1, r: -1 },
        }
    }
}

/// Canonical identity of one length-6 window.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WindowKey {
    /// First coordinate in the window.
    pub start: HexCoord,
    /// Axis along which the six cells are read.
    pub axis: Axis,
}

impl WindowKey {
    /// Coordinate at position `index` in this window.
    pub fn coord_at(self, index: u8) -> HexCoord {
        self.start + self.axis.vector().scale(index as i16)
    }

    /// All six coordinates in this window.
    pub fn cells(self) -> [HexCoord; WINDOW_LEN as usize] {
        let mut cells = [HexCoord::ZERO; WINDOW_LEN as usize];
        for index in 0..WINDOW_LEN as u8 {
            cells[index as usize] = self.coord_at(index);
        }
        cells
    }

    /// True when `coord` is one of this window's six cells.
    pub fn contains(self, coord: HexCoord) -> bool {
        self.cells().contains(&coord)
    }

    /// True when two windows share at least one cell.
    pub fn intersects(self, other: Self) -> bool {
        self.cells().iter().any(|cell| other.contains(*cell))
    }

    /// True when the windows do not overlap but have adjacent cells.
    pub fn touches(self, other: Self) -> bool {
        if self.intersects(other) {
            return false;
        }

        let left_cells = self.cells();
        let right_cells = other.cells();
        left_cells.iter().any(|left| {
            right_cells
                .iter()
                .any(|right| hex_distance(*left, *right) == 1)
        })
    }

    /// True when two windows either overlap or have adjacent cells.
    pub fn intersects_or_touches(self, other: Self) -> bool {
        self.intersects(other) || self.touches(other)
    }
}

/// Read view of one length-6 window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowEntry {
    key: WindowKey,
    masks: [u8; 2],
}

impl WindowEntry {
    /// Canonical start and axis.
    pub fn key(self) -> WindowKey {
        self.key
    }

    /// All six coordinates in this window.
    pub fn cells(self) -> [HexCoord; WINDOW_LEN as usize] {
        self.key.cells()
    }

    /// Mask for one player's stones.
    pub fn mask(self, player: Player) -> u8 {
        self.masks[player.index()]
    }

    /// Number of stones the player has in this window.
    pub fn count(self, player: Player) -> u8 {
        self.mask(player).count_ones() as u8
    }

    /// All occupied cells, regardless of owner.
    pub fn occupied_mask(self) -> u8 {
        self.masks[Player::Player0.index()] | self.masks[Player::Player1.index()]
    }

    /// Empty positions inside the six-cell window.
    pub fn empty_mask(self) -> u8 {
        !self.occupied_mask() & WINDOW_MASK
    }

    /// Coordinates occupied by `player` inside this window.
    pub fn stone_cells(self, player: Player) -> Vec<HexCoord> {
        self.cells_for_mask(self.mask(player))
    }

    /// Empty coordinates inside this window.
    pub fn empty_cells(self) -> Vec<HexCoord> {
        self.cells_for_mask(self.empty_mask())
    }

    /// All occupied coordinates with their owning players.
    pub fn occupied_cells(self) -> Vec<(HexCoord, Player)> {
        let mut cells = Vec::with_capacity(self.occupied_mask().count_ones() as usize);
        for player in [Player::Player0, Player::Player1] {
            cells.extend(
                self.stone_cells(player)
                    .into_iter()
                    .map(|coord| (coord, player)),
            );
        }
        cells
    }

    /// Player who owns this active window, if it is active.
    pub fn active_player(self) -> Option<Player> {
        match (
            self.masks[Player::Player0.index()] != 0,
            self.masks[Player::Player1.index()] != 0,
        ) {
            (true, false) => Some(Player::Player0),
            (false, true) => Some(Player::Player1),
            _ => None,
        }
    }

    /// True when the window contains stones from exactly one player.
    pub fn is_active(self) -> bool {
        self.active_player().is_some()
    }

    /// Player who owns this threat, if the window is currently a threat.
    pub fn threat_player(self) -> Option<Player> {
        let player = self.active_player()?;
        (self.count(player) >= 4).then_some(player)
    }

    /// True when this window has at least four stones from one player and none
    /// from the other.
    pub fn is_threat(self) -> bool {
        self.threat_player().is_some()
    }

    /// True when this active window has at least four stones for `player`.
    pub fn is_threat_for(self, player: Player) -> bool {
        self.threat_player() == Some(player)
    }

    /// True when this active window is completely filled by `player`.
    pub fn is_win_for(self, player: Player) -> bool {
        self.active_player() == Some(player) && self.count(player) == WINDOW_LEN as u8
    }

    /// True when this window shares at least one cell with `other`.
    pub fn intersects(self, other: Self) -> bool {
        self.key.intersects(other.key)
    }

    /// True when this window does not overlap `other` but has adjacent cells.
    pub fn touches(self, other: Self) -> bool {
        self.key.touches(other.key)
    }

    /// True when this window either overlaps `other` or has adjacent cells.
    pub fn intersects_or_touches(self, other: Self) -> bool {
        self.key.intersects_or_touches(other.key)
    }

    fn cells_for_mask(self, mask: u8) -> Vec<HexCoord> {
        let mut cells = Vec::with_capacity(mask.count_ones() as usize);
        for index in 0..WINDOW_LEN as u8 {
            if mask & (1u8 << index) != 0 {
                cells.push(self.key.coord_at(index));
            }
        }
        cells
    }
}

const EMPTY_WINDOW_KEY: WindowKey = WindowKey {
    start: HexCoord::ZERO,
    axis: Axis::Q,
};

/// Fixed-capacity list of window keys affected by one placement.
///
/// A placement can affect at most 18 windows, so this keeps `WindowUpdate`
/// stack-backed while still exposing slice-like read access.
#[derive(Clone, Eq)]
pub struct WindowKeyList {
    keys: [WindowKey; WINDOWS_PER_PLACEMENT],
    len: u8,
}

impl WindowKeyList {
    /// Create an empty list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of stored keys.
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// True when no keys are stored.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Stored keys as a slice.
    pub fn as_slice(&self) -> &[WindowKey] {
        &self.keys[..self.len()]
    }

    fn push(&mut self, key: WindowKey) {
        assert!(
            self.len() < WINDOWS_PER_PLACEMENT,
            "window key list capacity exceeded"
        );
        let index = self.len();
        self.keys[index] = key;
        self.len += 1;
    }
}

impl Default for WindowKeyList {
    fn default() -> Self {
        Self {
            keys: [EMPTY_WINDOW_KEY; WINDOWS_PER_PLACEMENT],
            len: 0,
        }
    }
}

impl fmt::Debug for WindowKeyList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

impl PartialEq for WindowKeyList {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl AsRef<[WindowKey]> for WindowKeyList {
    fn as_ref(&self) -> &[WindowKey] {
        self.as_slice()
    }
}

impl Deref for WindowKeyList {
    type Target = [WindowKey];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

/// Incremental result produced by one placement's window updates.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WindowUpdate {
    /// All windows touched by the placement.
    pub changed: WindowKeyList,
    /// Changed windows that are now threats for the placed player.
    pub threats: WindowKeyList,
    /// Changed windows that are now wins for the placed player.
    pub winning_windows: WindowKeyList,
}

impl WindowUpdate {
    /// True if this placement completed a six-in-line window.
    pub fn has_win(&self) -> bool {
        !self.winning_windows.is_empty()
    }

    /// True if this placement created or preserved a threat.
    pub fn has_threat(&self) -> bool {
        !self.threats.is_empty()
    }
}

/// Maintained index of all touched windows.
#[derive(Clone, Debug, Default)]
pub struct WindowStore {
    masks_by_key: AHashMap<WindowKey, [u8; 2]>,
}

impl WindowStore {
    /// Create an empty window store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of known/touched windows.
    pub fn len(&self) -> usize {
        self.masks_by_key.len()
    }

    /// True when no windows have been touched yet.
    pub fn is_empty(&self) -> bool {
        self.masks_by_key.is_empty()
    }

    /// Fetch a window entry by canonical key.
    pub fn entry(&self, key: WindowKey) -> Option<WindowEntry> {
        self.masks_by_key
            .get(&key)
            .copied()
            .map(|masks| WindowEntry { key, masks })
    }

    /// Iterate all known windows.
    pub fn entries(&self) -> impl Iterator<Item = WindowEntry> + '_ {
        self.masks_by_key.iter().map(|(key, masks)| WindowEntry {
            key: *key,
            masks: *masks,
        })
    }

    /// Current threat windows for one player.
    pub fn threat_entries(&self, player: Player) -> impl Iterator<Item = WindowEntry> + '_ {
        self.entries()
            .filter(move |entry| (*entry).is_threat_for(player))
    }

    /// Current threat windows for both players.
    pub fn threats(&self) -> impl Iterator<Item = (Player, WindowEntry)> + '_ {
        self.entries()
            .filter_map(|entry| entry.threat_player().map(|player| (player, entry)))
    }

    /// Update the 18 windows affected by one newly placed stone.
    pub(crate) fn update_for_placement(&mut self, coord: HexCoord, player: Player) -> WindowUpdate {
        let mut update = WindowUpdate::default();

        for axis in Axis::ALL {
            for offset in 0..WINDOW_LEN as u8 {
                let (key, bit) = window_containing(coord, axis, offset);
                let masks = self.masks_by_key.entry(key).or_insert([0; 2]);
                debug_assert_eq!((masks[0] | masks[1]) & bit, 0);
                masks[player.index()] |= bit;

                let entry = WindowEntry { key, masks: *masks };
                update.changed.push(key);

                if entry.is_threat_for(player) {
                    update.threats.push(key);
                }
                if entry.is_win_for(player) {
                    update.winning_windows.push(key);
                }
            }
        }

        update
    }
}

fn window_containing(coord: HexCoord, axis: Axis, offset: u8) -> (WindowKey, u8) {
    let start = coord - axis.vector().scale(offset as i16);
    (WindowKey { start, axis }, 1u8 << offset)
}

/// Serializable/debug-friendly view of an active threat window.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Threat {
    /// Player who owns the stones in this window.
    pub player: Player,
    /// Canonical start/axis of the window.
    pub key: WindowKey,
    /// The six coordinates that make up the window.
    pub cells: [HexCoord; WINDOW_LEN as usize],
    /// Player stone positions as a six-bit mask.
    pub stone_mask: u8,
    /// Player stone coordinates inside the window.
    pub stone_cells: Vec<HexCoord>,
    /// Empty positions as a six-bit mask.
    pub empty_mask: u8,
    /// Empty coordinates inside the window.
    pub empty_cells: Vec<HexCoord>,
    /// Number of `player` stones in the window. Always at least four.
    pub own_count: u8,
}

/// Return current threats for `player` from the board's maintained windows.
pub fn find_threats(board: &Board, player: Player) -> Vec<Threat> {
    let mut threats: Vec<_> = board
        .windows()
        .threat_entries(player)
        .map(|entry| Threat {
            player,
            key: entry.key(),
            cells: entry.cells(),
            stone_mask: entry.mask(player),
            stone_cells: entry.stone_cells(player),
            empty_mask: entry.empty_mask(),
            empty_cells: entry.empty_cells(),
            own_count: entry.count(player),
        })
        .collect();
    sort_threats(&mut threats);
    threats
}

/// Return current threats for both players from the board's maintained windows.
pub fn find_all_threats(board: &Board) -> Vec<Threat> {
    let mut threats: Vec<_> = board
        .windows()
        .threats()
        .map(|(player, entry)| Threat {
            player,
            key: entry.key(),
            cells: entry.cells(),
            stone_mask: entry.mask(player),
            stone_cells: entry.stone_cells(player),
            empty_mask: entry.empty_mask(),
            empty_cells: entry.empty_cells(),
            own_count: entry.count(player),
        })
        .collect();
    sort_threats(&mut threats);
    threats
}

fn sort_threats(threats: &mut [Threat]) {
    threats.sort_by_key(|threat| {
        (
            threat.player.index(),
            threat.key.axis.index(),
            threat.key.start.q,
            threat.key.start.r,
        )
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{apply_placement, HexoState, Placement};

    fn place_axis_line(
        store: &mut WindowStore,
        player: Player,
        start: HexCoord,
        axis: Axis,
        count: i16,
    ) {
        for index in 0..count {
            store.update_for_placement(start + axis.vector().scale(index), player);
        }
    }

    fn place_q_line(store: &mut WindowStore, player: Player, start: i16, end: i16) {
        for q in start..=end {
            store.update_for_placement(HexCoord::new(q, 0), player);
        }
    }

    fn line_coord(axis: Axis, index: i16) -> HexCoord {
        HexCoord::ZERO + axis.vector().scale(index)
    }

    fn side_vector(axis: Axis) -> HexCoord {
        match axis {
            Axis::Q | Axis::QR => Axis::R.vector(),
            Axis::R => Axis::Q.vector(),
        }
    }

    fn side_coord(axis: Axis, row: i16, index: i16) -> HexCoord {
        side_vector(axis).scale(row) + axis.vector().scale(index)
    }

    #[test]
    fn placement_updates_only_containing_windows() {
        let mut store = WindowStore::new();

        let update = store.update_for_placement(HexCoord::ZERO, Player::Player0);

        assert_eq!(update.changed.len(), WINDOWS_PER_PLACEMENT);
        assert_eq!(store.len(), WINDOWS_PER_PLACEMENT);
        assert!(
            store
                .entry(WindowKey {
                    start: HexCoord::ZERO,
                    axis: Axis::Q,
                })
                .unwrap()
                .mask(Player::Player0)
                & 1
                != 0
        );
    }

    #[test]
    fn threat_entries_scan_live_windows() {
        let mut store = WindowStore::new();
        place_q_line(&mut store, Player::Player0, 0, 3);

        let threats: Vec<_> = store.threat_entries(Player::Player0).collect();

        assert!(!threats.is_empty());
        assert!(threats
            .iter()
            .all(|entry| (*entry).is_threat_for(Player::Player0)));
        assert_eq!(store.threat_entries(Player::Player1).count(), 0);
    }

    #[test]
    fn threats_are_detected_on_all_axes() {
        for axis in Axis::ALL {
            let mut store = WindowStore::new();
            place_axis_line(&mut store, Player::Player0, HexCoord::ZERO, axis, 4);

            let entry = store
                .entry(WindowKey {
                    start: HexCoord::ZERO,
                    axis,
                })
                .unwrap();

            assert!(entry.is_threat_for(Player::Player0), "axis {:?}", axis);
            assert_eq!(
                entry.stone_cells(Player::Player0),
                (0..4)
                    .map(|index| line_coord(axis, index))
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn apply_placement_wins_on_all_axes() {
        for axis in Axis::ALL {
            let mut state = HexoState::new();
            let opponent = [
                side_coord(axis, 1, 0),
                side_coord(axis, 1, 2),
                side_coord(axis, 1, 4),
                side_coord(axis, 2, 0),
                side_coord(axis, 2, 2),
                side_coord(axis, 2, 4),
            ];

            apply_placement(
                &mut state,
                Placement {
                    coord: line_coord(axis, 0),
                },
            )
            .unwrap();
            apply_placement(&mut state, Placement { coord: opponent[0] }).unwrap();
            apply_placement(&mut state, Placement { coord: opponent[1] }).unwrap();
            apply_placement(
                &mut state,
                Placement {
                    coord: line_coord(axis, 1),
                },
            )
            .unwrap();
            apply_placement(
                &mut state,
                Placement {
                    coord: line_coord(axis, 2),
                },
            )
            .unwrap();
            apply_placement(&mut state, Placement { coord: opponent[2] }).unwrap();
            apply_placement(&mut state, Placement { coord: opponent[3] }).unwrap();
            apply_placement(
                &mut state,
                Placement {
                    coord: line_coord(axis, 3),
                },
            )
            .unwrap();
            apply_placement(
                &mut state,
                Placement {
                    coord: line_coord(axis, 4),
                },
            )
            .unwrap();
            apply_placement(&mut state, Placement { coord: opponent[4] }).unwrap();
            apply_placement(&mut state, Placement { coord: opponent[5] }).unwrap();

            let result = apply_placement(
                &mut state,
                Placement {
                    coord: line_coord(axis, 5),
                },
            )
            .unwrap();

            assert!(result.window_update.has_win(), "axis {:?}", axis);
            assert_eq!(result.outcome.unwrap().winner, Player::Player0);
            assert!(state.is_terminal(), "axis {:?}", axis);
        }
    }

    #[test]
    fn blocked_windows_are_not_threats() {
        let mut store = WindowStore::new();
        place_q_line(&mut store, Player::Player0, 0, 3);

        let key = WindowKey {
            start: HexCoord::new(-1, 0),
            axis: Axis::Q,
        };
        assert!(store.entry(key).unwrap().is_threat_for(Player::Player0));

        store.update_for_placement(HexCoord::new(4, 0), Player::Player1);

        assert!(!store.entry(key).unwrap().is_threat_for(Player::Player0));
        assert!(!store
            .threat_entries(Player::Player0)
            .any(|entry| entry.key() == key));
    }

    #[test]
    fn window_entries_expose_active_state_counts_and_cells() {
        let mut store = WindowStore::new();
        place_q_line(&mut store, Player::Player0, 0, 3);

        let entry = store
            .entry(WindowKey {
                start: HexCoord::new(-1, 0),
                axis: Axis::Q,
            })
            .unwrap();

        assert!(entry.is_active());
        assert_eq!(entry.active_player(), Some(Player::Player0));
        assert!(entry.is_threat());
        assert_eq!(entry.threat_player(), Some(Player::Player0));
        assert_eq!(entry.count(Player::Player0), 4);
        assert_eq!(
            entry.stone_cells(Player::Player0),
            vec![
                HexCoord::new(0, 0),
                HexCoord::new(1, 0),
                HexCoord::new(2, 0),
                HexCoord::new(3, 0),
            ]
        );
        assert_eq!(
            entry.empty_cells(),
            vec![HexCoord::new(-1, 0), HexCoord::new(4, 0)]
        );
    }

    #[test]
    fn windows_can_report_overlap_and_touching() {
        let window = WindowKey {
            start: HexCoord::ZERO,
            axis: Axis::Q,
        };
        let overlapping = WindowKey {
            start: HexCoord::new(3, 0),
            axis: Axis::Q,
        };
        let touching = WindowKey {
            start: HexCoord::new(6, 0),
            axis: Axis::Q,
        };
        let separate = WindowKey {
            start: HexCoord::new(8, 8),
            axis: Axis::Q,
        };

        assert!(window.intersects(overlapping));
        assert!(!window.touches(overlapping));
        assert!(!window.intersects(touching));
        assert!(window.touches(touching));
        assert!(window.intersects_or_touches(touching));
        assert!(!window.intersects_or_touches(separate));
    }

    #[test]
    fn store_can_iterate_threats_for_both_players() {
        let mut store = WindowStore::new();
        place_q_line(&mut store, Player::Player0, 0, 3);
        place_q_line(&mut store, Player::Player1, 20, 23);

        let threats: Vec<_> = store.threats().collect();

        assert!(threats
            .iter()
            .any(|(player, entry)| *player == Player::Player0 && (*entry).is_threat()));
        assert!(threats
            .iter()
            .any(|(player, entry)| *player == Player::Player1 && (*entry).is_threat()));
    }
}
