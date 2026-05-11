//! Incremental six-cell window tracking.
//!
//! Every possible win or threat lives inside a length-6 straight window. A new
//! placement only affects the 18 windows that contain that coordinate:
//! `3 axes * 6 offsets`. This module stores those windows once and maintains
//! lightweight indexes for active windows and threats so callers do not need to
//! rescan the whole board.

use super::board::Board;
use super::coord::HexCoord;
use super::state::Player;
use ahash::{AHashMap, AHashSet};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Number of cells in a win/threat window.
pub const WINDOW_LEN: i16 = 6;

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
}

/// Stable id into `WindowStore.entries`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WindowId(pub u32);

impl WindowId {
    fn index(self) -> usize {
        self.0 as usize
    }
}

/// Compact state for one length-6 window.
///
/// A window stores only two six-bit masks. Coordinates are derived from
/// `WindowKey` when needed, which avoids duplicating six coordinates across many
/// overlapping windows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WindowEntry {
    /// Canonical start and axis.
    pub key: WindowKey,
    /// Bits occupied by Player 0.
    pub p0_mask: u8,
    /// Bits occupied by Player 1.
    pub p1_mask: u8,
}

impl WindowEntry {
    /// Create an empty window entry.
    pub fn new(key: WindowKey) -> Self {
        Self {
            key,
            p0_mask: 0,
            p1_mask: 0,
        }
    }

    /// Mask for one player's stones.
    pub fn mask(self, player: Player) -> u8 {
        match player {
            Player::Player0 => self.p0_mask,
            Player::Player1 => self.p1_mask,
        }
    }

    fn mask_mut(&mut self, player: Player) -> &mut u8 {
        match player {
            Player::Player0 => &mut self.p0_mask,
            Player::Player1 => &mut self.p1_mask,
        }
    }

    /// Number of stones the player has in this window.
    pub fn count(self, player: Player) -> u8 {
        self.mask(player).count_ones() as u8
    }

    /// All occupied cells, regardless of owner.
    pub fn occupied_mask(self) -> u8 {
        self.p0_mask | self.p1_mask
    }

    /// Empty positions inside the six-cell window.
    pub fn empty_mask(self) -> u8 {
        !self.occupied_mask() & WINDOW_MASK
    }

    /// Player who owns this active window, if it is active.
    ///
    /// Active means at least one stone from exactly one player and zero stones
    /// from the other player.
    pub fn active_player(self) -> Option<Player> {
        match (self.p0_mask != 0, self.p1_mask != 0) {
            (true, false) => Some(Player::Player0),
            (false, true) => Some(Player::Player1),
            _ => None,
        }
    }

    /// True when this is an active window for `player`.
    pub fn is_active_for(self, player: Player) -> bool {
        self.active_player() == Some(player)
    }

    /// True when this active window has at least four stones for `player`.
    pub fn is_threat_for(self, player: Player) -> bool {
        self.is_active_for(player) && self.count(player) >= 4
    }

    /// True when this active window is completely filled by `player`.
    pub fn is_win_for(self, player: Player) -> bool {
        self.is_active_for(player) && self.count(player) == WINDOW_LEN as u8
    }

    /// Convert a bit mask into concrete coordinates.
    pub fn coords_for_mask(self, mask: u8) -> Vec<HexCoord> {
        (0..WINDOW_LEN as u8)
            .filter(|index| mask & (1u8 << index) != 0)
            .map(|index| self.key.coord_at(index))
            .collect()
    }

    /// Coordinates occupied by `player` inside this window.
    pub fn stone_cells(self, player: Player) -> Vec<HexCoord> {
        self.coords_for_mask(self.mask(player))
    }

    /// Empty coordinates inside this window.
    pub fn empty_cells(self) -> Vec<HexCoord> {
        self.coords_for_mask(self.empty_mask())
    }
}

/// Incremental result produced by one placement's window updates.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WindowUpdate {
    /// All windows touched by the placement.
    pub changed: Vec<WindowId>,
    /// Changed windows that are now threats for the placed player.
    pub threats: Vec<WindowId>,
    /// Changed windows that are now wins for the placed player.
    pub winning_windows: Vec<WindowId>,
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

/// Maintained index of all non-empty windows.
#[derive(Clone, Debug)]
pub struct WindowStore {
    entries: Vec<WindowEntry>,
    by_key: AHashMap<WindowKey, WindowId>,
    active_by_player: [AHashSet<WindowId>; 2],
    threat_by_player: [AHashSet<WindowId>; 2],
}

impl Default for WindowStore {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            by_key: AHashMap::new(),
            active_by_player: [AHashSet::new(), AHashSet::new()],
            threat_by_player: [AHashSet::new(), AHashSet::new()],
        }
    }
}

impl WindowStore {
    /// Create an empty window store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Rebuild a store from canonical entries, reconstructing all indexes.
    pub fn from_entries(entries: Vec<WindowEntry>) -> Self {
        let mut store = Self {
            entries,
            ..Self::default()
        };
        store.rebuild_indices();
        store
    }

    /// Number of known non-empty/touched windows.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when no windows have been touched yet.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Fetch a window entry by id.
    pub fn entry(&self, id: WindowId) -> Option<&WindowEntry> {
        self.entries.get(id.index())
    }

    /// Find the id for a canonical window key, if that window has been touched.
    pub fn id_for_key(&self, key: WindowKey) -> Option<WindowId> {
        self.by_key.get(&key).copied()
    }

    /// Fetch a window entry by canonical key.
    pub fn entry_by_key(&self, key: WindowKey) -> Option<(WindowId, &WindowEntry)> {
        let id = self.id_for_key(key)?;
        self.entry(id).map(|entry| (id, entry))
    }

    /// Iterate all known windows.
    pub fn entries(&self) -> impl Iterator<Item = (WindowId, &WindowEntry)> {
        self.entries
            .iter()
            .enumerate()
            .map(|(index, entry)| (WindowId(index as u32), entry))
    }

    /// Active windows for one player.
    pub fn active_windows(&self, player: Player) -> impl Iterator<Item = WindowId> + '_ {
        self.active_by_player[player.index()].iter().copied()
    }

    /// Number of active windows for one player.
    pub fn active_window_count(&self, player: Player) -> usize {
        self.active_by_player[player.index()].len()
    }

    /// Threat windows for one player.
    pub fn threat_windows(&self, player: Player) -> impl Iterator<Item = WindowId> + '_ {
        self.threat_by_player[player.index()].iter().copied()
    }

    /// Number of threat windows for one player.
    pub fn threat_window_count(&self, player: Player) -> usize {
        self.threat_by_player[player.index()].len()
    }

    /// Update the 18 windows affected by one newly placed stone.
    pub fn update_for_placement(&mut self, coord: HexCoord, player: Player) -> WindowUpdate {
        let mut update = WindowUpdate::default();

        for axis in Axis::ALL {
            for offset in 0..WINDOW_LEN as u8 {
                let start = coord - axis.vector().scale(offset as i16);
                let key = WindowKey { start, axis };
                let id = self.get_or_create(key);

                self.remove_indices(id);

                let bit = 1u8 << offset;
                let entry = &mut self.entries[id.index()];
                *entry.mask_mut(player) |= bit;

                self.add_indices(id);

                update.changed.push(id);
                if self.entries[id.index()].is_threat_for(player) {
                    update.threats.push(id);
                }
                if self.entries[id.index()].is_win_for(player) {
                    update.winning_windows.push(id);
                }
            }
        }

        update
    }

    fn get_or_create(&mut self, key: WindowKey) -> WindowId {
        if let Some(id) = self.by_key.get(&key).copied() {
            return id;
        }

        let id = WindowId(self.entries.len() as u32);
        self.entries.push(WindowEntry::new(key));
        self.by_key.insert(key, id);
        id
    }

    /// Recompute all derived indexes from `entries`.
    pub fn rebuild_indices(&mut self) {
        self.by_key.clear();
        for player in [Player::Player0, Player::Player1] {
            self.active_by_player[player.index()].clear();
            self.threat_by_player[player.index()].clear();
        }

        for index in 0..self.entries.len() {
            let id = WindowId(index as u32);
            self.by_key.insert(self.entries[index].key, id);
            self.add_indices(id);
        }
    }

    fn remove_indices(&mut self, id: WindowId) {
        if self.entry(id).is_none() {
            return;
        }

        for player in [Player::Player0, Player::Player1] {
            self.active_by_player[player.index()].remove(&id);
            self.threat_by_player[player.index()].remove(&id);
        }
    }

    fn add_indices(&mut self, id: WindowId) {
        let Some(entry) = self.entry(id).copied() else {
            return;
        };

        if let Some(player) = entry.active_player() {
            self.active_by_player[player.index()].insert(id);
            if entry.is_threat_for(player) {
                self.threat_by_player[player.index()].insert(id);
            }
        }
    }
}

impl Serialize for WindowStore {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.entries.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for WindowStore {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let entries = Vec::<WindowEntry>::deserialize(deserializer)?;
        Ok(Self::from_entries(entries))
    }
}

/// Serializable/debug-friendly view of an active threat window.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Threat {
    /// Player who owns the stones in this window.
    pub player: Player,
    /// Id of the stored window.
    pub id: WindowId,
    /// Canonical start/axis of the window.
    pub key: WindowKey,
    /// The six coordinates that make up the window.
    pub cells: [HexCoord; WINDOW_LEN as usize],
    /// Player stone positions as a six-bit mask.
    pub stone_mask: u8,
    /// Empty positions as a six-bit mask.
    pub empty_mask: u8,
    /// Number of `player` stones in the window. Always at least four.
    pub own_count: u8,
}

/// Return current threats for `player` from the board's maintained index.
pub fn find_threats(board: &Board, player: Player) -> Vec<Threat> {
    let mut threats: Vec<_> = board
        .windows()
        .threat_windows(player)
        .filter_map(|id| {
            let entry = board.windows().entry(id).copied()?;
            Some(Threat {
                player,
                id,
                key: entry.key,
                cells: entry.key.cells(),
                stone_mask: entry.mask(player),
                empty_mask: entry.empty_mask(),
                own_count: entry.count(player),
            })
        })
        .collect();
    threats.sort_by_key(|threat| {
        (
            threat.key.axis.index(),
            threat.key.start.q,
            threat.key.start.r,
            threat.id.0,
        )
    });
    threats
}
