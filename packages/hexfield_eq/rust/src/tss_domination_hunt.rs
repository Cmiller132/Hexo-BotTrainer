//! Domination / inferior-cell empirical hunt (DATA, not proofs).
//!
//! Mission: hunt local, cheap-to-detect configurations that let the solver
//! SOUNDLY delete a mover reply at a node ("y is dominated by x / prunable").
//! Method discipline: computation -> conjecture -> (later round) proof. This
//! module produces a counterexample-tested, frequency-ranked catalog; a
//! subsequent proof round takes the survivors.
//!
//! Scope of adjudication. We adjudicate single-cell replies at **defender
//! SecondStone nodes** (budget b = 1): the completing defensive placement.
//! There, a reply is one cell and the child is an attacker-`FirstStone` node,
//! so the wide engine (`WidthOptions::vcf_pair_complete`) can act as a sound
//! WIN referee for the attacker. This is exactly where U11's single-cell
//! sub-hitting dispatch (`min_hitting_set == b`, b = 1) lives.
//!
//! Adjudication direction (minimax). At a defender node P we compare KEEP-x
//! vs DELETE-y, claim "x is at least as good as y for the defender", i.e.
//! V_def(P+x) >= V_def(P+y). After the completing stone the attacker is to
//! move, so V_def(P+c) is read from the attacker's win-ability at the child:
//!   AttWin  (attacker forced win, SOUND)     -> worst for defender
//!   AttUnknown (no proof at cap)             -> middle
//!   AttCantWin (attacker proven loss / defender immediate win) -> best
//! REFUTATION of "keep x, delete y": V_def(P+x) < V_def(P+y). The crisp,
//! cap-robust witness is  attacker WINS after the kept x  but  attacker does
//! NOT win after the deleted y  (y was the better / saving defense). Deleting
//! y while keeping x would then let the solver conclude a FALSE attacker WIN
//! at P. `ProofStatus::Loss` needs a dual certificate and is rarely produced,
//! so the referee is used as a sound WIN-detector; an AttWin(x) vs
//! high-wide-cap AttUnknown(y) asymmetry is recorded as an empirical
//! refutation with its cap stated.
//!
//! Determinism: no RNG on any solved path; all sampling is fixed-seed
//! Fisher-Yates. Corpus + replay conventions are copied verbatim from the
//! validated `tss_leaf_width_hunt` (branch hunt/leaf-width, commit 8d97ac8d,
//! replay validated 300/300).
//!
//! Runners (all `#[ignore]`, `--test-threads=1 --nocapture`):
//!   * `dom_hunt_selftest`  — replay-convention guard + geometry sanity.
//!   * `dom_hunt_scan`      — corpus + random-playout scan: per-pattern fire
//!     rates, branching saved, two-stage (narrow-scan / wide-confirm)
//!     adjudication; emits the records .jsonl.
//!   * `dom_hunt_directed`  — a few hand-scripted legal sequences that force
//!     junction / counterfork / relay shapes, adjudicated the same way.

use std::collections::{BTreeSet, HashSet};
use std::time::Instant;

use hexo_engine::{
    apply_placement, hex_distance, Axis, HexCoord, HexoState, Placement, Player, TurnPhase,
    WindowKey,
};

use crate::threats_shared as threats;
use crate::tss_core::{ProofStatus, SolveCaps, SolveGoal};
use crate::tss_reference;
use crate::tss_reference_fast::{self, FastOrderingHint, FastReferenceConfig};
use crate::tss_solver::{TssSolver, WidthOptions};

// ==========================================================================
// Corpus parsing (stdlib only; verbatim from tss_leaf_width_hunt.rs).
// schema: {"game_hash":"..","moves":[[q,r],..],"winner":±1,"elo":[..]}
// ==========================================================================

struct Game {
    game_hash: String,
    moves: Vec<(i16, i16)>,
    winner: i8,
}

fn corpus_path() -> String {
    std::env::var("TSS_DOM_CORPUS").unwrap_or_else(|_| {
        "E:/Hexo-BotTrainer-hexgt/data/hexo-bootstrap-corpus/hexo_human_corpus.jsonl".to_string()
    })
}

fn parse_ints(slice: &str) -> Vec<i16> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in slice.chars() {
        if ch == '-' || ch.is_ascii_digit() {
            cur.push(ch);
        } else if !cur.is_empty() {
            out.push(cur.parse().expect("i16 token"));
            cur.clear();
        }
    }
    if !cur.is_empty() {
        out.push(cur.parse().expect("i16 token"));
    }
    out
}

fn parse_hash(line: &str) -> String {
    let key = "\"game_hash\":\"";
    match line.find(key) {
        Some(m) => {
            let after = &line[m + key.len()..];
            let end = after.find('"').unwrap_or(0);
            after[..end].to_string()
        }
        None => String::new(),
    }
}

fn parse_line(line: &str) -> Option<Game> {
    let key = "\"moves\":";
    let m = line.find(key)?;
    let after = &line[m + key.len()..];
    let start = after.find('[')?;
    let bytes = after.as_bytes();
    let mut depth = 0i32;
    let mut end = None;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        match b {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let arr = &after[start..=end?];
    let nums = parse_ints(arr);
    let moves: Vec<(i16, i16)> = nums.chunks_exact(2).map(|c| (c[0], c[1])).collect();

    let wkey = "\"winner\":";
    let w = line.find(wkey)?;
    let wafter = &line[w + wkey.len()..];
    let mut ws = String::new();
    for ch in wafter.chars() {
        if ch == '-' || ch.is_ascii_digit() {
            ws.push(ch);
        } else if !ws.is_empty() {
            break;
        }
    }
    let winner: i8 = ws.parse().ok()?;
    Some(Game {
        game_hash: parse_hash(line),
        moves,
        winner,
    })
}

fn load_corpus() -> Vec<Game> {
    let path = corpus_path();
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read corpus {path}: {e}"));
    let games: Vec<Game> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| parse_line(l).unwrap_or_else(|| panic!("bad corpus line")))
        .collect();
    eprintln!("DOM_CORPUS path={path} games={}", games.len());
    games
}

// --------------------------------------------------------------------------
// Deterministic RNG (no external dep; never on a scored/solved path).
// --------------------------------------------------------------------------

struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            0
        } else {
            self.next() % n
        }
    }
}

fn free_ram_gb() -> f64 {
    let out = std::process::Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB",
        ])
        .output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout)
            .trim()
            .parse::<f64>()
            .unwrap_or(64.0),
        Err(_) => 64.0,
    }
}

// ==========================================================================
// Geometry / window helpers.
// ==========================================================================

/// All cells within hex distance <= r of `c` (|B_8| = 217).
fn ball(c: HexCoord, r: i16) -> Vec<HexCoord> {
    let mut out = Vec::new();
    for dq in -r..=r {
        for dr in -r..=r {
            let z = HexCoord::new(c.q + dq, c.r + dr);
            if hex_distance(c, z) <= r {
                out.push(z);
            }
        }
    }
    out
}

/// The 18 canonical windows through `c` (3 axes x 6 offsets).
fn windows_through(c: HexCoord) -> Vec<WindowKey> {
    let mut out = Vec::with_capacity(18);
    for axis in Axis::ALL {
        let v = axis.vector();
        for o in 0..6i16 {
            let start = HexCoord::new(c.q - v.q * o, c.r - v.r * o);
            out.push(WindowKey { start, axis });
        }
    }
    out
}

/// Legal-support set Lambda(P) = every cell within radius 8 of some stone =
/// (legal empty cells) union (occupied cells).
fn support_set(state: &HexoState) -> HashSet<HexCoord> {
    let mut set: HashSet<HexCoord> = HashSet::new();
    let mut legal = Vec::new();
    state.write_legal_moves(&mut legal);
    for c in legal {
        set.insert(c);
    }
    for &c in state.board().occupied_cells() {
        set.insert(c);
    }
    set
}

/// Cells that placing at `c` would newly legalize: B_8(c) \ Lambda(P).
fn new_support_cells(c: HexCoord, support: &HashSet<HexCoord>) -> Vec<HexCoord> {
    ball(c, 8)
        .into_iter()
        .filter(|z| !support.contains(z))
        .collect()
}

/// `c` is frontier-inert: B_8(c) subset Lambda(P) (opens no new territory).
fn frontier_inert(c: HexCoord, support: &HashSet<HexCoord>) -> bool {
    ball(c, 8).into_iter().all(|z| support.contains(&z))
}

/// True when window `key` is dead (both colours present) in `state`.
fn window_dead(state: &HexoState, key: WindowKey) -> bool {
    match state.board().windows().entry(key) {
        Some(w) => w.count(Player::Player0) > 0 && w.count(Player::Player1) > 0,
        None => false, // untouched window = empty = not dead
    }
}

/// True when window `key` is alive for `player` (>=1 of player, 0 of the other).
fn window_alive_for(state: &HexoState, key: WindowKey, player: Player) -> bool {
    match state.board().windows().entry(key) {
        Some(w) => w.count(player) > 0 && w.count(player.other()) == 0,
        None => false,
    }
}

/// True when every one of the 18 windows through `c` is dead.
fn cell_dead(state: &HexoState, c: HexCoord) -> bool {
    windows_through(c)
        .into_iter()
        .all(|k| window_dead(state, k))
}

/// True when `c` touches no window currently alive for `player` (placing there
/// creates no line for `player`).
fn touches_no_alive_for(state: &HexoState, c: HexCoord, player: Player) -> bool {
    windows_through(c)
        .into_iter()
        .all(|k| !window_alive_for(state, k, player))
}

/// Attacker = the side NOT to move (the one carrying the threats at a defender
/// node). Live attacker threat windows (count >= 4, single-colour attacker).
fn attacker_threat_windows(state: &HexoState, attacker: Player) -> Vec<WindowKey> {
    state
        .board()
        .windows()
        .live_threat_entries()
        .filter(|(p, _)| *p == attacker)
        .map(|(_, w)| w.key())
        .collect()
}

/// Empty cells of window `key`.
fn window_empties(state: &HexoState, key: WindowKey) -> Vec<HexCoord> {
    match state.board().windows().entry(key) {
        Some(w) => w.empty_cells(),
        None => key.cells().to_vec(), // fully empty window
    }
}

/// Legal cells that create a `defender` counter-threat: an empty completing a
/// window currently alive-for-defender with count >= 3 (so placing there makes
/// count >= 4 = a defender four/five). These are the G3 counterfork candidates.
fn counter_threat_cells(
    state: &HexoState,
    defender: Player,
    legal: &HashSet<HexCoord>,
) -> Vec<HexCoord> {
    let mut out: Vec<HexCoord> = Vec::new();
    for entry in state.board().windows().entries() {
        if entry.count(defender) >= 3 && entry.count(defender.other()) == 0 {
            for e in entry.empty_cells() {
                if legal.contains(&e) && !out.contains(&e) {
                    out.push(e);
                }
            }
        }
    }
    out.sort_by_key(|c| (c.q, c.r));
    out
}

/// Cells (from `cands`) that kill every attacker threat window (full coverers).
fn full_coverers(state: &HexoState, node: &DefNode, cands: &[HexCoord]) -> Vec<HexCoord> {
    cands
        .iter()
        .copied()
        .filter(|&c| {
            node.threats
                .iter()
                .all(|&w| window_empties(state, w).contains(&c))
        })
        .collect()
}

// ==========================================================================
// Adjudication: attacker win-ability at a child (attacker to move).
// ==========================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AVal {
    AttWin,     // attacker forced win (sound)   -> worst for defender
    AttUnknown, // no proof at cap                -> middle
    AttCantWin, // attacker proven loss / defender immediate win -> best for defender
}

/// Defender-preference order: AttCantWin > AttUnknown > AttWin.
fn def_rank(v: AVal) -> u8 {
    match v {
        AVal::AttWin => 0,
        AVal::AttUnknown => 1,
        AVal::AttCantWin => 2,
    }
}

/// Apply one completing defensive stone; return the child and whether it is a
/// defender immediate win.
fn child_after(state: &HexoState, cell: HexCoord) -> (HexoState, Option<Player>) {
    let mut child = state.clone();
    let res = apply_placement(
        &mut child,
        Placement {
            coord: HexCoord::new(cell.q, cell.r),
        },
    )
    .expect("candidate reply must be legal");
    (child, res.outcome.map(|o| o.winner))
}

fn status_solve(child: &HexoState, wide: bool, node_cap: u64, horizon_slack: u32) -> ProofStatus {
    let caps = SolveCaps {
        node_cap,
        tt_bytes_cap: 256 << 20,
        semantic_horizon: child.placements_made().saturating_add(horizon_slack),
    };
    let mut solver = TssSolver::default();
    if wide {
        solver.set_width_options(WidthOptions::vcf_pair_complete());
    }
    solver.solve_goal(child, &caps, SolveGoal::Win).status
}

/// Evaluate a child for attacker win-ability, cheapest-first:
/// lambda-1 verdict -> (optional) solve.
fn eval_child(
    state: &HexoState,
    cell: HexCoord,
    defender: Player,
    wide: bool,
    node_cap: u64,
    horizon_slack: u32,
) -> AVal {
    let (child, win) = child_after(state, cell);
    if let Some(w) = win {
        // The defender's own completing stone can only win for the defender.
        return if w == defender {
            AVal::AttCantWin
        } else {
            AVal::AttWin
        };
    }
    // Cheap sound leaf: attacker to move at child.
    match threats::analyze(&child).verdict() {
        Some(v) if v > 0.0 => return AVal::AttWin,
        Some(_) => return AVal::AttCantWin, // attacker forced-loss this turn
        None => {}
    }
    if node_cap == 0 {
        return AVal::AttUnknown;
    }
    match status_solve(&child, wide, node_cap, horizon_slack) {
        ProofStatus::Win => AVal::AttWin,
        ProofStatus::Loss => AVal::AttCantWin,
        ProofStatus::Unknown => AVal::AttUnknown,
    }
}

// ==========================================================================
// Node model + pattern firings.
// ==========================================================================

/// A genuine defensive SecondStone node: attacker (side-not-to-move) carries
/// >=1 live threat, defender (side-to-move) has no own win-now.
struct DefNode {
    attacker: Player,
    defender: Player,
    threats: Vec<WindowKey>,
    min_hitting_set: Option<u8>,
    hitters: Vec<HexCoord>, // legal cells that kill >=1 attacker threat
    support: HashSet<HexCoord>,
    legal: HashSet<HexCoord>,
    first_stone: bool, // true = FirstStone (budget 2), false = SecondStone (budget 1)
}

/// A genuine defensive node (either phase): attacker (side-not-to-move) carries
/// >=1 live threat, defender (side-to-move) has no own win-now.
fn classify_def_node(state: &HexoState) -> Option<DefNode> {
    if state.is_terminal() {
        return None;
    }
    let first_stone = match state.phase() {
        TurnPhase::FirstStone => true,
        TurnPhase::SecondStone { .. } => false,
        TurnPhase::Opening => return None,
    };
    let defender = state.current_player();
    let attacker = defender.other();
    let a = threats::analyze(state);
    if a.own_win_now {
        return None; // defender just wins; not a defensive decision
    }
    let threats = attacker_threat_windows(state, attacker);
    if threats.is_empty() {
        return None;
    }
    let legal: HashSet<HexCoord> = {
        let mut v = Vec::new();
        state.write_legal_moves(&mut v);
        v.into_iter().collect()
    };
    let mut hitters: Vec<HexCoord> = Vec::new();
    for &w in &threats {
        for e in window_empties(state, w) {
            if legal.contains(&e) && !hitters.contains(&e) {
                hitters.push(e);
            }
        }
    }
    hitters.sort_by_key(|c| (c.q, c.r));
    Some(DefNode {
        attacker,
        defender,
        min_hitting_set: a.min_hitting_set,
        threats,
        hitters,
        support: support_set(state),
        legal,
        first_stone,
    })
}

/// Set of attacker-threat windows that placing at `c` kills (c is an empty of
/// the window). Represented as sorted (start.q,start.r,axis) tuples for subset.
fn kill_set(state: &HexoState, node: &DefNode, c: HexCoord) -> Vec<WindowKey> {
    node.threats
        .iter()
        .copied()
        .filter(|&w| window_empties(state, w).contains(&c))
        .collect()
}

fn is_subset(sub: &[WindowKey], sup: &[WindowKey]) -> bool {
    sub.iter().all(|w| sup.contains(w))
}

/// A DBD-superset firing: keep `x` (kills a superset of attacker threats),
/// delete `y`. Records the load-bearing guard flags so analysis can bucket by
/// which guard is required for soundness.
struct DbdFiring {
    keep: HexCoord,
    delete: HexCoord,
    kx: usize,
    ky: usize,
    frontier_ok: bool, // new_support(x) subset new_support(y)
    x_pure: bool,      // x touches no defender-alive window
    y_pure: bool,      // y touches no defender-alive window
    x_inert: bool,     // x frontier-inert
}

fn detect_dbd_superset(state: &HexoState, node: &DefNode) -> Vec<DbdFiring> {
    let mut out = Vec::new();
    let kills: Vec<(HexCoord, Vec<WindowKey>)> = node
        .hitters
        .iter()
        .map(|&c| (c, kill_set(state, node, c)))
        .collect();
    for (x, kx) in &kills {
        for (y, ky) in &kills {
            if x == y {
                continue;
            }
            // strict superset: kx superset ky and |kx| > |ky|
            if ky.len() >= kx.len() {
                continue;
            }
            if !is_subset(ky, kx) {
                continue;
            }
            let nsx = new_support_cells(*x, &node.support);
            let nsy_set: HashSet<HexCoord> =
                new_support_cells(*y, &node.support).into_iter().collect();
            let frontier_ok = nsx.iter().all(|z| nsy_set.contains(z));
            out.push(DbdFiring {
                keep: *x,
                delete: *y,
                kx: kx.len(),
                ky: ky.len(),
                frontier_ok,
                x_pure: touches_no_alive_for(state, *x, node.defender),
                y_pure: touches_no_alive_for(state, *y, node.defender),
                x_inert: frontier_inert(*x, &node.support),
            });
        }
    }
    out
}

/// A pair of dead + frontier-inert cells (DRQ / dead-region quotient seed):
/// both are pure "pass" placements, conjectured mutually equivalent.
fn detect_dead_equiv(state: &HexoState, node: &DefNode) -> Vec<(HexCoord, HexCoord)> {
    let mut dead_inert: Vec<HexCoord> = {
        let mut v = Vec::new();
        state.write_legal_moves(&mut v);
        v.into_iter()
            .filter(|&c| cell_dead(state, c) && frontier_inert(c, &node.support))
            .collect()
    };
    dead_inert.sort_by_key(|c| (c.q, c.r));
    let mut out = Vec::new();
    for i in 0..dead_inert.len() {
        for j in (i + 1)..dead_inert.len() {
            out.push((dead_inert[i], dead_inert[j]));
        }
    }
    out
}

/// P2 dead-spoke interchangeable-hit detector (the PROVEN pattern) — for fire
/// rate measurement only. Fires on a count-4 attacker window with exactly two
/// empties x,y, all OTHER incident windows through x,y dead, and equal new
/// support.
fn detect_p2(state: &HexoState, node: &DefNode) -> Vec<(HexCoord, HexCoord)> {
    let mut out = Vec::new();
    for &w in &node.threats {
        let entry = match state.board().windows().entry(w) {
            Some(e) => e,
            None => continue,
        };
        if entry.count(node.attacker) != 4 {
            continue;
        }
        let empties = entry.empty_cells();
        if empties.len() != 2 {
            continue;
        }
        let (x, y) = (empties[0], empties[1]);
        // all other incident windows through x and y are dead
        let others_dead = windows_through(x)
            .into_iter()
            .chain(windows_through(y))
            .filter(|&k| k != w)
            .all(|k| window_dead(state, k));
        if !others_dead {
            continue;
        }
        let nsx: HashSet<HexCoord> = new_support_cells(x, &node.support).into_iter().collect();
        let nsy: HashSet<HexCoord> = new_support_cells(y, &node.support).into_iter().collect();
        if nsx == nsy {
            out.push((x, y));
        }
    }
    out
}

// ==========================================================================
// Scan tallies + record emission.
// ==========================================================================

#[derive(Default, Clone, Copy)]
struct Tally {
    def_nodes: u64,
    forced_loss_nodes: u64, // min_hitting_set == None (defender already lost)
    dbd_firings: u64,
    dbd_guarded_firings: u64, // frontier_ok && x_pure && y_pure
    dead_equiv_firings: u64,
    p2_firings: u64,
    branching_saved_dbd_guarded: u64, // distinct deleted cells (guarded)
    dispatch_forced_nodes: u64,       // mhs == Some(1) forced defensive nodes
    dispatch_pairs_checked: u64,      // (full-coverer keep, counter/non-cover delete)
    dispatch_refutations: u64,        // keep loses, delete holds (would be unsound)
    coverer_multi_nodes: u64,         // forced nodes with >=2 full coverers
    coverer_pairs_checked: u64,       // adjudicated full-coverer pairs
    coverer_mismatch: u64,            // two covering hits with DIFFERENT referee value
    coverer_p2_mismatch: u64,         // mismatch where P2 dead-spoke held (would refute P2!)
}

fn status_name(s: ProofStatus) -> &'static str {
    match s {
        ProofStatus::Win => "WIN",
        ProofStatus::Loss => "LOSS",
        ProofStatus::Unknown => "UNKNOWN",
    }
}

fn aval_name(v: AVal) -> &'static str {
    match v {
        AVal::AttWin => "att_win",
        AVal::AttUnknown => "att_unknown",
        AVal::AttCantWin => "att_cant_win",
    }
}

// ==========================================================================
// Position sources.
// ==========================================================================

/// Replay a game prefix of `n` placements.
fn replay_prefix(moves: &[(i16, i16)], n: usize) -> HexoState {
    let mut state = HexoState::new();
    for &(q, r) in &moves[..n] {
        apply_placement(
            &mut state,
            Placement {
                coord: HexCoord::new(q, r),
            },
        )
        .expect("legal corpus replay");
    }
    state
}

/// A reachable random position: play random legal single stones from the
/// opening for `plies` placements (or until terminal). Deterministic in `seed`.
fn random_position(seed: u64, plies: u32) -> HexoState {
    let mut rng = XorShift(seed | 1);
    let mut state = HexoState::new();
    let mut legal = Vec::new();
    for _ in 0..plies {
        if state.is_terminal() {
            break;
        }
        state.write_legal_moves(&mut legal);
        if legal.is_empty() {
            break;
        }
        let pick = legal[rng.below(legal.len() as u64) as usize];
        apply_placement(&mut state, Placement { coord: pick }).expect("legal random move");
    }
    state
}

// ==========================================================================
// SELF TEST — replay convention + geometry sanity.
// ==========================================================================

#[test]
#[ignore = "dom-hunt self test; run with --nocapture"]
fn dom_hunt_selftest() {
    // geometry
    assert_eq!(ball(HexCoord::ZERO, 8).len(), 217, "|B_8| must be 217");
    assert_eq!(ball(HexCoord::ZERO, 1).len(), 7);
    assert_eq!(windows_through(HexCoord::ZERO).len(), 18);
    // all 18 windows actually contain the origin
    for k in windows_through(HexCoord::ZERO) {
        assert!(k.contains(HexCoord::ZERO));
    }

    // replay-convention guard (subset; the full 300/300 lives in leaf-width).
    let games = load_corpus();
    let decisive: Vec<&Game> = games
        .iter()
        .filter(|g| g.winner == 1 || g.winner == -1)
        .collect();
    assert!(decisive.len() >= 200);
    let mut rng = XorShift(0xD0D0_CAFE_1234_5678);
    let mut idx: Vec<usize> = (0..decisive.len()).collect();
    for i in (1..idx.len()).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        idx.swap(i, j);
    }
    let n = 200.min(idx.len());
    for &gi in idx.iter().take(n) {
        let g = decisive[gi];
        let mut state = HexoState::new();
        let mut ended = None;
        for &(q, r) in &g.moves {
            let res = apply_placement(
                &mut state,
                Placement {
                    coord: HexCoord::new(q, r),
                },
            )
            .unwrap_or_else(|e| panic!("illegal replay {}: {e:?}", g.game_hash));
            if let Some(o) = res.outcome {
                ended = Some(o);
                break;
            }
        }
        let o = ended.unwrap_or_else(|| panic!("game did not terminate {}", g.game_hash));
        let sign = if o.winner == Player::Player0 { 1 } else { -1 };
        assert_eq!(sign, g.winner, "winner mismatch {}", g.game_hash);
        assert!(state.is_terminal());
    }
    println!("DOM_SELFTEST_OK geometry+replay checked={n}");
}

// ==========================================================================
// MAIN SCAN — fire rates + two-stage adjudication.
// ==========================================================================

#[test]
#[ignore = "empirical domination hunt; serialized, --test-threads=1 --nocapture"]
fn dom_hunt_scan() {
    let games = load_corpus();

    let node_budget: usize = envn("TSS_DOM_NODES", 4000);
    let per_source_nodes: usize = envn("TSS_DOM_PER_SOURCE", 2000);
    let seed: u64 = envn64("TSS_DOM_SEED", 0x51ED_D06_D0_11A7);
    let scan_cap: u64 = envn64("TSS_DOM_SCAN_CAP", 4000);
    let confirm_cap: u64 = envn64("TSS_DOM_CONFIRM_CAP", 80_000);
    let horizon_slack: u32 = envn("TSS_DOM_HORIZON", 40) as u32;
    let random_positions: usize = envn("TSS_DOM_RANDOM_POS", 4000);
    let random_plies_max: u32 = envn("TSS_DOM_RANDOM_PLIES", 60) as u32;
    let records_path = std::env::var("TSS_DOM_RECORDS").unwrap_or_else(|_| {
        "E:/Hexo-BotTrainer-hexgt/.claude/worktrees/hunt-domination/dom_hunt_records.jsonl"
            .to_string()
    });
    let _ = node_budget;

    eprintln!(
        "DOM_SCAN_SETUP seed={seed} scan_cap={scan_cap} confirm_cap={confirm_cap} \
         horizon_slack={horizon_slack} per_source_nodes={per_source_nodes} \
         random_positions={random_positions} random_plies_max={random_plies_max}"
    );

    // Collect candidate defender-SecondStone nodes from two sources:
    //  (A) corpus decisive games, every SecondStone node along the replay;
    //  (B) random reachable positions.
    // Each candidate is (state, provenance-string).
    let mut nodes: Vec<(HexoState, String)> = Vec::new();

    // Source A: corpus.
    {
        // enumerate all SecondStone nodes in decisive games, then fixed-seed
        // subsample to per_source_nodes.
        let mut cands: Vec<(usize, usize)> = Vec::new(); // (game_idx, prefix_len)
        for (gi, g) in games.iter().enumerate() {
            if g.winner != 1 && g.winner != -1 {
                continue;
            }
            let mut state = HexoState::new();
            for (i, &(q, r)) in g.moves.iter().enumerate() {
                // Pre-filter: keep only genuine defensive SecondStone nodes so
                // every sampled node exercises the patterns.
                if matches!(state.phase(), TurnPhase::SecondStone { .. }) {
                    if let Some(node) = classify_def_node(&state) {
                        if !node.first_stone {
                            cands.push((gi, i));
                        }
                    }
                }
                if state.is_terminal() {
                    break;
                }
                apply_placement(
                    &mut state,
                    Placement {
                        coord: HexCoord::new(q, r),
                    },
                )
                .expect("legal replay");
            }
        }
        let mut rng = XorShift(seed ^ 0xA11A);
        for i in (1..cands.len()).rev() {
            let j = rng.below(i as u64 + 1) as usize;
            cands.swap(i, j);
        }
        cands.truncate(per_source_nodes);
        eprintln!(
            "DOM_SCAN sourceA corpus_secondstone_nodes_sampled={}",
            cands.len()
        );
        for (gi, plen) in cands {
            let g = &games[gi];
            let state = replay_prefix(&g.moves, plen);
            nodes.push((state, format!("corpus:{}:{}", g.game_hash, plen)));
        }
    }

    // Source B: random reachable positions; walk each to its SecondStone nodes.
    {
        let mut rng = XorShift(seed ^ 0xB22B);
        let mut taken = 0usize;
        for _ in 0..random_positions {
            if taken >= per_source_nodes {
                break;
            }
            let s = rng.next();
            let plies = 2 + rng.below(random_plies_max as u64) as u32;
            let state = random_position(s, plies);
            if matches!(state.phase(), TurnPhase::SecondStone { .. }) {
                if let Some(node) = classify_def_node(&state) {
                    if !node.first_stone {
                        nodes.push((state, format!("random:{s:016x}:{plies}")));
                        taken += 1;
                    }
                }
            }
        }
        eprintln!("DOM_SCAN sourceB random_secondstone_nodes={taken}");
    }

    eprintln!("DOM_SCAN total_candidate_nodes={}", nodes.len());

    let mut tally = Tally::default();
    let mut records: Vec<String> = Vec::new();
    let mut refutations: Vec<String> = Vec::new();
    let mut dead_equiv_checked = 0u64;
    let mut dead_equiv_mismatch = 0u64;
    let t_start = Instant::now();

    for (i, (state, prov)) in nodes.iter().enumerate() {
        if i % 200 == 0 {
            let ram = free_ram_gb();
            eprintln!(
                "DOM_SCAN progress={i}/{} refutations={} elapsed_s={:.0} free_ram_gb={ram:.1}",
                nodes.len(),
                refutations.len(),
                t_start.elapsed().as_secs_f64()
            );
            while free_ram_gb() < 8.0 {
                eprintln!("DOM_SCAN low RAM (<8GB), sleeping 60s");
                std::thread::sleep(std::time::Duration::from_secs(60));
            }
        }
        let node = match classify_def_node(state) {
            Some(n) => n,
            None => continue,
        };
        tally.def_nodes += 1;
        if node.min_hitting_set.is_none() {
            tally.forced_loss_nodes += 1;
        }

        // ---- P2 (proven) fire-rate ----
        let p2 = detect_p2(state, &node);
        tally.p2_firings += p2.len() as u64;

        // ---- DRQ dead-equiv fire-rate + spot adjudication ----
        let de = detect_dead_equiv(state, &node);
        tally.dead_equiv_firings += de.len() as u64;
        // adjudicate at most one dead-equiv pair per node (cheap, should match)
        if let Some(&(x, y)) = de.first() {
            let vx = eval_child(state, x, node.defender, false, scan_cap, horizon_slack);
            let vy = eval_child(state, y, node.defender, false, scan_cap, horizon_slack);
            dead_equiv_checked += 1;
            if vx != vy {
                dead_equiv_mismatch += 1;
                refutations.push(format!(
                    "{{\"pattern\":\"DRQ_dead_equiv\",\"prov\":\"{prov}\",\"keep\":[{},{}],\"delete\":[{},{}],\"val_keep\":\"{}\",\"val_delete\":\"{}\"}}",
                    x.q, x.r, y.q, y.r, aval_name(vx), aval_name(vy)
                ));
            }
        }

        // ---- DBD superset ----
        let dbd = detect_dbd_superset(state, &node);
        let mut guarded_deletes: HashSet<(i16, i16)> = HashSet::new();
        for f in &dbd {
            tally.dbd_firings += 1;
            let guarded = f.frontier_ok && f.x_pure && f.y_pure;
            if guarded {
                tally.dbd_guarded_firings += 1;
                guarded_deletes.insert((f.delete.q, f.delete.r));
            }

            // Stage 1: narrow scan of both children.
            let vx = eval_child(state, f.keep, node.defender, false, scan_cap, horizon_slack);
            let vy = eval_child(
                state,
                f.delete,
                node.defender,
                false,
                scan_cap,
                horizon_slack,
            );

            // Refutation candidate: kept x strictly worse for defender than
            // deleted y  (def_rank(x) < def_rank(y)).  Sharpest: x=AttWin,
            // y != AttWin.
            let stage1_refute = def_rank(vx) < def_rank(vy);
            if !stage1_refute {
                // still record a compact row for a sample of firings
                if records.len() < 4000 && (tally.dbd_firings % 40 == 0) {
                    records.push(dbd_row(prov, &node, f, vx, vy, "consistent", scan_cap));
                }
                continue;
            }

            // Stage 2: wide confirm. Re-solve x WIN (sound) at wide, and y at
            // wide+high cap; a persistent asymmetry is the counterexample.
            let vx2 = eval_child(
                state,
                f.keep,
                node.defender,
                true,
                confirm_cap,
                horizon_slack,
            );
            let vy2 = eval_child(
                state,
                f.delete,
                node.defender,
                true,
                confirm_cap,
                horizon_slack,
            );
            let label = if def_rank(vx2) < def_rank(vy2) {
                "REFUTED"
            } else {
                "stage1_only"
            };
            let row = dbd_row(prov, &node, f, vx2, vy2, label, confirm_cap);
            records.push(row.clone());
            if label == "REFUTED" {
                refutations.push(row);
                eprintln!(
                    "DOM_SCAN REFUTED DBD prov={prov} keep=({},{}) del=({},{}) kx={} ky={} guards[fr={} xp={} yp={}] vkeep={} vdel={}",
                    f.keep.q, f.keep.r, f.delete.q, f.delete.r, f.kx, f.ky,
                    f.frontier_ok, f.x_pure, f.y_pure, aval_name(vx2), aval_name(vy2)
                );
            }
        }
        tally.branching_saved_dbd_guarded += guarded_deletes.len() as u64;

        // ---- PAT-A: dispatch dismissal soundness (b=1) ----
        // At a forced node (mhs == 1) the engine's implicit_dispatch keeps a
        // minimum hitting set and dismisses non-hitting replies (U3). The
        // sharpest adversary is a DEFENDER COUNTER-THREAT cell (G3): a delete
        // candidate that does NOT cover all attacker threats but builds a
        // defender four/five. If keeping a full-coverer loses while that
        // counter-threat holds, the dismissal is unsound. Reasoning predicts
        // soundness at b=1 (any unhit attacker >=4 window is completed next
        // attacker turn, before the counter fires) — this measures it.
        if node.min_hitting_set == Some(1) {
            tally.dispatch_forced_nodes += 1;
            let full = full_coverers(state, &node, &node.hitters);

            // ---- Full-coverer interchangeability (P2 / sub-hitting algebra) ----
            // Two single cells that BOTH cover all attacker threats are both
            // successful immediate defenses. Are they interchangeable? P2 says
            // NO in general (only under dead-spoke conditions). A referee-visible
            // difference (one covering hit lets the attacker win, the other
            // holds) refutes naive "any min-hitting cell is interchangeable" and
            // shows P2's dead-spoke hypothesis is load-bearing.
            if full.len() >= 2 {
                tally.coverer_multi_nodes += 1;
                let p2_pairs: HashSet<(i16, i16, i16, i16)> = detect_p2(state, &node)
                    .into_iter()
                    .map(|(x, y)| {
                        let (a, b) = if (x.q, x.r) <= (y.q, y.r) {
                            (x, y)
                        } else {
                            (y, x)
                        };
                        (a.q, a.r, b.q, b.r)
                    })
                    .collect();
                for a in 0..full.len() {
                    for b in (a + 1)..full.len() {
                        let (x, y) = (full[a], full[b]);
                        tally.coverer_pairs_checked += 1;
                        let vx =
                            eval_child(state, x, node.defender, false, scan_cap, horizon_slack);
                        let vy =
                            eval_child(state, y, node.defender, false, scan_cap, horizon_slack);
                        if vx != vy {
                            // confirm wide before recording
                            let vx2 = eval_child(
                                state,
                                x,
                                node.defender,
                                true,
                                confirm_cap,
                                horizon_slack,
                            );
                            let vy2 = eval_child(
                                state,
                                y,
                                node.defender,
                                true,
                                confirm_cap,
                                horizon_slack,
                            );
                            if vx2 != vy2 {
                                tally.coverer_mismatch += 1;
                                let (lx, ly) = if (x.q, x.r) <= (y.q, y.r) {
                                    (x, y)
                                } else {
                                    (y, x)
                                };
                                let p2_prot = p2_pairs.contains(&(lx.q, lx.r, ly.q, ly.r));
                                if p2_prot {
                                    tally.coverer_p2_mismatch += 1;
                                }
                                let row = format!(
                                    "{{\"pattern\":\"COVERER_interchange\",\"label\":\"{}\",\"prov\":\"{prov}\",\"cell_a\":[{},{}],\"cell_b\":[{},{}],\"threats\":{},\"p2_protected\":{},\"val_a\":\"{}\",\"val_b\":\"{}\",\"cap\":{}}}",
                                    if p2_prot { "P2_REFUTED" } else { "naive_interchange_refuted" },
                                    x.q, x.r, y.q, y.r, node.threats.len(), p2_prot,
                                    aval_name(vx2), aval_name(vy2), confirm_cap
                                );
                                records.push(row.clone());
                                refutations.push(row);
                                eprintln!(
                                    "DOM_SCAN COVERER-mismatch prov={prov} a=({},{}) b=({},{}) p2_protected={p2_prot} va={} vb={}",
                                    x.q, x.r, y.q, y.r, aval_name(vx2), aval_name(vy2)
                                );
                            }
                        }
                    }
                }
            }

            let counters = counter_threat_cells(state, node.defender, &node.legal);
            if let Some(&keep) = full.first() {
                // delete candidates: counter-threat cells that are NOT full coverers.
                for &del in &counters {
                    if full.contains(&del) {
                        continue;
                    }
                    tally.dispatch_pairs_checked += 1;
                    let vk = eval_child(state, keep, node.defender, false, scan_cap, horizon_slack);
                    let vd = eval_child(state, del, node.defender, false, scan_cap, horizon_slack);
                    if def_rank(vk) < def_rank(vd) {
                        // confirm wide
                        let vk2 = eval_child(
                            state,
                            keep,
                            node.defender,
                            true,
                            confirm_cap,
                            horizon_slack,
                        );
                        let vd2 =
                            eval_child(state, del, node.defender, true, confirm_cap, horizon_slack);
                        if def_rank(vk2) < def_rank(vd2) {
                            tally.dispatch_refutations += 1;
                            let row = format!(
                                "{{\"pattern\":\"DISPATCH_counterfork\",\"label\":\"REFUTED\",\"prov\":\"{prov}\",\"keep_full\":[{},{}],\"delete_counter\":[{},{}],\"threats\":{},\"val_keep\":\"{}\",\"val_delete\":\"{}\",\"cap\":{}}}",
                                keep.q, keep.r, del.q, del.r, node.threats.len(),
                                aval_name(vk2), aval_name(vd2), confirm_cap
                            );
                            records.push(row.clone());
                            refutations.push(row);
                            eprintln!(
                                "DOM_SCAN REFUTED DISPATCH prov={prov} keep_full=({},{}) del_counter=({},{}) vkeep={} vdel={}",
                                keep.q, keep.r, del.q, del.r, aval_name(vk2), aval_name(vd2)
                            );
                        }
                    }
                }
            }
        }
    }

    // Emit records.
    let mut all = String::new();
    for r in &records {
        all.push_str(r);
        all.push('\n');
    }
    all.push_str("\n# --- refutations ---\n");
    for r in &refutations {
        all.push_str(r);
        all.push('\n');
    }
    std::fs::write(&records_path, all).expect("write records");

    println!("==== DOM_SCAN SUMMARY ====");
    println!("candidate_nodes         = {}", nodes.len());
    println!("def_nodes (fired filter)= {}", tally.def_nodes);
    println!("  forced_loss_nodes     = {}", tally.forced_loss_nodes);
    println!("P2 firings (proven)     = {}", tally.p2_firings);
    println!("DRQ dead-equiv firings  = {}", tally.dead_equiv_firings);
    println!("  dead-equiv adjudicated= {dead_equiv_checked}  mismatches={dead_equiv_mismatch}");
    println!("DBD firings (all)       = {}", tally.dbd_firings);
    println!("DBD firings (guarded)   = {}", tally.dbd_guarded_firings);
    println!(
        "DBD branching saved(g)  = {}",
        tally.branching_saved_dbd_guarded
    );
    println!("DISPATCH forced nodes   = {}", tally.dispatch_forced_nodes);
    println!("  dispatch pairs checked= {}", tally.dispatch_pairs_checked);
    println!("  dispatch refutations  = {}", tally.dispatch_refutations);
    println!("COVERER multi nodes     = {}", tally.coverer_multi_nodes);
    println!("  coverer pairs checked = {}", tally.coverer_pairs_checked);
    println!("  coverer mismatches    = {}", tally.coverer_mismatch);
    println!(
        "  coverer P2 mismatches = {} (MUST be 0)",
        tally.coverer_p2_mismatch
    );
    println!("REFUTATIONS             = {}", refutations.len());
    println!("records_path            = {records_path}");
    println!(
        "elapsed_s               = {:.1}",
        t_start.elapsed().as_secs_f64()
    );
    println!("==========================");
}

fn dbd_row(
    prov: &str,
    node: &DefNode,
    f: &DbdFiring,
    vx: AVal,
    vy: AVal,
    label: &str,
    cap: u64,
) -> String {
    format!(
        "{{\"pattern\":\"DBD_superset\",\"label\":\"{label}\",\"prov\":\"{prov}\",\
         \"attacker\":{},\"threats\":{},\"mhs\":{},\
         \"keep\":[{},{}],\"delete\":[{},{}],\"kx\":{},\"ky\":{},\
         \"frontier_ok\":{},\"x_pure\":{},\"y_pure\":{},\"x_inert\":{},\
         \"val_keep\":\"{}\",\"val_delete\":\"{}\",\"cap\":{}}}",
        node.attacker.index(),
        node.threats.len(),
        node.min_hitting_set.map(|v| v as i64).unwrap_or(-1),
        f.keep.q,
        f.keep.r,
        f.delete.q,
        f.delete.r,
        f.kx,
        f.ky,
        f.frontier_ok,
        f.x_pure,
        f.y_pure,
        f.x_inert,
        aval_name(vx),
        aval_name(vy),
        cap
    )
}

// ==========================================================================
// DIRECTED — hand-scripted legal shapes (junction / counterfork / relay).
// ==========================================================================

#[test]
#[ignore = "dom-hunt directed constructions; --nocapture"]
fn dom_hunt_directed() {
    // Each construction is a legal placement script reaching a defender
    // SecondStone node; we then fire DBD-superset and adjudicate.  Constructions
    // are documented inline. If a script fails to reach the intended node it is
    // reported (not asserted) so the harness stays informative.
    let scan_cap: u64 = envn64("TSS_DOM_SCAN_CAP", 4000);
    let confirm_cap: u64 = envn64("TSS_DOM_CONFIRM_CAP", 80_000);
    let horizon: u32 = envn("TSS_DOM_HORIZON", 40) as u32;

    let scripts: Vec<(&str, Vec<(i16, i16)>)> = directed_scripts();
    for (name, moves) in scripts {
        let mut state = HexoState::new();
        let mut ok = true;
        for &(q, r) in &moves {
            match apply_placement(
                &mut state,
                Placement {
                    coord: HexCoord::new(q, r),
                },
            ) {
                Ok(res) => {
                    if res.outcome.is_some() {
                        println!("DIRECTED {name}: terminated early during script");
                        ok = false;
                        break;
                    }
                }
                Err(e) => {
                    println!("DIRECTED {name}: illegal move ({q},{r}): {e:?}");
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        let node = match classify_def_node(&state) {
            Some(n) => n,
            None => {
                println!(
                    "DIRECTED {name}: not a defensive SecondStone node (phase={:?} terminal={})",
                    state.phase(),
                    state.is_terminal()
                );
                continue;
            }
        };
        let dbd = detect_dbd_superset(&state, &node);
        println!(
            "DIRECTED {name}: def-node attacker={} threats={} mhs={:?} hitters={} dbd_firings={}",
            node.attacker.index(),
            node.threats.len(),
            node.min_hitting_set,
            node.hitters.len(),
            dbd.len()
        );
        for f in &dbd {
            let vx = eval_child(&state, f.keep, node.defender, true, confirm_cap, horizon);
            let vy = eval_child(&state, f.delete, node.defender, true, confirm_cap, horizon);
            let _ = scan_cap;
            let refuted = def_rank(vx) < def_rank(vy);
            println!(
                "  DBD keep=({},{}) del=({},{}) kx={} ky={} guards[fr={} xp={} yp={}] vkeep={} vdel={} {}",
                f.keep.q, f.keep.r, f.delete.q, f.delete.r, f.kx, f.ky,
                f.frontier_ok, f.x_pure, f.y_pure, aval_name(vx), aval_name(vy),
                if refuted { "<<< REFUTED" } else { "" }
            );
        }
    }
    let _ = status_name(ProofStatus::Win);
}

/// Directed scripts (filled in after the scan shows which shapes matter).
fn directed_scripts() -> Vec<(&'static str, Vec<(i16, i16)>)> {
    Vec::new()
}

// ==========================================================================
// VERIFY — re-solve a single flagged position at escalating caps to confirm a
// counterexample is stable (not a cap artifact). Env-driven:
//   TSS_DOM_VERIFY_HASH   game_hash
//   TSS_DOM_VERIFY_PREFIX placements to replay
//   TSS_DOM_VERIFY_A      "q,r" of cell A (the sound attacker-win child)
//   TSS_DOM_VERIFY_B      "q,r" of cell B (the conjectured defender-hold child)
// ==========================================================================

fn parse_cell(s: &str) -> HexCoord {
    let mut it = s.split(',').map(|t| t.trim().parse::<i16>().expect("i16"));
    HexCoord::new(it.next().unwrap(), it.next().unwrap())
}

// ==========================================================================
// B2 EXACT EXPERIMENT -- repaired Section 7 protocol.
//
// These runners deliberately stay in this #[cfg(test)] module.  The dry
// inventory discovers every corpus row/prefix at which a genuine defensive
// FirstStone node occurs.  The exact runner evaluates one frozen first action
// and one stopped depth per invocation.  Solving at P+u with d+1 plies is the
// exhaustive second-stone aggregation F_d(u): the defender remains the mover
// at SecondStone, so minimax ranges over every legal v (including cells newly
// legalized by u) before the d-ply completed-child horizon.
// ==========================================================================

fn coord_key(c: HexCoord) -> (i16, i16) {
    (c.q, c.r)
}

fn b2_h(state: &HexoState, node: &DefNode) -> Vec<HexCoord> {
    let mut h = full_coverers(state, node, &node.hitters);
    h.sort_by_key(|&c| coord_key(c));
    h
}

fn b2_pair_covers(state: &HexoState, node: &DefNode, a: HexCoord, b: HexCoord) -> bool {
    node.threats.iter().all(|&w| {
        let empties = window_empties(state, w);
        empties.contains(&a) || empties.contains(&b)
    })
}

fn b2_split_firsts(state: &HexoState, node: &DefNode, h: &[HexCoord]) -> Vec<HexCoord> {
    let mut split = BTreeSet::new();
    for (i, &a) in node.hitters.iter().enumerate() {
        if h.contains(&a) {
            continue;
        }
        for &b in node.hitters.iter().skip(i + 1) {
            if !h.contains(&b) && b2_pair_covers(state, node, a, b) {
                split.insert(coord_key(a));
                split.insert(coord_key(b));
            }
        }
    }
    split
        .into_iter()
        .map(|(q, r)| HexCoord::new(q, r))
        .collect()
}

fn b2_hitting_sets(state: &HexoState, node: &DefNode) -> Vec<(HexCoord, HexCoord)> {
    let mut pairs = Vec::new();
    for (i, &a) in node.hitters.iter().enumerate() {
        for &b in node.hitters.iter().skip(i + 1) {
            if b2_pair_covers(state, node, a, b) {
                pairs.push((a, b));
            }
        }
    }
    pairs
}

fn b2_position_dump(moves: &[(i16, i16)], prefix: usize) -> String {
    moves[..prefix]
        .iter()
        .map(|&(q, r)| format!("[{q},{r}]"))
        .collect::<Vec<_>>()
        .join(",")
}

fn b2_coord_dump(cells: &[HexCoord]) -> String {
    cells
        .iter()
        .map(|c| format!("[{},{}]", c.q, c.r))
        .collect::<Vec<_>>()
        .join(",")
}

fn b2_pair_dump(pairs: &[(HexCoord, HexCoord)]) -> String {
    pairs
        .iter()
        .map(|(a, b)| format!("[[{},{}],[{},{}]]", a.q, a.r, b.q, b.r))
        .collect::<Vec<_>>()
        .join(",")
}

struct B2ForcingGame {
    id: String,
    moves: Vec<(i16, i16)>,
}

fn b2_forcing_corpus() -> Vec<B2ForcingGame> {
    let text = include_str!("../corpus/forcing_corpus_moves.txt");
    let mut lines = text.lines();
    let mut games = Vec::new();
    while let Some(header) = lines.next() {
        if !header.starts_with("POS ") {
            continue;
        }
        let mut id = None;
        let mut nstones = None;
        for field in header.split_whitespace().skip(1) {
            let (key, value) = field.split_once('=').expect("forcing k=v field");
            match key {
                "id" => id = Some(value.to_string()),
                "nstones" => nstones = Some(value.parse::<usize>().expect("forcing nstones")),
                _ => {}
            }
        }
        let nstones = nstones.expect("forcing row nstones");
        let mut moves = Vec::with_capacity(nstones);
        for _ in 0..nstones {
            let mut fields = lines.next().expect("forcing stone").split_whitespace();
            moves.push((
                fields.next().unwrap().parse().unwrap(),
                fields.next().unwrap().parse().unwrap(),
            ));
        }
        assert_eq!(lines.next().map(str::trim), Some("END"));
        games.push(B2ForcingGame {
            id: id.expect("forcing row id"),
            moves,
        });
    }
    games
}

#[test]
#[ignore = "b=2 first occurrence in the checked-in 19-row forcing corpus"]
fn dom_hunt_b2_forcing_inventory() {
    let games = b2_forcing_corpus();
    assert_eq!(games.len(), 19, "forcing corpus row count changed");
    let mut rows_with_b2 = 0usize;
    let mut records = Vec::new();
    for game in &games {
        let mut state = HexoState::new();
        let mut first = None;
        let mut count = 0usize;
        for prefix in 0..=game.moves.len() {
            if let Some(node) = classify_def_node(&state) {
                if node.first_stone && matches!(node.min_hitting_set, Some(1 | 2)) {
                    count += 1;
                    if first.is_none() {
                        let h = b2_h(&state, &node);
                        let split = b2_split_firsts(&state, &node, &h);
                        let pairs = b2_hitting_sets(&state, &node);
                        first = Some((
                            prefix,
                            node.min_hitting_set.unwrap(),
                            node.legal.len(),
                            h.len(),
                            split.len(),
                            pairs.len(),
                        ));
                        records.push(format!(
                            "{{\"kind\":\"forcing-b2-first\",\"id\":\"{}\",\"prefix\":{},\"mhs\":{},\"legal\":{},\"h\":{},\"h_cells\":[{}],\"split_firsts\":{},\"split_cells\":[{}],\"hitting_sets\":{},\"hitting_set_cells\":[{}],\"position\":[{}]}}",
                            game.id,
                            prefix,
                            node.min_hitting_set.unwrap(),
                            node.legal.len(),
                            h.len(),
                            b2_coord_dump(&h),
                            split.len(),
                            b2_coord_dump(&split),
                            pairs.len(),
                            b2_pair_dump(&pairs),
                            b2_position_dump(&game.moves, prefix)
                        ));
                    }
                }
            }
            if prefix == game.moves.len() || state.is_terminal() {
                break;
            }
            let (q, r) = game.moves[prefix];
            apply_placement(
                &mut state,
                Placement {
                    coord: HexCoord::new(q, r),
                },
            )
            .expect("legal forcing-corpus replay");
        }
        if let Some((prefix, mhs, legal, h, split, pairs)) = first {
            rows_with_b2 += 1;
            println!(
                "B2_FORCING id={} b2_nodes={} first_prefix={} first_mhs={} first_legal={} first_h={} first_split={} first_hsets={}",
                game.id, count, prefix, mhs, legal, h, split, pairs
            );
        } else {
            println!("B2_FORCING id={} b2_nodes=0 first=NONE", game.id);
        }
    }
    if let Ok(path) = std::env::var("TSS_DOM_B2_FORCING_MANIFEST") {
        let mut text = records.join("\n");
        text.push('\n');
        std::fs::write(&path, text)
            .unwrap_or_else(|e| panic!("write forcing manifest {path}: {e}"));
    }
    println!(
        "B2_FORCING_SUMMARY rows={} rows_with_b2={} rows_without_b2={}",
        games.len(),
        rows_with_b2,
        games.len() - rows_with_b2
    );
}

#[test]
#[ignore = "b=2 repaired-protocol corpus inventory; --nocapture"]
fn dom_hunt_b2_inventory() {
    let games = load_corpus();
    let mut rows_with_b2 = 0usize;
    let mut total_b2 = 0usize;
    let mut total_k1 = 0usize;
    let mut total_k1_split = 0usize;
    let mut total_k2 = 0usize;
    let mut total_k2_multi = 0usize;
    let mut records = Vec::new();

    for game in games.iter().filter(|g| matches!(g.winner, -1 | 1)) {
        let mut state = HexoState::new();
        let mut row_nodes = Vec::new();
        for prefix in 0..game.moves.len() {
            if let Some(node) = classify_def_node(&state) {
                if node.first_stone && matches!(node.min_hitting_set, Some(1 | 2)) {
                    let h = b2_h(&state, &node);
                    let split = b2_split_firsts(&state, &node, &h);
                    let pairs = b2_hitting_sets(&state, &node);
                    let k1 = node.min_hitting_set == Some(1);
                    let k1_split = k1 && !split.is_empty();
                    let k2 = node.min_hitting_set == Some(2);
                    let k2_multi = k2 && pairs.len() >= 2;
                    total_b2 += 1;
                    total_k1 += usize::from(k1);
                    total_k1_split += usize::from(k1_split);
                    total_k2 += usize::from(k2);
                    total_k2_multi += usize::from(k2_multi);
                    let mut legal: Vec<_> = node.legal.iter().copied().collect();
                    legal.sort_by_key(|&c| coord_key(c));
                    let record = format!(
                        "{{\"kind\":\"b2-parent\",\"game_hash\":\"{}\",\"prefix\":{},\"player\":{},\"mhs\":{},\"threats\":{},\"legal\":{},\"hitters\":{},\"h\":{},\"h_cells\":[{}],\"split_firsts\":{},\"split_cells\":[{}],\"hitting_sets\":{},\"hitting_set_cells\":[{}],\"position\":[{}]}}",
                        game.game_hash,
                        prefix,
                        node.defender.index(),
                        node.min_hitting_set.unwrap(),
                        node.threats.len(),
                        legal.len(),
                        node.hitters.len(),
                        h.len(),
                        b2_coord_dump(&h),
                        split.len(),
                        b2_coord_dump(&split),
                        pairs.len(),
                        b2_pair_dump(&pairs),
                        b2_position_dump(&game.moves, prefix)
                    );
                    records.push(record);
                    row_nodes.push((
                        prefix,
                        node.min_hitting_set.unwrap(),
                        legal.len(),
                        h.len(),
                        split.len(),
                        pairs.len(),
                    ));
                }
            }

            let (q, r) = game.moves[prefix];
            let result = apply_placement(
                &mut state,
                Placement {
                    coord: HexCoord::new(q, r),
                },
            )
            .expect("legal corpus replay during b2 inventory");
            if result.outcome.is_some() {
                break;
            }
        }
        if let Some(first) = row_nodes.first() {
            rows_with_b2 += 1;
            println!(
                "B2_ROW hash={} nodes={} first_prefix={} first_mhs={} first_legal={} first_h={} first_split={} first_hsets={}",
                game.game_hash,
                row_nodes.len(),
                first.0,
                first.1,
                first.2,
                first.3,
                first.4,
                first.5
            );
        }
    }

    if let Ok(path) = std::env::var("TSS_DOM_B2_MANIFEST") {
        let mut text = records.join("\n");
        text.push('\n');
        std::fs::write(&path, text).unwrap_or_else(|e| panic!("write manifest {path}: {e}"));
        println!("B2_MANIFEST path={path} rows={}", records.len());
    }
    println!(
        "B2_INVENTORY corpus_rows={} b2_nodes={} k1={} k1_with_split={} k2={} k2_multi={} seed=7766554433221100",
        rows_with_b2, total_b2, total_k1, total_k1_split, total_k2, total_k2_multi
    );
}

#[test]
#[ignore = "DRQ/P2 eligible-pair manifest counts for repaired b2 controls"]
fn dom_hunt_b2_control_inventory() {
    #[derive(Clone)]
    struct AuditParent {
        legal: usize,
        hash: String,
        prefix: usize,
        state: HexoState,
    }
    fn retain_first_four(parents: &mut Vec<AuditParent>, item: AuditParent) {
        parents.push(item);
        parents.sort_by(|a, b| {
            (a.legal, a.hash.as_str(), a.prefix).cmp(&(b.legal, b.hash.as_str(), b.prefix))
        });
        parents.truncate(4);
    }

    let games = load_corpus();
    let mut k1 = Vec::new();
    let mut k2 = Vec::new();
    for game in games.iter().filter(|g| matches!(g.winner, -1 | 1)) {
        let mut state = HexoState::new();
        for prefix in 0..game.moves.len() {
            if let Some(node) = classify_def_node(&state) {
                if node.first_stone {
                    let h = b2_h(&state, &node);
                    if node.min_hitting_set == Some(1)
                        && !b2_split_firsts(&state, &node, &h).is_empty()
                    {
                        retain_first_four(
                            &mut k1,
                            AuditParent {
                                legal: node.legal.len(),
                                hash: game.game_hash.clone(),
                                prefix,
                                state: state.clone(),
                            },
                        );
                    } else if node.min_hitting_set == Some(2)
                        && b2_hitting_sets(&state, &node).len() >= 2
                    {
                        retain_first_four(
                            &mut k2,
                            AuditParent {
                                legal: node.legal.len(),
                                hash: game.game_hash.clone(),
                                prefix,
                                state: state.clone(),
                            },
                        );
                    }
                }
            }
            let (q, r) = game.moves[prefix];
            let result = apply_placement(
                &mut state,
                Placement {
                    coord: HexCoord::new(q, r),
                },
            )
            .expect("legal corpus replay during control inventory");
            if result.outcome.is_some() {
                break;
            }
        }
    }

    let mut drq_pairs = 0usize;
    for parent in &k1 {
        let node = classify_def_node(&parent.state).unwrap();
        let mut eligible_here = 0usize;
        for h in b2_h(&parent.state, &node) {
            let (after_h, winner) = child_after(&parent.state, h);
            assert!(winner.is_none());
            let mut legal = Vec::new();
            after_h.write_legal_moves(&mut legal);
            let dead = legal
                .into_iter()
                .filter(|&c| cell_dead(&after_h, c))
                .count();
            eligible_here += dead.saturating_mul(dead.saturating_sub(1)) / 2;
        }
        drq_pairs += eligible_here;
        println!(
            "B2_DRQ_CONTROL hash={} prefix={} eligible_pairs={}",
            parent.hash, parent.prefix, eligible_here
        );
    }

    let mut p2_pairs = 0usize;
    for parent in &k2 {
        let node = classify_def_node(&parent.state).unwrap();
        let mut eligible_here = 0usize;
        for &first in &node.hitters {
            let (after_first, winner) = child_after(&parent.state, first);
            if winner.is_none() {
                if let Some(second_node) = classify_def_node(&after_first) {
                    eligible_here += detect_p2(&after_first, &second_node).len();
                }
            }
        }
        p2_pairs += eligible_here;
        println!(
            "B2_P2_CONTROL hash={} prefix={} eligible_pairs={}",
            parent.hash, parent.prefix, eligible_here
        );
    }
    println!(
        "B2_CONTROL_SUMMARY k1_parents={} drq_eligible_pairs={} k2_parents={} p2_eligible_pairs={} drq_status={} p2_status={}",
        k1.len(),
        drq_pairs,
        k2.len(),
        p2_pairs,
        if drq_pairs == 0 { "NOT_TESTED" } else { "ELIGIBLE" },
        if p2_pairs == 0 { "NOT_TESTED" } else { "ELIGIBLE" }
    );
}

fn b2_fast_config() -> FastReferenceConfig {
    let tt_bytes_cap = envn64("TSS_REFERENCE_FAST_TT_BYTES", 512 << 20) as usize;
    let d6_canonical = std::env::var("TSS_REFERENCE_FAST_D6")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(true);
    FastReferenceConfig {
        tt_bytes_cap,
        d6_canonical,
        ordering_hint: FastOrderingHint::None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum B2Bounded {
    Complete(ProofStatus),
    Incomplete,
}

fn b2_stock_bounded(
    state: &mut HexoState,
    root: Player,
    plies_left: u32,
    nodes: &mut u64,
    node_cap: u64,
    deadline: std::time::Instant,
) -> B2Bounded {
    if *nodes >= node_cap || std::time::Instant::now() >= deadline {
        return B2Bounded::Incomplete;
    }
    *nodes = nodes.saturating_add(1);
    if let Some(winner) = tss_reference::direct_winner(state) {
        return B2Bounded::Complete(if winner == root {
            ProofStatus::Win
        } else {
            ProofStatus::Loss
        });
    }
    if plies_left == 0 {
        return B2Bounded::Complete(ProofStatus::Unknown);
    }
    let mut moves = tss_reference::legal_moves(state);
    if moves.is_empty() {
        return B2Bounded::Complete(ProofStatus::Unknown);
    }

    let mover = state.current_player();
    moves.sort_by_key(|&coord| {
        (
            std::cmp::Reverse(b2_direct_extension_length(state, mover, coord)),
            coord.q,
            coord.r,
        )
    });
    let maximizing = state.current_player() == root;
    let mut saw_unknown = false;
    let mut saw_incomplete = false;
    for coord in moves {
        let (_result, delta) = state
            .apply_with_delta(Placement { coord })
            .expect("stock legal enumerator produced an illegal placement");
        let child = b2_stock_bounded(state, root, plies_left - 1, nodes, node_cap, deadline);
        state.undo(delta);
        match child {
            B2Bounded::Complete(ProofStatus::Win) if maximizing => {
                return B2Bounded::Complete(ProofStatus::Win);
            }
            B2Bounded::Complete(ProofStatus::Loss) if !maximizing => {
                return B2Bounded::Complete(ProofStatus::Loss);
            }
            B2Bounded::Complete(ProofStatus::Unknown) => saw_unknown = true,
            B2Bounded::Incomplete => saw_incomplete = true,
            B2Bounded::Complete(_) => {}
        }
    }
    if saw_incomplete {
        B2Bounded::Incomplete
    } else if saw_unknown {
        B2Bounded::Complete(ProofStatus::Unknown)
    } else if maximizing {
        B2Bounded::Complete(ProofStatus::Loss)
    } else {
        B2Bounded::Complete(ProofStatus::Win)
    }
}

fn b2_offset_coord(start: HexCoord, dq: i32, dr: i32, offset: i32) -> Option<HexCoord> {
    Some(HexCoord {
        q: i16::try_from(i32::from(start.q) + dq * offset).ok()?,
        r: i16::try_from(i32::from(start.r) + dr * offset).ok()?,
    })
}

fn b2_direct_extension_length(state: &HexoState, player: Player, coord: HexCoord) -> u8 {
    const AXES: [(i32, i32); 3] = [(1, 0), (0, 1), (1, -1)];
    let mut best = 1u8;
    for (dq, dr) in AXES {
        let mut length = 1u8;
        for sign in [-1, 1] {
            for distance in 1..6 {
                let Some(cell) = b2_offset_coord(coord, dq * sign, dr * sign, distance) else {
                    break;
                };
                if state.board().get(cell) != Some(player) {
                    break;
                }
                length = length.saturating_add(1);
            }
        }
        best = best.max(length);
    }
    best
}

#[test]
#[ignore = "b=2 stock/fast attacker-Loss qualification; --nocapture"]
fn dom_hunt_b2_q0() {
    let games = load_corpus();
    let per_bucket = envn("TSS_DOM_B2_Q0_PER_BUCKET", 4);
    let node_cap = envn64("TSS_DOM_B2_Q0_NODE_CAP", 1_000_000);
    let deadline_ms = envn64("TSS_DOM_B2_DEADLINE_MS", 540_000);
    let overall_deadline = Instant::now() + std::time::Duration::from_millis(deadline_ms);
    let mut counts = [[0usize; 2]; 2];
    let mut rows = Vec::new();

    'games: for game in games.iter().filter(|g| matches!(g.winner, -1 | 1)) {
        let mut state = HexoState::new();
        for prefix in 0..game.moves.len() {
            if Instant::now() >= overall_deadline {
                break 'games;
            }
            let phase_index = match state.phase() {
                TurnPhase::FirstStone => Some(0usize),
                TurnPhase::SecondStone { .. } => Some(1usize),
                TurnPhase::Opening => None,
            };
            if let Some(phase_index) = phase_index {
                let player_index = state.current_player().index();
                if counts[player_index][phase_index] < per_bucket
                    && threats::analyze(&state).forced_loss()
                {
                    for depth in 1..=4u32 {
                        if counts[player_index][phase_index] >= per_bucket {
                            break;
                        }
                        let mut working = state.clone();
                        let mut nodes = 0u64;
                        let stock = b2_stock_bounded(
                            &mut working,
                            state.current_player(),
                            depth,
                            &mut nodes,
                            node_cap,
                            overall_deadline,
                        );
                        let phase = if phase_index == 0 {
                            "FirstStone"
                        } else {
                            "SecondStone"
                        };
                        match stock {
                            B2Bounded::Complete(ProofStatus::Loss) => {
                                let fast = tss_reference_fast::solve_for_player_until(
                                    &state,
                                    state.current_player(),
                                    depth,
                                    b2_fast_config(),
                                    overall_deadline,
                                );
                                match fast.status {
                                    Some(ProofStatus::Loss) => {
                                        let slot = counts[player_index][phase_index] + 1;
                                        counts[player_index][phase_index] += 1;
                                        let row = format!(
                                            "Q0_LOSS {{\"game_hash\":\"{}\",\"prefix\":{},\"position\":[{}],\"player\":{},\"phase\":\"{}\",\"slot\":{},\"depth\":{},\"stock_nodes\":{},\"fast_nodes\":{},\"fast_tt_hits\":{},\"classification\":\"QUALIFIED\"}}",
                                            game.game_hash,
                                            prefix,
                                            b2_position_dump(&game.moves, prefix),
                                            player_index,
                                            phase,
                                            slot,
                                            depth,
                                            nodes,
                                            fast.nodes,
                                            fast.tt_hits
                                        );
                                        println!("{row}");
                                        rows.push(row);
                                    }
                                    Some(status) => println!(
                                        "Q0_DISQUALIFIED hash={} prefix={} player={} phase={} depth={} stock=LOSS stock_nodes={} fast={} fast_nodes={} reason=FAST_MISMATCH",
                                        game.game_hash,
                                        prefix,
                                        player_index,
                                        phase,
                                        depth,
                                        nodes,
                                        status_name(status),
                                        fast.nodes
                                    ),
                                    None => println!(
                                        "Q0_UNQUALIFIABLE hash={} prefix={} player={} phase={} depth={} stock=LOSS stock_nodes={} fast_nodes={} wall_s={:.6} reason=FAST_DEADLINE",
                                        game.game_hash,
                                        prefix,
                                        player_index,
                                        phase,
                                        depth,
                                        nodes,
                                        fast.nodes,
                                        fast.elapsed.as_secs_f64()
                                    ),
                                }
                            }
                            B2Bounded::Complete(status) => println!(
                                "Q0_DISQUALIFIED hash={} prefix={} player={} phase={} depth={} stock={} stock_nodes={} reason=STOCK_NOT_LOSS",
                                game.game_hash,
                                prefix,
                                player_index,
                                phase,
                                depth,
                                status_name(status),
                                nodes
                            ),
                            B2Bounded::Incomplete => println!(
                                "Q0_UNQUALIFIABLE hash={} prefix={} player={} phase={} depth={} stock_nodes={} reason={}",
                                game.game_hash,
                                prefix,
                                player_index,
                                phase,
                                depth,
                                nodes,
                                if nodes >= node_cap {
                                    "STOCK_NODE_CAP"
                                } else {
                                    "STOCK_DEADLINE"
                                }
                            ),
                        }
                    }
                }
            }
            let (q, r) = game.moves[prefix];
            let result = apply_placement(
                &mut state,
                Placement {
                    coord: HexCoord::new(q, r),
                },
            )
            .expect("legal corpus replay during Q0");
            if result.outcome.is_some() {
                break;
            }
        }
        if counts.iter().flatten().all(|&n| n >= per_bucket) {
            break;
        }
    }
    println!(
        "Q0_SUMMARY p0_first={} p0_second={} p1_first={} p1_second={} required_each={} rows={} complete={}",
        counts[0][0],
        counts[0][1],
        counts[1][0],
        counts[1][1],
        per_bucket,
        rows.len(),
        counts.iter().flatten().all(|&n| n >= per_bucket)
    );
    for (player, phases) in counts.iter().enumerate() {
        for (phase_index, &completed) in phases.iter().enumerate() {
            for slot in (completed + 1)..=per_bucket {
                println!(
                    "Q0_MISSING player={} phase={} slot={} classification=GENUINELY_UNQUALIFIABLE reason=NO_QUALIFIED_ROW_BEFORE_45_MINUTE_DEADLINE",
                    player,
                    if phase_index == 0 {
                        "FirstStone"
                    } else {
                        "SecondStone"
                    },
                    slot
                );
            }
        }
    }
    if let Ok(path) = std::env::var("TSS_DOM_B2_Q0_MANIFEST") {
        let mut text = rows.join("\n");
        text.push('\n');
        std::fs::write(&path, text).unwrap_or_else(|e| panic!("write b2 q0 manifest {path}: {e}"));
        println!("Q0_MANIFEST path={path} rows={}", rows.len());
    }
    assert!(
        counts.iter().flatten().all(|&n| n >= per_bucket),
        "Q0 Loss qualification shortfall blocks use of b2 attacker-Loss rows"
    );
}

#[test]
#[ignore = "one exact b=2 F_d(first) case; --nocapture"]
fn dom_hunt_b2_exact() {
    let hash = std::env::var("TSS_DOM_B2_HASH").expect("TSS_DOM_B2_HASH");
    let prefix: usize = envn("TSS_DOM_B2_PREFIX", usize::MAX);
    let first = parse_cell(&std::env::var("TSS_DOM_B2_FIRST").expect("TSS_DOM_B2_FIRST"));
    let depth: u32 = envn("TSS_DOM_B2_DEPTH", 3) as u32;
    assert!(
        depth >= 3,
        "covered comparisons are discriminatory only at d>=3"
    );

    let games = load_corpus();
    let game = games
        .iter()
        .find(|g| g.game_hash == hash)
        .expect("b2 hash not found");
    let state = replay_prefix(&game.moves, prefix);
    let node = classify_def_node(&state).expect("not a defensive node");
    assert!(node.first_stone, "b2 parent must be FirstStone");
    assert_eq!(
        node.min_hitting_set,
        Some(1),
        "primary experiment is K1 spare pruning"
    );
    assert!(
        node.legal.contains(&first),
        "frozen first action is not legal"
    );
    assert!(
        node.hitters.contains(&first),
        "non-hitter first is P3-dominated by an H-first alias"
    );

    let h = b2_h(&state, &node);
    let split = b2_split_firsts(&state, &node, &h);
    let role = if h.contains(&first) {
        "H"
    } else if split.contains(&first) {
        "SPLIT"
    } else {
        "PARTIAL_NO_SPLIT"
    };
    let support_delta = new_support_cells(first, &node.support).len();
    let dead = cell_dead(&state, first);
    let quiet = touches_no_alive_for(&state, first, node.defender);
    let g3 = counter_threat_cells(&state, node.defender, &node.legal).contains(&first);

    let (after_first, winner) = child_after(&state, first);
    assert!(
        winner.is_none(),
        "!own_win_now b2 parent produced a terminal first stone"
    );
    assert!(matches!(after_first.phase(), TurnPhase::SecondStone { .. }));
    let config = b2_fast_config();
    let started = Instant::now();
    let result = tss_reference_fast::solve_for_player(
        &after_first,
        node.attacker,
        depth.saturating_add(1),
        config,
    );
    let wall = started.elapsed().as_secs_f64();
    println!(
        "B2_EXACT {{\"game_hash\":\"{}\",\"prefix\":{},\"position\":[{}],\"first\":[{},{}],\"role\":\"{}\",\"depth_after_pair\":{},\"status_for_attacker\":\"{}\",\"def_rank\":{},\"nodes\":{},\"tt_hits\":{},\"tt_entries\":{},\"tt_bytes\":{},\"tt_clears\":{},\"d6\":{},\"wall_s\":{:.6},\"dead\":{},\"quiet\":{},\"g3\":{},\"support_delta\":{},\"h_count\":{},\"split_first_count\":{}}}",
        hash,
        prefix,
        b2_position_dump(&game.moves, prefix),
        first.q,
        first.r,
        role,
        depth,
        status_name(result.status),
        match result.status { ProofStatus::Win => 0, ProofStatus::Unknown => 1, ProofStatus::Loss => 2 },
        result.nodes,
        result.tt_hits,
        result.tt_entries,
        result.tt_accounted_bytes,
        result.tt_clears,
        config.d6_canonical,
        wall,
        dead,
        quiet,
        g3,
        support_delta,
        h.len(),
        split.len()
    );
}

#[test]
#[ignore = "one exact completed b=2 macromove case; --nocapture"]
fn dom_hunt_b2_pair_exact() {
    let hash = std::env::var("TSS_DOM_B2_HASH").expect("TSS_DOM_B2_HASH");
    let prefix: usize = envn("TSS_DOM_B2_PREFIX", usize::MAX);
    let first = parse_cell(&std::env::var("TSS_DOM_B2_FIRST").expect("TSS_DOM_B2_FIRST"));
    let second = parse_cell(&std::env::var("TSS_DOM_B2_SECOND").expect("TSS_DOM_B2_SECOND"));
    let depth: u32 = envn("TSS_DOM_B2_DEPTH", 3) as u32;
    assert!(
        depth >= 3,
        "covered comparisons are discriminatory only at d>=3"
    );

    let games = load_corpus();
    let game = games
        .iter()
        .find(|g| g.game_hash == hash)
        .expect("b2 hash not found");
    let state = replay_prefix(&game.moves, prefix);
    let node = classify_def_node(&state).expect("not a defensive node");
    assert!(node.first_stone, "b2 parent must be FirstStone");
    assert!(
        node.legal.contains(&first),
        "first action is not legal at turn start"
    );

    let h = b2_h(&state, &node);
    let coverage = if h.contains(&first) || h.contains(&second) {
        "H_CONTAINING"
    } else {
        "SPLIT"
    };
    assert!(
        b2_pair_covers(&state, &node, first, second),
        "exact candidate must cover the complete initial threat family"
    );
    let (after_first, first_winner) = child_after(&state, first);
    assert!(
        first_winner.is_none(),
        "first action terminated unexpectedly"
    );
    let second_support = support_set(&after_first);
    let mut after_pair = after_first;
    let result = apply_placement(&mut after_pair, Placement { coord: second })
        .expect("second action is not legal after first");
    assert!(
        result.outcome.is_none(),
        "covered b2 pair terminated unexpectedly"
    );
    assert_eq!(after_pair.current_player(), node.attacker);
    assert_eq!(after_pair.phase(), TurnPhase::FirstStone);

    let first_delta = new_support_cells(first, &node.support).len();
    let second_delta = new_support_cells(second, &second_support).len();
    let config = b2_fast_config();
    let exact = tss_reference_fast::solve_for_player(&after_pair, node.attacker, depth, config);
    println!(
        "B2_PAIR {{\"game_hash\":\"{}\",\"prefix\":{},\"position\":[{}],\"pair\":[[{},{}],[{},{}]],\"coverage\":\"{}\",\"depth\":{},\"status_for_attacker\":\"{}\",\"def_rank\":{},\"nodes\":{},\"tt_hits\":{},\"tt_entries\":{},\"tt_bytes\":{},\"tt_clears\":{},\"d6\":{},\"first_support_delta\":{},\"second_support_delta\":{}}}",
        hash,
        prefix,
        b2_position_dump(&game.moves, prefix),
        first.q,
        first.r,
        second.q,
        second.r,
        coverage,
        depth,
        status_name(exact.status),
        match exact.status { ProofStatus::Win => 0, ProofStatus::Unknown => 1, ProofStatus::Loss => 2 },
        exact.nodes,
        exact.tt_hits,
        exact.tt_entries,
        exact.tt_accounted_bytes,
        exact.tt_clears,
        config.d6_canonical,
        first_delta,
        second_delta
    );
}

fn b2_adjudication_deadline() -> Instant {
    Instant::now() + std::time::Duration::from_millis(envn64("TSS_DOM_B2_DEADLINE_MS", 2_700_000))
}

fn b2_bounded_status(status: Option<ProofStatus>) -> (&'static str, i8) {
    match status {
        Some(ProofStatus::Loss) => ("LOSS", 2),
        Some(ProofStatus::Unknown) => ("UNKNOWN", 1),
        Some(ProofStatus::Win) => ("WIN", 0),
        None => ("INCOMPLETE", -1),
    }
}

#[test]
#[ignore = "45-minute adjudication: every hitter F_3 in four frozen K1 parents"]
fn dom_hunt_b2_adjudicate_f3() {
    // Split hitters precede H hitters so that a deadline still identifies the
    // smallest exactly completable subset. The row set itself is frozen.
    let cases = [
        ("32f44c499244b611", 9usize, HexCoord::new(-2, 1)),
        ("32f44c499244b611", 9, HexCoord::new(4, 1)),
        ("19b085e7aa9f6215", 9, HexCoord::new(-1, 0)),
        ("19b085e7aa9f6215", 9, HexCoord::new(5, 0)),
        ("498a61ae0b5cf4ef", 9, HexCoord::new(-2, 2)),
        ("498a61ae0b5cf4ef", 9, HexCoord::new(4, -4)),
        ("fd688f189544bf72", 9, HexCoord::new(-2, 0)),
        ("fd688f189544bf72", 9, HexCoord::new(4, 0)),
        ("32f44c499244b611", 9, HexCoord::new(2, 1)),
        ("19b085e7aa9f6215", 9, HexCoord::new(3, 0)),
        ("498a61ae0b5cf4ef", 9, HexCoord::new(2, -2)),
        ("fd688f189544bf72", 9, HexCoord::new(2, 0)),
    ];
    let games = load_corpus();
    let deadline = b2_adjudication_deadline();
    let started = Instant::now();
    let mut completed = 0usize;
    let mut total_nodes = 0u64;
    let mut rows = Vec::new();

    for (hash, prefix, first) in cases {
        let game = games
            .iter()
            .find(|game| game.game_hash == hash)
            .expect("frozen b2 audit game missing");
        let state = replay_prefix(&game.moves, prefix);
        let node = classify_def_node(&state).expect("frozen row is not defensive");
        assert_eq!(node.min_hitting_set, Some(1));
        assert!(node.hitters.contains(&first));
        let h = b2_h(&state, &node);
        let role = if h.contains(&first) { "H" } else { "SPLIT" };
        let (after_first, winner) = child_after(&state, first);
        assert!(winner.is_none());
        let exact = tss_reference_fast::solve_for_player_until(
            &after_first,
            node.attacker,
            4,
            b2_fast_config(),
            deadline,
        );
        let (status, rank) = b2_bounded_status(exact.status);
        completed += usize::from(exact.status.is_some());
        total_nodes = total_nodes.saturating_add(exact.nodes);
        let row = format!(
            "B2_ADJ_F3 {{\"game_hash\":\"{}\",\"prefix\":{},\"position\":[{}],\"first\":[{},{}],\"role\":\"{}\",\"depth_after_pair\":3,\"status_for_attacker\":\"{}\",\"def_rank\":{},\"nodes\":{},\"tt_hits\":{},\"tt_entries\":{},\"tt_bytes\":{},\"tt_clears\":{},\"wall_s\":{:.6}}}",
            hash,
            prefix,
            b2_position_dump(&game.moves, prefix),
            first.q,
            first.r,
            role,
            status,
            rank,
            exact.nodes,
            exact.tt_hits,
            exact.tt_entries,
            exact.tt_accounted_bytes,
            exact.tt_clears,
            exact.elapsed.as_secs_f64()
        );
        println!("{row}");
        rows.push(row);
    }
    println!(
        "B2_ADJ_F3_SUMMARY required={} completed={} incomplete={} nodes={} wall_s={:.6}",
        cases.len(),
        completed,
        cases.len() - completed,
        total_nodes,
        started.elapsed().as_secs_f64()
    );
    if let Ok(path) = std::env::var("TSS_DOM_B2_F3_RESULTS") {
        let mut text = rows.join("\n");
        text.push('\n');
        std::fs::write(&path, text).unwrap_or_else(|e| panic!("write b2 F3 results {path}: {e}"));
    }
}

#[test]
#[ignore = "45-minute adjudication: frozen d=4 quiet/frontier comparisons"]
fn dom_hunt_b2_adjudicate_d4() {
    let cases = [
        (
            "32f44c499244b611",
            9usize,
            HexCoord::new(-2, 1),
            HexCoord::new(4, 1),
        ),
        (
            "32f44c499244b611",
            9,
            HexCoord::new(2, 1),
            HexCoord::new(-2, 1),
        ),
        (
            "32f44c499244b611",
            9,
            HexCoord::new(2, 1),
            HexCoord::new(4, 1),
        ),
        (
            "19b085e7aa9f6215",
            9,
            HexCoord::new(-1, 0),
            HexCoord::new(5, 0),
        ),
        (
            "19b085e7aa9f6215",
            9,
            HexCoord::new(3, 0),
            HexCoord::new(-1, 0),
        ),
        (
            "498a61ae0b5cf4ef",
            9,
            HexCoord::new(-2, 2),
            HexCoord::new(4, -4),
        ),
        (
            "498a61ae0b5cf4ef",
            9,
            HexCoord::new(2, -2),
            HexCoord::new(-2, 2),
        ),
        (
            "fd688f189544bf72",
            9,
            HexCoord::new(-2, 0),
            HexCoord::new(4, 0),
        ),
        (
            "fd688f189544bf72",
            9,
            HexCoord::new(2, 0),
            HexCoord::new(-2, 0),
        ),
        (
            "d7e1b56c925b7f32",
            19,
            HexCoord::new(-1, 0),
            HexCoord::new(-2, 3),
        ),
        (
            "d7e1b56c925b7f32",
            19,
            HexCoord::new(-1, 0),
            HexCoord::new(-1, 2),
        ),
    ];
    let games = load_corpus();
    let deadline = b2_adjudication_deadline();
    let started = Instant::now();
    let mut completed = 0usize;
    let mut total_nodes = 0u64;
    let mut rows = Vec::new();

    for (hash, prefix, first, second) in cases {
        let game = games
            .iter()
            .find(|game| game.game_hash == hash)
            .expect("frozen b2 d4 game missing");
        let state = replay_prefix(&game.moves, prefix);
        let node = classify_def_node(&state).expect("frozen d4 row is not defensive");
        assert!(b2_pair_covers(&state, &node, first, second));
        let h = b2_h(&state, &node);
        let coverage = if h.contains(&first) || h.contains(&second) {
            "H_CONTAINING"
        } else {
            "SPLIT"
        };
        let (mut after_pair, first_winner) = child_after(&state, first);
        assert!(first_winner.is_none());
        let second_result = apply_placement(&mut after_pair, Placement { coord: second })
            .expect("frozen d4 second action is illegal");
        assert!(second_result.outcome.is_none());
        let exact = tss_reference_fast::solve_for_player_until(
            &after_pair,
            node.attacker,
            4,
            b2_fast_config(),
            deadline,
        );
        let (status, rank) = b2_bounded_status(exact.status);
        completed += usize::from(exact.status.is_some());
        total_nodes = total_nodes.saturating_add(exact.nodes);
        let row = format!(
            "B2_ADJ_D4 {{\"game_hash\":\"{}\",\"prefix\":{},\"position\":[{}],\"pair\":[[{},{}],[{},{}]],\"coverage\":\"{}\",\"depth\":4,\"status_for_attacker\":\"{}\",\"def_rank\":{},\"nodes\":{},\"tt_hits\":{},\"tt_entries\":{},\"tt_bytes\":{},\"tt_clears\":{},\"wall_s\":{:.6}}}",
            hash,
            prefix,
            b2_position_dump(&game.moves, prefix),
            first.q,
            first.r,
            second.q,
            second.r,
            coverage,
            status,
            rank,
            exact.nodes,
            exact.tt_hits,
            exact.tt_entries,
            exact.tt_accounted_bytes,
            exact.tt_clears,
            exact.elapsed.as_secs_f64()
        );
        println!("{row}");
        rows.push(row);
    }
    println!(
        "B2_ADJ_D4_SUMMARY required={} completed={} incomplete={} nodes={} wall_s={:.6}",
        cases.len(),
        completed,
        cases.len() - completed,
        total_nodes,
        started.elapsed().as_secs_f64()
    );
    if let Ok(path) = std::env::var("TSS_DOM_B2_D4_RESULTS") {
        let mut text = rows.join("\n");
        text.push('\n');
        std::fs::write(&path, text).unwrap_or_else(|e| panic!("write b2 d4 results {path}: {e}"));
    }
}

#[test]
#[ignore = "dom-hunt single-position verify; --nocapture"]
fn dom_hunt_verify() {
    let hash = std::env::var("TSS_DOM_VERIFY_HASH").unwrap_or_default();
    let prefix: usize = envn("TSS_DOM_VERIFY_PREFIX", 0);
    let a = parse_cell(&std::env::var("TSS_DOM_VERIFY_A").unwrap_or_else(|_| "0,0".into()));
    let b = parse_cell(&std::env::var("TSS_DOM_VERIFY_B").unwrap_or_else(|_| "0,0".into()));

    let games = load_corpus();
    let g = games
        .iter()
        .find(|g| g.game_hash == hash)
        .expect("hash not found");
    let state = replay_prefix(&g.moves, prefix);
    let node = classify_def_node(&state).expect("not a defensive node");
    println!(
        "VERIFY hash={hash} prefix={prefix} phase_first={} attacker={} threats={} mhs={:?} coverers_full={}",
        node.first_stone,
        node.attacker.index(),
        node.threats.len(),
        node.min_hitting_set,
        full_coverers(&state, &node, &node.hitters).len()
    );
    println!("  threat windows:");
    for &w in &node.threats {
        let e = state.board().windows().entry(w).unwrap();
        println!(
            "    axis={:?} start=({},{}) count_att={} empties={:?}",
            w.axis,
            w.start.q,
            w.start.r,
            e.count(node.attacker),
            e.empty_cells()
                .iter()
                .map(|c| (c.q, c.r))
                .collect::<Vec<_>>()
        );
    }
    let a_covers = node
        .threats
        .iter()
        .all(|&w| window_empties(&state, w).contains(&a));
    let b_covers = node
        .threats
        .iter()
        .all(|&w| window_empties(&state, w).contains(&b));
    println!(
        "  A=({},{}) covers_all={a_covers}   B=({},{}) covers_all={b_covers}",
        a.q, a.r, b.q, b.r
    );

    for &(wide, cap) in &[
        (false, 4_000u64),
        (true, 60_000),
        (true, 250_000),
        (true, 1_000_000),
    ] {
        let t = Instant::now();
        let va = eval_child(&state, a, node.defender, wide, cap, 60);
        let ta = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let vb = eval_child(&state, b, node.defender, wide, cap, 60);
        let tb = t.elapsed().as_secs_f64();
        println!(
            "  cap={cap:>8} wide={wide:<5} A->{:<12}({ta:.1}s)  B->{:<12}({tb:.1}s)",
            aval_name(va),
            aval_name(vb)
        );
    }
}

// ==========================================================================
// small env helpers
// ==========================================================================

fn envn(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
fn envn64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
