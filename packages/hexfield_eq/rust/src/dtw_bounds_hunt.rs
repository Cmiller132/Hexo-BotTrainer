//! Distance-to-win lower-bound hunt (NQ1) — DATA + candidate soundness, not proofs.
//!
//! Goal: find a cheap, position-computable function LB(P) with the eventual
//! theorem "the attacker cannot win within LB(P) placements against any
//! defense", measured in the same units the production leaf-solver deadline
//! uses (plies from the leaf).  If LB(P) exceeds the +8/+12-ply deadline, the
//! whole leaf solve is skippable for free.
//!
//! This module produces CANDIDATES with empirical soundness validation.  It
//! does not prove anything.  The soundness ORACLE is exact: the production WIN
//! solver strictly respects `semantic_horizon` (tss_solver.rs:2435/2644 refute
//! any completion beyond it), so horizon-laddering a solved WIN yields the
//! minimal horizon `h*` at which the win is provable — a sound UPPER bound on
//! the true forced-win distance dtw (dtw <= h*).  Any candidate with
//! LB(P) > h* on any solved win is REFUTED (LB > dtw).
//!
//! Units.  `plies` = single placements from the position (both players), the
//! unit of the deadline.  Hexo places 2 stones/turn; the winning placement is
//! always the ATTACKER's, so plies-to-win is a deterministic function of the
//! turn phase and the number of the attacker's own placements needed.
//!
//! Ignored helpers (run explicitly, --test-threads=1 --nocapture):
//!   * `dtw_pilot`               — tiny end-to-end sanity (oracle + LB agree on
//!                                  a handful of forcing-corpus / synthetic
//!                                  positions).
//!   * `dtw_soundness`           — every solved WIN (forcing corpus + leaf-width
//!                                  records + fresh corpus solves) vs every
//!                                  candidate LB; records violations.
//!   * `dtw_firerate`            — >=2000 sampled leaf FirstStone nodes: fire
//!                                  rate LB>h for h in {2,4,6,8,12,16} plies +
//!                                  eval cost (us/node warm).
//!
//! Corpus: HuggingFace `timmyburn/hexo-bootstrap-corpus`, local jsonl (verbatim
//! loader from tss_leaf_width_hunt.rs).  Replay: P0 opens at (0,0), then
//! alternating two-stone turns; apply each (q,r) in order.

use std::time::Instant;

use hexo_engine::{apply_placement, HexCoord, HexoState, Placement, Player, TurnPhase};

use crate::tss_core::{ProofStatus, SolveCaps, SolveGoal};
use crate::tss_solver::{TssSolver, WidthOptions};

type Cell = (i16, i16);
const AXES: [Cell; 3] = [(1, 0), (0, 1), (1, -1)];

// ==========================================================================
// Corpus parsing (stdlib only; verbatim from tss_leaf_width_hunt.rs).
// ==========================================================================

struct Game {
    game_hash: String,
    moves: Vec<(i16, i16)>,
    winner: i8, // +1 = Player0 wins, -1 = Player1 wins
}

fn corpus_path() -> String {
    std::env::var("TSS_DTW_CORPUS").unwrap_or_else(|_| {
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
    eprintln!("DTW_CORPUS path={path} games={}", games.len());
    games
}

/// Leaf-width record: we only need the replay prefix and the recorded winner.
struct LeafWidthRecord {
    game_hash: String,
    ply: u32,
    winner: i8,
    prefix: Vec<(i16, i16)>,
}

fn parse_lw_record(line: &str) -> Option<LeafWidthRecord> {
    if !line.contains("\"kind\":\"wide_only_win\"") {
        return None;
    }
    let winner = {
        let key = "\"winner\":";
        let m = line.find(key)?;
        let after = &line[m + key.len()..];
        let mut s = String::new();
        for ch in after.chars() {
            if ch == '-' || ch.is_ascii_digit() {
                s.push(ch);
            } else if !s.is_empty() {
                break;
            }
        }
        s.parse::<i8>().ok()?
    };
    let ply = {
        let key = "\"ply\":";
        let m = line.find(key)?;
        let after = &line[m + key.len()..];
        let mut s = String::new();
        for ch in after.chars() {
            if ch.is_ascii_digit() {
                s.push(ch);
            } else if !s.is_empty() {
                break;
            }
        }
        s.parse::<u32>().ok()?
    };
    let prefix = {
        let key = "\"prefix\":";
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
        parse_ints(arr)
            .chunks_exact(2)
            .map(|c| (c[0], c[1]))
            .collect::<Vec<_>>()
    };
    Some(LeafWidthRecord {
        game_hash: parse_hash(line),
        ply,
        winner,
        prefix,
    })
}

fn load_leaf_width_records() -> Vec<LeafWidthRecord> {
    let path = std::env::var("TSS_DTW_LW_RECORDS").unwrap_or_else(|_| {
        "E:/Hexo-BotTrainer-hexgt/.claude/worktrees/hunt-leaf-width/LEAF_WIDTH_RECORDS.jsonl"
            .to_string()
    });
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("DTW_LW skip (read {path}: {e})");
            return Vec::new();
        }
    };
    let recs: Vec<LeafWidthRecord> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(parse_lw_record)
        .collect();
    eprintln!("DTW_LW path={path} wide_only_win_records={}", recs.len());
    recs
}

// ==========================================================================
// Forcing corpus loader (verbatim structure from tss_corpus.rs).
// ==========================================================================

struct ForcingPos {
    id: String,
    expect_win: bool,
    state: HexoState,
}

fn load_forcing_corpus() -> Vec<ForcingPos> {
    let path = std::env::var("TSS_CORPUS_FILE").unwrap_or_else(|_| {
        format!(
            "{}/rust/corpus/forcing_corpus_moves.txt",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("DTW_FORCING skip (read {path}: {e})");
            return Vec::new();
        }
    };
    let mut out = Vec::new();
    let mut lines = text.lines();
    while let Some(header) = lines.next() {
        let header = header.trim();
        if header.is_empty() {
            continue;
        }
        assert!(header.starts_with("POS "), "bad header: {header}");
        let mut id = String::new();
        let mut expect = String::new();
        let mut nstones = 0usize;
        for tok in header.split_whitespace().skip(1) {
            let (k, v) = tok.split_once('=').expect("k=v token");
            match k {
                "id" => id = v.to_string(),
                "expect" => expect = v.to_string(),
                "nstones" => nstones = v.parse().unwrap(),
                _ => {}
            }
        }
        let mut state = HexoState::new();
        for _ in 0..nstones {
            let line = lines.next().expect("stone line");
            let mut it = line.split_whitespace();
            let q: i16 = it.next().unwrap().parse().unwrap();
            let r: i16 = it.next().unwrap().parse().unwrap();
            apply_placement(&mut state, Placement { coord: HexCoord { q, r } })
                .unwrap_or_else(|e| panic!("{id}: illegal replay at ({q},{r}): {e:?}"));
        }
        assert_eq!(lines.next().map(str::trim), Some("END"), "{id}: missing END");
        out.push(ForcingPos {
            id,
            expect_win: expect == "WIN",
            state,
        });
    }
    eprintln!("DTW_FORCING path={path} positions={}", out.len());
    out
}

// ==========================================================================
// Deterministic RNG.
// ==========================================================================

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
// Window census: for player `me`, every length-6 lattice window that is ALIVE
// (no opponent stone) and touched (>=1 me stone), as (cnt, empties).
// ==========================================================================

struct AliveWindow {
    cnt: u8,
    empties: Vec<Cell>,
}

fn stones_of(state: &HexoState, me: Player) -> (std::collections::BTreeSet<Cell>, std::collections::BTreeSet<Cell>) {
    use std::collections::BTreeSet;
    let mut mine: BTreeSet<Cell> = BTreeSet::new();
    let mut opp: BTreeSet<Cell> = BTreeSet::new();
    for &c in state.board().occupied_cells() {
        match state.board().get(c) {
            Some(p) if p == me => {
                mine.insert((c.q, c.r));
            }
            Some(_) => {
                opp.insert((c.q, c.r));
            }
            None => {}
        }
    }
    (mine, opp)
}

/// Every alive (opp-free) length-6 window with >=1 me stone.  Deduped by
/// (axis, start).  `empties` are the window cells with no stone at all.
fn alive_windows(mine: &std::collections::BTreeSet<Cell>, opp: &std::collections::BTreeSet<Cell>) -> Vec<AliveWindow> {
    use std::collections::BTreeSet;
    let mut seen: BTreeSet<(u8, i16, i16)> = BTreeSet::new();
    let mut out = Vec::new();
    for &a in mine {
        for (ax, &v) in AXES.iter().enumerate() {
            for k in 0..6i16 {
                let start = (a.0 - k * v.0, a.1 - k * v.1);
                if !seen.insert((ax as u8, start.0, start.1)) {
                    continue;
                }
                let mut cnt = 0u8;
                let mut has_opp = false;
                let mut empties: Vec<Cell> = Vec::new();
                for j in 0..6i16 {
                    let c = (start.0 + j * v.0, start.1 + j * v.1);
                    if opp.contains(&c) {
                        has_opp = true;
                        break;
                    }
                    if mine.contains(&c) {
                        cnt += 1;
                    } else {
                        empties.push(c);
                    }
                }
                if !has_opp && cnt >= 1 {
                    out.push(AliveWindow { cnt, empties });
                }
            }
        }
    }
    out
}

/// Max cnt over alive windows (0 if none).
fn max_cnt(wins: &[AliveWindow]) -> u8 {
    wins.iter().map(|w| w.cnt).max().unwrap_or(0)
}

// ==========================================================================
// Turn-structure ply map.  The winning placement is always the attacker's
// own, so the ply of the attacker's m-th future placement is fixed by the
// current (player, phase) and the strict two-stone turn alternation.
// ==========================================================================

/// Owner of each future ply 1..=len, given who is to move and the phase.
fn owner_sequence(cp: Player, phase: TurnPhase, len: usize) -> Vec<Player> {
    let mut owners = Vec::with_capacity(len);
    let mut player = cp;
    // Remaining placements THIS player owns before the turn passes.
    let mut rem = match phase {
        TurnPhase::FirstStone => 2,
        TurnPhase::SecondStone { .. } => 1,
        TurnPhase::Opening => 1,
    };
    while owners.len() < len {
        owners.push(player);
        rem -= 1;
        if rem == 0 {
            player = player.other();
            rem = 2;
        }
    }
    owners
}

/// Ply index (1-based) of `target`'s m-th future placement.  m>=1.
fn ply_of_mth(cp: Player, phase: TurnPhase, target: Player, m: u32) -> u32 {
    if m == 0 {
        return 0;
    }
    let owners = owner_sequence(cp, phase, 64);
    let mut seen = 0u32;
    for (i, &p) in owners.iter().enumerate() {
        if p == target {
            seen += 1;
            if seen == m {
                return (i + 1) as u32;
            }
        }
    }
    u32::MAX
}

// ==========================================================================
// LB candidates.  All return plies-from-position (the deadline unit) for the
// designated attacker `me` (who must be able to move eventually).  A return of
// 0 means "no lower bound / already may win now".
// ==========================================================================

/// PROVEN floor.  Single-window fill: any attacker win completes some alive
/// window W; that needs 6-cnt(W) of the attacker's OWN placements; the fastest
/// is 6 - max_cnt.  The attacker's m-th own placement lands no earlier than
/// `ply_of_mth`, so no attacker win before that ply, against ANY defense
/// (defense only delays).  Sound; capped near 10-12 plies (a fresh window is
/// 6 attacker placements).
fn lb0_attacker_placements(wins: &[AliveWindow]) -> u32 {
    let c = max_cnt(wins) as u32;
    6u32.saturating_sub(c).max(1)
}

fn lb0_plies(state: &HexoState, me: Player) -> u32 {
    let (mine, opp) = stones_of(state, me);
    if mine.is_empty() {
        // must build a fresh window: 6 own placements.
        return ply_of_mth(state.current_player(), state.phase(), me, 6);
    }
    let wins = alive_windows(&mine, &opp);
    let m = lb0_attacker_placements(&wins);
    ply_of_mth(state.current_player(), state.phase(), me, m)
}

/// HEURISTIC candidate "block+1": if the attacker's fastest window is slow
/// (cnt<=3, so filling it spans >=2 of the attacker's turns and the defender
/// gets to block it), require one extra attacker placement.  Motivated by
/// "a lone slow threat is blocked".  Soundness DUBIOUS (overlapping clusters
/// can co-advance faster) — tested, not assumed.
fn lb_block1_plies(state: &HexoState, me: Player) -> u32 {
    let (mine, opp) = stones_of(state, me);
    if mine.is_empty() {
        return ply_of_mth(state.current_player(), state.phase(), me, 6);
    }
    let wins = alive_windows(&mine, &opp);
    let c = max_cnt(&wins) as u32;
    let mut m = 6u32.saturating_sub(c).max(1);
    if c <= 3 {
        m += 1; // one forced defender block on the fastest window
    }
    ply_of_mth(state.current_player(), state.phase(), me, m.min(6))
}

/// HEURISTIC candidate "triple-threat escalation": a forced win needs, at the
/// pre-win defender node, an immediate-threat family with min hitting set >= 3
/// (defender blocks 2/turn).  Estimate the attacker placements to build 3
/// alive windows to count-5 with distinct completion cells, then +1 to
/// complete.  Greedy over the census (distinct empty-cell sets).  Soundness
/// DUBIOUS (count-4 forks win in one turn without 3 count-5s; overlaps) —
/// tested, not assumed.
fn lb_triple_plies(state: &HexoState, me: Player) -> u32 {
    let (mine, opp) = stones_of(state, me);
    let wins = alive_windows(&mine, &opp);
    // Sort alive windows by cnt desc; greedily take up to 3 with distinct
    // completion-cell sets (empties), summing fills-to-5.
    let mut idx: Vec<usize> = (0..wins.len()).collect();
    idx.sort_by(|&a, &b| wins[b].cnt.cmp(&wins[a].cnt));
    let mut chosen_empty_sets: Vec<std::collections::BTreeSet<Cell>> = Vec::new();
    let mut fills = 0u32;
    for &i in &idx {
        if chosen_empty_sets.len() == 3 {
            break;
        }
        let eset: std::collections::BTreeSet<Cell> = wins[i].empties.iter().copied().collect();
        // require a completion cell distinct from all already-chosen windows'
        // empties (so the three threats have >=3 distinct empties overall).
        if chosen_empty_sets.iter().any(|s| s == &eset) {
            continue;
        }
        chosen_empty_sets.push(eset);
        fills += 5u32.saturating_sub(wins[i].cnt as u32);
    }
    // pad missing windows as fresh count-5 builds (5 fills each).
    while chosen_empty_sets.len() < 3 {
        chosen_empty_sets.push(std::collections::BTreeSet::new());
        fills += 5;
    }
    let m = (fills + 1).min(6 * 3); // +1 to complete a surviving threat
    ply_of_mth(state.current_player(), state.phase(), me, m.min(6).max(1))
}

// ==========================================================================
// Ground-truth dtw oracle via horizon ladder.
// ==========================================================================

const ORACLE_NODE_CAP: u64 = 1_500_000;
const ORACLE_TT_BYTES: usize = 256 << 20;

/// Ladder the WIN solver over semantic_horizon = base+h for h=1..=hmax.
/// Returns (h*, exact) where h* is the minimal horizon that proves the
/// side-to-move's win (a sound upper bound on dtw_plies), and `exact` is true
/// iff the h*-1 solve exhausted (UNKNOWN, nodes<cap) so dtw == h* precisely.
/// None if no WIN within hmax (too deep for this cap).
fn dtw_oracle(state: &HexoState, hmax: u32) -> Option<(u32, bool)> {
    let base = state.placements_made();
    let mut prev_exhausted = true; // horizon 0: no win in 0 plies, trivially exhausted.
    for h in 1..=hmax {
        let caps = SolveCaps {
            node_cap: ORACLE_NODE_CAP,
            tt_bytes_cap: ORACLE_TT_BYTES,
            semantic_horizon: base + h,
        };
        let mut solver = TssSolver::default();
        solver.set_width_options(WidthOptions::vcf_pair_complete());
        let r = solver.solve_goal(state, &caps, SolveGoal::Win);
        match r.status {
            ProofStatus::Win => return Some((h, prev_exhausted)),
            _ => {
                prev_exhausted = r.stats.nodes < ORACLE_NODE_CAP;
            }
        }
    }
    None
}

// ==========================================================================
// ONE-LINE TWO-GAP LEMMA — exhaustive, engine-free.
//
// Reduction (why 1D suffices): a window reaching count-5 (resp. count-4) after
// one attacker pair from a root with maxcnt<=3 (resp. <=2) must contain BOTH
// placed stones (pre-count <= maxcnt, +2 needed).  Two distinct cells lie on
// at most one common axis line, so ALL such windows sit on that single line,
// and only that line's cells matter.  Off-line stones affect only windows not
// containing both placements.  Cells outside [x-5, y+5] cannot belong to a
// window containing both x and y.
//
// Claim A (maxcnt<=3): the post-pair count-5 family has <= 2 distinct empty
// cells — two defender placements service it (kill every count-5).
// Claim B (maxcnt<=2): the post-pair count-4 family has min hitting set <= 2
// — two defender placements put a stone in every count-4 window.
//
// Verified exhaustively over every stone configuration of the relevant span
// satisfying the root window cap, for every placement distance d = 1..=5.
// ==========================================================================

#[test]
#[ignore = "exhaustive one-line two-gap lemma; run with --nocapture"]
fn dtw_line_lemma() {
    // Cells are integers on a line, TERNARY state: 0 empty, 1 attacker,
    // 2 defender.  Placements at fixed x=0, y=d (both empty pre-move).
    // Relevant span: [-5, d+5].  Root cap: every DEFENDER-FREE window inside
    // the span has <= k_cap attacker stones (dead windows are unconstrained,
    // matching the real maxcnt definition).  Families range over defender-free
    // windows containing both 0 and d.
    for (k_cap, claim) in [(3u32, "A_count5_empties<=2"), (2u32, "B_count4_hitset<=2")] {
        let mut configs_checked = 0u64;
        let mut fams_seen = 0u64;
        let mut worst_empties = 0usize;
        let mut worst_hitset = 0usize;
        for d in 1..=5i32 {
            let lo = -5i32;
            let hi = d + 5; // inclusive
            let ncells = (hi - lo + 1) as usize; // d+11 <= 16
            // free cells = span minus {0, d} (those must be empty pre-move).
            let free: Vec<i32> = (lo..=hi).filter(|&c| c != 0 && c != d).collect();
            let nfree = free.len();
            assert!(nfree <= 14, "span too large");
            let total: u64 = 3u64.pow(nfree as u32);
            for code in 0..total {
                // ternary decode
                let mut cell = vec![0u8; ncells]; // 0 empty / 1 att / 2 def
                let mut c = code;
                for &fc in &free {
                    cell[(fc - lo) as usize] = (c % 3) as u8;
                    c /= 3;
                }
                // root cap: every defender-free window <= k_cap attacker stones.
                let mut ok = true;
                for s in lo..=(hi - 5) {
                    let mut acnt = 0u32;
                    let mut has_def = false;
                    for cc in s..s + 6 {
                        match cell[(cc - lo) as usize] {
                            1 => acnt += 1,
                            2 => {
                                has_def = true;
                                break;
                            }
                            _ => {}
                        }
                    }
                    if !has_def && acnt > k_cap {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    continue;
                }
                configs_checked += 1;
                // place attacker at x=0, y=d.
                cell[(0 - lo) as usize] = 1;
                cell[(d - lo) as usize] = 1;
                // families over defender-free windows containing both 0 and d.
                let target = if k_cap == 3 { 5u32 } else { 4u32 };
                let mut fam_empties: Vec<Vec<i32>> = Vec::new();
                for s in (d - 5).max(lo)..=0 {
                    if s + 5 > hi {
                        continue;
                    }
                    let mut acnt = 0u32;
                    let mut has_def = false;
                    let mut empties: Vec<i32> = Vec::new();
                    for cc in s..s + 6 {
                        match cell[(cc - lo) as usize] {
                            1 => acnt += 1,
                            2 => {
                                has_def = true;
                                break;
                            }
                            _ => empties.push(cc),
                        }
                    }
                    if !has_def && acnt == target {
                        fam_empties.push(empties);
                    }
                }
                if fam_empties.is_empty() {
                    continue;
                }
                fams_seen += 1;
                if k_cap == 3 {
                    // Claim A: distinct empties across all count-5s <= 2.
                    let mut distinct: Vec<i32> =
                        fam_empties.iter().flatten().copied().collect();
                    distinct.sort_unstable();
                    distinct.dedup();
                    worst_empties = worst_empties.max(distinct.len());
                    assert!(
                        distinct.len() <= 2,
                        "CLAIM A REFUTED: d={d} code={code} count5 empties {distinct:?}"
                    );
                } else {
                    // Claim B: min hitting set over 2-empty families <= 2.
                    let mut cells: Vec<i32> = fam_empties.iter().flatten().copied().collect();
                    cells.sort_unstable();
                    cells.dedup();
                    let hits = |a: i32, b: i32| {
                        fam_empties
                            .iter()
                            .all(|es| es.contains(&a) || es.contains(&b))
                    };
                    let mut hs = 3;
                    'outer: for i in 0..cells.len() {
                        for j in i..cells.len() {
                            if hits(cells[i], cells[j]) {
                                hs = if i == j { 1 } else { 2 };
                                break 'outer;
                            }
                        }
                    }
                    worst_hitset = worst_hitset.max(hs);
                    assert!(
                        hs <= 2,
                        "CLAIM B REFUTED: d={d} code={code} fam={fam_empties:?}"
                    );
                }
            }
        }
        println!(
            "LINE_LEMMA claim={claim} configs_checked={configs_checked} families_seen={fams_seen} worst_empties={worst_empties} worst_hitset={worst_hitset} verdict=EXHAUSTIVELY-VERIFIED"
        );
    }
    println!("LINE_LEMMA_DONE");
}

// ==========================================================================
// Replay helper.
// ==========================================================================

fn replay(moves: &[(i16, i16)]) -> Option<HexoState> {
    let mut state = HexoState::new();
    for &(q, r) in moves {
        if state.is_terminal() {
            return None;
        }
        apply_placement(&mut state, Placement { coord: HexCoord::new(q, r) }).ok()?;
    }
    if state.is_terminal() {
        return None;
    }
    Some(state)
}

fn prefix_json(moves: &[(i16, i16)]) -> String {
    let mut s = String::from("[");
    for (i, &(q, r)) in moves.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("[{q},{r}]"));
    }
    s.push(']');
    s
}

// The candidate registry: (name, is_proven, fn).
type LbFn = fn(&HexoState, Player) -> u32;
fn candidates() -> Vec<(&'static str, bool, LbFn)> {
    vec![
        ("lb0_single_window", true, lb0_plies as LbFn),
        ("lb_block1_heur", false, lb_block1_plies as LbFn),
        ("lb_triple_heur", false, lb_triple_plies as LbFn),
    ]
}

// ==========================================================================
// PILOT — tiny sanity end-to-end.
// ==========================================================================

#[test]
#[ignore = "dtw pilot; run with --nocapture --test-threads=1"]
fn dtw_pilot() {
    // Ply-map sanity: FirstStone, attacker = side to move.
    let s = HexoState::new();
    // opening is Player0 at (0,0); build a small deep-ish position.
    let mut st = s;
    for &(q, r) in &[(0, 0), (3, 0), (0, 1), (3, 1), (0, 2)] {
        apply_placement(&mut st, Placement { coord: HexCoord::new(q, r) }).unwrap();
    }
    println!(
        "PILOT phase={:?} cp={:?} ply1@FirstStone maps: m1={} m2={} m3={} m6={}",
        st.phase(),
        st.current_player(),
        ply_of_mth(Player::Player0, TurnPhase::FirstStone, Player::Player0, 1),
        ply_of_mth(Player::Player0, TurnPhase::FirstStone, Player::Player0, 2),
        ply_of_mth(Player::Player0, TurnPhase::FirstStone, Player::Player0, 3),
        ply_of_mth(Player::Player0, TurnPhase::FirstStone, Player::Player0, 6),
    );
    // Expected: m1=1 m2=2 m3=5 m6=10 (FirstStone attacker).
    assert_eq!(ply_of_mth(Player::Player0, TurnPhase::FirstStone, Player::Player0, 3), 5);
    assert_eq!(ply_of_mth(Player::Player0, TurnPhase::FirstStone, Player::Player0, 6), 10);
    // SecondStone attacker: m6 -> ply 12.
    let sec = TurnPhase::SecondStone { first: HexCoord::new(0, 0) };
    assert_eq!(ply_of_mth(Player::Player0, sec, Player::Player0, 6), 12);
    assert_eq!(ply_of_mth(Player::Player0, sec, Player::Player0, 1), 1);
    assert_eq!(ply_of_mth(Player::Player0, sec, Player::Player0, 2), 4);

    // Forcing corpus: oracle + LB on the WIN entries (small hmax pilot).
    let forcing = load_forcing_corpus();
    let hmax: u32 = std::env::var("DTW_PILOT_HMAX").ok().and_then(|v| v.parse().ok()).unwrap_or(18);
    let mut shown = 0;
    for pos in &forcing {
        if !pos.expect_win {
            continue;
        }
        let me = pos.state.current_player();
        let lb0 = lb0_plies(&pos.state, me);
        let t0 = Instant::now();
        let oracle = dtw_oracle(&pos.state, hmax);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        println!(
            "PILOT_FORCING id={} cp={:?} maxcnt={} lb0_plies={lb0} oracle={:?} ({ms:.0}ms)",
            pos.id,
            me,
            {
                let (mine, opp) = stones_of(&pos.state, me);
                max_cnt(&alive_windows(&mine, &opp))
            },
            oracle,
        );
        if let Some((h, _)) = oracle {
            assert!(lb0 <= h, "PILOT lb0 {lb0} > oracle {h} on {}", pos.id);
        }
        shown += 1;
        if shown >= 6 {
            break;
        }
    }
    println!("PILOT_DONE shown={shown}");
}

// ==========================================================================
// SOUNDNESS — every solved WIN vs every candidate LB.
// ==========================================================================

struct WinSample {
    source: &'static str,
    tag: String,
    moves: Vec<(i16, i16)>, // replay prefix (empty for forcing-corpus states carried separately)
    state: HexoState,
}

#[test]
#[ignore = "dtw candidate soundness vs solved wins; --test-threads=1 --nocapture"]
fn dtw_soundness() {
    // Every candidate LB is <= 12 plies (it maps m<=6 through ply_of_mth), so a
    // violation requires dtw < 12.  Laddering to hmax=14 catches every possible
    // violation; positions unresolved by then have dtw>14 and are auto-safe.
    let hmax: u32 = std::env::var("DTW_HMAX").ok().and_then(|v| v.parse().ok()).unwrap_or(14);
    let fresh_n: usize = std::env::var("DTW_FRESH_N").ok().and_then(|v| v.parse().ok()).unwrap_or(250);
    let fresh_cap: u64 = std::env::var("DTW_FRESH_CAP").ok().and_then(|v| v.parse().ok()).unwrap_or(30_000);
    let fresh_tried_cap: usize = std::env::var("DTW_FRESH_TRIED_CAP").ok().and_then(|v| v.parse().ok()).unwrap_or(9000);
    let seed: u64 = std::env::var("DTW_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(0x51ED_C0DE_2026_0716);
    let out_path = std::env::var("DTW_SOUNDNESS_OUT").unwrap_or_else(|_| {
        "E:/Hexo-BotTrainer-hexgt/.claude/worktrees/hunt-dtw-bounds/DTW_SOUNDNESS.jsonl".to_string()
    });

    let cands = candidates();
    let mut samples: Vec<WinSample> = Vec::new();

    // Source 1: forcing corpus WIN entries.
    for pos in load_forcing_corpus() {
        if pos.expect_win {
            samples.push(WinSample {
                source: "forcing",
                tag: pos.id.clone(),
                moves: Vec::new(),
                state: pos.state,
            });
        }
    }

    // Source 2: leaf-width wide_only_win records (replay the prefix).
    for rec in load_leaf_width_records() {
        if let Some(state) = replay(&rec.prefix) {
            // These are attacker-to-move FirstStone nodes; the side to move is
            // the winner (wide proved a WIN for the mover).
            samples.push(WinSample {
                source: "leafwidth",
                tag: format!("{}@{}", rec.game_hash, rec.ply),
                moves: rec.prefix.clone(),
                state,
            });
            let _ = rec.winner;
        }
    }

    // Source 3: fresh corpus solves — sample FirstStone nodes, solve, keep WINs.
    let games = load_corpus();
    let mut cands_nodes: Vec<(u32, u32)> = Vec::new(); // (game_idx, prefix_len)
    for (gi, g) in games.iter().enumerate() {
        if g.winner != 1 && g.winner != -1 {
            continue;
        }
        let mut state = HexoState::new();
        for (i, &(q, r)) in g.moves.iter().enumerate() {
            if state.is_terminal() {
                break;
            }
            if matches!(state.phase(), TurnPhase::FirstStone) {
                cands_nodes.push((gi as u32, i as u32));
            }
            apply_placement(&mut state, Placement { coord: HexCoord::new(q, r) }).expect("legal");
        }
    }
    let mut idx: Vec<usize> = (0..cands_nodes.len()).collect();
    let mut rng = XorShift(seed | 1);
    for i in (1..idx.len()).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        idx.swap(i, j);
    }
    let mut fresh_found = 0usize;
    let mut fresh_tried = 0usize;
    for &k in &idx {
        if fresh_found >= fresh_n || fresh_tried >= fresh_tried_cap {
            break;
        }
        fresh_tried += 1;
        if fresh_tried % 500 == 0 {
            eprintln!("DTW_FRESH scan tried={fresh_tried} found={fresh_found} ram={:.1}", free_ram_gb());
        }
        let (gi, pl) = cands_nodes[k];
        let g = &games[gi as usize];
        let moves: Vec<(i16, i16)> = g.moves[..pl as usize].to_vec();
        let Some(state) = replay(&moves) else { continue };
        // quick screen: solve for mover WIN at fresh_cap; horizon base+hmax so
        // we only keep shallow wins (dtw<=hmax) — exactly the tight tests.
        let caps = SolveCaps {
            node_cap: fresh_cap,
            tt_bytes_cap: ORACLE_TT_BYTES,
            semantic_horizon: state.placements_made() + hmax,
        };
        let mut solver = TssSolver::default();
        solver.set_width_options(WidthOptions::vcf_pair_complete());
        let r = solver.solve_goal(&state, &caps, SolveGoal::Win);
        if r.status == ProofStatus::Win {
            fresh_found += 1;
            samples.push(WinSample {
                source: "fresh",
                tag: format!("{}@{}", g.game_hash, state.placements_made()),
                moves,
                state,
            });
        }
    }
    eprintln!(
        "DTW_SOUNDNESS samples: total={} (forcing+leafwidth+fresh); fresh_found={fresh_found}/{fresh_tried}",
        samples.len()
    );

    // Evaluate every candidate against every sample's oracle.
    #[derive(Default, Clone)]
    struct CStat {
        checked: u64,
        violations: u64,
        min_slack: i64, // min (oracle - lb); negative => violation
    }
    let mut stats: Vec<CStat> = vec![CStat { min_slack: i64::MAX, ..Default::default() }; cands.len()];
    let mut viol_records: Vec<String> = Vec::new();
    let mut done = 0usize;
    // Resolved-dtw records (for the distribution table in the report).
    let mut dtw_records: Vec<String> = Vec::new();
    let mut resolved = 0u64;

    for s in &samples {
        if done % 25 == 0 {
            let ram = free_ram_gb();
            eprintln!("DTW_ORACLE progress={done}/{} resolved={resolved} ram={ram:.1}", samples.len());
            while free_ram_gb() < 9.0 {
                eprintln!("DTW_WAIT low RAM (<9GB), sleeping 60s");
                std::thread::sleep(std::time::Duration::from_secs(60));
            }
        }
        done += 1;
        let me = s.state.current_player();
        let lbs: Vec<u32> = cands.iter().map(|(_, _, f)| f(&s.state, me)).collect();
        let max_lb = lbs.iter().copied().max().unwrap_or(0);
        // Ladder only as far as the largest candidate LB (capped at hmax): a
        // WIN at h<=max_lb pins dtw tightly; no WIN by max_lb => dtw>max_lb>=
        // every LB, so all candidates are safe on this sample.
        let cap_h = max_lb.max(1).min(hmax);
        let oracle = dtw_oracle(&s.state, cap_h);
        match oracle {
            Some((h, exact)) => {
                resolved += 1;
                dtw_records.push(format!(
                    "{{\"source\":\"{}\",\"tag\":\"{}\",\"dtw_h\":{h},\"exact\":{exact},\"lb0\":{},\"cp\":{}}}",
                    s.source, s.tag, lbs[0], me.index()
                ));
                for (ci, (name, proven, _)) in cands.iter().enumerate() {
                    let lb = lbs[ci];
                    let slack = h as i64 - lb as i64;
                    let st = &mut stats[ci];
                    st.checked += 1;
                    if slack < st.min_slack {
                        st.min_slack = slack;
                    }
                    if lb > h {
                        st.violations += 1;
                        let rec = format!(
                            "{{\"candidate\":\"{name}\",\"proven\":{proven},\"source\":\"{}\",\"tag\":\"{}\",\"lb_plies\":{lb},\"oracle_h\":{h},\"oracle_exact\":{exact},\"cp\":{},\"prefix\":{}}}",
                            s.source, s.tag, me.index(), prefix_json(&s.moves),
                        );
                        viol_records.push(rec.clone());
                        if *proven {
                            println!("!!!!! DTW_ALARM proven-candidate VIOLATION: {rec}");
                        }
                    }
                }
            }
            None => {
                // dtw > cap_h == max_lb >= every candidate LB: all safe here.
                for (ci, _) in cands.iter().enumerate() {
                    let lb = lbs[ci];
                    let st = &mut stats[ci];
                    st.checked += 1;
                    let slack = cap_h as i64 + 1 - lb as i64; // >=1 lower bound on real slack
                    if slack < st.min_slack {
                        st.min_slack = slack;
                    }
                }
            }
        }
    }

    // Report.
    println!("=== DTW SOUNDNESS ===");
    println!(
        "DTW_SND_META samples={} resolved={resolved} hmax={hmax} fresh_n={fresh_n} fresh_cap={fresh_cap} oracle_node_cap={ORACLE_NODE_CAP} seed={seed}",
        samples.len()
    );
    for (ci, (name, proven, _)) in cands.iter().enumerate() {
        let st = &stats[ci];
        let verdict = if st.violations == 0 {
            format!("VALIDATED-ON-{}-WINS", st.checked)
        } else {
            format!("REFUTED ({} violations)", st.violations)
        };
        println!(
            "DTW_SND candidate={name} proven={proven} checked={} violations={} min_slack={} verdict={verdict}",
            st.checked,
            st.violations,
            if st.min_slack == i64::MAX { 9999 } else { st.min_slack },
        );
    }
    let body = viol_records.join("\n");
    std::fs::write(&out_path, if body.is_empty() { String::from("") } else { format!("{body}\n") })
        .unwrap_or_else(|e| panic!("write {out_path}: {e}"));
    let dtw_path = out_path.replace("DTW_SOUNDNESS.jsonl", "DTW_RESOLVED.jsonl");
    std::fs::write(&dtw_path, format!("{}\n", dtw_records.join("\n")))
        .unwrap_or_else(|e| panic!("write {dtw_path}: {e}"));
    println!("DTW_SND_RECORDS violations_written={} path={out_path} resolved_records={} path={dtw_path}", viol_records.len(), dtw_records.len());
    println!("DTW_SND_DONE");

    // Hard assert: a PROVEN candidate must never violate.
    for (ci, (name, proven, _)) in cands.iter().enumerate() {
        if *proven {
            assert_eq!(
                stats[ci].violations, 0,
                "PROVEN candidate {name} violated soundness — census/plymap bug"
            );
        }
    }
}

// ==========================================================================
// FIRE RATE x COST — >=2000 sampled leaf FirstStone nodes.
// ==========================================================================

const FIRE_HS: [u32; 6] = [2, 4, 6, 8, 12, 16];

#[test]
#[ignore = "dtw fire-rate x cost; --test-threads=1 --nocapture"]
fn dtw_firerate() {
    let want: usize = std::env::var("DTW_FIRE_N").ok().and_then(|v| v.parse().ok()).unwrap_or(4000);
    let seed: u64 = std::env::var("DTW_FIRE_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(0x9E37_79B9_7F4A_7C15);
    let out_path = std::env::var("DTW_FIRE_OUT").unwrap_or_else(|_| {
        "E:/Hexo-BotTrainer-hexgt/.claude/worktrees/hunt-dtw-bounds/DTW_FIRERATE.json".to_string()
    });
    let games = load_corpus();
    let cands = candidates();

    // Enumerate FirstStone non-terminal nodes (both players to move), tagged by
    // phase band, across all games (decisive or not — leaves are leaves).
    #[derive(Clone, Copy)]
    struct Node {
        gi: u32,
        pl: u32,
        band: u8,
        maxcnt: u8,
    }
    let band_of = |ply: u32| -> u8 {
        if ply <= 12 { 0 } else if ply <= 40 { 1 } else { 2 }
    };
    let mut nodes: Vec<Node> = Vec::new();
    for (gi, g) in games.iter().enumerate() {
        let mut state = HexoState::new();
        for (i, &(q, r)) in g.moves.iter().enumerate() {
            if state.is_terminal() {
                break;
            }
            if matches!(state.phase(), TurnPhase::FirstStone) && (i as u32) < g.moves.len() as u32 {
                let me = state.current_player();
                let (mine, opp) = stones_of(&state, me);
                let mc = max_cnt(&alive_windows(&mine, &opp));
                nodes.push(Node {
                    gi: gi as u32,
                    pl: i as u32,
                    band: band_of(state.placements_made()),
                    maxcnt: mc,
                });
            }
            apply_placement(&mut state, Placement { coord: HexCoord::new(q, r) }).expect("legal");
        }
    }
    // Deterministic shuffle, take `want`.
    let mut idx: Vec<usize> = (0..nodes.len()).collect();
    let mut rng = XorShift(seed | 1);
    for i in (1..idx.len()).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        idx.swap(i, j);
    }
    idx.truncate(want.min(idx.len()));
    let sample: Vec<Node> = idx.iter().map(|&k| nodes[k]).collect();
    eprintln!("DTW_FIRE pool={} sample={}", nodes.len(), sample.len());

    // Rebuild states once, compute LBs, tally.
    let ncand = cands.len();
    // fire[cand][h_idx] and per band.
    let mut fire = vec![[0u64; FIRE_HS.len()]; ncand];
    let mut fire_band = vec![vec![[0u64; FIRE_HS.len()]; 3]; ncand];
    let mut band_n = [0u64; 3];
    // maxcnt histogram (explains the cap).
    let mut maxcnt_hist = [0u64; 8];
    // timing accumulators (ns) per candidate.
    let mut eval_ns: Vec<Vec<u64>> = vec![Vec::new(); ncand];

    let n = sample.len();
    for (si, node) in sample.iter().enumerate() {
        if si % 500 == 0 {
            eprintln!("DTW_FIRE progress={si}/{n} ram={:.1}", free_ram_gb());
        }
        let g = &games[node.gi as usize];
        let Some(state) = replay(&g.moves[..node.pl as usize]) else { continue };
        let me = state.current_player();
        band_n[node.band as usize] += 1;
        maxcnt_hist[(node.maxcnt as usize).min(7)] += 1;
        for (ci, (_, _, f)) in cands.iter().enumerate() {
            // warm timing: per-call ns over a repeat burst (acc prevents the
            // optimizer from eliding the call).
            let reps = 64u32;
            let t0 = Instant::now();
            let mut acc = 0u32;
            for _ in 0..reps {
                acc = acc.wrapping_add(f(&state, me));
            }
            let per = (t0.elapsed().as_nanos() as u64) / reps as u64;
            eval_ns[ci].push(per);
            std::hint::black_box(acc);
            let lb = f(&state, me);
            for (hi, &h) in FIRE_HS.iter().enumerate() {
                if lb > h {
                    fire[ci][hi] += 1;
                    fire_band[ci][node.band as usize][hi] += 1;
                }
            }
        }
    }

    let total: u64 = band_n.iter().sum();
    let frac = |num: u64, den: u64| if den == 0 { 0.0 } else { num as f64 / den as f64 };
    let med = |v: &mut Vec<u64>| {
        if v.is_empty() { return 0; }
        v.sort_unstable();
        v[v.len() / 2]
    };

    println!("=== DTW FIRE RATE x COST ===");
    println!("DTW_FIRE_META sample={total} bands=[{},{},{}] hs={:?}", band_n[0], band_n[1], band_n[2], FIRE_HS);
    println!(
        "DTW_FIRE_MAXCNT hist(maxcnt=0..7)=[{}]",
        maxcnt_hist.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
    );
    let mut json = String::from("{\n");
    json.push_str(&format!("  \"sample\": {total},\n  \"hs\": {:?},\n  \"candidates\": {{\n", FIRE_HS));
    for (ci, (name, proven, _)) in cands.iter().enumerate() {
        let med_ns = med(&mut eval_ns[ci]);
        let rates: Vec<String> = FIRE_HS
            .iter()
            .enumerate()
            .map(|(hi, h)| format!("h{}={:.4}", h, frac(fire[ci][hi], total)))
            .collect();
        println!(
            "DTW_FIRE candidate={name} proven={proven} eval_med_ns={med_ns} eval_med_us={:.3} rates[{}]",
            med_ns as f64 / 1000.0,
            rates.join(" "),
        );
        for b in 0..3 {
            let brates: Vec<String> = FIRE_HS
                .iter()
                .enumerate()
                .map(|(hi, h)| format!("h{}={:.4}", h, frac(fire_band[ci][b][hi], band_n[b])))
                .collect();
            println!("DTW_FIRE_BAND candidate={name} band={b} n={} [{}]", band_n[b], brates.join(" "));
        }
        json.push_str(&format!("    \"{name}\": {{ \"proven\": {proven}, \"eval_med_ns\": {med_ns}, \"rates\": {{ "));
        let jr: Vec<String> = FIRE_HS
            .iter()
            .enumerate()
            .map(|(hi, h)| format!("\"h{}\": {:.5}", h, frac(fire[ci][hi], total)))
            .collect();
        json.push_str(&jr.join(", "));
        json.push_str(" } },\n");
    }
    json.push_str("    \"_end\": true\n  }\n}\n");
    std::fs::write(&out_path, json).unwrap_or_else(|e| panic!("write {out_path}: {e}"));
    println!("DTW_FIRE_JSON path={out_path}");
    println!("DTW_FIRE_DONE");
}
