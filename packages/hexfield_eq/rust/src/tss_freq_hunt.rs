//! Empirical-grounding hunt: how often the TSS phenomena occur in real human
//! play (HuggingFace `timmyburn/hexo-bootstrap-corpus`, local copy).
//!
//! DATA, not proofs.  Two ignored helpers, run explicitly:
//!   * `freq_cheap` — pure threat/geometry over every replayed node: defender
//!     width denominators (k vs B), pileup incidence, ES Phi (<1) incidence,
//!     and the D6-canonical opening-family table.  No engine solves.
//!   * `freq_vcf` — the certificate-grade wide solver (`vcf_pair_complete`,
//!     10k cap, 512 MiB TT, one solve at a time) over a fixed-seed sample of
//!     mid/late attacker-to-move (FirstStone) nodes: VCF-exists incidence,
//!     human-found-the-win rate, and the quiet-move (lambda^2) signature.
//!
//! Role/threat semantics are exactly `crate::threats_shared` (the single source
//! of TSS threat truth) and the wide solver is exactly the corpus gate's
//! `WidthOptions::vcf_pair_complete()`.  Phi is ported verbatim from
//! hunt/gap-raw's `gap_raw_hunt.rs` (attacker = Player1, defender = Player0).

use std::collections::{BTreeSet, HashMap};

use hexo_engine::{apply_placement, HexCoord, HexoState, Placement, Player, TurnPhase};

use crate::threats_shared::analyze;
use crate::tss_core::{DeepSolve, ProofStatus, SolveCaps};
use crate::tss_solver::{TssSolver, WidthOptions};
use crate::tss_verify::d6_transform_coord;

// --------------------------------------------------------------------------
// Corpus parsing (stdlib only; the schema is fixed:
// {"game_hash":..,"moves":[[q,r],..],"winner":±1,..}).
// --------------------------------------------------------------------------

struct Game {
    moves: Vec<(i16, i16)>,
    /// engine convention: +1 = Player0 (opener) wins, -1 = Player1 wins.
    winner: i8,
}

fn corpus_path() -> String {
    std::env::var("TSS_FREQ_CORPUS").unwrap_or_else(|_| {
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
    Some(Game { moves, winner })
}

fn load_corpus() -> Vec<Game> {
    let path = corpus_path();
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read corpus {path}: {e}"));
    let games: Vec<Game> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| parse_line(l).unwrap_or_else(|| panic!("bad corpus line")))
        .collect();
    eprintln!("FREQ_CORPUS path={path} games={}", games.len());
    games
}

// --------------------------------------------------------------------------
// ES potential Phi (verbatim port of gap_raw_hunt.rs; attacker = Player1).
// --------------------------------------------------------------------------

const PHI_AXES: [(i16, i16); 3] = [(1, 0), (0, 1), (1, -1)];

/// Returns (A, B) with 27*Phi = A + B*sqrt(3), over attacker(Player1)-alive
/// length-6 windows (no defender stone, >=1 attacker stone).
fn phi_ab(state: &HexoState) -> (i128, i128) {
    let mut att: BTreeSet<(i16, i16)> = BTreeSet::new();
    let mut def: BTreeSet<(i16, i16)> = BTreeSet::new();
    for &c in state.board().occupied_cells() {
        match state.board().get(c) {
            Some(Player::Player1) => {
                att.insert((c.q, c.r));
            }
            Some(Player::Player0) => {
                def.insert((c.q, c.r));
            }
            None => {}
        }
    }
    let mut seen: BTreeSet<(u8, i16, i16)> = BTreeSet::new();
    let mut n = [0u64; 7];
    for &a in &att {
        for (ax, &v) in PHI_AXES.iter().enumerate() {
            for k in 0..6i16 {
                let start = (a.0 - k * v.0, a.1 - k * v.1);
                if !seen.insert((ax as u8, start.0, start.1)) {
                    continue;
                }
                let mut acnt = 0u64;
                let mut has_def = false;
                for j in 0..6i16 {
                    let c = (start.0 + j * v.0, start.1 + j * v.1);
                    if def.contains(&c) {
                        has_def = true;
                        break;
                    }
                    if att.contains(&c) {
                        acnt += 1;
                    }
                }
                if !has_def && acnt >= 1 && acnt <= 6 {
                    n[acnt as usize] += 1;
                }
            }
        }
    }
    let a = 3 * n[2] as i128 + 9 * n[4] as i128;
    let b = n[1] as i128 + 3 * n[3] as i128 + 9 * n[5] as i128;
    (a, b)
}

/// Exact Phi < 1 test: A + B*sqrt(3) < 27.
fn phi_lt_one(a: i128, b: i128) -> bool {
    if a >= 27 {
        return false;
    }
    let d = 27 - a;
    3 * b * b < d * d
}

fn phi_f64(a: i128, b: i128) -> f64 {
    (a as f64 + b as f64 * 3f64.sqrt()) / 27.0
}

// --------------------------------------------------------------------------
// D6 canonicalization for opening families.
// --------------------------------------------------------------------------

fn d6_pt(p: (i16, i16), sym: u8) -> Option<(i16, i16)> {
    d6_transform_coord(HexCoord::new(p.0, p.1), sym).map(|c| (c.q, c.r))
}

/// Canonical unordered stone pair under the 12 D6 symmetries (opener at origin
/// is D6-fixed, so this classifies the reply up to board symmetry).
fn d6_canon_pair(a: (i16, i16), b: (i16, i16)) -> ((i16, i16), (i16, i16)) {
    let mut best: Option<((i16, i16), (i16, i16))> = None;
    for sym in 0..12u8 {
        if let (Some(ta), Some(tb)) = (d6_pt(a, sym), d6_pt(b, sym)) {
            let mut pr = [ta, tb];
            pr.sort();
            let cand = (pr[0], pr[1]);
            best = Some(match best {
                None => cand,
                Some(cur) => cur.min(cand),
            });
        }
    }
    best.expect("at least identity symmetry")
}

/// Canonical labeled two-turn class: (P2 reply pair, P1 second-turn pair),
/// each pair unordered within its own turn but the two turns kept distinct.
type TwoTurn = (((i16, i16), (i16, i16)), ((i16, i16), (i16, i16)));
fn d6_canon_two_turn(p2: [(i16, i16); 2], p1: [(i16, i16); 2]) -> TwoTurn {
    let mut best: Option<TwoTurn> = None;
    for sym in 0..12u8 {
        let t = (|| {
            let a = [d6_pt(p2[0], sym)?, d6_pt(p2[1], sym)?];
            let b = [d6_pt(p1[0], sym)?, d6_pt(p1[1], sym)?];
            let mut a = a;
            a.sort();
            let mut b = b;
            b.sort();
            Some(((a[0], a[1]), (b[0], b[1])))
        })();
        if let Some(cand) = t {
            best = Some(match best {
                None => cand,
                Some(cur) => cur.min(cand),
            });
        }
    }
    best.expect("at least identity symmetry")
}

// --------------------------------------------------------------------------
// Measurement 1/3a/3c/4 — pure threat + geometry, every replayed node.
// --------------------------------------------------------------------------

#[test]
#[ignore = "empirical corpus hunt; run explicitly with --nocapture"]
fn freq_cheap() {
    let games = load_corpus();

    // Defender-width denominators (side to move faces >=1 opponent >=4 window).
    let mut threatened_all = 0u64; // opp_threat_count >= 1
    let mut threatened_ownwin = 0u64; // ...but side to move also wins now
    // Genuine defender nodes (threatened, no own win-now), keyed by (B, k-class):
    // k-class: 0 => k<B (unforced), 1 => k==B (forced-exact), 2 => None (forced loss)
    let mut def_bk = [[0u64; 3]; 3]; // index by B (0,1,2), then k-class
    let mut legal_counts: Vec<u32> = Vec::new(); // |Legal| at every genuine def node
    let mut threat_hist: HashMap<usize, u64> = HashMap::new(); // opp_threat_count -> count (genuine def)
    let mut total_nodes = 0u64;
    let mut nonterminal_nodes = 0u64;

    // Pileup (>=3 simultaneous opponent >=4 windows facing a defender turn).
    let mut pileup_b2 = 0u64; // FirstStone genuine-defender nodes with opp>=3
    let mut pileup_any = 0u64; // any genuine-defender node with opp>=3
    let mut pileup_bk = [[0u64; 3]; 3]; // pileup nodes only, by (B, k-class)
    let mut games_with_pileup = 0u64;

    // ES Phi at Defender(Player0)-FirstStone nodes.
    let mut def_fs_nodes = 0u64;
    let mut phi_lt1 = 0u64;
    let mut def_fs_dev = 0u64; // with >=6 attacker(Player1) stones (windows meaningful)
    let mut phi_lt1_dev = 0u64;

    // Opening families.
    let mut fam_p2: HashMap<((i16, i16), (i16, i16)), u64> = HashMap::new();
    let mut fam_two: HashMap<TwoTurn, u64> = HashMap::new();
    let mut opener_nonzero = 0u64;

    for g in &games {
        if g.moves.first() != Some(&(0, 0)) {
            opener_nonzero += 1;
        }
        // Opening families (need >=3 stones for P2 reply, >=5 for two turns).
        if g.moves.len() >= 3 {
            let key = d6_canon_pair(g.moves[1], g.moves[2]);
            *fam_p2.entry(key).or_insert(0) += 1;
        }
        if g.moves.len() >= 5 {
            let key = d6_canon_two_turn([g.moves[1], g.moves[2]], [g.moves[3], g.moves[4]]);
            *fam_two.entry(key).or_insert(0) += 1;
        }

        let mut state = HexoState::new();
        let mut this_game_pileup = false;
        for &(q, r) in &g.moves {
            total_nodes += 1;
            if state.is_terminal() {
                break;
            }
            nonterminal_nodes += 1;

            // Phi at Player0-FirstStone (defender) nodes.
            if state.current_player() == Player::Player0
                && matches!(state.phase(), TurnPhase::FirstStone)
            {
                def_fs_nodes += 1;
                let (a, b) = phi_ab(&state);
                if phi_lt_one(a, b) {
                    phi_lt1 += 1;
                }
                let p1_stones = state
                    .board()
                    .occupied_cells()
                    .iter()
                    .filter(|&&c| state.board().get(c) == Some(Player::Player1))
                    .count();
                if p1_stones >= 6 {
                    def_fs_dev += 1;
                    if phi_lt_one(a, b) {
                        phi_lt1_dev += 1;
                    }
                }
            }

            let an = analyze(&state);
            if an.opp_threat_count >= 1 {
                threatened_all += 1;
                if an.own_win_now {
                    threatened_ownwin += 1;
                } else {
                    // genuine defender node
                    let b = an.b as usize; // 1 or 2
                    let kclass = match an.min_hitting_set {
                        Some(k) if (k as usize) < b => 0, // unforced (slack)
                        Some(_) => 1,                     // forced-exact (k == B)
                        None => 2,                        // forced loss (k > B)
                    };
                    def_bk[b][kclass] += 1;
                    legal_counts.push(state.legal_move_count() as u32);
                    *threat_hist.entry(an.opp_threat_count).or_insert(0) += 1;
                    if an.opp_threat_count >= 3 {
                        pileup_any += 1;
                        pileup_bk[b][kclass] += 1;
                        this_game_pileup = true;
                        if an.b == 2 {
                            pileup_b2 += 1;
                        }
                    }
                }
            }

            apply_placement(
                &mut state,
                Placement {
                    coord: HexCoord::new(q, r),
                },
            )
            .unwrap_or_else(|e| panic!("illegal replay ({q},{r}): {e:?}"));
        }
        if this_game_pileup {
            games_with_pileup += 1;
        }
    }

    // ---- report ----
    let def_b2_total = def_bk[2][0] + def_bk[2][1] + def_bk[2][2];
    let def_b1_total = def_bk[1][0] + def_bk[1][1] + def_bk[1][2];
    let def_total = def_b2_total + def_b1_total;
    let unforced = def_bk[2][0] + def_bk[1][0]; // k<B (only possible at B=2)

    println!("=== FREQ_CHEAP corpus summary ===");
    println!(
        "FREQ_NODES total_move_slots={total_nodes} nonterminal_decision_nodes={nonterminal_nodes} opener_not_origin={opener_nonzero}"
    );
    println!(
        "FREQ_THREATENED all(opp>=1)={threatened_all} of_which_own_win_now={threatened_ownwin} genuine_defender={def_total}"
    );
    println!(
        "FREQ_DEF_B split B2(FirstStone)={def_b2_total} B1(SecondStone)={def_b1_total}"
    );
    println!("--- defender width: k vs B (genuine defender nodes) ---");
    println!(
        "FREQ_KB B=2  k<B(unforced,k=1)={} k=B(forced-exact,k=2)={} k>B(forced-loss)={}",
        def_bk[2][0], def_bk[2][1], def_bk[2][2]
    );
    println!(
        "FREQ_KB B=1  k<B(impossible)={} k=B(forced-exact,k=1)={} k>B(forced-loss)={}",
        def_bk[1][0], def_bk[1][1], def_bk[1][2]
    );
    let frac = |num: u64, den: u64| if den == 0 { 0.0 } else { num as f64 / den as f64 };
    println!(
        "FREQ_UNFORCED unforced(k<B)={unforced} / genuine_defender={def_total} = {:.4}  | over_B2_only = {:.4}",
        frac(unforced, def_total),
        frac(def_bk[2][0], def_b2_total)
    );
    println!(
        "FREQ_UNFORCED_INTERP forced_exact_frac(all)={:.4} forced_loss_frac(all)={:.4}",
        frac(def_bk[2][1] + def_bk[1][1], def_total),
        frac(def_bk[2][2] + def_bk[1][2], def_total)
    );

    println!("--- |Legal| distribution at genuine defender nodes ---");
    legal_counts.sort_unstable();
    let lc = &legal_counts;
    let pct = |p: f64| -> u32 {
        if lc.is_empty() {
            return 0;
        }
        let idx = ((p * (lc.len() as f64 - 1.0)).round() as usize).min(lc.len() - 1);
        lc[idx]
    };
    let lsum: u64 = lc.iter().map(|&x| x as u64).sum();
    println!(
        "FREQ_LEGAL n={} min={} p10={} p50={} p90={} max={} mean={:.1}",
        lc.len(),
        lc.first().copied().unwrap_or(0),
        pct(0.10),
        pct(0.50),
        pct(0.90),
        lc.last().copied().unwrap_or(0),
        if lc.is_empty() { 0.0 } else { lsum as f64 / lc.len() as f64 }
    );
    println!("--- opponent threat-window count at genuine defender nodes ---");
    let mut tk: Vec<_> = threat_hist.iter().collect();
    tk.sort_by_key(|(k, _)| **k);
    for (k, v) in tk {
        println!("FREQ_THREATN opp_windows={k} count={v} frac={:.4}", frac(*v, def_total));
    }

    println!("--- pileup (>=3 simultaneous opponent >=4 windows) ---");
    println!(
        "FREQ_PILEUP b2_firststone={pileup_b2} any_defnode={pileup_any} games_with_pileup={games_with_pileup}/{} pileup_frac_of_B2def={:.4} pileup_frac_of_genuinedef={:.5}",
        games.len(),
        frac(pileup_b2, def_b2_total),
        frac(pileup_any, def_total)
    );
    // Overlapping length-6 windows inflate the raw count (a single 4-in-a-row =
    // ~3 windows), so cross-tab pileup nodes by hitting number k: k=1 = one
    // stone kills the whole cluster (cheap), k=2/loss = a genuine multi-target
    // fork.
    let pileup_k1 = pileup_bk[2][0] + pileup_bk[1][1]; // k<B at B=2 OR k=1(==B) at B=1
    let pileup_hard = pileup_bk[2][1] + pileup_bk[2][2] + pileup_bk[1][2];
    println!(
        "FREQ_PILEUP_K single-hit(k=1)={pileup_k1} two-target(k=2,B2)={} forced-loss={} hard(k>=2 or loss)={pileup_hard} hard_frac_of_pileup={:.4}",
        pileup_bk[2][1],
        pileup_bk[2][2] + pileup_bk[1][2],
        frac(pileup_hard, pileup_any)
    );

    println!("--- ES Phi at Defender(Player0)-FirstStone nodes ---");
    println!(
        "FREQ_PHI def_fs_nodes={def_fs_nodes} phi_lt1={phi_lt1} frac={:.4} | dev(>=6 atk stones): nodes={def_fs_dev} phi_lt1={phi_lt1_dev} frac={:.4}",
        frac(phi_lt1, def_fs_nodes),
        frac(phi_lt1_dev, def_fs_dev)
    );

    println!("--- opening families (D6-canonical) ---");
    println!(
        "FREQ_FAM_COUNT distinct_P2_reply={} distinct_two_turn={}",
        fam_p2.len(),
        fam_two.len()
    );
    let games_ge3 = games.iter().filter(|g| g.moves.len() >= 3).count() as u64;
    let mut fp: Vec<_> = fam_p2.into_iter().collect();
    fp.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for (i, (key, cnt)) in fp.iter().take(20).enumerate() {
        println!(
            "FREQ_FAM_P2 rank={} count={cnt} frac={:.4} stones={:?}",
            i + 1,
            frac(*cnt, games_ge3),
            key
        );
    }
    let games_ge5 = games.iter().filter(|g| g.moves.len() >= 5).count() as u64;
    let mut ft: Vec<_> = fam_two.into_iter().collect();
    ft.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for (i, (key, cnt)) in ft.iter().take(12).enumerate() {
        println!(
            "FREQ_FAM_TWO rank={} count={cnt} frac={:.4} P2={:?} P1={:?}",
            i + 1,
            frac(*cnt, games_ge5),
            key.0,
            key.1
        );
    }
    println!("FREQ_CHEAP_DONE");
}

// --------------------------------------------------------------------------
// Measurement 2 / 3b — VCF incidence + human-found-win + lambda^2 signature.
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
}

#[derive(Clone, Copy)]
struct VcfCand {
    game_idx: u32,
    prefix_len: u32,      // moves applied to reach this FirstStone node
    winner: i8,           // +1 Player0, -1 Player1
    mover_is_p0: bool,    // side to move at the node
    plies_to_end: u32,    // moves.len() - prefix_len
}

fn free_ram_gb() -> f64 {
    // best-effort; returns a large number if the query fails (never blocks work)
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

#[test]
#[ignore = "empirical corpus VCF hunt; serialized, run with --test-threads=1 --nocapture"]
fn freq_vcf() {
    let games = load_corpus();
    let min_stones: u32 = std::env::var("TSS_FREQ_MIN_STONES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let sample_n: usize = std::env::var("TSS_FREQ_VCF_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500);
    let seed: u64 = std::env::var("TSS_FREQ_VCF_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0x9E37_79B9_7F4A_7C15);
    let node_cap: u64 = std::env::var("TSS_FREQ_VCF_CAP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000);
    let tt_bytes_cap: usize = std::env::var("TSS_FREQ_TT_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512 << 20);

    // Enumerate every mid/late FirstStone decision node (candidate attacker-to-move).
    let mut cands: Vec<VcfCand> = Vec::new();
    for (gi, g) in games.iter().enumerate() {
        let mut state = HexoState::new();
        for (i, &(q, r)) in g.moves.iter().enumerate() {
            if !state.is_terminal()
                && matches!(state.phase(), TurnPhase::FirstStone)
                && state.placements_made() >= min_stones
                && (i as u32) < g.moves.len() as u32 // a next move exists (non-terminal)
            {
                cands.push(VcfCand {
                    game_idx: gi as u32,
                    prefix_len: i as u32,
                    winner: g.winner,
                    mover_is_p0: state.current_player() == Player::Player0,
                    plies_to_end: (g.moves.len() - i) as u32,
                });
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
    let pool = cands.len();

    // Deterministic Fisher-Yates shuffle, take first sample_n.
    let mut rng = XorShift(seed | 1);
    for i in (1..cands.len()).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        cands.swap(i, j);
    }
    let sample: Vec<VcfCand> = cands.into_iter().take(sample_n).collect();

    eprintln!(
        "FREQ_VCF_SETUP pool={pool} sample_n={} seed={seed} node_cap={node_cap} tt_bytes_cap={tt_bytes_cap} min_stones={min_stones}",
        sample.len()
    );

    let solve_win = |state: &HexoState| -> (ProofStatus, u64) {
        let mut solver = TssSolver::default();
        solver.set_width_options(WidthOptions::vcf_pair_complete());
        let caps = SolveCaps {
            node_cap,
            tt_bytes_cap,
            semantic_horizon: u32::MAX,
        };
        let r = solver.solve(state, &caps);
        (r.status, r.stats.nodes)
    };

    let mut win_exists = 0u64;
    let mut unknown = 0u64;
    let mut loss = 0u64; // should be 0 in Both+vcf mode
    let mut win_human_found = 0u64; // among win_exists: human's actual first stone keeps the win
    let mut win_with_next = 0u64; // win_exists nodes that had a next move to test

    // lambda^2 signature: side-to-move eventually won the game, bucketed by
    // plies-to-end, split by whether a VCF win was already provable here.
    // buckets: [1..=6, 7..=12, 13..=20, 21..=40, 41+]
    let mut won_by_mover = [0u64; 5];
    let mut won_by_mover_vcf = [0u64; 5]; // of those, VCF win exists (forcing)
    let bucket = |p: u32| -> usize {
        match p {
            0..=6 => 0,
            7..=12 => 1,
            13..=20 => 2,
            21..=40 => 3,
            _ => 4,
        }
    };

    let mut done = 0u64;
    for c in &sample {
        if done % 100 == 0 {
            let ram = free_ram_gb();
            eprintln!("FREQ_VCF_PROGRESS done={done} free_ram_gb={ram:.1}");
            while free_ram_gb() < 8.0 {
                eprintln!("FREQ_VCF_WAIT low RAM, sleeping 60s");
                std::thread::sleep(std::time::Duration::from_secs(60));
            }
        }
        // Rebuild the node state.
        let g = &games[c.game_idx as usize];
        let mut state = HexoState::new();
        for &(q, r) in &g.moves[..c.prefix_len as usize] {
            apply_placement(
                &mut state,
                Placement {
                    coord: HexCoord::new(q, r),
                },
            )
            .expect("legal replay");
        }
        assert!(!state.is_terminal());
        let (status, _nodes) = solve_win(&state);

        // lambda^2 accounting: does the SIDE TO MOVE go on to win this game?
        let mover_wins = (c.mover_is_p0 && c.winner == 1) || (!c.mover_is_p0 && c.winner == -1);
        if mover_wins {
            let b = bucket(c.plies_to_end);
            won_by_mover[b] += 1;
            if status == ProofStatus::Win {
                won_by_mover_vcf[b] += 1;
            }
        }

        match status {
            ProofStatus::Win => {
                win_exists += 1;
                // Did the human play a winning first stone? Apply the actual
                // next move and re-solve for the same (attacker) side to move.
                let next = g.moves[c.prefix_len as usize];
                let mut child = state.clone();
                if apply_placement(
                    &mut child,
                    Placement {
                        coord: HexCoord::new(next.0, next.1),
                    },
                )
                .is_ok()
                    && !child.is_terminal()
                {
                    win_with_next += 1;
                    let (cs, _) = solve_win(&child);
                    if cs == ProofStatus::Win {
                        win_human_found += 1;
                    }
                } else if !state.is_terminal() {
                    // next move won immediately => trivially the winning first move
                    win_with_next += 1;
                    win_human_found += 1;
                }
            }
            ProofStatus::Unknown => unknown += 1,
            ProofStatus::Loss => loss += 1,
        }
        done += 1;
    }

    let n = sample.len() as u64;
    let frac = |num: u64, den: u64| if den == 0 { 0.0 } else { num as f64 / den as f64 };
    println!("=== FREQ_VCF results ===");
    println!(
        "FREQ_VCF_INCIDENCE sample={n} pool={pool} win_exists={win_exists} unknown={unknown} loss={loss} win_rate={:.4}",
        frac(win_exists, n)
    );
    println!(
        "FREQ_VCF_HUMAN win_with_testable_next={win_with_next} human_kept_win={win_human_found} human_found_rate={:.4}",
        frac(win_human_found, win_with_next)
    );
    println!("--- lambda^2 signature: mover eventually WON, by plies-to-end ---");
    let labels = ["1-6", "7-12", "13-20", "21-40", "41+"];
    for i in 0..5 {
        let quiet = won_by_mover[i] - won_by_mover_vcf[i];
        println!(
            "FREQ_L2 bucket={} won_nodes={} vcf_forcing={} quiet_required(no_vcf)={} quiet_frac={:.4}",
            labels[i], won_by_mover[i], won_by_mover_vcf[i], quiet, frac(quiet, won_by_mover[i])
        );
    }
    println!("FREQ_VCF_DONE");
}
