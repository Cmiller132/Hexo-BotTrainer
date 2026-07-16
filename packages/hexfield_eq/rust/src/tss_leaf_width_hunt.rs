//! Leaf-width empirical hunt (DATA, not proofs).
//!
//! Question: at MCTS-leaf node budgets, what does the production NARROW leaf
//! solver (`TssSolver::default()`, `SolveGoal::Both`) miss that the normative
//! WIDE engine (`WidthOptions::vcf_pair_complete()`) catches, and at what
//! wall-clock cost?  Both are invoked exactly as the corpus gate invokes them
//! (`crate::tss_spare_corpus` / `crate::tss_freq_hunt`).
//!
//! One ignored helper, run explicitly:
//!   * `leaf_width_miss_rate` — stratified fixed-seed sample of attacker-to-move
//!     (FirstStone) nodes from decisive human-corpus games; runs NARROW and WIDE
//!     at matched `node_cap` in {500,2000,10000}, records per-cell WIN%,
//!     both-UNKNOWN%, wide-only WIN share, disagreements (soundness alarm), and
//!     median/p95 wall clock per engine; emits the width-record .jsonl.
//!   * `leaf_width_validate_replay` — replay-convention guard: a fixed-seed
//!     sample of >=200 decisive games replayed to the last move must be terminal
//!     with the recorded winner (a 6-in-a-row).  Asserts.
//!
//! Corpus: HuggingFace `timmyburn/hexo-bootstrap-corpus`, local jsonl.
//! Replay convention (verbatim from `tss_freq_hunt.rs`): P0 single opening
//! placement at (0,0), then alternating two-stone turns; the engine tracks
//! phase/parity, so we simply `apply_placement` each `(q,r)` in order.

use std::time::Instant;

use hexo_engine::{apply_placement, HexCoord, HexoState, Placement, Player, TurnPhase};

use crate::tss_core::{DeepSolve, ProofStatus, SolveCaps, SolveGoal};
use crate::tss_solver::{TssSolver, WidthOptions};

// --------------------------------------------------------------------------
// Corpus parsing (stdlib only; schema:
// {"game_hash":"..16 hex..","moves":[[q,r],..],"winner":±1,"elo":[..]}).
// --------------------------------------------------------------------------

struct Game {
    game_hash: String,
    moves: Vec<(i16, i16)>,
    /// engine convention: +1 = Player0 (opener) wins, -1 = Player1 wins.
    winner: i8,
}

fn corpus_path() -> String {
    std::env::var("TSS_LEAFW_CORPUS").unwrap_or_else(|_| {
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
    eprintln!("LEAFW_CORPUS path={path} games={}", games.len());
    games
}

// --------------------------------------------------------------------------
// Deterministic RNG (no external dep, no RNG on any scored/solved path).
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

fn status_name(s: ProofStatus) -> &'static str {
    match s {
        ProofStatus::Win => "WIN",
        ProofStatus::Loss => "LOSS",
        ProofStatus::Unknown => "UNKNOWN",
    }
}

// --------------------------------------------------------------------------
// Replay-convention validation (>=200 games; assert terminal + winner).
// --------------------------------------------------------------------------

#[test]
#[ignore = "leaf-width replay-convention guard; run with --nocapture"]
fn leaf_width_validate_replay() {
    let games = load_corpus();
    let decisive: Vec<&Game> = games
        .iter()
        .filter(|g| g.winner == 1 || g.winner == -1)
        .collect();
    assert!(
        decisive.len() >= 200,
        "need >=200 decisive games, have {}",
        decisive.len()
    );
    let seed: u64 = std::env::var("TSS_LEAFW_VALIDATE_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0xA1B2_C3D4_E5F6_0718);
    let want: usize = std::env::var("TSS_LEAFW_VALIDATE_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);

    let mut idx: Vec<usize> = (0..decisive.len()).collect();
    let mut rng = XorShift(seed | 1);
    for i in (1..idx.len()).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        idx.swap(i, j);
    }
    let n = want.min(idx.len());
    let mut checked = 0usize;
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
            .unwrap_or_else(|e| {
                panic!(
                    "LEAFW_VALIDATE illegal replay hash={} ({q},{r}): {e:?}",
                    g.game_hash
                )
            });
            if let Some(outcome) = res.outcome {
                ended = Some(outcome);
                break;
            }
        }
        let outcome = ended.unwrap_or_else(|| {
            panic!(
                "LEAFW_VALIDATE game did not terminate hash={} moves={}",
                g.game_hash,
                g.moves.len()
            )
        });
        let winner_sign = if outcome.winner == Player::Player0 {
            1
        } else {
            -1
        };
        assert_eq!(
            winner_sign, g.winner,
            "LEAFW_VALIDATE winner mismatch hash={} engine_winner={:?} corpus_winner={}",
            g.game_hash, outcome.winner, g.winner
        );
        // A terminal in Hexo is set only on six-in-a-line, so a terminal with
        // the right winner is the 6-in-a-row check.
        assert!(state.is_terminal());
        checked += 1;
    }
    println!(
        "LEAFW_VALIDATE_OK checked={checked} of decisive={} total={}",
        decisive.len(),
        games.len()
    );
}

// --------------------------------------------------------------------------
// Miss-rate sweep.
// --------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Cand {
    game_idx: u32,
    prefix_len: u32, // moves applied to reach this FirstStone node
    band: u8,        // 0: ply<=12, 1: 13-40, 2: >40
    winner: i8,
    mover_is_p0: bool,
}

fn band_of(ply: u32) -> u8 {
    if ply <= 12 {
        0
    } else if ply <= 40 {
        1
    } else {
        2
    }
}

const BAND_LABELS: [&str; 3] = ["ply<=12", "ply13-40", "ply>40"];
const NBANDS: usize = 3;

#[derive(Default, Clone, Copy)]
struct Cell {
    n: u64,
    narrow_win: u64,
    narrow_loss: u64,
    narrow_unknown: u64,
    narrowwin_only_goal_win: u64, // narrow solve_goal(Win) == WIN
    wide_win: u64,
    wide_loss: u64,
    wide_unknown: u64,
    both_unknown: u64,
    wide_only_win: u64,   // narrow != WIN && wide == WIN  (headline miss)
    narrow_only_win: u64, // narrow == WIN && wide != WIN  (narrow's speed value)
    contradiction: u64,   // narrow==Loss&wide==Win OR narrow==Win&wide==Loss
}

fn pct(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((p * (sorted.len() as f64 - 1.0)).round() as usize).min(sorted.len() - 1);
    sorted[idx]
}

#[test]
#[ignore = "empirical leaf-width miss-rate hunt; serialized, --test-threads=1 --nocapture"]
fn leaf_width_miss_rate() {
    let games = load_corpus();

    let per_band: usize = std::env::var("TSS_LEAFW_PER_BAND")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500);
    let seed: u64 = std::env::var("TSS_LEAFW_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0x9E37_79B9_7F4A_7C15);
    let horizon_slack: u32 = std::env::var("TSS_LEAFW_HORIZON_SLACK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    let tt_bytes_cap: usize = std::env::var("TSS_LEAFW_TT_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256 << 20);
    let caps: Vec<u64> = std::env::var("TSS_LEAFW_CAPS")
        .ok()
        .map(|v| {
            v.split(',')
                .map(|x| x.trim().parse::<u64>().expect("numeric cap"))
                .collect()
        })
        .unwrap_or_else(|| vec![500, 2000, 10000]);
    let record_cap: u64 = std::env::var("TSS_LEAFW_RECORD_CAP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);
    let records_path = std::env::var("TSS_LEAFW_RECORDS_PATH").unwrap_or_else(|_| {
        "E:/Hexo-BotTrainer-hexgt/.claude/worktrees/hunt-leaf-width/LEAF_WIDTH_RECORDS.jsonl"
            .to_string()
    });

    // Enumerate every attacker-to-move (FirstStone, non-terminal, has a next
    // move) node in decisive games, tagged by phase band.
    let mut cands: Vec<Cand> = Vec::new();
    for (gi, g) in games.iter().enumerate() {
        if g.winner != 1 && g.winner != -1 {
            continue; // decisive games only
        }
        let mut state = HexoState::new();
        for (i, &(q, r)) in g.moves.iter().enumerate() {
            if !state.is_terminal()
                && matches!(state.phase(), TurnPhase::FirstStone)
                && (i as u32) < g.moves.len() as u32
            {
                let ply = state.placements_made();
                cands.push(Cand {
                    game_idx: gi as u32,
                    prefix_len: i as u32,
                    band: band_of(ply),
                    winner: g.winner,
                    mover_is_p0: state.current_player() == Player::Player0,
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

    // Partition by band, deterministic Fisher-Yates per band, take per_band.
    let mut sample: Vec<Cand> = Vec::new();
    let mut pool_sizes = [0usize; NBANDS];
    for band in 0..NBANDS as u8 {
        let mut band_cands: Vec<Cand> = cands.iter().copied().filter(|c| c.band == band).collect();
        pool_sizes[band as usize] = band_cands.len();
        let mut rng = XorShift((seed ^ ((band as u64).wrapping_mul(0x9E37_79B9))) | 1);
        for i in (1..band_cands.len()).rev() {
            let j = (rng.next() % (i as u64 + 1)) as usize;
            band_cands.swap(i, j);
        }
        band_cands.truncate(per_band);
        sample.extend(band_cands);
    }

    eprintln!(
        "LEAFW_SETUP pool_total={} pool_bands=[{},{},{}] per_band={per_band} sample={} caps={:?} tt_bytes={tt_bytes_cap} horizon_slack={horizon_slack} seed={seed}",
        cands.len(),
        pool_sizes[0],
        pool_sizes[1],
        pool_sizes[2],
        sample.len(),
        caps
    );

    // cells[cap_idx][band]
    let ncaps = caps.len();
    let mut cells = vec![[Cell::default(); NBANDS]; ncaps];
    // timing: micros per solve, per engine per cap (aggregate over bands)
    let mut narrow_us: Vec<Vec<u64>> = vec![Vec::new(); ncaps];
    let mut wide_us: Vec<Vec<u64>> = vec![Vec::new(); ncaps];
    let mut narrowgoal_us: Vec<Vec<u64>> = vec![Vec::new(); ncaps];
    // node counts (stats.nodes) per solve, per engine per cap
    let mut narrow_nodes: Vec<Vec<u64>> = vec![Vec::new(); ncaps];
    let mut wide_nodes: Vec<Vec<u64>> = vec![Vec::new(); ncaps];

    let mut records: Vec<String> = Vec::new();
    let mut wide_only_records = 0u64;
    let mut narrow_only_records = 0u64;
    let mut contradiction_records: Vec<String> = Vec::new();

    let mk_caps = |node_cap: u64, base_ply: u32| SolveCaps {
        node_cap,
        tt_bytes_cap,
        semantic_horizon: base_ply.saturating_add(horizon_slack),
    };

    let mut done = 0u64;
    for c in &sample {
        if done % 50 == 0 {
            let ram = free_ram_gb();
            eprintln!(
                "LEAFW_PROGRESS done={done}/{} free_ram_gb={ram:.1}",
                sample.len()
            );
            while free_ram_gb() < 8.0 {
                eprintln!("LEAFW_WAIT low RAM (<8GB), sleeping 60s");
                std::thread::sleep(std::time::Duration::from_secs(60));
            }
        }
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
        assert!(matches!(state.phase(), TurnPhase::FirstStone));
        let base_ply = state.placements_made();
        let band = c.band as usize;

        for (ci, &node_cap) in caps.iter().enumerate() {
            let caps_s = mk_caps(node_cap, base_ply);

            // NARROW (production leaf; SolveGoal::Both via DeepSolve::solve).
            let mut narrow = TssSolver::default();
            let t0 = Instant::now();
            let nr = narrow.solve(&state, &caps_s);
            narrow_us[ci].push(t0.elapsed().as_micros() as u64);

            // WIDE (normative offline engine).
            let mut wide = TssSolver::default();
            wide.set_width_options(WidthOptions::vcf_pair_complete());
            let t1 = Instant::now();
            let wr = wide.solve(&state, &caps_s);
            wide_us[ci].push(t1.elapsed().as_micros() as u64);
            narrow_nodes[ci].push(nr.stats.nodes);
            wide_nodes[ci].push(wr.stats.nodes);

            // NARROW WIN-only diagnostic (isolates the Both budget-split from
            // structural OR-generator width).
            let mut narrow_goal = TssSolver::default();
            let t2 = Instant::now();
            let ngr = narrow_goal.solve_goal(&state, &caps_s, SolveGoal::Win);
            narrowgoal_us[ci].push(t2.elapsed().as_micros() as u64);

            let cell = &mut cells[ci][band];
            cell.n += 1;
            match nr.status {
                ProofStatus::Win => cell.narrow_win += 1,
                ProofStatus::Loss => cell.narrow_loss += 1,
                ProofStatus::Unknown => cell.narrow_unknown += 1,
            }
            match wr.status {
                ProofStatus::Win => cell.wide_win += 1,
                ProofStatus::Loss => cell.wide_loss += 1,
                ProofStatus::Unknown => cell.wide_unknown += 1,
            }
            if ngr.status == ProofStatus::Win {
                cell.narrowwin_only_goal_win += 1;
            }
            if nr.status == ProofStatus::Unknown && wr.status == ProofStatus::Unknown {
                cell.both_unknown += 1;
            }
            if nr.status != ProofStatus::Win && wr.status == ProofStatus::Win {
                cell.wide_only_win += 1;
            }
            if nr.status == ProofStatus::Win && wr.status != ProofStatus::Win {
                cell.narrow_only_win += 1;
            }
            // Soundness alarm: proven-WIN vs proven-LOSS contradiction on the
            // same side-to-move root.
            let contradiction = (nr.status == ProofStatus::Loss && wr.status == ProofStatus::Win)
                || (nr.status == ProofStatus::Win && wr.status == ProofStatus::Loss);
            if contradiction {
                cell.contradiction += 1;
                contradiction_records.push(format!(
                    "{{\"kind\":\"contradiction\",\"game_hash\":\"{}\",\"ply\":{},\"cap\":{},\"narrow\":\"{}\",\"wide\":\"{}\",\"prefix\":{}}}",
                    g.game_hash,
                    base_ply,
                    node_cap,
                    status_name(nr.status),
                    status_name(wr.status),
                    prefix_json(&g.moves[..c.prefix_len as usize]),
                ));
            }

            // Width-record list at the record cap.
            if node_cap == record_cap {
                if nr.status == ProofStatus::Unknown && wr.status == ProofStatus::Win {
                    wide_only_records += 1;
                    let cert_nodes = wr.cert.as_ref().map(|ct| ct.nodes.len()).unwrap_or(0);
                    records.push(format!(
                        "{{\"kind\":\"wide_only_win\",\"game_hash\":\"{}\",\"ply\":{},\"band\":\"{}\",\"winner\":{},\"mover_is_p0\":{},\"narrow_status\":\"UNKNOWN\",\"narrow_nodes\":{},\"wide_nodes\":{},\"wide_cert_nodes\":{},\"prefix\":{}}}",
                        g.game_hash,
                        base_ply,
                        BAND_LABELS[band],
                        c.winner,
                        c.mover_is_p0,
                        nr.stats.nodes,
                        wr.stats.nodes,
                        cert_nodes,
                        prefix_json(&g.moves[..c.prefix_len as usize]),
                    ));
                }
                if nr.status == ProofStatus::Win && wr.status != ProofStatus::Win {
                    narrow_only_records += 1;
                    records.push(format!(
                        "{{\"kind\":\"narrow_only_win\",\"game_hash\":\"{}\",\"ply\":{},\"band\":\"{}\",\"winner\":{},\"mover_is_p0\":{},\"wide_status\":\"{}\",\"narrow_nodes\":{},\"wide_nodes\":{},\"prefix\":{}}}",
                        g.game_hash,
                        base_ply,
                        BAND_LABELS[band],
                        c.winner,
                        c.mover_is_p0,
                        status_name(wr.status),
                        nr.stats.nodes,
                        wr.stats.nodes,
                        prefix_json(&g.moves[..c.prefix_len as usize]),
                    ));
                }
            }
        }
        done += 1;
    }

    // Write records jsonl (wide-only + narrow-only + any contradictions).
    {
        let mut all = records.clone();
        all.extend(contradiction_records.iter().cloned());
        let body = all.join("\n");
        std::fs::write(&records_path, format!("{body}\n"))
            .unwrap_or_else(|e| panic!("write records {records_path}: {e}"));
    }

    // ---- report ----
    let frac = |num: u64, den: u64| {
        if den == 0 {
            0.0
        } else {
            num as f64 / den as f64
        }
    };
    println!("=== LEAFW miss-rate results ===");
    println!(
        "LEAFW_META sample={} per_band={per_band} caps={:?} tt_bytes={tt_bytes_cap} horizon=ply+{horizon_slack} seed={seed}",
        sample.len(),
        caps
    );

    for (ci, &node_cap) in caps.iter().enumerate() {
        // per-band cells
        let mut agg = Cell::default();
        for band in 0..NBANDS {
            let cell = &cells[ci][band];
            agg.n += cell.n;
            agg.narrow_win += cell.narrow_win;
            agg.narrow_loss += cell.narrow_loss;
            agg.narrow_unknown += cell.narrow_unknown;
            agg.narrowwin_only_goal_win += cell.narrowwin_only_goal_win;
            agg.wide_win += cell.wide_win;
            agg.wide_loss += cell.wide_loss;
            agg.wide_unknown += cell.wide_unknown;
            agg.both_unknown += cell.both_unknown;
            agg.wide_only_win += cell.wide_only_win;
            agg.narrow_only_win += cell.narrow_only_win;
            agg.contradiction += cell.contradiction;
            println!(
                "LEAFW_CELL cap={node_cap} band={} n={} narrow_win={} ({:.4}) wide_win={} ({:.4}) narrowgoalwin={} ({:.4}) both_unknown={} ({:.4}) wide_only_win={} ({:.4}) narrow_only_win={} ({:.4}) narrow_loss={} wide_loss={} contradiction={}",
                BAND_LABELS[band],
                cell.n,
                cell.narrow_win,
                frac(cell.narrow_win, cell.n),
                cell.wide_win,
                frac(cell.wide_win, cell.n),
                cell.narrowwin_only_goal_win,
                frac(cell.narrowwin_only_goal_win, cell.n),
                cell.both_unknown,
                frac(cell.both_unknown, cell.n),
                cell.wide_only_win,
                frac(cell.wide_only_win, cell.n),
                cell.narrow_only_win,
                frac(cell.narrow_only_win, cell.n),
                cell.narrow_loss,
                cell.wide_loss,
                cell.contradiction,
            );
        }
        println!(
            "LEAFW_CAP cap={node_cap} n={} narrow_win={} ({:.4}) wide_win={} ({:.4}) narrowgoalwin={} ({:.4}) both_unknown={} ({:.4}) wide_only_win={} ({:.4}) narrow_only_win={} ({:.4}) narrow_loss={} wide_loss={} contradiction={}",
            agg.n,
            agg.narrow_win,
            frac(agg.narrow_win, agg.n),
            agg.wide_win,
            frac(agg.wide_win, agg.n),
            agg.narrowwin_only_goal_win,
            frac(agg.narrowwin_only_goal_win, agg.n),
            agg.both_unknown,
            frac(agg.both_unknown, agg.n),
            agg.wide_only_win,
            frac(agg.wide_only_win, agg.n),
            agg.narrow_only_win,
            frac(agg.narrow_only_win, agg.n),
            agg.narrow_loss,
            agg.wide_loss,
            agg.contradiction,
        );

        // timing
        let mut nu = narrow_us[ci].clone();
        nu.sort_unstable();
        let mut wu = wide_us[ci].clone();
        wu.sort_unstable();
        let mut gu = narrowgoal_us[ci].clone();
        gu.sort_unstable();
        let nmed = pct(&nu, 0.50);
        let np95 = pct(&nu, 0.95);
        let wmed = pct(&wu, 0.50);
        let wp95 = pct(&wu, 0.95);
        let gmed = pct(&gu, 0.50);
        let gp95 = pct(&gu, 0.95);
        let ratio_med = if nmed == 0 {
            f64::INFINITY
        } else {
            wmed as f64 / nmed as f64
        };
        let ratio_p95 = if np95 == 0 {
            f64::INFINITY
        } else {
            wp95 as f64 / np95 as f64
        };
        println!(
            "LEAFW_TIME cap={node_cap} narrow_med_us={nmed} narrow_p95_us={np95} wide_med_us={wmed} wide_p95_us={wp95} narrowgoal_med_us={gmed} narrowgoal_p95_us={gp95} ratio_wide/narrow_med={ratio_med:.2} ratio_p95={ratio_p95:.2}"
        );

        // node-count distribution (explains the timing: forcing-tree size).
        let mut nn = narrow_nodes[ci].clone();
        nn.sort_unstable();
        let mut wn = wide_nodes[ci].clone();
        wn.sort_unstable();
        println!(
            "LEAFW_NODES cap={node_cap} narrow_med={} narrow_p95={} narrow_max={} wide_med={} wide_p95={} wide_max={}",
            pct(&nn, 0.50),
            pct(&nn, 0.95),
            nn.last().copied().unwrap_or(0),
            pct(&wn, 0.50),
            pct(&wn, 0.95),
            wn.last().copied().unwrap_or(0),
        );
    }

    println!(
        "LEAFW_RECORDS wide_only_win(cap={record_cap})={wide_only_records} narrow_only_win(cap={record_cap})={narrow_only_records} path={records_path}"
    );

    let total_contra: u64 = (0..ncaps)
        .flat_map(|ci| (0..NBANDS).map(move |b| (ci, b)))
        .map(|(ci, b)| cells[ci][b].contradiction)
        .sum();
    if total_contra > 0 {
        println!("!!!!! LEAFW_ALARM SOUNDNESS CONTRADICTIONS = {total_contra} !!!!!");
        for r in &contradiction_records {
            println!("LEAFW_ALARM_ROW {r}");
        }
    } else {
        println!("LEAFW_ALARM none (no WIN/LOSS contradiction across all cells)");
    }
    println!("LEAFW_DONE");
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

// --------------------------------------------------------------------------
// Measurement 3 (stretch): cheap ES no-win screen.
//
// Exact-surd ES potential Phi (verbatim port of gap_raw_hunt.rs /
// tss_freq_hunt.rs; attacker = Player1, defender = Player0).  Phi<1 is a
// sound instant no-win region for the DEFENDER (Player0) to move: the
// attacker's summed window potential is below the winning threshold, so no
// attacker VCF can exist yet.  We measure how often it holds at defender leaf
// nodes and what a screen evaluation costs versus a narrow solve.
// --------------------------------------------------------------------------

const PHI_AXES: [(i16, i16); 3] = [(1, 0), (0, 1), (1, -1)];

/// Returns (A, B) with 27*Phi = A + B*sqrt(3), over attacker(Player1)-alive
/// length-6 windows (no defender stone, >=1 attacker stone).
fn phi_ab(state: &HexoState) -> (i128, i128) {
    use std::collections::BTreeSet;
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

#[test]
#[ignore = "leaf-width ES no-win screen; run with --nocapture --test-threads=1"]
fn leaf_es_screen() {
    let games = load_corpus();
    let seed: u64 = std::env::var("TSS_LEAFW_ES_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0x2545_F491_4F6C_DD1D);
    let time_sample: usize = std::env::var("TSS_LEAFW_ES_TIME_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);

    // Incidence over every defender (Player0-FirstStone) leaf node.
    let mut def_nodes = 0u64;
    let mut phi_lt1 = 0u64;
    let mut def_dev = 0u64; // >=6 attacker stones (windows meaningful)
    let mut phi_lt1_dev = 0u64;
    // Collect (game_idx, prefix_len) of defender nodes for the timing sample.
    let mut def_index: Vec<(u32, u32)> = Vec::new();

    for (gi, g) in games.iter().enumerate() {
        if g.winner != 1 && g.winner != -1 {
            continue;
        }
        let mut state = HexoState::new();
        for (i, &(q, r)) in g.moves.iter().enumerate() {
            if state.is_terminal() {
                break;
            }
            if state.current_player() == Player::Player0
                && matches!(state.phase(), TurnPhase::FirstStone)
            {
                def_nodes += 1;
                def_index.push((gi as u32, i as u32));
                let (a, b) = phi_ab(&state);
                if phi_lt_one(a, b) {
                    phi_lt1 += 1;
                }
                let p1 = state
                    .board()
                    .occupied_cells()
                    .iter()
                    .filter(|&&c| state.board().get(c) == Some(Player::Player1))
                    .count();
                if p1 >= 6 {
                    def_dev += 1;
                    if phi_lt_one(a, b) {
                        phi_lt1_dev += 1;
                    }
                }
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

    // Timing: Phi screen vs a narrow solve (cap 500) on a fixed-seed sample.
    let mut idx: Vec<usize> = (0..def_index.len()).collect();
    let mut rng = XorShift(seed | 1);
    for i in (1..idx.len()).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        idx.swap(i, j);
    }
    let take = time_sample.min(idx.len());
    let mut phi_us: Vec<u64> = Vec::with_capacity(take);
    let mut narrow_us: Vec<u64> = Vec::with_capacity(take);
    for &k in idx.iter().take(take) {
        let (gi, pl) = def_index[k];
        let g = &games[gi as usize];
        let mut state = HexoState::new();
        for &(q, r) in &g.moves[..pl as usize] {
            apply_placement(
                &mut state,
                Placement {
                    coord: HexCoord::new(q, r),
                },
            )
            .expect("legal replay");
        }
        let t0 = Instant::now();
        let (a, b) = phi_ab(&state);
        let _ = phi_lt_one(a, b);
        phi_us.push(t0.elapsed().as_nanos() as u64);

        let caps = SolveCaps {
            node_cap: 500,
            tt_bytes_cap: 256 << 20,
            semantic_horizon: state.placements_made() + 50,
        };
        let mut narrow = TssSolver::default();
        let t1 = Instant::now();
        let _ = narrow.solve(&state, &caps);
        narrow_us.push(t1.elapsed().as_nanos() as u64);
    }
    phi_us.sort_unstable();
    narrow_us.sort_unstable();
    let frac = |num: u64, den: u64| {
        if den == 0 {
            0.0
        } else {
            num as f64 / den as f64
        }
    };
    let med = |v: &[u64]| if v.is_empty() { 0 } else { v[v.len() / 2] };

    println!("=== LEAFW ES no-win screen (Phi<1) ===");
    println!(
        "LEAFW_ES def_nodes={def_nodes} phi_lt1={phi_lt1} frac={:.6} | developed(>=6 atk): nodes={def_dev} phi_lt1={phi_lt1_dev} frac={:.6}",
        frac(phi_lt1, def_nodes),
        frac(phi_lt1_dev, def_dev)
    );
    println!(
        "LEAFW_ES_TIME sample={take} phi_med_ns={} narrow_med_ns={} phi_vs_narrow={:.5}",
        med(&phi_us),
        med(&narrow_us),
        if med(&narrow_us) == 0 {
            0.0
        } else {
            med(&phi_us) as f64 / med(&narrow_us) as f64
        }
    );
    println!("LEAFW_ES_DONE");
}
