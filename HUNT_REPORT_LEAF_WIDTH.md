# Leaf-width hunt: what the narrow production leaf solver misses, and what it costs

**Branch** `hunt/leaf-width` · **HEAD** `1e082d40` (round-9b wide engine) · **2026-07-16**
**Scope:** measurement only. No production code changed, no proofs, no commits. The
only new code is a test-gated harness module
`packages/hexfield_eq/rust/src/tss_leaf_width_hunt.rs` (registered under
`#[cfg(test)]` in `lib.rs`).

---

## TL;DR

At MCTS-leaf node budgets, over 1,500 real human attacker-to-move positions:

1. **The narrow production leaf structurally misses 6–9% of forced wins that the
   wide VCF engine proves at the same node budget.** At cap 2,000: narrow proves
   a win in **6.7%** of attacker nodes, wide in **14.9%** — a **wide-only WIN share
   of 8.1%** (122 positions), with **zero** wins narrow found that wide missed.
2. **The miss is width, not budget.** Giving the narrow solver its *entire* budget
   on the WIN goal (`SolveGoal::Win` instead of `Both`) finds **exactly the same
   wins** (6.8%). The blind spot is the OR-generator (count≥3 windows only), as
   designed.
3. **The wall-clock story is dominated by transposition-table allocation, not
   search.** The narrow solver eagerly zeroes a ~128 MiB shared TT + local TT on
   every fresh solve; the wide engine skips the shared TT and grows its table
   lazily. With that allocation isolated (or amortized by solver reuse — the
   production pattern), the **warm** median costs are: narrow-WIN ≈ **0.07 ms**
   (matching the owner's figure exactly), wide ≈ **0.16 ms**. Wide's real cost is
   in the **p95 tail** on hard positions (up to ~170 ms at cap 2,000), which is
   exactly where it finds the wins narrow can't.
4. **No soundness alarm.** Zero WIN/LOSS contradictions across all 4,500 matched
   solve pairs (1,500 nodes × 3 caps).
5. **The ES no-win screen does not pay at leaves.** Exact-surd Φ<1 fires on
   **0.024%** of defender leaf nodes (25 / 104,452). Cheap per call (~0.03 ms) but
   far too rare to screen anything.

**Recommendation (short):** a leaf-width rung is worth building. The mechanism the
records implicate is squarely **count-2 pair-builds and quiet connectors** that
narrow's OR-generator never emits — not deep horizons alone. See
[Recommendation](#recommendation).

---

## Method

### The two solvers (both at HEAD in `tss_solver.rs`)

| | NARROW (production leaf) | WIDE (normative engine) |
|---|---|---|
| Construction | `TssSolver::default()` | `TssSolver::default()` + `set_width_options(WidthOptions::vcf_pair_complete())` |
| Goal via `solve()` | `SolveGoal::Both` → budget split `((n+1)/2, n/2)` between WIN and dual-LOSS | `SolveGoal::Both` under vcf → **all** budget to WIN (`(n,0)`) |
| OR-generator | empties of claimant count≥3 windows only | pair-complete VCF (sees count-2 pair-builds, quiet setups) |
| Engine | classic AND/OR proof-number DFS | `WidePnSearch` staged-deepening df-pn |
| TT | `split_tt_cap` → shared + local (both eagerly sized) | shared cap = 0; lazy local `WidePnSearch` TT |

Both are invoked exactly as the corpus gate invokes them
(`tss_spare_corpus.rs`, `tss_freq_hunt.rs`): a **fresh solver per solve**, results
read only from `DeepResult::status` (WIN / LOSS / UNKNOWN). Only proven statuses
count as "caught"; UNKNOWN is a result, not a verdict.

### Corpus and replay

- Source: `E:/Hexo-BotTrainer-hexgt/data/hexo-bootstrap-corpus/hexo_human_corpus.jsonl`
  — **6,902 games**, all decisive (winner ±1). Schema `{game_hash, moves:[[q,r]…],
  winner:±1, elo}`, opener always (0,0).
- Replay convention (P0 single opening placement, then alternating two-stone
  turns) is handled by the engine's phase machine: we simply `apply_placement`
  each `(q,r)` in order. Parser/loader ported from `tss_freq_hunt.rs`
  (branch `hunt/corpus-freq`, commit `3f66a410`).
- **Replay validated** (`leaf_width_validate_replay`): 300 fixed-seed decisive
  games replayed to the last move must be terminal with the recorded winner. A
  Hexo terminal is set *only* on six-in-a-line, so a terminal with the correct
  winner **is** the 6-in-a-row check. Result: `checked=300`, all pass, all 6,902
  games decisive.

### Sampling

- **Node = attacker-to-move leaf** = a non-terminal `FirstStone`-phase state (the
  side to move is about to spend a fresh pair; the natural VCF entry point). Pool:
  **212,356** such nodes across decisive games (`ply≤12`: 41,412 / `13–40`: 83,457
  / `>40`: 87,487).
- **Stratified by game phase**, three bands by `placements_made`: `ply≤12`,
  `ply13–40`, `ply>40`. Per band: deterministic Fisher-Yates shuffle
  (XorShift, per-band seed derived from master seed `0x9E3779B97F4A7C15`), take
  **500** → **N = 1,500**.
- No RNG on any scored/solved path; the sample set is fully deterministic.

### Caps and horizon policy

- `node_cap ∈ {500, 2000, 10000}`, both engines matched per cell.
- `tt_bytes_cap = 256 MiB` (spec).
- **`semantic_horizon = placements_made() + 50`** (fixed slack). Rationale: this
  makes **`node_cap` the sole binding compute constraint**, matching the MCTS-leaf
  framing where the leaf budget is *nodes*, not a ply deadline; it decouples the
  measurement from human game length. The wide depth cap is then a generous 50
  plies (`MAX_CERT_DEPTH = 256` is a further structural bound). **Verified
  non-binding:** quadrupling the slack to 200 at cap 2,000 leaves `wide_win`
  identical (223 → 223). Wide WIN rising monotonically with `node_cap`
  (12.7 → 14.9 → 16.1%) confirms the node cap, not the horizon, is what binds.
- Environment: Windows 11, native `cargo 1.95.0`, `--release`, `--test-threads=1`.
  Free RAM ≥ 8 GB enforced (one cargo process; RAM guard sleeps if it drops).

---

## Measurement 1 — miss-rate table

Regen: `TSS_LEAFW_PER_BAND=500 CARGO_TARGET_DIR=.target-hunt cargo test --release -p hexfield_eq leaf_width_miss_rate -- --ignored --nocapture --test-threads=1`
(raw: `LEAF_WIDTH_MISSRATE_RAW_256MiB.txt`).

### Aggregate (N = 1,500)

| node_cap | narrow WIN | wide WIN | narrow-WIN-only-goal | both UNKNOWN | **wide-only WIN** | narrow-only WIN | narrow LOSS | wide LOSS | contradiction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500   | 99 (6.60%)  | 190 (12.67%) | 99 (6.60%)  | 1267 (84.47%) | **91 (6.07%)**  | 0 | 43 | 31 | 0 |
| 2000  | 101 (6.73%) | 223 (14.87%) | 102 (6.80%) | 1234 (82.27%) | **122 (8.13%)** | 0 | 43 | 31 | 0 |
| 10000 | 102 (6.80%) | 241 (16.07%) | 102 (6.80%) | 1213 (80.87%) | **139 (9.27%)** | 0 | 46 | 31 | 0 |

- **Wide-only WIN share** (narrow ≠ WIN, wide = WIN) is the headline: what narrow
  structurally misses. It **grows with budget** (6.1 → 8.1 → 9.3%) — narrow's
  deficit widens as the wide engine is allowed to search deeper.
- **Narrow-only WIN = 0 at every cap.** Narrow never proves a win wide misses.
- **narrow-WIN-only-goal ≈ narrow WIN** everywhere → the `Both` budget split is
  *not* the cause; the miss is the count≥3-window OR-generator.
- `wide LOSS = 31` is not a dual search (wide is WIN-only under vcf); it is the
  root λ¹ immediate check finding the attacker already lost. It never coincides
  with a narrow WIN (contradiction = 0).

### Phase breakdown (WIN% narrow / wide / **wide-only**)

| band \ cap | 500 | 2000 | 10000 |
|---|---|---|---|
| `ply≤12`   | 0.00 / 1.80 / **1.80** | 0.00 / 4.00 / **4.00** | 0.00 / 4.00 / **4.00** |
| `ply13–40` | 7.40 / 17.40 / **10.00** | 7.40 / 19.60 / **12.20** | 7.40 / 20.80 / **13.40** |
| `ply>40`   | 12.40 / 18.80 / **6.40** | 12.80 / 21.00 / **8.20** | 13.00 / 23.40 / **10.40** |

- **Mid-game (`ply13–40`) is where the gap is largest** — wide-only WIN reaches
  **13.4%** of attacker nodes at cap 10k. This is the phase where forced wins
  first become available and narrow's generator is blindest.
- Early (`ply≤12`) has few forced wins at all (both engines mostly UNKNOWN), but
  wide still finds 4% that narrow finds 0% of.
- Late (`ply>40`) narrow does best in absolute terms (boards are dense, count≥3
  windows common) but wide still adds +6–10 pp.

---

## Wall-clock economics

This is the subtle part, and it inverts the naive reading. **The narrow solver's
measured median wall-clock is dominated by transposition-table allocation, not by
search.**

### As-measured, fresh solver per solve, 256 MiB TT (spec configuration)

| node_cap | narrow med / p95 | wide med / p95 | narrow-WIN med / p95 | wide/narrow med |
|---:|---:|---:|---:|---:|
| 500   | 13.68 ms / 25.1 ms  | **0.23 ms** / 47.7 ms  | 8.00 ms / 20.6 ms  | 0.02 |
| 2000  | 13.62 ms / 63.7 ms  | **0.22 ms** / 166 ms   | 8.00 ms / 55.6 ms  | 0.02 |
| 10000 | 13.68 ms / 273 ms   | **0.22 ms** / 279 ms   | 8.00 ms / 242 ms   | 0.02 |

The narrow median is **flat at ~13.7 ms regardless of cap**, while its *node*
median is only **35** (below). Thirty-five nodes cannot cost 13.7 ms. The cause,
confirmed below: `TssSolver::default().solve()` calls `split_tt_cap(256 MiB)` →
allocates and zeroes a ~128 MiB shared proof cache (262,144 slots) **plus** a
local table, and `Both` builds the local table twice (WIN + dual-LOSS attempts).
Wide (`vcf_pair_complete`) sets shared cap = 0 and grows its `WidePnSearch` TT
lazily, so it pays almost no allocation.

### Node-count distribution (deterministic)

| node_cap | narrow med / p95 / max | wide med / p95 / max |
|---:|---:|---:|
| 500   | 35 / 500 / 500   | 2 / 500 / 500   |
| 2000  | 35 / 2000 / 2000 | 2 / 2000 / 2000 |
| 10000 | 35 / 6378 / 10000 | **2 / 3084 / 10000** |

The median wide solve resolves in **2 nodes** (immediate VCF or immediate
exhaustion of the pair-complete tree at the root); the median narrow solve
explores 35. On the hardest positions wide is also **more node-efficient**
(p95 3,084 vs 6,378 at cap 10k).

### Allocation-isolated (4 MiB TT) — the warm / reused-solver economics

Re-running the identical 1,500-node sample with `tt_bytes_cap = 4 MiB` gives
**identical WIN/node counts** (one position differs only at cap 10k — a TT-eviction
artifact, which is why 256 MiB stays the official cap) but strips the allocation
cost. This is the honest proxy for a **reused/warm solver** — the production MCTS
pattern, where the TT is allocated once and amortized across thousands of leaves.

| node_cap | narrow med / p95 | wide med / p95 | **narrow-WIN med / p95** |
|---:|---:|---:|---:|
| 500   | 0.68 ms / 12.2 ms | 0.16 ms / 48.1 ms | **0.078 ms** / 12.9 ms |
| 2000  | 0.61 ms / 50.8 ms | 0.16 ms / 169 ms  | **0.073 ms** / 48.0 ms |
| 10000 | 0.60 ms / 258 ms  | 0.16 ms / 281 ms  | **0.085 ms** / 235 ms  |

Regen: add `TSS_LEAFW_TT_BYTES=4194304` to the Measurement-1 command
(raw: `LEAF_WIDTH_TIMING_RAW_4MiB.txt`).

**Reading it:**
- The owner's "narrow ≈ 0.07 ms" is the **warm narrow-WIN median** — reproduced
  here exactly (0.073–0.085 ms). Good anchor.
- At that same warm operating point, **wide's median is ~0.16 ms** — about **2×**
  a warm narrow-WIN leaf, and *faster* than warm narrow-`Both` (~0.6 ms, which
  pays for two local-table searches). Both are sub-millisecond at the median.
- **Wide's cost lives entirely in the p95 tail**: ~48 ms (cap 500), ~169 ms
  (cap 2000), ~281 ms (cap 10k) — the positions that consume the full node budget.
  These are exactly the wide-only wins. A leaf-width rung must therefore be
  **budget-gated** (a small `node_cap`, or a shallow first stage) to keep the tail
  bounded; at cap 500 the wide p95 is ~48 ms and it still delivers a 6.1% wide-only
  WIN share.
- **Is wide-at-2k viable inside an MCTS leaf?** At the median, yes trivially
  (~0.16 ms). At p95, ~169 ms is *not* a per-leaf budget you can pay on every
  visit — but you would only invoke it on the small fraction of leaves that are
  plausible attacker positions, and a lower cap (500) cuts the tail to ~48 ms
  while keeping most of the win. Wide-at-500 is the realistic leaf rung.

---

## Measurement 2 — width-record list

Machine-readable: **`LEAF_WIDTH_RECORDS.jsonl`** (worktree root, 122 records).
Each `wide_only_win` row: `game_hash, ply, band, winner, mover_is_p0,
narrow_nodes, wide_nodes, wide_cert_nodes, prefix` (the full replay prefix).
Definition: **narrow = UNKNOWN at cap 2,000, wide = WIN at cap 2,000.**

- **122 wide-only records; 0 narrow-only records; 0 contradictions.**
- Phase: 20 `ply≤12` · 61 `ply13–40` · 41 `ply>40`. Ply range 9–181, median 29.
- `wide_nodes`: min 2, median 114, p90 991, max 1,851.
- `wide_cert_nodes` (certificate size): min 3, median 55, p90 406, max 1,576.
- **84 of 122** are positions where the side to move went on to win the human game
  (wide proves the win is *already forced*); the other **38** are forced wins the
  **human missed**.

**Two mechanisms, one root cause.** Both trace to narrow's OR-generator only
emitting empties of count≥3 windows:

1. **Immediate structural blindness (the smoking gun).** 8 records have
   `wide_nodes ≤ 2, cert = 3` — wide sees an outright winning move at the root —
   while narrow ground through its **entire** budget (`narrow_nodes` up to 2,000)
   and still returned UNKNOWN. Example: `game_hash=3255c18654583d27`, ply 9, wide
   wins in 2 nodes (cert 3), narrow explores 5 nodes and gives up; and multiple
   ply-21/29/37 rows where narrow hits its 2,000-node cap on a move it can never
   generate. **41 of 122** resolve in ≤ 50 wide nodes — shallow VCFs invisible to
   narrow's generator. These are **count-2 pair-builds / quiet connectors**.
2. **Deep forcing lines.** The large-certificate rows (cert 883 / 1,225 / 1,576 at
   early ply, `narrow_nodes` 5–35) are long VCFs. Narrow's whole tree *exhausts*
   at ~35 nodes and returns UNKNOWN because its root generation is too narrow to
   even start the line; wide sustains a 1,500-node forcing sequence.

The converse count (narrow WIN, wide slower/UNKNOWN at 2,000) is **0** — there is
no position in the sample where narrow's speed buys a win wide misses. Narrow's
value is purely the cheap warm median on the ~93% of attacker nodes that are
both-UNKNOWN, not any unique proving power.

---

## Measurement 3 — cheap ES no-win screen

Regen: `CARGO_TARGET_DIR=.target-hunt cargo test --release -p hexfield_eq leaf_es_screen -- --ignored --nocapture --test-threads=1`.
Exact-surd ES potential Φ ported verbatim from `gap_raw_hunt.rs` / `tss_freq_hunt.rs`
(attacker = Player1, defender = Player0; `27·Φ = A + B√3`, Φ<1 ⇔ a sound instant
attacker-no-win region for the defender to move).

Over **104,452** defender (Player0-FirstStone) leaf nodes in decisive games, Φ<1
holds at exactly **25** — **0.024%** (1 in ~4,200). Restricting to *developed*
positions (≥ 6 attacker stones, where windows are meaningful) it is rarer still:
**5 / 90,648 = 0.0055%**. A Φ evaluation costs a median **32.5 µs** — about
**0.23%** of a (256 MiB) narrow solve, i.e. genuinely cheap. But screening 0.024%
of defender nodes saves 0.024% of solves: **ES screening does not pay at leaves.**
This confirms the prior figure (0.024%) exactly. The exact-surd Φ potential is a
useful *offline / opening* filter, not a per-leaf screen.

---

## Soundness

Zero WIN/LOSS contradictions across all 4,500 matched solve pairs (1,500 nodes ×
3 caps), at both 256 MiB and 4 MiB TT and at horizon slack 50 and 200. `wide LOSS`
(31, root λ¹) never coincides with `narrow WIN`; `narrow LOSS` (43–46) never
coincides with `wide WIN`. No alarm.

---

## Recommendation

**Build a leaf-width rung. The evidence is one-sided.** At matched node budgets the
wide VCF engine strictly dominates the narrow leaf on strength (wide-only WIN share
6.1 / 8.1 / 9.3% at cap 500 / 2k / 10k; narrow-only WIN = 0), and the miss is
**structural width, not budget** (`SolveGoal::Win` on the full budget finds nothing
extra). The narrow OR-generator's count≥3 restriction is leaving 8% of forced wins
on the table at a 2,000-node leaf budget — and the deficit *grows* with budget.

**Per cap:**
- **cap 500 — yes, this is the realistic leaf rung.** 6.1% wide-only WIN for a wide
  median of ~0.16 ms and a **p95 of ~48 ms**. Bounded tail, big strength gain.
- **cap 2000 — yes, if the tail is affordable.** 8.1% wide-only WIN, but p95
  ~169 ms. Viable only if invoked selectively (attacker-plausible leaves) or as the
  second stage of a staged deepening.
- **cap 10000 — offline / analysis only.** 9.3% wide-only WIN but p95 ~281 ms; too
  heavy for a per-visit leaf.

**What to build (mechanism the records implicate):** the records point first and
hardest at **count-2 pair-builds and quiet connectors** — 41/122 misses are ≤ 50
wide nodes and 8 are outright 2-node root wins that narrow's generator can never
emit. So the highest-leverage, cheapest rung is **not** a full `WidePnSearch`
port: it is **widening the narrow OR-generator to include count-2 pair-build
empties (and quiet threat-creating setups)**, i.e. the `vcf_pair_complete`
generator applied to narrow's fast engine. That captures the shallow tail (the
bulk of the misses) at near-narrow cost. The deep-forcing-line misses (large
certs) need the staged df-pn depth and are the province of a heavier, budget-gated
second stage.

**Also worth flagging to the owner (not the mission, but load-bearing):** as
currently written, a *fresh* `TssSolver::default()` per leaf pays ~13 ms of TT
zeroing that dwarfs its 35-node search. Any leaf integration must **reuse a
persistent solver** (or shrink `tt_bytes_cap`) to hit the ~0.07 ms warm median.
The wide engine, which skips the shared TT, does not have this cliff.

**ES screen:** do not build a per-leaf ES no-win screen. Φ<1 is real and cheap but
fires on 0.024% of defender nodes — it screens nothing at a leaf.

---

## Regeneration

All commands from the worktree root
`E:\Hexo-BotTrainer-hexgt\.claude\worktrees\hunt-leaf-width`, Bash shell,
`CARGO_TARGET_DIR=.target-hunt`, `--release`, `--test-threads=1`. (Windows
PowerShell: set env vars with `$env:NAME=…` first.)

```bash
# Build the test harness
CARGO_TARGET_DIR=.target-hunt cargo test --release -p hexfield_eq --no-run

# Replay-convention validation (300 games)
CARGO_TARGET_DIR=.target-hunt cargo test --release -p hexfield_eq \
  leaf_width_validate_replay -- --ignored --nocapture --test-threads=1

# Measurement 1 + 2 (OFFICIAL: N=1500, 256 MiB, horizon ply+50) — writes LEAF_WIDTH_RECORDS.jsonl
TSS_LEAFW_PER_BAND=500 CARGO_TARGET_DIR=.target-hunt cargo test --release -p hexfield_eq \
  leaf_width_miss_rate -- --ignored --nocapture --test-threads=1

# Allocation-isolated / warm-solver timing (identical sample, 4 MiB TT)
TSS_LEAFW_PER_BAND=500 TSS_LEAFW_TT_BYTES=4194304 \
  TSS_LEAFW_RECORDS_PATH=/tmp/tt4m.jsonl \
  CARGO_TARGET_DIR=.target-hunt cargo test --release -p hexfield_eq \
  leaf_width_miss_rate -- --ignored --nocapture --test-threads=1

# Horizon-sensitivity check (slack 200, cap 2000; wide_win must stay 223)
TSS_LEAFW_PER_BAND=500 TSS_LEAFW_CAPS=2000 TSS_LEAFW_HORIZON_SLACK=200 \
  TSS_LEAFW_RECORDS_PATH=/tmp/h200.jsonl \
  CARGO_TARGET_DIR=.target-hunt cargo test --release -p hexfield_eq \
  leaf_width_miss_rate -- --ignored --nocapture --test-threads=1

# Measurement 3 — ES no-win screen
CARGO_TARGET_DIR=.target-hunt cargo test --release -p hexfield_eq \
  leaf_es_screen -- --ignored --nocapture --test-threads=1
```

Env knobs (all have documented defaults): `TSS_LEAFW_PER_BAND` (500),
`TSS_LEAFW_CAPS` ("500,2000,10000"), `TSS_LEAFW_TT_BYTES` (256 MiB),
`TSS_LEAFW_HORIZON_SLACK` (50), `TSS_LEAFW_SEED` (0x9E3779B97F4A7C15),
`TSS_LEAFW_RECORD_CAP` (2000), `TSS_LEAFW_RECORDS_PATH`, `TSS_LEAFW_CORPUS`.

## Files

- `packages/hexfield_eq/rust/src/tss_leaf_width_hunt.rs` — harness (new, test-only).
- `packages/hexfield_eq/rust/src/lib.rs` — one-line `#[cfg(test)] mod` registration.
- `HUNT_REPORT_LEAF_WIDTH.md` — this report.
- `LEAF_WIDTH_RECORDS.jsonl` — 122 width records (machine-readable).
- `LEAF_WIDTH_MISSRATE_RAW_256MiB.txt` — official run stdout.
- `LEAF_WIDTH_TIMING_RAW_4MiB.txt` — allocation-isolated timing stdout.
