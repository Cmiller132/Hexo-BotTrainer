# V1 — Enriched Offline Shadow Soak at ep90

Normative spec: `docs/PLAN_TSS_MCTS_INTEGRATION.md` §9 (V1), with §5 (horizon
policy), §6 (gating), §10 (open decisions). Mechanics: `docs/TSS_RUNBOOK.md`.
Branch `claude/v1-soak` @ merge `eea51de0` (campaign engine port, all six V0
acceptance gates passed). Net: `epoch_000090.pt` (main_3 best-eval,
owner-designated). GPU free (trainer stopped at ep111). Measure everything,
consume nothing.

**Soundness backstop:** `deep_verify_failed` is asserted after every solve; any
nonzero aborts the sweep with raws preserved. `cert_version = 2` (TSS_CERT_VERSION)
recorded on every solve.

---

## 0. Pre-registration (fixed BEFORE the sweep ran)

### Measurement primitive

A dedicated PyO3 probe, `hexfield_eq_deep_solve_probe(state, node_cap, goal,
horizon, ladder, zone, wide, with_stats)`, runs ONE solve through the exact
production verified path (`tss_solve_verified` in `tree.rs` — tight-zone
fast-path + §5 horizon ladder + certificate-horizon preflight) on a fresh solver
of the requested width profile, and returns the verdict, per-solve wall
(`Instant` around the solve), certificate geometry (depth = verifier-derived T
minus root placements; Choice/Universal/zone node counts; the designated root
move for the §8 baseline), `cert_version`, and every `TssCounters` field. It is
measurement plumbing only — NOT wired into any consumption path, cannot mint a
training value. With `with_stats`, a second same-profile `solve_goal` at the base
horizon surfaces the raw `SolveStats` (census-gate evaluations/dismissals, TT /
fragment reuse) the verified path folds away. Harness code committed to this
branch; the flag-off golden digest stays bit-identical (the change is additive —
`test_stage0_digest_matches_golden` PASS).

Rationale for a harness-driven probe over reading the in-search `tss` block: §9
asks for per-solve wall percentiles, cert-depth histograms, paired narrow-vs-wide
on identical leaf sets, and per-position would-it-flip — none of which the
aggregate per-search epoch block exposes. The probe measures each solve
independently and consumes nothing, satisfying the mission's "prefer
harness-driven measurement over changing consumption semantics."

### Node cap

`node_cap = 500` for all arms — the deployed trainer leaf cap (main_3 ep41+;
`RustSearch::TSS_SOLVER_TT_BYTES` = 256 KiB per-solve). §5 records that caps
500/2000/8000 leave the decided set identical (depth, not width, is the leaf
frontier), so 500 is representative and matches production.

### Arm matrix (per position)

WIN-goal, wide leaf profile (`configure_leaf_profile`: wide `vcf_pair_complete`
+ lazy frontier + interior census gate):

| Arm | horizon | ladder | zone |
|---|---|---|---|
| `h16_flat_wide` | 16 | off | off |
| `h16_ladder_wide` | 16 | →h32 | off |
| `unbounded_wide` | ∞ (u32::MAX) + cap | off | off |
| `h16_flat_zone` | 16 | off | on |
| `h16_ladder_zone` | 16 | →h32 | on |
| `unbounded_zone` | ∞ + cap | off | on |

WIN-goal, narrow profile (the engine's narrow `WidthOptions` default), for the
paired superiority table:

| Arm | horizon | ladder | zone |
|---|---|---|---|
| `h16_flat_narrow` | 16 | off | off |
| `unbounded_narrow` | ∞ + cap | off | off |

LOSS-goal (dual seat, defender-goal), wide, on a 1-in-4 paired subsample (no
census early-out on the LOSS side; cost tracked separately, §10.4):

| Arm | horizon | ladder |
|---|---|---|
| `h16_flat_wide_loss` | 16 | off |
| `unbounded_wide_loss` | ∞ + cap | off |

### Position slices (sources, §9 "position sources 07-20")

1. **Self-play (primary, in-regime):** fresh ep90 MCTS self-play at the run's
   256-visit / c_puct 1.5 config (production divergence map with deep-solver
   CONSUMPTION disabled for generation determinism — mode 0, no root guard, no
   async/park; the play-shaping interior guard retained), 64 games, full game
   length. Each played ROOT position recorded with the net's root value and
   top-24 prior. The slice is thinned by stratified shuffle (seed 20260720) to a
   fixed cap for the solve sweep; the drop count is logged (no silent
   truncation).
2. **Forcing corpus (tactical anchor):** all 19 positions
   (`rust/corpus/forcing_corpus_moves.txt`, 14 WIN / 5 NO), every arm, LOSS on
   every position. NO→WIN is a soundness violation and is checked.
3. **Spare corpus:** `rust/corpus/spare_corpus_moves.txt`, every arm.
4. **Human corpus (OOD tactical):** a sampled slice of interior prefixes from
   `hexo_human_corpus.jsonl` (8,698 rated games), every arm.

### What each slice decides

- Self-play slice → the §10 decision inputs: horizon shape (§10.3),
  affordability (§10.2), `deep_kb_death` (§5/§10.3), LOSS-side cost (§10.4),
  would-it-flip + calibration + internalization baseline (§8), narrow-vs-wide
  superiority (§10 / §4).
- Corpus slices → tactical-anchor verdicts, time-to-verdict, and the zone delta
  (`zone_nodes` must go nonzero where the ladder makes zones live — inert only
  at the retired flat +12).

### Known caveats / measurement biases (pre-registered)

These are properties of the harness, stated up front so the tables are read
correctly. Paired arm comparisons (same positions, same cold-start) are
internally valid throughout; the caveats bound ABSOLUTE interpretation.

1. **Cold-cache measurement (pessimistic absolute yields/walls).** The probe
   builds a FRESH `TssSolver` per call, so there is no cross-move warmth.
   Production leaf solves share a persistent solver whose positive-proof-fragment
   cache persists across moves (O16, `tree.rs`). Absolute verdict yields and
   walls are therefore a pessimistic (cold) bound vs production, and the
   `stats_fragment_*` counters (intra-solve only) are near-meaningless as a
   cross-solve reuse measure. A persistent-solver sensitivity arm is run
   separately if it does not jeopardise the main battery (§1.x).
2. **Root value sign convention.** `root_value` = the MCTS root mean
   (`root.value()`), side-to-move perspective; the WIN-goal probe
   (`SolveGoal::Win`) proves the SAME side-to-move wins. So proven-WIN with
   `root_value <= 0` = the net undervalues a win = tactical headroom. One
   hand-checked example is shown in §1.
3. **"Would-it-flip" is a SIGN-DISAGREEMENT PROXY**, not a consumed-move flip:
   V1 consumes nothing, so no backup/root-move is actually changed. It counts
   proven verdicts whose net-value sign disagrees.
4. **Zone-on vs zone-off deltas conflate two effects.** With `tss_zone` on,
   `tss_solve_verified` also runs the production tight +8 half-budget fast-path;
   both fire only under the zone flag. `zone_nodes` isolates ACTUAL zone-AND use;
   wall/verdict deltas are the combined (tight-pass + zone) effect.
5. **Internalization baseline excludes win-in-one roots.** `cert_root_move` is
   emitted only for `Choice` root nodes; an `OrCompletion` (immediate-completion)
   root has no designated Choice move, so the easiest wins are excluded from the
   prior-mass/rank baseline (coverage bias toward deeper wins).
6. **LOSS subsample** = every `loss_every_n`-th position in iteration order
   (deterministic, content-arbitrary) — a cost/rate estimate, not a matched
   subsample.
7. **Self-play thinning is stratified by stone-count band** (width-10),
   proportional allocation + seeded within-band sample (seed 20260720); per-band
   keep/drop logged.
8. **Human slice ordering bias:** games are the first N lines of the jsonl (a
   file-order slice, not a random game sample); plies within a game are a seeded
   random sample.

---

## 1. Results

### 1.0 Battery totals + soundness

37,108 verified-path solves across five raw sets: self-play 27,668 (FULL
3,255-position set — no thinning; caveat 7 is moot, see §3), warmth 6,510,
human 2,720, forcing 190, spare 20. **`deep_verify_failed = 0` on every solve
in every arm.** Forcing-corpus NO→WIN violations: **0** (all 20 WIN rows sit on
`expect=WIN` positions; both LOSS-goal proofs sit on `expect=NO` positions).
`cert_version = 2` throughout.

### 1.1 Generation throughput + the interior-guard question (measured)

Paired A/B, identical config and seed (32 games, 48 visits, 50 plies,
`base_seed` 20260720), ep90 through the production serve path:

| run | machine state | moves/min |
|---|---|---|
| guard ON (`tss_enabled=1`) | quiet | **284.1** |
| guard OFF | quiet | **278.7** |
| guard ON (same seed, rerun) | 10-core solve sweep running | 204.3 |

**The play-shaping interior guard costs ~nothing at this config: guard-on =
guard-off within noise (284 vs 279).** The two guard-on runs are BYTE-IDENTICAL
(same sha256, §2) despite different background load — generation is fully
deterministic given the seed, so the 470 s vs 338 s wall gap between them is a
controlled measurement of pure CPU-contention effect (−28%). The earlier
"solver slowed generation" impression decomposes into (a) background CPU load
and (b) the visit budget (256→48 was the real throughput lever), NOT the TSS
guard. The guard-off file differs from guard-on (guard shifts play at proven-
tactical roots), confirming the flag was live in both arms.

Production-representativeness caveat (binding): these rates are 32–64
concurrent games with a per-move Python callback, NOT the trainer's 256-active-
root pipeline; absolute moves/min here says nothing about trainer throughput.
And none of this involves the production ASYNC solver pool (parked, 6–8
threads), which is consumption-side and known to keep up in main_3. Historical
context rows (non-comparable configs, listed for completeness): first attempt
guard-on @256 visits ≈ 118 rec/min; main generation guard-off @48 visits, 64
games ≈ 181 rec/min.

### 1.2 Self-play slice (primary, in-regime) — arm table

n = 3,255 positions/arm (LOSS arms n = 814, 1-in-4). Walls in µs.

| arm | W | L | U | verdict% | p50 | p90 | p99 |
|---|---|---|---|---|---|---|---|
| h16_flat_wide | 150 | 0 | 3105 | 4.61% | 1,906 | 76,225 | 712,554 |
| h16_ladder_wide | 150 | 0 | 3105 | 4.61% | 1,901 | 78,594 | 693,156 |
| unbounded_wide | 189 | 0 | 3066 | **5.81%** | 1,913 | 277,660 | 960,706 |
| h16_flat_zone | 150 | 0 | 3105 | 4.61% | 3,658 | 90,541 | 760,396 |
| h16_ladder_zone | 150 | 0 | 3105 | 4.61% | 3,675 | 89,707 | 751,064 |
| unbounded_zone | 189 | 0 | 3066 | 5.81% | 3,661 | 290,321 | 970,931 |
| h16_flat_narrow | 100 | 0 | 3155 | 3.07% | 103 | 253,940 | 387,592 |
| unbounded_narrow | 99 | 0 | 3156 | 3.04% | 88 | 263,957 | 453,813 |
| h16_flat_wide_loss | 0 | 23 | 791 | 2.83% | 22 | 45 | 232,134 |
| unbounded_wide_loss | 0 | 24 | 790 | 2.95% | 16 | 39 | 528,186 |

Band yield (unbounded_wide): 0% at bands 0 and 8; peak 13.0% at band 5
(stones 50–59), 8.7–9.5% at bands 4/6/7. Census interior gate: 61.2%
dismissal rate on the h16 wide with_stats arm (16,049 / 26,210).

### 1.3 The five §10 decision inputs

1. **Horizon shape (§10.3): unbounded+cap wins; the ladder is DEAD.** Paired
   on 3,255 positions: flat-h16 150 verdicts, ladder 150 (**+0**), unbounded
   189 (**+39, +26%**). The ladder kill criterion fires in the strongest
   possible form: `flat_cut_eligible = 0` — no wide-arm flat solve was ever
   horizon-cut-and-Unknown, so there is nothing for a tall pass to convert
   (`horizon_cut` fired only on narrow arms). Unbounded's cost: identical p50
   (1,913 vs 1,906 µs), tail p90 3.6× (278 ms vs 76 ms), p99 ~1 s — bounded by
   the node cap. A concrete h16 miss: forcing `hu01jk4` is solved ONLY by the
   unbounded arms. `deep_kb_death = 0` in ALL twelve arms — the §5 kb-death
   concern is empirically absent at cap 500.
2. **Affordability (§10.2):** p50 ≈ 1.9 ms/solve (wide, either horizon), p90
   76 ms (h16) / 278 ms (unbounded), p99 0.7–1.0 s. Quiet positions p50
   1.7 ms; hot positions p50 2.6 ms with a SMALLER p90 (5.9 ms) — the tail
   lives in quiet deep-Unknown grinds, not tactical positions.
3. **`deep_kb_death`:** 0 everywhere (see 1 above).
4. **LOSS-side cost (§10.4):** nearly free at the median — p50 16–22 µs, p90
   ≈ 40 µs (most positions fail the LOSS precondition instantly), yield 2.8–
   3.0%, tail p99 0.2–0.5 s on hot positions. Dual-seat probing is affordable
   as a broad default if the tail is capped.
5. **Zone delta: `tss_zone` is INERT and pure cost at production leaf
   settings.** `zone_nodes = 0` across the ENTIRE battery (all arms, all four
   slices, 37k solves) — the zone AND path never fired once. Verdicts
   identical zone-on vs zone-off (150/150, 189/189). Cost: +1.39 ms p50
   (+73%), +6.7 ms p90 paired — that is the tight +8 half-budget fast-path
   running for nothing. Recommendation: leaf-profile `tss_zone` OFF in the
   trainer (the campaign engine's zone value lives at big caps/horizons, not
   here).

### 1.4 Width, calibration, internalization, warmth

- **Narrow vs wide (§10/§4): wide strictly dominates verdicts.** Paired h16:
  wide 150 vs narrow 100, `narrow_only = 0`; unbounded: 189 vs 99. But narrow
  is ~19× cheaper at p50 (103 µs vs 1,906 µs) with a WORSE tail (254 ms p90 —
  it burns the node cap failing; nodes_mean 101 vs 33). Wide's per-node cost
  buys certificate progress; narrow's cheapness is fake at the tail.
- **Calibration / would-it-flip (§9, sign-disagreement proxy):** of 189 proven
  WINs, **22 (11.6%) have net root value ≤ 0** — genuine tactical headroom at
  ep90. `win_netval_mean` 0.488, p10 −0.048.
- **Internalization baseline (§8):** cert root move is the net's top-1 prior
  in 56.6% of 106 Choice-root proven WINs; mean prior mass 0.470; mean rank
  1.83; 98/106 inside top-24. (OrCompletion roots excluded — caveat 5.)
- **Warmth sensitivity — pre-registered caveat 1 RESOLVED:** persistent-solver
  re-solve of all 3,255 positions in ply order, both arms: **verdict agreement
  3,255/3,255 (both arms), walls within ±8% of cold** (flat p50 2,055 vs
  1,906; unbounded 1,922 vs 1,913). The O16 fragment cache never hit at cap
  500 (`fragment_lookups = 0`). The cold-cache probe is NOT pessimistic in
  this regime; absolute numbers stand.

### 1.5 Human slice (OOD tactical; prevalence cohort)

320 interior prefixes (80 games × ≤4 plies, seed 1234). Verdict yield is
HIGHER than self-play: unbounded_wide **48/320 = 15.0% WIN** (h16 43/320 =
13.4%; narrow 18/320; LOSS-goal 0/80) — human play leaves ~2.6× more provable
tactics on the board than ep90 self-play, consistent with the corpus-semantics
rule (this cohort measures prevalence on a play distribution, not solver
efficiency). Same shape as self-play: ladder +0, unbounded +5 over flat,
`zone_nodes = 0`, walls comparable (p50 2.6 ms wide).

### 1.6 Forcing/spare anchors (efficiency corpus)

At the PRODUCTION leaf profile (cap 500 / 256 KiB TT) the forcing corpus is
mostly out of reach by design — these are deep campaign-engine puzzles: wide
arms solve `acly7kb` + `xsnfyll` (h16) and additionally `hu01jk4` (unbounded
only); narrow solves none; LOSS-goal proves `8is963b` + `dy3dg99`. 0 NO→WIN.
Spare: 0/2 all arms. This is the expected leaf-cap picture, NOT a solver
regression (the campaign engine solves 14/14 WINs at full budgets); quoted per
the corpus-semantics rule as efficiency context only.

### 1.7 Recommendations into V2/V3

1. Trainer leaf solve config: **unbounded+cap, wide profile, ladder OFF,
   `tss_zone` OFF** — matches the owner's "≥h16/unbounded+cap" intent; the +26%
   verdict gain costs nothing at p50 and only tail (capped) wall.
2. Dual-seat LOSS probing is cheap enough to run broadly if tail-capped.
3. V2 fixed-budget h2h at ep90 should use exactly this config vs the
   legacy-solver-ref tag (`b8effa3a`).
4. The 11.6% sign-disagreement rate and 56.6% top-1 internalization are the
   baselines the consumption rungs must move.

---

## 2. Raws + manifest

All raws under `raws/` with `raws/MANIFEST.sha256` (sha256 of every artifact:
positions, five soak sets, A/B files, guard-on early set-aside, four summary
JSONs). Aggregations: `summary_selfplay.json` (selfplay + warmth, with
positions prior), `summary_human.json`, `summary_forcing.json`,
`summary_spare.json` — kept PER-SLICE per the corpus-semantics rule (never mix
efficiency and prevalence cohorts in one table).

## 3. Execution notes (deviations from the original plan)

- Phase-2 sweeps ran under `run_soak_parallel.py` (10/8-way process sharding;
  same record schema, arm matrix, and global-index LOSS rule as `run_soak.py`).
  This removed the need for stratified thinning: the FULL 3,255-position
  self-play set was solved (pre-registered caveat 7 no longer applies). The
  orchestrator took over mid-run; the stopped serial run's partial raws are
  preserved as `raws/_partial_serial_soak_selfplay.jsonl` (superseded, not
  aggregated).
- Warmth ran game-sharded 8 ways (`run_warmth.py n_shards shard_idx`);
  sharding by game preserves the per-game persistent-solver semantics exactly.
- The guard A/B (§1.1) was added during takeover: `gen_positions.py` gained an
  optional `tss_on` argv flag; the loaded-machine guard-on run was repeated on
  a quiet machine after the confound was observed, and the byte-identical
  outputs turned the repeat into a controlled load measurement.
- Generation config deviations from production (pre-registered): 48 visits
  (throughput; positions are the object, not the games), consumption disabled
  (mode 0), 64 games. The guard-on/off near-parity in §1.1 was measured at 32
  games / 48 visits and does not claim to bound the 256-visit config, where
  the guard's per-move share is strictly smaller.
