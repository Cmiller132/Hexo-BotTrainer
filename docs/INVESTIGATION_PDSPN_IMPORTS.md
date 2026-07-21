# INVESTIGATION — pdspn/hexo-solver imports for the TSS engine

Date: 2026-07-21. Owner-directed (dispositions below). Prepared for Codex
lanes when usage frees (~today).

**Owner goal (verbatim intent):** improve the PRODUCTION solver — much more
efficient and better. Main use case = production runs (cap-500 leaves in
main_4); it must also be good at deep solves (offline labeling, atlas,
94gnnol-class disproofs).

**Scope guard for lane authors:** all four items below are ordering/memory
levers with ZERO verifier surface. The verifier (`tss_verify.rs`) is not
touched by any of them. Every experiment inherits the standing gates:
flag-off golden-digest bit-identity, the 6,443-position identity battery for
any default-flip, harness adoption metric = verified WIN/LOSS coverage on
fixed position sets, bench for anything touching production knobs.

**Production-honesty note (read first):** at production shape (cap 500,
256 KiB TT) the V1 battery measured TT hit/entry ≈ 0.01 (near-pure overhead)
and quiet solves self-terminate in ~0.1–4 ms. Items 1–4 therefore mostly
attack the DEEP operating point (cap 2k–100k, 256 MiB–2 GiB), which is where
today's evidence says we lag. The production-wall levers remain the P7
sequels (second_candidates churn ~8%, first-candidate enumeration ~15%,
D_FORCED_GEN ~20%, P6 cross-node memoization) — do not sell a deep-only win
as a production win.

## Evidence base: first direct matched-host battery vs hexo-solver (2026-07-21)

Source engine: `hexo-solver` crate of github.com/SootyOwl/hexo-strix
(drivers: `idtt` = production ID+TT solver, `dfpn` = Nagai df-pn with 1+ε,
`pdspn` = Winands PDS-PN two-level). Same 19-position forcing corpus (15
fixtures verified stone-identical to their repo's originals; the corpus
descends from their race corpus — tss_corpus.rs header). Same host, both
sides under live main_4 trainer load (walls load-inflated ~equally; verdicts
deterministic). Their side: 120 s wall / 50 M nodes / 256 MB TT per
position+driver, pinned binary from a fresh clone. Our side: canonical
512 MiB profile, fixed 10k→100k→1M ladder (0l4291i excluded — needs the
2 GiB official profile; RAM-gated while the trainer ran).

Headline rows:

- **Their deployed self-play config** (wide, depth 6, 2,000 nodes, root-only)
  decided **5/19** — missed 11 of 14 wins, incl. mates-in-4/5/6.
- **Our gate: 13/13 available wins PROVEN with verified certs** (269 s total
  under load), plus **2 of 5 "NO" rows proven as dual-certificate LOSSes**
  (a claim their engines cannot express). Zero false claims.
- **pdspn decided 19/19** (0l4291i WIN 305 s at 256 MB; 94gnnol NO 25 s).
- **The seeding datum (motivates item 1):** on 0l4291i, dfpn burned
  50,000,001 kernel nodes in 16 s and FAILED; pdspn — identical MID
  recursion, identical kernel — expanded **1,058 level-1 nodes** and WON.
  The only difference is bounded-probe initialization of frontier pn/dn.
  ~4 orders of magnitude fewer informed steps.
- **Width trade-off measured:** their engines prove "No" on 94gnnol (25 s)
  and mvp2lvc (1.5 s) where we sit Unknown at 1M nodes/512 MiB — our wider
  (more win-complete) attacker universe makes disproof-by-exhaustion
  proportionally harder. l9mxn59 width-exhausts for us in 25 ms, so the gap
  is position-dependent, not structural.
- **Easy-win latency:** their idtt is typically 2–5× faster on sub-second
  wins. Two known taxes on our side: fresh-per-rung ladder methodology
  (~30% measured historically; `CapResumeSession` exists but is
  cfg(test)-gated) and cert emission + independent verification inside our
  timed wall.

Raw artifacts (session-local; regen recipe in the appendix):
`%TEMP%/hexo-strix-clone/strix_battery.csv`, `/tmp/our_corpus_gate18.log`,
`/tmp/our_corpus_gate.log` (aborted 19-position run, 0l ladder walls).

## Item 1 — PN²-style bounded-probe pn/dn initialization

**Owner disposition:** uncertain value for the MCTS/production path; test on
DEEP solves if promising.

Their mechanism (`prover/pdspn.rs`, `prover/dfpn.rs solve_mode(pn2=true)`):
first descent into any frontier node not yet in the TT runs a bounded
best-first PN search (`pn2_nodes` = 50k default), stores ONLY the resulting
root (pn, dn, work) into the TT, discards the level-2 tree.

Our plug point: `WidePnEntry.prior` (tss_solver.rs, WidePnPrior — currently
static state-derived, span ~1..37, restored on depth-cutoff reopen). The
R-ORDER-PRIOR ordering-hints mechanism is an alternative consumer but
reorders only; prior injection steers pn/dn arithmetic directly, which is
what pdspn demonstrates matters.

Soundness class: initialization/ordering only — priors never become
verdicts; proofs still close through the normal machinery + verifier.
Flag-off bit-identity mandatory.

Why it may finally clear our ordering-oracle bar: SOLVER_NOTES P5 rejected
ep90 policy priors (12↑/26↓) and threat statics (19↑/99↓) and set the bar at
"proof-participation data or nothing." A bounded probe IS proof-participation
data, measured on the exact subtree.

Experiment (deep lane): A/B at cap 50k–1M on (a) the 95 cap-bound grinds,
(b) 0l4291i + lz60mfb + zrugh2x, (c) the two Unknown NO rows (94gnnol,
mvp2lvc — probe-seeded DISPROOF is where pdspn shone). Probe cap sweep
{128, 500, 2000}; probe nodes COUNT against the total budget (else the A/B
is rigged). Metrics: verdict coverage at fixed total budget, nodes-to-
verdict, wall. Kill: no coverage gain at matched total budget.

## Item 2 — 1+ε child thresholds (Pawlewicz & Lew 2007)

**Owner disposition:** same scope as item 1, and "I think this is a good
idea."

Their mechanism (dfpn.rs:34,282,342): EPS = 0.25;
`child_thpn = thpn.min((1+ε)·second_best_pn)` instead of `second_best + 1`.
Cuts TT re-expansion thrash; their df-pn and pdspn both run it.

Our state: threshold advance is +1 while priors span 1..37 — the exact
mismatch RESULTS_LOG names as ranked candidate 2 (Kishimoto survey). A
threshold-delta scaffold ALREADY EXISTS in the engine
(`ThresholdDelta::from_env`, `TSS_THRESHOLD_COUNTERS`, cfg(test)) from the
R-TS1 round. ⚠ CAVEAT for the lane: **R-TS1 closed "threshold scale null"**
in the campaign — the lane must first read that round's raws/report and
state precisely what WAS measured null (scale bands at which budgets, which
cohort) before re-running; the pdspn empirical success justifies exactly ONE
clean re-test at deep budgets (their ε=0.25 semantics, not a rescaled band),
not a reopened family. If R-TS1 already covered (1+ε) semantics at deep
budgets, record that and close this item without new cargo.

Experiment: counter-first (sentinel/threshold-cross counters already in the
engine), then A/B nodes-to-verdict + coverage at cap 50k–1M on the same
cohort as item 1. Production expectation: ~nil (cap-500 searches rarely
re-expand enough to thrash); measure only at deep budgets unless counters
say otherwise.

## Item 3 — Work-weighted TT replacement — INVESTIGATE

**Owner disposition:** investigate.

Their mechanism (`prover/pn.rs ProofTt`): 2-way set-associative table,
`Slot { key, pn, dn, work }`; `work` records effort invested in the entry,
replacement protects expensive knowledge under saturation.

Our observed failure mode today: 0l4291i at 512 MiB — `peak_tt_bytes` pegged
at cap by the 1M rung, verdict Unknown; pdspn decided the same position with
HALF that memory. Memory ceiling, not node budget, is the visible binding
constraint on our deep tail.

Investigation steps:
1. AUDIT: document our full-key TT's actual replacement policy under
   byte-cap saturation (tss_solver.rs, "Full-key transposition table"
   section ~:10389) — what is evicted, what survives, is any work/effort
   signal consulted? (Genuinely unknown at time of writing — the audit is
   the deliverable, before any design.)
2. INSTRUMENT (cfg(test)): eviction counts + "work lost per eviction" on the
   0l/94gnnol ladder.
3. PROTOTYPE work-weighted (or depth/proof-size-weighted) eviction; A/B
   verdict + wall at 256/512 MiB on the deep set.
Production relevance: ~nil (TT near-pure overhead at cap 500). Deep lane
only.

## Item 4 — Partial-result (pn, dn, work) summaries for Unknown subtrees — INVESTIGATE

**Owner disposition:** investigate.

Idea: extend the fragment-store philosophy (currently PROVEN-only, env-gated
off, measured zero verdict flips at cap 500) to store bounded-cost summaries
of INCONCLUSIVE subtrees: exact key → (pn, dn, work). pdspn's memory
discipline (discard the sub-tree, keep one root summary) bounds cost per
node by construction — compatible with the 256 KiB-per-solver production law
if it ever graduates, but target the deep lane first.

Design questions for the lane (answer before code):
- Keying: `WidePositionKey` exact-key reuse; interaction with
  `deferred_by_position` lazy thunks.
- Staleness under staged deepening: a (pn,dn) summary computed at depth
  stage d is not valid at stage d' > d (DepthCutoff/reopen semantics) —
  summaries must carry the stage or be restricted to unbounded-horizon
  solves (production/atlas profile is unbounded, so this may be moot — state
  it precisely).
- Interaction with dual_pass (WIN-goal summaries must not leak into the
  LOSS-goal attempt's init — goals search different claims).
- Relation to P6 cross-node generation memoization: candidate-generation
  reuse (~8–20% wall) and pn/dn summaries are separable — do not entangle
  the lanes.
Soundness class: initialization only, same as item 1.

## Sequencing recommendation (for the Codex round)

1. Item 2 first (cheapest: read R-TS1, counters, one A/B — possibly closes
   without code).
2. Item 1 (the big expected win; the 0l/94gnnol data is the motivation).
3. Item 3 audit (small, informs both 1 and 4's memory story).
4. Item 4 design-doc-only round, gated on 1+3 results.
Also queued independently: 0l4291i official 2 GiB gate rerun on the
now-quiet host (settles the last pdspn-vs-us row), and the reference-
capstone honesty note in RESULTS_LOG can then be updated with today's
matched-host table (paper work itself remains deferred).

## Appendix — regen recipe for the Strix-side battery

```
git clone https://github.com/SootyOwl/hexo-strix.git
# scratch crate beside it:
#   Cargo.toml: [dependencies] hexo-solver = { path = "hexo-strix/hexo-rs/hexo-solver" }
#   main.rs: for each position JSON (io::Position::load) × driver in
#   {prod: idtt wide depth6 budget2000; idtt/dfpn/pdspn: 120s, 50M nodes,
#   256MB TT via ProverConfig}, call prover::run, print CSV
#   (id,driver,verdict,depth,elapsed_s,nodes,tt_hits).
# Positions: generated from packages/hexfield_eq/rust/corpus/
#   forcing_corpus_moves.txt — stone owners derive from Connect6 placement
#   order (stone 0 = P1, then pairs P2,P2,P1,P1,...); emit io-format
#   {stones:[[q,r,"P1"|"P2"],...], attacker, placements_remaining}.
#   15 direct-id positions verified stone-identical to their
#   scripts/fixtures/forcing_puzzles/*.json (site exports; map the
#   first-mover site player to P1).
cargo build --release  # then: strix_compare <positions_dir> 120 50000000 prod,idtt,dfpn,pdspn
```

Our side: `tss_corpus_check` per TSS_RUNBOOK (512 MiB default; official gate
`TSS_BACKWALK_TT_BYTES=2147483648`, ≥10 GB host-free RAM). A subset corpus
can be supplied via `TSS_CORPUS_FILE` (note: the canonical harness asserts
19 positions when no override file is given).
