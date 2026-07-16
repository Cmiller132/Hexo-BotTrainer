# Round 9 progress — orchestrator-authored engine rewrite (no Codex)

Build base: `c37e0799` (round-8b consolidated engine, 0l WIN banked).
Mandate: Fix A (racer) + incremental threat maintenance + rewrite of hot
areas; >=10x wall-clock target; study proof docs for further levers.

## Profiling first (methodology)

Temporary cfg(test) instrumentation (`GEN_PROFILE` line in the corpus
harness) attributed g2xx6wl's 13.3s solve: **94% inside
`attack_pair_children`** — of which ~11s was the apply-and-analyze pair
double loop and ~1.7s the per-first-stone candidate regeneration. A second
finding: every PN expansion re-descended from the root (staged depth 60-78),
costing ~milliseconds per expansion on deep positions.

## Landed changes (commit b803f463)

1. **df-pn threshold descent** (`work` replaces the single-expansion
   stepper): expansions stay at the frontier while the node's pn/dn remain
   below sibling-derived thresholds (standard df-pn recurrence, floored at
   child+1 for policy-selected children; commitment domains pass thresholds
   through). Depth cutoffs still bubble for staged deepening. Visit-order
   only; certificates untouched. Historical `step` kept for focused tests
   via a soft expansion cap.
2. **Stateless pair classification** (`WideTurnGate`): at pair generation
   the claimant provably has no live >=4 window (expand leafs those first),
   so the post-pair defender analysis is fully derivable from turn-start
   window snapshots. The |C1|x|C2| loop now runs zero engine applies and
   zero full-window scans. Replicates turn_created_claimant_threat /
   turn_forces_small_defender_reply / immediate_winner + typed_lambda_leaf
   (sparse L13 obstruction + horizon) / completed_turn_prior exactly —
   verified bit-identical node counts (2917/4149/2041/80) on the spot set.
3. **Incremental-threat exploitation**: engine gained
   `WindowStore::live_threat_entries()` (O(active threats) iterator over the
   already-maintained index); `threats_shared::analyze` rewired to it
   (outputs order-insensitive; `tactical_cells` deliberately kept on the
   ordered full scan for narrow byte-identity). HashMap candidate dedup
   (was quadratic), single-pass `attacker_fork_degree` (was a full ranked
   generation per attacker-node prior), hoisted claimant-stone list,
   defender pair plan on apply/undo (was a full engine clone per kernel
   cell).

## Benchmarks after (1)-(3), DEFAULT settings, 512MiB TT

| Position | Before (8b) | After | Speedup |
|---|---|---|---|
| hard child @1M (prefix14+extras) | 225,924 n / 1,272.8s | **WIN 185,790 n / 97.9s** | **13.0x** |
| spot set (0hz/g2xx/jh7/xsn) | 20.2s | 3.6s | 5.6x |
| 12-entry matrix | 12/12 (8+ min) | **12/12 (~44s)** | ~11x |
| full unit suite | 95/0 | 95/0 | — |

Matrix node counts shifted modestly under df-pn visit order (e.g. jnzzmcm
13,646->14,317, hayes 13,524->12,844, strongloss_b 682->1,128); all rungs
hold with huge margin.

## Fix A racer (in progress)

`WideRacer`: bounded, uncertified, depth-first turn racer over the exact
wide universe (same gate + generators; no certificates/arena/keys), top-K
budgets (8 firsts x 16 seconds, 12 pair recursions, defender fan-out cap
12), zobrist-memoized. Integrated as a scheduling oracle at fresh attacker
pair nodes (depth <= 24): a racer win drops the winning child's pn prior to
1 so df-pn drives the oracle line first. Racer verdicts NEVER mint
certificates (soundness unaffected by construction). cfg(test) kill-switch:
TSS_WIDE_AB_DISABLE_RACER. A/B on the hard child in flight.

## Proof-docs review (task: anything left to exploit?)

Verdict: the wide VCF path already consumes every proven applicable result
(tau-init, K_b kernel, L13 3/5 sparse witnesses, P3 canonicalization). The
U12-U18 ranked-zone machinery targets UNFORCED AND nodes (deferred Group-2
scope) which do not occur under turn-forcing. Useful note: U18 confirms
certificate-DAG sharing is sound if TT memory becomes binding again. No
further proof-derived speed lever for wide mode today; remaining wins are
engineering.

## Fix A racer — MEASURED, DEFAULT OFF

Direct A/B on the isolated hard child (1M cap, 512MiB):

| Config | Nodes | Wall |
|---|---:|---:|
| racer off | 185,790 | 97.9s |
| probe everywhere (1,500 turns) | 185,943 | 189.8s |
| tie-gated (600 turns) | 185,986 | 131.0s |

Node counts unchanged in every configuration: the tau/fork/tier/commitment
ordering stack already discriminates on this corpus, so oracle probes buy
no visit-order value and only burn wall. Also structural: with df-pn the
certified search is itself frontier-efficient, and the racer shares the
same generator cost profile rather than being 100x cheaper (the pdspn
economics don't transfer). DECISION per the owner's one-system A/B rule:
racer default OFF behind cfg(test) TSS_WIDE_AB_RACER opt-in, evidence
documented here; recommend deletion at the next consolidation.

## Headline results (rewritten engine, racer off)

| Position | Round-8b | Round-9 | Speedup |
|---|---|---|---|
| hard child @1M | 225,924 n / 1,272.8s | WIN 185,790 n / 97.9s | **13.0x** |
| 12-entry matrix | 12/12, ~8 min | 12/12, ~44s | **~11x** |
| lz60mfb 1M rung | 122,132 n / 192.1s | **WIN 109,460 n** / 32.5s | **5.9x** (new best) |
| full 0l @4M/2GiB | 2,335,295 n / 6,969.6s | **WIN 1,831,556 n** / 788.1s | **8.8x** (fewer nodes) |
| full unit suite | 95/0 | 95/0 | — |

Narrow default-mode byte-identity vs round-5 signature: **EXACT** — all 101
normalized rows identical on every substantive field (timing-only and
timing-derived-gate fields stripped on both sides per the round-8b
normalizer protocol); sig at .codex-round9/narrow-default.sig.

## Pending

- Official all-19 single-process gate (in flight — also discharges the
  round-8b pending replay, now feasible in well under an hour).
- Instrumentation cleanup decision (GEN_PROFILE timers are cfg(test)-only).

## OFFICIAL ALL-19 SINGLE-PROCESS GATE: PASS

`CORPUS_DONE failures=0`, Rust test `ok`, wall **1,870.96s (~31 min)** in one
process at the documented 2GiB test resource profile — the replay round 8b
had to hand off because it projected >4 hours. Per-entry banked rungs:

| Entry | Rung | Status | Nodes |
|---|---:|---|---:|
| 0hz3hty | 10k | WIN | 2,917 |
| 0l4291i_live | **20M** | **WIN** | **1,831,556** |
| 8is963b | 10k | LOSS (NO ok) | 1 |
| 94gnnol | 1M | UNKNOWN (NO ok) | 1,000,000 |
| acly7kb | 10k | WIN | 75 |
| dy3dg99 | 10k | LOSS (NO ok) | 1 |
| g2xx6wl | 10k | WIN | 4,149 |
| hu01jk4 | 10k | WIN | 380 |
| jh7yo7y | 10k | WIN | 2,041 |
| jnzzmcm | 100k | WIN | 14,317 |
| l9mxn59 | 1M | UNKNOWN (NO ok) | 225 |
| lz60mfb | 1M | WIN | 109,460 |
| mvp2lvc | 1M | UNKNOWN (NO ok) | 19,895 |
| xsnfyll | 10k | WIN | 80 |
| zrugh2x | 100k | WIN | 38,893 |
| strongloss_a_prefix6 | 100k | WIN | 18,973 |
| strongloss_b_prefix8 | 10k | WIN | 1,128 |
| hayes_20260712_turn16 | 100k | WIN | 12,844 |
| hayes_20260712_placement31 | 100k | WIN | 12,844 |

14/14 WIN certified on the ladder; 5/5 NO non-WIN; full log
`final-matrix-19-rewrite.log`. Round-8b's pending-official-replay section is
discharged by this run.
