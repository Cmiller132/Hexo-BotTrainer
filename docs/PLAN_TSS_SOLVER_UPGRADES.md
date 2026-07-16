# PLAN: TSS Solver Upgrades — Unified Master Plan (wide-engine era)

Status: **LIVING DOC** (2026-07-16). This is a ground-up rewrite that
**supersedes the FINAL (R3 PASS) plan of 2026-07-14**, whose anchor engine —
the single-pass narrow DFS prover of branch `claude/tss-v2-build` — is no
longer the normative solver. The superseded document remains readable (with a
SUPERSEDED banner) at its old home in the `hexfield-eq-main-review` worktree;
everything from it that is still true is carried forward here, and every
U-item it defined is dispositioned in Part III.
Author: Claude (Fable), under owner rulings of 2026-07-16.

## Mission (owner rulings 2026-07-16, binding)

1. **Dramatically stronger solver on ALL axes**: real-position solving (human
   corpus + self-play at fixed budget), deep offline certified solving (the
   opening atlas), training-loop leaf strength, and beating idtt/dfpn/pdspn
   everywhere. The solver must remain MCTS-leaf-integrable: useful at LOW
   node counts, compute-efficient.
2. Leaf-improvement axes: verdict rate at fixed cap; cheaper solves → wider
   gating; the budget envelope (node_cap / park timeout / sample-16) is
   re-tunable when data justifies. Direction: "deeper search and more elegant
   design — dramatically improve."
3. **Policy-side consumption bar: PROOF-BACKED SETS ONLY** (defender-side
   certified zones and the like). Value consumption stays certificate-only
   through the single-mint verifier gate. No heuristic hints ever enter
   policy targets.
4. Paper division of labor: the paper's subject is the deep df-pn corpus
   solver and search-space reduction for the ideal solver; MCTS-leaf
   integration is mostly out of the paper; theorems get light per-solver
   notes. **This plan serves engineering** — it owns everything, including
   what the paper omits.
5. Deployment: improved leaf integration lands at the Phase-3 `main_3`
   relaunch; flags / one-rung-per-relaunch discipline unchanged
   (`docs/TSS_RUNBOOK.md`, this worktree, is the deployment instrument).
6. `RZOP_SOLVER_OPTIMIZATION.md` §9's priority ranking supersedes the old
   plan's defender phasing (U14→U13→U12→U15) — reconciled against the wide
   engine in §I.7, because the corpus is already 14/14 via the wide engine
   and several §9 premises describe the deleted narrow generator.

## Normative sources

- **Engine**: `WidePnSearch` in `packages/hexfield_eq/rust/src/tss_solver.rs`
  on branch `claude/tss-vcf-width`, round-9b tip `ac3f455f`, gate-verified at
  `dba6111d` (`.codex-round9b-gate/GATE.md`); Group-2 round-3 additions live
  uncommitted in this worktree (see IN-FLIGHT register).
- **Theory**: `docs/PROOF_TSS_DEFENDER_ZONES.md` (rounds 5–8 revision):
  D9–D21, L9′, L10–L17, T3–T11, zones `Z_dir ∪ Z_seed ∪ Z_touch ∪ Z_virgin`
  under (Z2)/(Z4)/(Z5′), §6a forcing-gate calculus, §12 open problems, §12a
  tightness frontier. Domination P1–P3: `docs/proof_parts/DOMINATION.md`; ES
  layer: `docs/proof_parts/ES_POTENTIAL.md` + `ES_GLOBAL_BOUNDARY.md`.
- **Formalization**: `E:\tss-lean\` (LEDGER.md is the decl-by-decl status
  map). T3/T4/T5/T9 and both dismissal corollaries are kernel-checked
  (`TssZones.T3`, `TssZones.T3_soundDismissal`, `TssZones.T4`, `TssZones.T5`,
  `TssZones.T9`, `TssZones.T9_soundDismissal`); T10 (DAG unfolding) is in
  flight. The coming `TssZones/SolverInterface.lean` + `SOLVER_HANDOFF.md`
  will be the **authoritative machine-auditable crosswalk** (Lean decl ↔ doc
  item ↔ U-item); Part III's anchors are a preview and defer to it on
  landing.
- **Prior art / priorities**: `E:\hexo-bot\docs\paper\RZOP_SOLVER_OPTIMIZATION.md`
  (§5 capstone spec, §9 ranking) and `RZOP_COMPARISON.md` (Wu & Lin 2010).
- **Empirical grounding**: corpus frequencies
  (`hunt/corpus-freq`, commit `3f66a410`, report
  `HUNT_REPORT_CORPUS_FREQ.md`); radius sharpness + virgin absorption
  (`hunt/r1b-r2`, `HUNT_REPORT_R1B_R2.md`); Group-2 progress files
  (`.codex-group2/round{1,2,3}-progress.md`, this worktree).
- **Deployment substrate**: `docs/TSS_RUNBOOK.md` (this worktree).

## Soundness classes (carried forward unchanged)

- **[S]** proven — value-preserving by a theorem of the proof doc, with its
  stated caveats.
- **[H]** heuristic — affects only *completeness* (which proofs are found
  within budget), never soundness, because the independent verifier re-checks
  every claim before minting. [H] items still change which verified values
  get minted and hence search/training trajectories — "heuristic" means no
  false value can be minted, not that behaviour is unchanged.
- **[T]** tooling/measurement — no behavioural change.

## Design invariants (every item below must preserve them)

1. **Single mint.** Values enter search/training only through the verifier
   gate (`hard_value_from_verified`). No second path, ever.
2. **No silent caps.** Defender reply sets are never truncated by count.
   Restrictions are *predicates* (set definitions the verifier re-derives
   from the position alone), never budgets. Ranks order; they never cap.
3. **Failure degrades to Unknown.** Any verify failure ⇒ Unknown + fatal
   counter, never a value.
4. **Opening excluded.** Every zone/theorem argument here is post-opening.
5. **Finder/verifier separation.** The verifier replays and re-derives every
   set and every edge from the position; finder hints are inadmissible as
   verification inputs. (Was implicit in the old plan; Group-2 round 3 made
   it the explicit contract.)
6. **Ladder discipline.** Every behavioral flag ships default-off and rides
   shadow → verify → consume; certificate-mutation rejection suites are
   mandatory before any consumption; flag-off narrow byte-identity is sacred.

## IN-FLIGHT register (what will update this doc, and where)

This document must not rot the day these land. Each lane names the exact
sections its landing updates.

| Lane | State at writing | Sections to update on landing |
|---|---|---|
| **G2R3** (Group-2 round 3): quiet-turn OR edges + ranked unforced defender zones, shadow→verify→consume | **LANDED, all 4 steps GREEN, committed `bfd03ca9`** (headline witness `double_fork_compact` = WIN/409 nodes, strict verifier ACCEPTED, first rung 10k; step-4 shrink `seed_band_radius(d)=8·(d−1)` in production, all prior certs re-verify; post-shrink all-19 gate PASS failures=0 in 442.6 s, orchestrator-reverified) | §I.2 (round outcome), §I.7 (RZOP ranks 1–2 close-out), Part III rows U12, U13, U19, U20, U21; §II.3 A1's shadow-statistics gate (witness zone = 62/478 legal at the k<b node) — fold on next revision pass |
| **T10 Lean** (DAG unfolding; `E:\tss-lean\`) | Structural/D9/D17-core projections kernel-checked; the full transport (roles, zones, T3/T9 conjunctions through the unfolding) unstated. Discovered semantics: **DAG labels are max-dominant bounds over path copies, NOT per-copy equalities** — this IS the U18/U22 merge-semantics spec | §I.4 (TT/DAG sharing design goes from spec to buildable), Part III rows U18, U22; §II.3 A4's soundness clause |
| **Leaf-width measurement** (worktree `hunt-leaf-width`, branch `hunt/leaf-width`; report `HUNT_REPORT_LEAF_WIDTH.md`) | **LANDED** (N=1,500 human-corpus attacker nodes, 3 caps, 0 contradictions): wide-only WIN = 6.07% / 8.13% / 9.27% at caps 500/2k/10k — structural width, not budget (a `SolveGoal::Win` full-budget control finds nothing more); warm medians narrow ≈0.07 ms vs wide ≈0.16 ms, wide's cost = p95 tail on exactly the positions it wins; 122 width records (mechanism: count-2 pair-builds / quiet connectors + deep VCFs); ES Φ<1 screen fires 0.024% — does not pay at leaves. Recommendation: cap-500 leaf-width rung via count-2 pair-build widening of the narrow OR-generator, NOT a WidePnSearch port; persistent-solver reuse mandatory (fresh-solve TT-zeroing cliff ≈13 ms) | §II.2 (axis sizing), §II.3 A2 (feature value), §II.5 (budget-envelope retuning), Part III row U8 — fold on next revision pass |
| **SolverInterface.lean / SOLVER_HANDOFF.md** (Lean campaign final passes) | Specified in `E:\tss-lean\CAMPAIGN.md`; not started | Part III's crosswalk column defers to it wholesale |

---

# Part I — The deep df-pn corpus solver

## I.1 The engine as built (round-9b, gate-verified)

The normative **offline** solver is **`WidePnSearch`** (`tss_solver.rs`): a
single-level, certificate-grade, staged-deepening df-pn **arena** engine. It
replaced the narrow DFS prover for corpus/deep root solving over rounds 5–9b
of the VCF-width campaign, and since C1's completion (G2R6 `3c180c66`) it is
the ONLY engine: the trainer's production leaf/root-guard/async surface still
constructs `TssSolver::default()`, but that default now dispatches through
`WidePnSearch::prove_narrow_compat`, which hosts the byte-identical narrow
DFS scheduler (identity proven — §I.3 C1). Offline mechanisms, by their
in-code names:

- **Persistent proof-number frontier.** The arena holds pn/dn per entry; the
  retained PN frontier is the search arena, not the transposition table, so
  memory-profile choices cannot alter frontier progress. Thresholded descent
  makes local progress without re-descending from the root.
- **Staged deepening.** An outer stage loop (`next_wide_stage_depth`,
  `reopen_depth_cutoffs`) advances the depth horizon on selected cutoffs; a
  staged depth cutoff is *unresolved, not a disproof* — cached dn=0 entries
  are reopened per stage (`stage_refreshes` telemetry).
- **Forcing-wide turn generation with stateless pair classification.**
  `WideTurnGate::build/evaluate_pair` classifies complete two-placement
  attacker turns statelessly; round-9b's stateless second-candidate
  derivation removed the per-child replay (the 4.5x round).
- **Canonical pair dedup.** Both legal orders of a placement pair are
  deduplicated via `canonical_frame`/`canonical_coord_key`; P3 (same-turn
  commutation) is thereby **structural in the TT** rather than a verifier
  arm.
- **Priors.** `pn_from_fork_degree` (attacker fork-degree ordering,
  `MAX_TURN_FORK_DEGREE = 36`) initializes pn; `dn_from_tau` initializes dn
  from the exact `min_hitting_set` (τ). A **root-only width-tier prior**
  (`first_width_tier`, `prefer_width_tier_at_depth`) orders the root's newly
  admitted width classes without perturbing established deep ordering.
- **Commitment domains.** At Universals with fanned-out obligations
  (latched once ≥4 distinct obligations are live —
  `universal_commitment_active`, `has_commitment_fanout`), selection commits
  to sequential obligation order so wide AND nodes drain one obligation at a
  time instead of thrashing the frontier.
- **`K_b` kernel at forced nodes.** The exact K2 kernel (T6) in canonical
  defender order; `implicit_dispatch` (premise `min_hitting_set == b`)
  carries the U3 staple-by-theorem dismissal of all non-hitting replies.
- **L13 sparse LOSS witnesses.** `sparse_loss_witnesses` emits ≤3 windows at
  b=1 and ≤5 at b=2 (caps improved from 3/6 to **3/5**; proof-doc R4b,
  Lean `TssZones.L13_capThree`/`L13_capFive` PROVEN).

**Certified performance (provenance-pinned).** Official all-19 corpus gate:
**PASS, `failures=0`**, single process, 436.8 s wall at the 2 GiB TT profile
(`TSS_BACKWALK_TT_BYTES=2147483648`), commit `ac3f455f`, gate record
`dba6111d` (`.codex-round9b-gate/GATE.md`; the G2R3 flags-off re-run
reproduced PASS at 445.4 s). 14/14 WIN certified, 5/5 NO non-WIN, zero false
wins. Cumulative vs the round-8b engine (round-9 progress notes,
`.codex-round9/round9-progress.md`): 12-entry matrix ~8 min → 12.35 s
(~40x); hard child @1M 1,272.8 s → 26.1 s (**48.7x**). The hardest entry
`0l4291i_live` full solve: ~6,970 s (round-8b) → 794.3 s (round-9 gate
`4daf1961`) → **177.7 s** (round-9b gate) — **faster than the reference
pdspn's 264 s on its hardest position**, at certificate grade. The 512 MiB
default TT was directly proven (round-8b telemetry, TT-saturation root
cause) to stop indexing 0l's working set around ~1M nodes; 2 GiB is the
official deep-solve test profile. That finding is why TT capacity/sharing is
now a first-class bottleneck (§I.4).

**Consolidation outcome** (§I.3): `WideRacer` and its test-only A/B hook are
deleted. Round-8b had already removed the losing round-5–8 DAG/graph-PN and
bounded-probe variants. The narrow DFS and count>=3 generator remain because
they are the live trainer leaf path; deleting them requires a separately
gated wide-engine-with-narrow-options migration.

## I.2 The Group-2 arc: closing the λ² structural gap

Corpus frequencies (§I.6 provenance) made this the highest-value front:
quiet-move share of real wins rises **8.7% → 87%** with distance-to-win, and
**25.7%** of threatened defender nodes are unforced (`k < B`). The wide
engine — forcing-complete and gate-perfect on the forcing corpus — was
structurally blind here: at a `k < b` defender node `implicit_dispatch`'s
premise fails and no quiet attacker continuation was ever generated.

**Round 1** (`.codex-group2/round1-progress.md`, base `ac3f455f`): isolated
the active witness. `double_fork_compact` (36-placement legal replay,
attacker SecondStone root): the historical narrow finder proves WIN in 2,884
nodes at absolute horizon 45; the normative wide engine dies **UNKNOWN in 2
nodes** — the first defender node after root choice `(4,0)` has `b=2, k=1`,
and the engine has no way to represent the hit-plus-spare defender turn or
the quiet attacker continuation. Also proved: stock-`tss_reference`
ground-truthing of this position is infeasible (structural lower bound
≥ 804 × 399 × 398 ≈ **127.7M nonterminal attacker nodes** before a verdict —
no TT, no pruning, by design).

**Round 2** (base `b4ec2e73`): built the validated escape hatch — a
test-only exact accelerator `tss_reference_fast` (independent legal
reconstruction, exact keys, optional independent D6 canonicalization,
bounded exact TT) that passed a **209/209 differential gate** against the
stock recurrence, yet still could not close the 478-cell universal wall at
depth 9. Outcome: two honest NO controls frozen in
`rust/corpus/spare_corpus_moves.txt` (`compact_urgent_spare`,
`strongloss_a_backoff_7`; UNKNOWN at every rung — the spare corpus has **no
WIN_PENDING row and must not be used as a positive acceptance gate**), plus
the round-3 design memo. Key consequence: for genuine λ² positions,
**acceptance comes from the independent verifier, not an oracle** — a wide
WIN whose certificate the strict verifier accepts is the deliverable.

**Round 3** (base `1e082d40`, landed as `bfd03ca9`,
`.codex-group2/round3-progress.md`) — THE ENGINE ROUND:

- Two tri-state (Off/Shadow/Consume) flags: **`quiet_turn_or_edges`**
  (complete two-placement attacker OR edges that may finish nonforcing,
  fired only after the forcing path is exhausted/refuted at an OR node —
  an OR *universe extension*, not a cap) and
  **`ranked_unforced_defender_zone`** (at `k < b` AND nodes, the certified
  T3/T4 union `Z_dir ∪ Z_seed ∪ Z_touch ∪ Z_virgin` under uniform wrappers as
  the complete searched set; rank orders, never caps).
- **Shadow** (green): zone derivation from completed certificates — D10 live
  roles (designated attacker placements + WIN/LOSS leaf-entry witness
  empties), D14 local budgets `B` computed bottom-up, four-part union
  recorded. Witness coverage: ply-37 node `k/b = 1/2`, zone 62 vs 478 legal
  (ratio 0.130; components dir/seed/touch/virgin = 19/0/50/0); ply-38 node
  18 vs 479 (0.038). All-19 forcing corpus: **0 quiet fires, 0 `k<b` zone
  nodes** — independent confirmation that the forcing corpus cannot exercise
  this machinery (the capstone needs λ² positions, §I.5).
- **Verify** (green): the verifier replaces the obsolete pre-round-3
  contract (`Z1` hitting row, full-DAG cores, `8·d` seed band) with the
  revised one — it independently replays the subtree, reconstructs D10
  roles, derives D14 `B`, and re-derives the full four-part union plus D9's
  deterministic nonempty fallback; zone nodes require `k<b`, exact stored
  local `B`, and an exact build-horizon binding; quiet edges replay under
  (Z4). Byte-for-byte finder/verifier zone agreement on every shadow node.
  Seven-mutation rejection suite green (omitted mandatory `Z_touch` cell,
  omitted defender edge, dropped quiet edge, wrong budget, wrong horizon,
  forged leaf, out-of-zone substitution). Verifier module 11/0.
- **Consume** (green): **`double_fork_compact` = WIN / 409 nodes / 24 ms at
  the first (10k) rung, strict verifier ACCEPTED.** A first attempt
  (WIN/409, verifier-REJECTED for a zone node carrying parent commutation
  allowances) correctly stopped the ladder; the mixed contract was removed —
  same-turn commutation is deliberately disabled on zone nodes (it is a
  separate forced-dispatch contract). No-regression matrix green: all-19
  gate PASS flags-off (445.4 s), spare NO controls unchanged, default suite
  95/0, narrow byte-identity untouched.
- **Step 4** (green): the R1b **seed-radius one-relay shrink**,
  production `seed_band_radius(d) = 8·(d−1)` for `d ≥ 1` (0 at `d ≤ 1`) in
  BOTH finder and verifier, justified by L9′'s `8(B−1)` bound and gated
  separately (chain fixtures + mutation suite + full all-19 re-run; any
  previously-verifying certificate that fails ⇒ STOP and revert — that would
  itself be a finding).

**Round 4 (likely shape).** Fold G2R3 (orchestrator gate + commit); then, in
some order justified by data: (a) shadow statistics at scale — zone sizes,
component splits, fire rates over corpus-sampled `k<b` nodes (feeds A1's
gate, §II.3); (b) a second, structurally different λ² witness (the spare
corpus still has no WIN row — a verifier-accepted quiet win from the human
corpus's 25.7% pool would both diversify evidence and seed the capstone,
§I.5); (c) exact per-role/per-window clocks behind a flag (U16) with the
uniform-vs-exact zone-size delta measured; (d) the `Z_virgin`/R2 question at
the definitional level, now that certificates carrying exposure labels
exist (the R1b/R2 hunt's blocker was exactly the absence of certificate
exposure labels).

## I.3 Consolidation: the wide engine becomes THE solver

Once G2R3 is folded and gate-green at the tip:

- **C1 — DONE (owner-approved, G2R5 `ace1f5b2` + G2R6 `3c180c66`).** The
  narrow prover now lives inside the wide engine:
  `WidePnSearch::prove_narrow_compat` hosts the byte-identical narrow DFS
  scheduler (exact status/node-count/certificate/TT-layout identity proven
  by the 512-position round-5 harness across fixtures, caps, TT profiles,
  and cache-warm behavior); `TssSolver::default()`/`prove_for` dispatch
  through it; the legacy `SearchContext` route, duplicate
  `prove_for_at_depth`, and historical count>=3 wrapper are deleted
  (−107 lines). No caller cap/dispatch change (trainer leaf, root guard,
  async workers audited). Full gate battery green at `3c180c66` including
  orchestrator-rerun all-19. The identity harness survives as an absolute
  regression test. Keep `tss_reference.rs` (stock, deliberately
  unoptimized) and `tss_reference_fast` (validated 209/209) as the two
  independent test-only oracles.
- **C2 — DONE: `WideRacer` (Fix A) deleted.** The racer field, probe branch,
  memo/Zobrist implementation, constants, and `TSS_WIDE_AB_RACER` test hook
  are gone. Its one residual idea — zone-cardinality-informed cross-leaf
  scheduling — remains recorded in U8 and consumes *verifier-derived*
  quantities only.
- **C3 — delete losing experimental scaffolds** of rounds 5–8 (round-8's
  losing TT variants etc.) and the round-2/3 harness dead ends that did not
  freeze into corpora. Frozen corpora, gate records, and progress memos
  stay.
- **C4 — single profile, single ladder.** `TSS_RUNBOOK.md` is authoritative:
  512 MiB ordinary offline default; 2 GiB official deep-solve profile via
  `TSS_BACKWALK_TT_BYTES=2147483648`; 256 KiB per trainer solve; and the
  forcing ladder 10k→100k→1M→20M (NO rows stop at 1M). The spare corpus
  keeps its honest semantics (NO controls; WIN_PENDING only with an
  exhaustive-oracle or verifier-accepted row).
- **C5 — paper-quote hygiene.** Any number quoted into the paper re-derives
  from a gate at the exact quoted commit (the round-9b gate at `ac3f455f`
  discharged this for the current headline set; G2R3's fold must repeat it
  at its tip).

## I.4 The named next bottleneck: TT capacity and U18/U22 DAG sharing

Round-8b proved the deep-solve regime is **TT-bound**: 512 MiB stops
indexing the 0l working set around ~1M nodes; the fix so far is a bigger
profile (2 GiB), which is a ceiling, not a design. The design answer has two
coupled halves:

- **U18 — certificate DAGs** (proof-doc D18/T10). Share repeated subproofs
  in the certificate arena instead of duplicating them. The soundness
  contract is now sharper than the old plan knew: T10's in-flight Lean
  formalization discovered that **DAG labels are max-dominant bounds over
  all path copies, not per-copy equalities** — a shared node's
  budget/rank/exposure labels must dominate every incoming path's exact
  recurrence, and obligations union over reachable descendants while
  coupling histories stay path-local. That rank-inequality semantics IS the
  merge-semantics spec for any TT/cache sharing of zone-bearing fragments.
  Gated on T10's completion (IN-FLIGHT register).
- **U22 — TT policy + ProvenFragment persistence** (new item; pairs with
  A4, §II.3). Inside a single deep solve: replacement policy aware of proof
  obligations (never evict entries pinned by the live frontier's proof
  DAG); byte-accounted admission for zone-bearing fragments (they are
  fatter — remeasure before promotion). Across solves: verified
  sub-certificates minted inside UNKNOWN solves persist in memo/TT across
  visits and moves, so later solves resume from proven frontiers. Round-6
  cert-import scaffolding exists (`cfg(test)`) as the starting point. The
  cross-path sharing half inherits U18's T10 semantics; the
  within-lineage half (same path re-visited at a later move) needs only the
  existing exact-key + build-horizon binding discipline (U4/U13).

Sizing note: this is the only front whose payoff compounds with every other
front — atlas solves, capstone runs, and leaf-mode cumulative search (A4)
all hit the same wall.

## I.5 Capstone measurement spec (RZOP §5, reconciled)

The capstone measures **how close to minimal we search**, on positions that
actually exercise the machinery. RZOP §5's items, corrected for the wide
engine:

1. **Corpus composition.** The 19-entry forcing corpus is structurally
   dispatch-only (G2R3 shadow: 0 zone nodes, 0 quiet fires on all 19) — it
   validates the forcing engine, not the zone machinery. The capstone set =
   the 19 forcing entries **+** `double_fork_compact` (first
   verifier-accepted λ² WIN) **+** 1–3 curated quiet-win positions from the
   human corpus's 25.3%-VCF / 87%-quiet pools (§I.6), each either
   oracle-ground-truthed or verifier-accepted **+** the two honest NO
   controls **+** opening-atlas spot-checks (§I.6).
2. **Minimality headline.** Per-AND-node and aggregate `|searched|/|Legal|`
   (G2R3's witness already logs 0.130 and 0.038 at its two zone nodes; the
   corpus-wide distribution is the deliverable).
3. **Uniform vs exact clocks.** Recompute each zone under the uniform
   `8(B−1)`-wrapper vs exact per-role `r_N` / per-window `E^D` (D15/D16,
   U16) and publish the size delta — the quantified minimality gain that
   RZOP's static zones structurally cannot produce.
4. **λ-order proxy column.** Certificate-derived: count of `k < B` Universal
   nodes + nonforcing OR edges + max spare-turn nesting — proves the tested
   position needed the new machinery (guards against dispatch-only
   capstones).
5. **Robustness control.** Re-verify capstone certificates under an
   outward-pushed legality frontier (more legal cells for both players) —
   the Hexo analogue of RZOP's infinite-board recheck, stressing (Z4)/virgin
   clauses. (Not the self-defeating "all cells strictly interior" phrasing;
   virgin windows live at the frontier by construction.)
6. **Cross-solver bar** (owner mission): matched-position wall/node
   comparison vs idtt/dfpn/pdspn where reference implementations exist; the
   0l-vs-pdspn 177.7 s vs 264 s datum is the template.

## I.6 Opening atlas

Corpus grounding (`hunt/corpus-freq`, `3f66a410`): the D6-canonical P2-reply
family distribution is strongly concentrated — **top 2 families = 36.1% of
all 6,902 games, top 5 = 50.0%, top 10 = 61.1%**. Solve order for the
certified atlas = the ranked list: `{(-1,0),(0,1)}` (19.4%) →
`{(-1,0),(-1,1)}` (36.1% cum.) → `{(-2,0),(-2,2)}` (43.7%) →
`{(-3,0),(-1,1)}` (47.5%) → `{(-2,1),(0,-1)}` (50.0%) → … (full table in the
hunt report).

Atlas rungs (each a separate, gateable deliverable):

1. **A-0**: per-family root solves at the deep profile (2 GiB, staged to
   20M+ rungs as needed), WIN/LOSS/UNKNOWN with certificates archived;
   honest UNKNOWN is an acceptable verdict — no bar-lowering.
2. **A-1**: frontier expansion inside solved families (certified subtree
   persistence is the A4/U22 consumer here — atlas work is exactly "many
   deep solves sharing proven frontiers").
3. **A-2**: atlas spot-checks feed the capstone (§I.5) and, Phase-3-side,
   opening-book consumption by serve/eval (out of this doc's scope until an
   owner ruling asks for it).

Prerequisite honesty: atlas economics are TT-bound (§I.4) and quiet-width
bound (§I.2). Do not schedule A-0 at scale before G2R3 folds and U22's
within-lineage persistence exists; before that, atlas time is mostly re-paying
the same subtrees.

## I.7 RZOP §9 reconciliation (owner ruling 6)

§9's ranking was written 2026-07-15 against the OLD narrow engine ("the
corpus proves 0 of 14"; "single-pass DFS, no pn frontier"). The wide engine
overtook several premises; the ranking's *spirit* — attacker width first —
was vindicated by G2R3. Row-by-row:

| RZOP §9 rank | Item | Status under the wide engine |
|---:|---|---|
| 1 | OR-generator width: count-2 pair-builds | **OVERTAKEN, then completed.** The wide engine's forcing-wide `WideTurnGate` already took the corpus to 14/14; the residual λ² blindness is closed by G2R3's `quiet_turn_or_edges` (U19). The specific `strength < 3` gate diagnosis described the deleted narrow generator. |
| 2 | λ-order iterative deepening | **PARTIALLY REALIZED, differently.** The wide engine has staged *depth* deepening; the order axis is realized as forcing-first-then-quiet within `quiet_turn_or_edges` (quiet universe fires only on forcing exhaustion) plus the leaf horizon ladder. A literal order-outer-loop remains available as a scheduling refinement if G2R4 data shows quiet-fire thrash; not currently needed. |
| 3 | T2 macromove defender collapse | **LIVE backlog** (U24, Part III) — still the one genuine defender fan-out collapse we lack; capped by frontier-inertness (distinct radius-8 balls), so honest impact medium/low. Needs a fresh Hexo lemma (hostile-reviewed) — RZS/Lemma 12 is template, not proof (locality breaks it). |
| 4 | Null-defender discovery probe | **MOSTLY MOOT.** Its purpose was to amortize discovery for a DFS finder; the pn-frontier engine self-schedules and G2R3 finds the quiet witness in 409 nodes. Keep the monotonicity lemma note; build only if atlas-scale profiles show discovery cost dominating. |
| 5 | Residual re-attack frontier on UNKNOWN | **LIVE** (U23) — now *more* valuable: with quiet width landed, the blocking-reply work-list is actionable, and it is exactly the routing signal A4/atlas resumption wants. |
| 6 | RZS/Lemma 12 template for U11 sub-hitting | LIVE with U11/U24; `[UNPROVEN]` label stands. |
| 7 | Frontier-inertness telemetry | Fold into capstone telemetry (§I.5). |
| 8 | Incremental generator state along DFS path | **UNCLEAR — re-profile.** The claim targeted the old generator's per-node rebuild; the wide generator's cost profile differs (stateless pair classification). Re-measure before building; node-throughput only, converts no UNKNOWN. |
| 9 | Racer + zone-cardinality PN into U8 | Racer: **delete** (C2). Zone-cardinality scheduling: folds into U8 (Part II), verifier-derived quantities only. |
| 10 | Promotion-composable zone schema | Carried into U22's admission design. |
| 11–13 | Capstone corpus / measurement / robustness | Adopted wholesale as §I.5. |
| 14 | Paper framing bundle | Paper's business (owner ruling 4); out of this plan. |

---

# Part II — The MCTS-leaf Phase-3 program

## II.1 Current integration reality (cite the RUNBOOK; do not duplicate it)

`docs/TSS_RUNBOOK.md` (this worktree) is normative for flags, rung order,
and health metrics. Summary of the substrate only: gated leaves (threat
trigger + hash subsample `tss_solver_sample_16`, deterministic
`tss_solver_node_cap` = 2000 default) ride a mode ladder
(`tss_solver_mode` 0/1/2/3 = off/shadow/+LOSS/+WIN) with root guard, interior
guard, async worker pool, and the park rung (wait-at-leaf first-touch
consumption, `tss_solver_park_timeout_ms` liveness bail). `tss_zone` runs
the **horizon ladder** per solve: a tight `+8`-deadline attempt on half the
node budget first, then the flat `+12` solve only if the tight attempt is
Unknown. `deep_verify_failed` MUST be 0; every consumption passes the single
mint. Deployment is one flag per relaunch at epoch boundaries, and the
Phase-3 `main_3` relaunch is the landing slot for everything below (owner
ruling 5).

## II.2 Improvement axes (owner ruling 2)

1. **Verdict rate at fixed cap.** The wide engine's economics change the
   leaf equation: G2R3's witness closed in 409 nodes/24 ms where the narrow
   finder needed 2,884 — but wide turns also cost more per node on wide
   frontiers. **Sized by the leaf-width measurement (IN-FLIGHT)**: narrow-
   vs-wide miss rates at caps 500/2k/10k on the human corpus + wall-clock
   economics decide which engine profile gates leaves at Phase-3 and at what
   cap.
2. **Cheaper solves → wider gating.** Every Part-I speedup converts directly
   into more gated leaves per second at fixed CPU share (raise
   `tss_solver_sample_16` coverage, or lower the trigger threshold) — the
   preferred spend of Part-I wins, per the mission's "useful at LOW node
   counts."
3. **Budget envelope re-tuning** (§II.5) — data-gated, never speculative.
4. **Sub-verdict yield** (§II.3) — the genuinely new axis: an UNKNOWN leaf
   solve still proved things; stop discarding them.

## II.3 The four sub-verdict artifacts (owner-endorsed 2026-07-16)

Program bar, restated: **policy targets consume proof-backed sets only;
values remain certificate-only through the single mint.** Each artifact
below states its consumption path, soundness argument under that bar, its
prerequisite, and an honest cost/benefit sketch.

### A1 — Certified zone policy masks at unforced defender nodes [S]

- **What/consumption.** At `k < B` defender nodes, the T3/T4 zone union is a
  *proven-complete* defender reply set: every reply outside it is
  certificate-dismissed (T3's contract). Consume exactly like
  `tss_policy_target_sharpen` (Lever 1) — narrow the recorded π′ target
  support to the zone — but at `k < B` nodes, which are **25.7% of
  threatened defender nodes** (43.3% of start-of-turn B=2 defender nodes).
- **Soundness.** Defender-side ONLY, and only where a verified certificate
  for the attacking side names the zone. **Attacker-side masks are unsound
  as priors**: quiet λ² moves are the exact residue outside any
  threat-derived attacker set, and the corpus says 87% of long wins run
  through them — masking the attacker would train the blindness G2R3 just
  cured. Proof anchors: T3/T4 (Lean `TssZones.T4` PROVEN); the mask is a
  non-losing-support claim, with the same loss-delay caveat U6 recorded
  (flattening preferences among all-losing replies).
- **Prerequisite.** G2R3 shadow statistics at scale (zone sizes, component
  splits, fire rates on corpus-sampled `k<b` nodes). Witness datapoints so
  far: zone/legal = 0.130 and 0.038 — if that ratio holds broadly, the mask
  is a ~10–25x support sharpening on a quarter of threatened defense.
- **Cost/benefit.** Cheap to consume (existing Lever-1 plumbing);
  certificate availability is the binding constraint — zones exist only
  where a gated leaf minted (or sub-minted, A4) a certificate covering the
  node. Benefit compounds with A4 (more persistent certificates → more
  masked nodes).

### A2 — Solver scalars as NN input features [T→H]

- **What/consumption.** Inputs, not targets — no soundness exposure: pn/dn
  at the abandoned root, depth reached, hitting number `k`, live
  threat-window counts, exact-surd Φ (the `A + B√3` form ports from
  `gap_raw_hunt.rs`). Plumb into the board-input feature block behind a
  flag.
- **Soundness.** None required beyond determinism: features change function
  inputs, not targets; the single mint is untouched. [H]-adjacent only in
  that they steer search through the net.
- **Prerequisite.** Leaf-width measurement sizes which scalars carry signal
  at leaf caps (IN-FLIGHT); feature-diet lessons from the main_3 board-input
  review apply (don't ship dead inputs).
- **Cost/benefit.** Cheap plumbing; risk is feature bloat, mitigated by the
  measurement gate. Φ's own corpus base-rate honesty: `Φ<1` occurs at 0.02%
  of defender FirstStone nodes — Φ is a *gradient* feature, not a boundary
  flag, at leaf scale.

### A3 — Bounded no-win facts from the horizon ladder [S with an exhaustion guard]

- **What/consumption.** The `tss_zone` ladder already runs a tight `+8`
  deadline before the flat `+12`. A tight rung that **exhausts** (the staged
  frontier completes with no proof at that deadline — dn=0 at the stage, not
  a node-cap bail) is a proven fact: "no forced win within 8 plies from
  here." Consume as an auxiliary value target (small regression head) or as
  in-tree exploration shaping (damp the win-probability prior of the solved
  horizon).
- **Soundness.** The fact is exactly the staged-deepening semantics
  ("a staged depth cutoff is unresolved, not a disproof" — so only
  *exhaustion*, never a cutoff or cap-out, mints the bounded fact). It is a
  bounded statement, never a Loss: the guard is structural, same shape as
  U9's old "never mint from a bounded futility check" rule. Auxiliary-target
  consumption does not enter the hard-value mint at all.
- **Prerequisite.** Telemetry to separate exhausted-tight-rungs from
  capped-tight-rungs (cheap counter); shadow histograms of how often the
  fact fires at production caps.
- **Cost/benefit.** Near-free at solve time (the rung already runs); the
  open question is training value — gate on the shadow fire-rate before
  building the head.

### A4 — ProvenFragment persistence [S; the leaf-side face of U22]

- **What/consumption.** Verified sub-certificates minted inside UNKNOWN
  solves persist in memo/TT across visits **and across moves**; later solves
  resume from proven frontiers. "Many shallow searches" become cumulative
  deep search at unchanged per-leaf budget — the highest-leverage leaf idea
  in this plan, because it converts the leaf regime's weakness (tiny caps)
  into amortization.
- **Soundness.** Within-lineage reuse (same game, later move): exact-key
  binding + build-horizon binding, already the verified-`Done`-entry
  discipline of the async memo (binding re-checked at every consumption).
  **Cross-path sharing** (transpositions): the spec is T10's max-dominant-
  label DAG semantics (U18) — labels at a shared node must dominate every
  incoming path; do not ship cross-path sharing before T10 lands
  (IN-FLIGHT register). Round-6 cert-import scaffolding (`cfg(test)`) is the
  implementation seed.
- **Prerequisite.** T10 (for the sharing half); U22's byte-accounted
  admission (zone-bearing fragments are fatter); memory ceilings per the
  runbook's solver-memory discipline (≤8192 memo entries, per-solve TT byte
  caps — persistence must not break the earlyoom budget).
- **Cost/benefit.** Implementation-heavy (the one genuinely hard leaf item),
  but it multiplies A1 (certificate availability), the atlas (§I.6), and
  verdict rate (axis 1) simultaneously.

## II.4 Lever-2 (proof-value target swap) — status unchanged

Runbook rung 9, **NOT YET BUILT** (build at its rung): value target :=
`tss_proof` where nonzero at expand, proof-valid-under-truncation mask, both
backends. Gate unchanged: the `tss.proof_disagreements` stream must show
deep proofs disagreeing with outcomes often enough to matter. Both labels
are already captured per row, so no data is lost while it waits. A3's
auxiliary head is deliberately weaker than Lever-2 and does not pre-empt it.

## II.5 Budget-envelope retuning — a data-gated rung

`tss_solver_node_cap` (2000), `tss_solver_park_timeout_ms` (100),
`tss_solver_sample_16` (16) were sized against the OLD engine's shadow
histograms. The wide engine's node economics differ in both directions
(§II.2 axis 1). Re-tune as ONE rung, after the leaf-width measurement
lands, from: verdict-rate-vs-cap curves at 500/2k/10k, wall-clock per
verdict, and park-bail rates at production thread counts. Until then the
envelope stays frozen — no speculative retunes ride other rungs (ladder
discipline, invariant 6).

## II.6 Deployment map (Phase-3)

Phase-3 relaunch order of the NEW items, after the runbook's existing rungs
(each line = one rung, one relaunch, shadow-first where semantics allow):

1. Wide-engine leaf profile swap (engine choice + envelope retune as sized
   by §II.5) — shadow mode first (`tss_solver_mode=1` twin-run).
2. A3 telemetry counters (pure shadow; no consumption).
3. A1 shadow: record would-be-masked support vs actual π′ (the
   `win_retained_mass_mean` analogue for zones) — consume only if mass
   actually moves.
4. A2 features behind a flag (needs a `target_regime`-style note only if
   input schema versioning demands it).
5. A4 within-lineage persistence; cross-path sharing strictly after T10.
6. A3 consumption (aux head or shaping) if its shadow fire-rate justified
   the head; Lever-2 at its own gate.

Every rung keeps invariants 1–6 and the runbook's watch metrics;
`deep_verify_failed` stays the hard-stop counter.

---

# Part III — U-item status ledger and crosswalk

One row per item. STATUS ∈ {DONE, STRUCK, SUPERSEDED, LIVE, IN-FLIGHT,
BLOCKED}; "where it lives" names the engine mechanism, campaign round, or
Part of this plan that owns it now. Proof anchors cite the proof doc tag and
the Lean decl where one exists (`TssZones.*`; status per
`E:\tss-lean\LEDGER.md` at 2026-07-16 — the coming `SolverInterface.lean` /
`SOLVER_HANDOFF.md` supersedes this column when it lands). Soundness
classes are the original ones unless re-derived.

**Tallies: 6 DONE, 2 STRUCK, 2 SUPERSEDED, 9 LIVE, 5 IN-FLIGHT (24 rows).**

| # | Original intent | Class | STATUS | Where it lives now | Proof anchor |
|---|---|---|---|---|---|
| U1 | Zone generator at `k<B` AND nodes (hitting ∪ 𝒜 ∪ ℬ ∪ core ∪ band) | [H] | **SUPERSEDED** | Formula superseded by U12's ranked union (already amended in the old doc §7); the k<B generator itself now ships as G2R3's `ranked_unforced_defender_zone` | T3/T4 (via U12) |
| U2 | Zone-carrying certificates + full D9 verifier checklist | [S] | **SUPERSEDED** | Checklist rows replaced by the revised contract (D10 roles, D14 budgets, four-part union, (Z2)/(Z4)/(Z5′)); shipped in G2R3's verifier rewrite. The D9 grammar obligations (typed leaves, nonempty S(N), no-defender-terminal, exact resolutions) survive verbatim inside the new contract | D9–D16, T3; Lean D9 grammar decls (STATED), `TssZones.T3` PROVEN |
| U3 | Staple-by-theorem at dispatch nodes; no complement enumeration | [S] | **DONE** | Wide engine `implicit_dispatch` (premise `min_hitting_set == b`); the per-omitted-move replay is gone from the normative path | U3 lemma; T1/T6/L3; Lean λ¹ soundness (`lambdaOne_win_sound`/`loss_sound` PROVEN), T6 kernel calculus decls |
| U4 | Path-derived ply clock, horizon semantics, two-stamp cache rule | [S] | **DONE** | Clock/horizon semantics in the wide engine; zone nodes carry an exact build-horizon binding (G2R3); the two-stamp fragment rule stands narrowed by U13's local budgets | D9, L7, L11; `TssZones.L7_*`, `L11_*` PROVEN |
| U5 | P3 same-turn commutation: pair-canonical generation | [S] | **DONE** | Structural in the wide TT via canonical pair dedup (`canonical_frame`/`canonical_coord_key`); note: commutation deliberately disabled on G2R3 zone nodes (separate contract) | P3 (DOMINATION.md); Lean §9 P3 row UNSTATED — formalization backlog |
| U6 | Interior forced-move guard default-on + λ¹ policy mask | [S] | **LIVE** | Part II / runbook rungs 2–3 (`tss_interior_guard`, `tss_policy_target_sharpen`); A1 extends the mask idea to `k<B` nodes | U3 lemma, T1, T6; loss-delay caveat recorded |
| U7 | OR-node ordering without child replay | — | **STRUCK** | Already implemented pre-plan (old R1 finding); the wide engine's fork-degree/tau priors supersede the mechanism anyway; DEEP_WIN ordering-regression telemetry residue → capstone (§I.5) | — |
| U8 | Trigger + regime detector (leaf gating) | [H] | **LIVE** | Part II axis 1–2; absorbs RZOP §6.3's fold (zone-cardinality scheduling from verifier-derived quantities; racer deleted instead of folded). **Gated on the leaf-width measurement (IN-FLIGHT)** | correctness-safety conditions carried from old plan verbatim |
| U9 | ES-potential futility check (Cor. 2 integer test) | [S-bounded] | **STRUCK** | Struck 07-16: the ES *global* forever-blocking claim is greedy-refuted (`ES_GLOBAL_BOUNDARY` Thm 1; GAP-RAW open), removing the intended growth path; and corpus data pre-triggers the old kill criterion — `Φ<1` fires at 0.02% of defender FirstStone nodes (<1% bar). Honesty note: the bounded Thm-2 form (first five attacker placements safe) remains mathematically valid; Φ survives as an A2 *feature*, never a futility gate | ES_POTENTIAL Thm 2/Cor 2; refutation ES_GLOBAL_BOUNDARY Thm 1 |
| U10 | Adversarial fixtures + differential harness; mutation suites are the gate | [T] | **DONE** (institutionalized) | Standard practice: G2R3's 7-mutation suite, round-2's 209/209 differential, hunt fixtures (`hunt_r1b_chain_sharpness` etc.), one-sided matched-horizon differentials. Every new consuming flag repeats it (invariant 6) | mutation testing is the gate; differentials are evidence |
| U11 | True domination arms P1/P2; sub-hitting dispatch `[UNPROVEN]` | [S]/[UNPROVEN] | **LIVE** (backlog) | Part I backlog; extended by U24 (macromove). RZS/Lemma 12 is a case-split *template* only — its zone-irrelevance step does not survive radius-8; `[UNPROVEN]` label stands until a fresh Hexo lemma passes hostile review | P1/P2 (DOMINATION.md); frontier-inertness Lemma 7 |
| U12 | Ranked zone generator + verifier (`Z_dir ∪ Z_seed ∪ Z_touch ∪ Z_virgin`, (Z2)/(Z4)/(Z5′)) | [S] | **IN-FLIGHT** (landed in G2R3, fold pending) | G2R3 `ranked_unforced_defender_zone`: shadow/verify/consume all green; verifier independently re-derives the union + D9 fallback; witness WIN verifier-accepted. Uniform wrappers this round; exact clocks = U16 | T3/T4/T7, D10–D16; `TssZones.T3`, `T4` PROVEN; `T5` PROVEN; zone-component decls STATED |
| U13 | Local budget labelling (D14/L11); cache reuse via budget inequalities | [S core; cache needs-derivation] | **IN-FLIGHT** (core) / cache lane LIVE | Core landed in G2R3 (exact stored local `B`, bottom-up D14, verifier-checked). The online-cache-reuse derivation (replacing final-assembly recheck) remains open — final assembly still rechecks all inequalities; failure ⇒ Unknown | D14, L11; `TssZones.L11_*` PROVEN |
| U14 | Sparse LOSS witnesses (≤3 at b=1, ≤6 at b=2) | [S] | **DONE** (improved) | Wide engine `sparse_loss_witnesses`; caps improved to **3/5** (R4b pinned relatively) | L13; `TssZones.L13_capThree`/`L13_capFive` PROVEN; sharpness fixtures STATED |
| U15 | Kernel T6 at forced nodes (`K_b`) | [S] | **DONE** | Wide engine exact K2 kernel in canonical defender order, beside `implicit_dispatch`; `mhs>b` hard guard per T6 | T6; Lean `t6Kernel_*` calculus decls landed (full T6 region contract still being stated) |
| U16 | Exact ranks and exposures (D15/D16 clocks) | [S when checks pass] | **LIVE** (backlog, promoted in value) | Behind a future flag with uniform-B as differential oracle; now also the capstone's uniform-vs-exact delta (§I.5 item 3) AND the only route to settling R2's full-union sharpness (blocked on D16 exposure labels only certificates supply — `hunt/r1b-r2` §3.3) | D15/D16, L11; ledger rows STATED |
| U17 | Branch-indexed substitution envelopes (D17) | [S] | **LIVE** (backlog) | Unimplemented in engine; **proof basis upgraded**: T9 + both dismissal corollaries now kernel-checked. Still the largest verifier surface; both `+1` transition charges mandatory (R7 pinned); gate on profiles showing whole-subtree unions dominate zone width | D17/T9; `TssZones.T9`, `T9_soundDismissal` PROVEN |
| U18 | Certificate DAGs (D18/T10) | [S] | **LIVE — PROMOTED to Tier A** | §I.4: TT capacity/sharing is now a first-class bottleneck (512 MiB ≈ 1M-node saturation, round-8b telemetry). Merge semantics spec = T10's max-dominant labels. **Sharing half BLOCKED on Lean T10** (IN-FLIGHT) | D18/T10; `unfoldDAG_*` core kernel-checked, T10 transport pending |
| U19 | **NEW** — Quiet-turn OR edges (attacker width) | [H, verifier-gated] | **IN-FLIGHT** (landed in G2R3) | `quiet_turn_or_edges`: complete two-placement attacker turn universe, fired on forcing exhaustion at OR nodes; replayed under (Z4) by the verifier. The plan's first attacker-side item; closes the λ² structural gap (RZOP ranks 1–2 vindicated) | verifier gate is soundness; (Z4) replay; corpus λ² gradient 8.7%→87% |
| U20 | **NEW** — Seed-radius one-relay shrink (`8·d` → `8·(d−1)`) | [S] | **IN-FLIGHT** (G2R3 step 4) | Production `seed_band_radius` in finder AND verifier; separately gated (chain fixtures keep binding seed at `8(B−1)`, shed at `8(B−2)`; full gate re-run; any previously-verifying cert failing ⇒ STOP+revert+writeup) | L9′ (`8(B−1)` bound; SHARP per `hunt/r1b-r2` §2 — attained in all 364 probe positions at B=2; absolute-pin attempt honestly BLOCKED); R1b OPEN as theory, shrink-evidence unconditional for the implementation |
| U21 | **NEW** — `Z_virgin` finite test (formerly absorbed) | [S] | **IN-FLIGHT** (inside U12's landing) | The pre-round-3 shipped verifier ABSORBED `Z_virgin` entirely (full-legal fallback where theory licenses the finite `8(E^D−6)` inversion test — `hunt/r1b-r2` §3.1). G2R3's re-derived union computes it under uniform wrappers; the exact-`E^D` refinement belongs to U16 | D16, L12; `TssZones.zVirgin` + completeness under L11 premise STATED; fixed-window sharpness `L12_fixedWindowExposureSeven_sharp` PROVEN |
| U22 | **NEW** — TT policy + ProvenFragment persistence | [S within-lineage; T10-gated cross-path] | **LIVE** | §I.4 (deep side) + A4 (leaf side); round-6 cert-import scaffolding (`cfg(test)`) is the seed; byte-accounted admission for fat zone fragments; obligation-pinned replacement | exact-key + build-horizon binding (U4/U13); cross-path = T10 |
| U23 | **NEW** — Residual re-attack frontier | [T→H] | **LIVE** | On UNKNOWN, emit the blocking defender reply set as a routing-only field; re-solve only those at deeper caps next pass. Mints nothing. Payoff unlocked by U19 (quiet width now exists to exploit the routing) | none needed (routing only) |
| U24 | **NEW** — Macromove / class-partition defender collapse | [UNPROVEN → needs fresh lemma] | **LIVE** (backlog) | Extends U11: merge distinct defenses sharing one winning continuation (RZOP T2 ancestry); sound only where absorbed fillers are frontier-inert, which caps the win (honest impact medium/low). Hostile-review the lemma before implementation | DOMINATION.md Lemma 7 (frontier-inertness); Wu & Lin Lemma 12 as architecture template only |

**Carried-forward verbatim analyses.** The old plan's per-item soundness
analyses that survive the engine change remain normative where this table
cites them: the U3 lemma text, U5's side-condition list (order frozen at the
b=2 parent; mirror state-bound and materialized; outcome- not key-dedup for
joint-second-win pairs), U6's default-on condition list and loss-delay
caveat, U8's correctness-safety conditions, U13's inequality-direction rule
(`required_B_at_reuse ≤ zone_build_B`; reversing it recreates the
omitted-fragment defect), U17's C2/C3 `+1` counterexamples, and U18's
merge-consistency test list. Consult the superseded doc for their full text;
they transfer to the wide engine unchanged because they are properties of
the certificate contract, not of the search algorithm.

**Deleted-not-dispositioned.** The old plan's §3 phasing table (P0–P3 rungs
of the narrow engine) and §7 "recommended extension phasing" are void — the
implemented-P0–P3 flags and stored zone certificates of the old engine are
not promoted across the new verifier contract without re-verification (as
the old doc itself required), and the old engine is slated for deletion
(C1). RZOP §9 replaces them as the priority source (owner ruling 6; §I.7).

---

# Measured-number provenance table

Every number in this doc, with its source of truth:

| Number | Value | Provenance |
|---|---|---|
| All-19 gate | PASS, failures=0, 436.8 s (2 GiB profile) | `.codex-round9b-gate/GATE.md`, commit `ac3f455f`, gate record `dba6111d`; G2R3 flags-off re-run 445.4 s (`.codex-group2/round3-progress.md` C5) |
| 0l full solve | 6,970 s → 794.3 s → 177.7 s (8b → 9 → 9b) | round-8b records; gate `4daf1961`; gate `dba6111d`. pdspn reference 264 s on its hardest position (owner-cited comparison) |
| Matrix / hard-child speedups | ~40x / 48.7x cumulative vs 8b | `.codex-round9/round9-progress.md` headline table |
| TT saturation | 512 MiB stops indexing 0l's working set (~1M nodes) | round-8b telemetry (`.codex-round8/round8-final.md`) |
| Racer A/B | 185,790 nodes / 97.9 s racer-on AND racer-off (no benefit) | `.codex-round9/round9-progress.md` "Fix A racer — MEASURED, DEFAULT OFF" |
| `double_fork_compact` | old finder WIN 2,884 n @ horizon 45; wide UNKNOWN/2 pre-R3; **WIN/409 n/24 ms verifier-accepted** post-R3 at 10k | `.codex-group2/round1-progress.md`, `round3-progress.md` C4 |
| Stock-reference infeasibility | ≥127,676,808 nonterminal attacker nodes | `.codex-group2/round1-progress.md` C3 lower-bound audit |
| Exact-oracle differential | 209/209 agreements | `.codex-group2/round2-progress.md` C1 |
| Zone/legal ratios (witness) | 0.130 (62/478), 0.038 (18/479) | `.codex-group2/round3-progress.md` C1 coverage table |
| Unforced `k<B` share | 25.7% of threatened defender nodes; 43.3% of B=2 start-of-turn | `HUNT_REPORT_CORPUS_FREQ.md` §1 (branch `hunt/corpus-freq`, `3f66a410`; 6,902 games, 431,495 nodes) |
| VCF-exists / human conversion | 25.3% ± 1.9% (lower bound, 10k cap) / 64.2% | same, §2 (n=2000 fixed-seed sample) |
| Quiet-move share of wins | 8.7% → 87.0% by distance-to-win | same, §3b |
| `Φ<1` incidence | 0.024% of defender FirstStone nodes | same, §3c |
| Opening families | top 2 = 36.1%, top 5 = 50.0%, top 10 = 61.1% | same, §4 |
| Seed-band sharpness | `8(B−1)` attained in 364/364 positions at B=2; shipped `8·d` carries ≥1 removable relay; absolute pin BLOCKED | `HUNT_REPORT_R1B_R2.md` §2, §Absolute-pin (branch `hunt/r1b-r2`, base `dba6111d`) |
| LOSS caps | 3 (b=1) / 5 (b=2) | proof doc R4a/R4b; Lean `L13_capThree`/`L13_capFive` |
| Lean status | T3/T4/T5/T9 + dismissal corollaries PROVEN; T10 in flight | `E:\tss-lean\LEDGER.md` (2026-07-16) |

# Known source conflicts (recorded, not hidden)

1. **Corpus size**: auto-memory said 8,698 games; the dataset's own metadata
   and the freq report say **6,902** (sha `54fae7ae…a5b7`). This doc uses
   6,902. (Already flagged inside `HUNT_REPORT_CORPUS_FREQ.md` §0.)
2. **RZOP_SOLVER_OPTIMIZATION.md** describes the deleted narrow engine in
   its §0/§1/§6.3 premises (0/14 corpus; "no pn frontier"; stale line
   numbers). Its §9 ranking is adopted per owner ruling 6 *as reconciled in
   §I.7*; do not cite its engine claims as current.
3. **Fix A lineage**: round-8's memo says "Fix A neither needed nor
   implemented"; round 9 then built and A/B'd it. Both true in sequence;
   current state = built, measured useless, `cfg(test)`-gated, delete (C2).
4. **Shipped seed radius description**: the R1b hunt derives it as `8·d`
   with `d` the budget to global T (`d ≥ B`); G2R3 describes its shadow
   wrapper as `8·B` (local-budget form). Consistent — the round-3 zone
   nodes carry local `B`, and step 4's shrink targets the one removable
   relay in either formulation.
5. **U9's mathematics vs its striking**: the bounded ES Theorem-2/Cor-2 test
   is not refuted; the *strike* rests on the global-layer refutation plus
   the 0.02% empirical fire rate. Recorded in the U9 row to keep the ledger
   honest.

# Change log

- 2026-07-16: Ground-up rewrite for the wide-engine era (this document).
  Supersedes FINAL (R3 PASS) 07-14. Old doc stubbed in place at
  `hexfield-eq-main-review` worktree with a SUPERSEDED banner.
