# PLAN: TSS Solver Upgrades — Unified Master Plan (wide-engine era)

Status: **LIVING DOC** (2026-07-17). This is a ground-up rewrite that
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
  on branch `claude/tss-vcf-width`; the efficiency-program consolidation is
  `b45b9bf0` (`MERGE_RESOLUTION.md`) and the leaf-surface landing is
  `5172d42d` (`HUNT_REPORT_LEAF_SURFACE.md`). The round-9b baseline remains
  `ac3f455f`, gate-verified at `dba6111d`
  (`.codex-round9b-gate/GATE.md`).
- **Theory**: `docs/PROOF_TSS_DEFENDER_ZONES.md` (rounds 5–8 revision):
  D9–D21, L9′, L10–L17, T3–T11, zones `Z_dir ∪ Z_seed ∪ Z_touch ∪ Z_virgin`
  under (Z2)/(Z4)/(Z5′), §6a forcing-gate calculus, §12 open problems, §12a
  tightness frontier. Domination P1–P3: `docs/proof_parts/DOMINATION.md`; ES
  layer: `docs/proof_parts/ES_POTENTIAL.md` + `ES_GLOBAL_BOUNDARY.md`.
- **Formalization**: `E:\tss-lean\` (LEDGER.md is the decl-by-decl status
  map). T3/T4/T5/T9 and both dismissal corollaries are kernel-checked
  (`TssZones.T3`, `TssZones.T3_soundDismissal`, `TssZones.T4`, `TssZones.T5`,
  `TssZones.T9`, `TssZones.T9_soundDismissal`). T10 (DAG unfolding) is
  kernel-checked at Lean commit `69adffc7` and licenses the finished-DAG
  sharing contract used by the shared-fragment store (`BUILD_SHARED_FRAGMENTS.md`,
  Rust completion `e4ef021f`). The coming `TssZones/SolverInterface.lean` +
  `SOLVER_HANDOFF.md`
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
   The sole exception is R-FIX1 (`2454fa91`): an always-on correctness repair
   that makes finder evidence match the unchanged verifier contract
   (`FIX_ZONE_CLOCK.md`, `MERGE_RESOLUTION.md`).

## IN-FLIGHT register (what will update this doc, and where)

This document must not rot the day these land. Each lane names the exact
sections its landing updates.

| Lane | State at writing | Sections to update on landing |
|---|---|---|
| **G2R3** (Group-2 round 3): quiet-turn OR edges + ranked unforced defender zones, shadow→verify→consume | **LANDED, all 4 steps GREEN, committed `bfd03ca9`** (headline witness `double_fork_compact` = WIN/409 nodes, strict verifier ACCEPTED, first rung 10k; step-4 shrink `seed_band_radius(d)=8·(d−1)` in production, all prior certs re-verify; post-shrink all-19 gate PASS failures=0 in 442.6 s, orchestrator-reverified) | §I.2 (round outcome), §I.7 (RZOP ranks 1–2 close-out), Part III rows U12, U13, U19, U20, U21; §II.3 A1's shadow-statistics gate (witness zone = 62/478 legal at the k<b node) — fold on next revision pass |
| **T10 + shared fragments** | **LANDED.** Lean T10 is kernel-checked at `69adffc7`; the T10-licensed monotone shared-fragment store is built at `e4ef021f`, consolidated at `b45b9bf0`, and remains default-off. Deep TT-saturation value is NULL; the current leaf profile gains no verdict (`BUILD_SHARED_FRAGMENTS.md`; `HUNT_REPORT_SHARED_FRAGMENTS.md`; `HUNT_REPORT_LEAF_SURFACE.md` at `5172d42d`) | §I.4; Part III rows U18, U22; §II.3 A4 |
| **Leaf-width measurement** (worktree `hunt-leaf-width`, branch `hunt/leaf-width`; report `HUNT_REPORT_LEAF_WIDTH.md`) | **LANDED** (N=1,500 human-corpus attacker nodes, 3 caps, 0 contradictions): wide-only WIN = 6.07% / 8.13% / 9.27% at caps 500/2k/10k — structural width, not budget (a `SolveGoal::Win` full-budget control finds nothing more); warm medians narrow ≈0.07 ms vs wide ≈0.16 ms, wide's cost = p95 tail on exactly the positions it wins; 122 width records (mechanism: count-2 pair-builds / quiet connectors + deep VCFs); ES Φ<1 screen fires 0.024% — does not pay at leaves. Recommendation: cap-500 leaf-width rung via count-2 pair-build widening of the narrow OR-generator, NOT a WidePnSearch port; persistent-solver reuse mandatory (fresh-solve TT-zeroing cliff ≈13 ms) | §II.2 (axis sizing), §II.3 A2 (feature value), §II.5 (budget-envelope retuning), Part III row U8 — fold on next revision pass |
| **SolverInterface.lean / SOLVER_HANDOFF.md** (Lean campaign final passes) | Specified in `E:\tss-lean\CAMPAIGN.md`; not started | Part III's crosswalk column defers to it wholesale |
| **NQ2 quiet-locality (hunt/quiet-locality)** | **CLOSED NEGATIVE, `833020ed`**: join-locality restriction of the quiet consume universe is REFUTED (frozen required-remote witness: unique win at `d_stone=6` in no live window, all 537 local alternatives lose; verifier-accepted cert). The complete quiet enumeration at consume nodes is not shrinkable by join/adjacency locality — treat any future locality proposal as guilty until it survives this witness. Salvage: `K_reply` kernel PROVEN (urgent SecondStone: wins-now + cells hitting every defender count-4/5 window; witness 538→1) — candidate for a future gated round | §I.2's quiet-universe wording stands (complete enumeration validated); the witness is a paper-grade sharp example |
| **Overnight efficiency program** | **LANDED through `5172d42d`.** Lazy frontier, interior census gate, R-FIX1, K_reply consume, the NQ3/NQ5/NQ8 negative verdicts, and the Phase-3 leaf configuration are dispositioned below; merge battery at `b45b9bf0` | §I.4; §II.7; Part III rows U11 and U25–U31; official-profile subsection |

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
cause) to stop indexing 0l's working set around ~1M nodes. The efficiency
program moved the official recommendation to the fully gated 1 GiB lazy-on
profile (`5f836b70`); the 2 GiB flags-off profile remains the legacy
comparison (§I.4, `HUNT_REPORT_LAZY_MEMORY_WALL.md`).

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
- **C4 — official profiles and ladder.** `TSS_RUNBOOK.md` remains the
  deployment authority. For deep solves, recommend 1 GiB only with
  `TSS_LAZY_FRONTIER=1`; retain the 2 GiB flags-off profile as the legacy
  baseline (`5f836b70`, `HUNT_REPORT_LAZY_MEMORY_WALL.md`). Trainer solves
  retain their 256 KiB ceiling, with the selected leaf configuration in
  §II.7 (`5172d42d`). The forcing ladder remains 10k→100k→1M→20M (NO rows
  stop at 1M), retained at consolidation `b45b9bf0`. The checked-in spare
  gate has only two near-vacuous NO rows;
  substantive expansion belongs to the paper capstone (`b45b9bf0`, commit
  message and `MERGE_RESOLUTION.md`).
- **C5 — paper-quote hygiene.** Any number quoted into the paper re-derives
  from a gate at the exact quoted commit (the round-9b gate at `ac3f455f`
  discharged the historical headline set; consolidation `b45b9bf0` and the
  leaf landing `5172d42d` pin the efficiency-program evidence).

## I.4 Efficiency-program resolution: sharing, admission, and bounded horizons

### U18/U22 shared-fragment store — BUILT, default-off

The shared-fragment store landed at `e4ef021f` and was consolidated at
`b45b9bf0`. It stores self-contained verified positive fragments under exact
position/claimant/profile keys, applies the max-dominant-label and obligation-
union contract to the reachable DAG, pins live payloads, accounts retained
bytes, and sends the final certificate through the unchanged strict verifier.
Lean T10 at `69adffc7` is the soundness license for finished-DAG composition,
not for arbitrary payload union (`BUILD_SHARED_FRAGMENTS.md`;
`HUNT_REPORT_SHARED_FRAGMENTS.md`).

Deep-profile TT-saturation value is **NULL**: fragments did not close
`0l4291i_live` through the completed 1M reduced-TT campaigns at either
512 MiB or 1 GiB (`e4ef021f`). Leaf value is **none at the current 256 KiB
profile**: 22 hits in 875 lookups produced a small expansion reduction and
zero additional hard verdicts, so the recommendation remains
`TSS_SHARED_FRAGMENTS=0` (`5172d42d`, `HUNT_REPORT_LEAF_SURFACE.md`). The
monotone contract remains valid evidence — warm UNKNOWN may become a
strict-verifier-accepted hard verdict — but does not justify default-on.

### Lazy frontier and interior census gate — BUILT

NQ4 discovered that 62.6–67.3% of retained wide entries were never expanded
(`f30e3fb1`, `HUNT_REPORT_TURN_QUOTIENT.md`). R-LF1 built lazy admission
behind default-off `TSS_LAZY_FRONTIER` at `86a6418c`; R-LF2 then made
**1 GiB + `TSS_LAZY_FRONTIER=1`** the official recommended deep profile at
`5f836b70`. At 512 MiB, the filtered `0l4291i_live` row was **8.4x** faster
(8.38x exact at `5f836b70`) with lazy admission, but that budget did not
receive a full 19-row gate and
is not the official profile (`5f836b70`, `HUNT_REPORT_LAZY_MEMORY_WALL.md`).
Lazy frontier remains a component of the winning leaf configuration
(`5172d42d`), where its direct measured value was TT-pressure control rather
than an extra verdict.

The DTW census bound became the default-off interior census gate at
`90f559be`. It is inert at the unbounded official corpus profile
(`semantic_horizon=u32::MAX`; zero evaluations at `90f559be`) and live on
horizon-bounded solves, saving **79–93%** of forcing-cohort expansions
(78.9–93.4% exact at `90f559be`) in the R-IG1 campaign
(`90f559be`, `BUILD_INTERIOR_GATE.md`). It is the other
efficiency component of the Phase-3 leaf winner (`5172d42d`).

### Correctness and closed experimental lanes

R-FIX1 (`2454fa91`) is always on: the materializer now stamps verifier-exact
D14 local zone budgets and the assembled certificate horizon. It is the
program's only unconditional production behavior change; the verifier stayed
unchanged (`FIX_ZONE_CLOCK.md`; consolidation `b45b9bf0` in
`MERGE_RESOLUTION.md`). All other efficiency levers remain independently
flag-gated.

K_reply, support hashing, D6 search-TT folding, broad horizon laddering, and
b=2 domination are dispositioned individually in Part III (U28–U31 and U11).

### Official deep-solve profiles after the program

- **Recommended:** `TSS_BACKWALK_TT_BYTES=1073741824` (1 GiB) together with
  asserted `TSS_LAZY_FRONTIER=1`; the full 19-row gate passed at `5f836b70`
  and was reproduced after consolidation at `b45b9bf0`
  (`HUNT_REPORT_LAZY_MEMORY_WALL.md`; `MERGE_RESOLUTION.md`).
- **Legacy comparison:** retain
  `TSS_BACKWALK_TT_BYTES=2147483648` (2 GiB), flags off, for continuity with
  the round-9b baseline and regression battery (`ac3f455f`, `dba6111d`,
  consolidated rerun `b45b9bf0`).
- **Coverage caveat:** the checked-in spare-corpus gate contains only two
  near-vacuous NO rows; it is a regression check, not substantive λ²
  coverage. Real spare-corpus expansion remains a paper-capstone item
  (`b45b9bf0`, commit message; `MERGE_RESOLUTION.md`).

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

1. **A-0**: per-family root solves at the recommended 1 GiB lazy-on deep
   profile (`5f836b70`), staged to 20M+ rungs as needed, with
   WIN/LOSS/UNKNOWN certificates archived;
   honest UNKNOWN is an acceptable verdict — no bar-lowering.
2. **A-1**: frontier expansion inside solved families (certified subtree
   persistence is the A4/U22 consumer here — atlas work is exactly "many
   deep solves sharing proven frontiers").
3. **A-2**: atlas spot-checks feed the capstone (§I.5) and, Phase-3-side,
   opening-book consumption by serve/eval (out of this doc's scope until an
   owner ruling asks for it).

Prerequisite honesty: atlas economics remain TT-bound (§I.4) and quiet-width
bound (§I.2). G2R3 is folded and U22 is built, but the completed fragment
campaign did not move the deep saturation wall (`e4ef021f`); use the 1 GiB
lazy-on profile and treat further fragment value as unproved.

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
Unknown. This existing leaf sequence is not NQ8's proposed broad deep-solve
ladder, which is DEAD (U31, `23ffc65b`). `deep_verify_failed` MUST be 0;
every consumption passes the single
mint. Deployment is one flag per relaunch at epoch boundaries, and the
Phase-3 `main_3` relaunch is the landing slot for everything below (owner
ruling 5).

## II.2 Improvement axes (owner ruling 2)

1. **Verdict rate at fixed cap.** The leaf-surface campaign has now sized
   this axis: pair-complete wide PN plus lazy frontier plus the interior
   census gate is the selected profile, with the exact flags and headline
   table in §II.7 (`5172d42d`, `HUNT_REPORT_LEAF_SURFACE.md`).
2. **Cheaper solves → wider gating.** Every Part-I speedup converts directly
   into more gated leaves per second at fixed CPU share (raise
   `tss_solver_sample_16` coverage, or lower the trigger threshold) — the
   preferred spend of Part-I wins, per the mission's "useful at LOW node
   counts."
3. **Budget envelope re-tuning** (§II.5) — the solver-side leaf cap/profile
   is decided; the horizon-height choice remains open (`5172d42d`).
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
- **Prerequisite.** The leaf-surface campaign selected the solver profile at
  `5172d42d`, but did not establish NN signal for these scalars; feature-diet
  lessons from the main_3 board-input review still apply (don't ship dead
  inputs).
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
- **PROVEN pre-solve variant (07-16, `ffdd414a` on hunt/dtw-bounds).** The
  census two-gap distance bound (`PROOF_DTW_CENSUS_BOUND.md`) mints the
  same artifact class WITHOUT running the solve: FirstStone theorem T1 +
  sound SecondStone theorem T2 (the naive SecondStone form is FALSE at
  c=3 — machine-checked reachable ply-5 witness; the +1 increment holds
  only for c<=2 there). Engine-visible consumer contract 8.1/8.2:
  ~17 µs census (plausibly ~1 µs on WindowStore) skips the current
  player's `SolveGoal::Win` attempt at the exact +8 rung for ~49% of
  human-corpus leaves (91% opening band), each skip carrying the proven
  "no forced win within h plies" fact. Cannot gate +12 (structural cap
  ply 10/12). Sole OPEN = production mint/verify integration (Phase-3).

### A4 — ProvenFragment persistence [S; the leaf-side face of U22]

- **Built form.** The default-off T10-licensed shared-fragment store persists
  exact-state verified positive subproofs across solves and supports shared
  DAG imports under the monotone verdict contract (`e4ef021f`,
  `BUILD_SHARED_FRAGMENTS.md`; `HUNT_REPORT_SHARED_FRAGMENTS.md`).
- **Soundness.** Exact key/claimant/profile and horizon checks gate lookup;
  max-dominant labels and obligation unions are reconstructed; live payloads
  are pinned; the unchanged strict verifier remains the single mint. Lean
  T10 at `69adffc7` licenses the finished-DAG composition.
- **Economics.** Deep saturation value was NULL at `e4ef021f`. At the 256 KiB
  leaf profile, 22/875 hits yielded no additional hard verdict, so the Phase-3
  configuration keeps fragments off (`5172d42d`). Reopening requires a
  leaf-budget size/admission study that demonstrates verdict value.

## II.4 Lever-2 (proof-value target swap) — status unchanged

Runbook rung 9, **NOT YET BUILT** (build at its rung): value target :=
`tss_proof` where nonzero at expand, proof-valid-under-truncation mask, both
backends. Gate unchanged: the `tss.proof_disagreements` stream must show
deep proofs disagreeing with outcomes often enough to matter. Both labels
are already captured per row, so no data is lost while it waits. A3's
auxiliary head is deliberately weaker than Lever-2 and does not pre-empt it.

## II.5 Budget-envelope retuning — solver profile decided

The campaign selected the h=8 solver-side leaf envelope at commit `5172d42d`:
pair-complete wide PN, lazy frontier, interior census gate, node cap 500,
and TT cap 262144 bytes; §II.7 records the exact configuration. The older
`tss_solver_park_timeout_ms` and `tss_solver_sample_16` deployment controls
remain runbook-owned. Relative horizon 16 remains an open trainer-side
decision because it materially raises verdict rate at additional cost
(`5172d42d`, `HUNT_REPORT_LEAF_SURFACE.md`).

## II.6 Deployment map (Phase-3)

Phase-3 relaunch order of the NEW items, after the runbook's existing rungs
(each line = one rung, one relaunch, shadow-first where semantics allow):

1. Land the exact §II.7 configuration selected at `5172d42d` — shadow mode
   first (`tss_solver_mode=1` twin-run).
2. A3 telemetry counters (pure shadow; no consumption).
3. A1 shadow: record would-be-masked support vs actual π′ (the
   `win_retained_mass_mean` analogue for zones) — consume only if mass
   actually moves.
4. A2 features behind a flag (needs a `target_regime`-style note only if
   input schema versioning demands it).
5. A4/shared fragments stay default-off unless a later leaf-budget study
   clears the verdict-value gate (`e4ef021f`, `5172d42d`).
6. A3 consumption (aux head or shaping) if its shadow fire-rate justified
   the head; Lever-2 at its own gate.

Every rung keeps invariants 1–6 and the runbook's watch metrics;
`deep_verify_failed` stays the hard-stop counter.

## II.7 PHASE-3 LEAF CONFIG

The leaf-surface campaign recommends configuration D at landing commit
`5172d42d` (`HUNT_REPORT_LEAF_SURFACE.md`):

```text
width = WidthOptions::vcf_pair_complete()
TSS_LAZY_FRONTIER=1
TSS_INTERIOR_CENSUS_GATE=1
TSS_SHARED_FRAGMENTS=0
TSS_K_REPLY_CONSUME=0
goal = SolveGoal::Win
relative_horizon = 8
node_cap = 500
tt_bytes_cap = 262144
```

Keep one persistent `TssSolver` per real leaf batch/worker; do not reconstruct
it for every solve (`5172d42d`).

Headline verdict-rate matrix (all cells are 300 solves and all figures are
from `5172d42d`, `HUNT_REPORT_LEAF_SURFACE.md`):

| relative horizon | cap | narrow A | selected D |
|---:|---:|---:|---:|
| 8 | 500 | 5.00% | 5.33% |
| 8 | 2,000 | 5.00% | 5.33% |
| 8 | 8,000 | 5.33% | 5.33% |
| 16 | 500 | 5.33% | 13.00% |
| 16 | 2,000 | 6.33% | 13.33% |
| 16 | 8,000 | 7.00% | 13.33% |

At native h=8, D@500 is the recommended production query. The open
horizon-height question is **h=8 versus h=16**: the h=16 arm roughly doubles
the measured verdict rate, but choosing that extra trainer-side work was not
authorized by the campaign (`5172d42d`).

---

# Part III — U-item status ledger and crosswalk

One row per item. STATUS now also records BUILT, ALWAYS-ON FIX, CLOSED,
REFUTED, DEAD, and PARKED verdicts from the efficiency program; "where it
lives" names the engine mechanism, campaign round, or
Part of this plan that owns it now. Proof anchors cite the proof doc tag and
the Lean decl where one exists (`TssZones.*`; status per
`E:\tss-lean\LEDGER.md` at 2026-07-16 — the coming `SolverInterface.lean` /
`SOLVER_HANDOFF.md` supersedes this column when it lands). Soundness
classes are the original ones unless re-derived.

**Tallies: 6 DONE, 4 BUILT, 1 ALWAYS-ON FIX, 1 CLOSED, 1 REFUTED, 2 DEAD,
2 STRUCK, 2 SUPERSEDED, 6 LIVE, 5 IN-FLIGHT, 1 PARKED (31 rows).**

| # | Original intent | Class | STATUS | Where it lives now | Proof anchor |
|---|---|---|---|---|---|
| U1 | Zone generator at `k<B` AND nodes (hitting ∪ 𝒜 ∪ ℬ ∪ core ∪ band) | [H] | **SUPERSEDED** | Formula superseded by U12's ranked union (already amended in the old doc §7); the k<B generator itself now ships as G2R3's `ranked_unforced_defender_zone` | T3/T4 (via U12) |
| U2 | Zone-carrying certificates + full D9 verifier checklist | [S] | **SUPERSEDED** | Checklist rows replaced by the revised contract (D10 roles, D14 budgets, four-part union, (Z2)/(Z4)/(Z5′)); shipped in G2R3's verifier rewrite. The D9 grammar obligations (typed leaves, nonempty S(N), no-defender-terminal, exact resolutions) survive verbatim inside the new contract | D9–D16, T3; Lean D9 grammar decls (STATED), `TssZones.T3` PROVEN |
| U3 | Staple-by-theorem at dispatch nodes; no complement enumeration | [S] | **DONE** | Wide engine `implicit_dispatch` (premise `min_hitting_set == b`); the per-omitted-move replay is gone from the normative path | U3 lemma; T1/T6/L3; Lean λ¹ soundness (`lambdaOne_win_sound`/`loss_sound` PROVEN), T6 kernel calculus decls |
| U4 | Path-derived ply clock, horizon semantics, two-stamp cache rule | [S] | **DONE** | Clock/horizon semantics in the wide engine; zone nodes carry an exact build-horizon binding (G2R3); the two-stamp fragment rule stands narrowed by U13's local budgets | D9, L7, L11; `TssZones.L7_*`, `L11_*` PROVEN |
| U5 | P3 same-turn commutation: pair-canonical generation | [S] | **DONE** | Structural in the wide TT via canonical pair dedup (`canonical_frame`/`canonical_coord_key`); note: commutation deliberately disabled on G2R3 zone nodes (separate contract) | P3 (DOMINATION.md); Lean §9 P3 row UNSTATED — formalization backlog |
| U6 | Interior forced-move guard default-on + λ¹ policy mask | [S] | **LIVE** | Part II / runbook rungs 2–3 (`tss_interior_guard`, `tss_policy_target_sharpen`); A1 extends the mask idea to `k<B` nodes | U3 lemma, T1, T6; loss-delay caveat recorded |
| U7 | OR-node ordering without child replay | — | **STRUCK** | Already implemented pre-plan (old R1 finding); the wide engine's fork-degree/tau priors supersede the mechanism anyway; DEEP_WIN ordering-regression telemetry residue → capstone (§I.5) | — |
| U8 | Trigger + regime detector (leaf gating) | [H] | **LIVE** | Part II axis 1–2; the solver profile is now selected by the leaf-surface campaign (`5172d42d`, §II.7). Production trigger/coverage tuning remains runbook-owned; zone-cardinality scheduling may use verifier-derived quantities only | correctness-safety conditions carried from old plan verbatim; `HUNT_REPORT_LEAF_SURFACE.md` |
| U9 | ES-potential futility check (Cor. 2 integer test) | [S-bounded] | **STRUCK** | Struck 07-16: the ES *global* forever-blocking claim is greedy-refuted (`ES_GLOBAL_BOUNDARY` Thm 1; GAP-RAW open), removing the intended growth path; and corpus data pre-triggers the old kill criterion — `Φ<1` fires at 0.02% of defender FirstStone nodes (<1% bar). Honesty note: the bounded Thm-2 form (first five attacker placements safe) remains mathematically valid; Φ survives as an A2 *feature*, never a futility gate | ES_POTENTIAL Thm 2/Cor 2; refutation ES_GLOBAL_BOUNDARY Thm 1 |
| U10 | Adversarial fixtures + differential harness; mutation suites are the gate | [T] | **DONE** (institutionalized) | Standard practice: G2R3's 7-mutation suite, round-2's 209/209 differential, hunt fixtures (`hunt_r1b_chain_sharpness` etc.), one-sided matched-horizon differentials. Every new consuming flag repeats it (invariant 6) | mutation testing is the gate; differentials are evidence |
| U11 | Domination: b=1 dispatch and b=2 spare-stone extension | [S b=1]/[OPEN b=2] | **PARKED** (b=2); b=1 READY | b=1 `L-DISPATCH-B1` is proven at `7e240388`, hostile-review confirmed at `17a6c6de`, and ready for Phase-3 dispatch wiring. b=2 is PARKED OPEN-COMPUTATION-LIMITED at `af6f777c`: zero reversals, but no complete d>=4 directional comparison; reopen only with a certified-engine exact reference (or independent theory justification) | `PROOF_DISPATCH_DOMINATION_ROUND1.md` + `REVIEW_DISPATCH_DOMINATION_ROUND1.md` at `17a6c6de`; `EXPERIMENT_DOMINATION_B2.md` at `af6f777c`. **Inline conflict:** report §6.4 says PROOF-ROUND-READY, explicitly superseded by the `af6f777c` commit message's orchestrator ruling |
| U12 | Ranked zone generator + verifier (`Z_dir ∪ Z_seed ∪ Z_touch ∪ Z_virgin`, (Z2)/(Z4)/(Z5′)) | [S] | **IN-FLIGHT** (landed in G2R3, fold pending) | G2R3 `ranked_unforced_defender_zone`: shadow/verify/consume all green; verifier independently re-derives the union + D9 fallback; witness WIN verifier-accepted. Uniform wrappers this round; exact clocks = U16 | T3/T4/T7, D10–D16; `TssZones.T3`, `T4` PROVEN; `T5` PROVEN; zone-component decls STATED |
| U13 | Local budget labelling (D14/L11); cache reuse via budget inequalities | [S core; cache needs-derivation] | **IN-FLIGHT** (core) / cache lane LIVE | Core landed in G2R3 (exact stored local `B`, bottom-up D14, verifier-checked). The online-cache-reuse derivation (replacing final-assembly recheck) remains open — final assembly still rechecks all inequalities; failure ⇒ Unknown | D14, L11; `TssZones.L11_*` PROVEN |
| U14 | Sparse LOSS witnesses (≤3 at b=1, ≤6 at b=2) | [S] | **DONE** (improved) | Wide engine `sparse_loss_witnesses`; caps improved to **3/5** (R4b pinned relatively) | L13; `TssZones.L13_capThree`/`L13_capFive` PROVEN; sharpness fixtures STATED |
| U15 | Kernel T6 at forced nodes (`K_b`) | [S] | **DONE** | Wide engine exact K2 kernel in canonical defender order, beside `implicit_dispatch`; `mhs>b` hard guard per T6 | T6; Lean `t6Kernel_*` calculus decls landed (full T6 region contract still being stated) |
| U16 | Exact ranks and exposures (D15/D16 clocks) | [S when checks pass] | **LIVE** (backlog, promoted in value) | Behind a future flag with uniform-B as differential oracle; now also the capstone's uniform-vs-exact delta (§I.5 item 3) AND the only route to settling R2's full-union sharpness (blocked on D16 exposure labels only certificates supply — `hunt/r1b-r2` §3.3) | D15/D16, L11; ledger rows STATED |
| U17 | Branch-indexed substitution envelopes (D17) | [S] | **LIVE** (backlog) | Unimplemented in engine; **proof basis upgraded**: T9 + both dismissal corollaries now kernel-checked. Still the largest verifier surface; both `+1` transition charges mandatory (R7 pinned); gate on profiles showing whole-subtree unions dominate zone width | D17/T9; `TssZones.T9`, `T9_soundDismissal` PROVEN |
| U18 | Certificate DAGs (D18/T10) | [S] | **BUILT**, default-off | T10-licensed shared-fragment store: exact-state positive fragments, max-dominant labels, obligation union, pinned/byte-accounted payloads, final strict replay. Deep saturation value NULL (`e4ef021f`); current leaf value none (`5172d42d`) | Lean T10 `69adffc7`; `BUILD_SHARED_FRAGMENTS.md` + `HUNT_REPORT_SHARED_FRAGMENTS.md` at `e4ef021f`; leaf evidence `HUNT_REPORT_LEAF_SURFACE.md` at `5172d42d` |
| U19 | **NEW** — Quiet-turn OR edges (attacker width) | [H, verifier-gated] | **IN-FLIGHT** (landed in G2R3) | `quiet_turn_or_edges`: complete two-placement attacker turn universe, fired on forcing exhaustion at OR nodes; replayed under (Z4) by the verifier. The plan's first attacker-side item; closes the λ² structural gap (RZOP ranks 1–2 vindicated) | verifier gate is soundness; (Z4) replay; corpus λ² gradient 8.7%→87% |
| U20 | **NEW** — Seed-radius one-relay shrink (`8·d` → `8·(d−1)`) | [S] | **IN-FLIGHT** (G2R3 step 4) | Production `seed_band_radius` in finder AND verifier; separately gated (chain fixtures keep binding seed at `8(B−1)`, shed at `8(B−2)`; full gate re-run; any previously-verifying cert failing ⇒ STOP+revert+writeup) | L9′ (`8(B−1)` bound; SHARP per `hunt/r1b-r2` §2 — attained in all 364 probe positions at B=2; absolute-pin attempt honestly BLOCKED); R1b OPEN as theory, shrink-evidence unconditional for the implementation |
| U21 | **NEW** — `Z_virgin` finite test (formerly absorbed) | [S] | **IN-FLIGHT** (inside U12's landing) | The pre-round-3 shipped verifier ABSORBED `Z_virgin` entirely (full-legal fallback where theory licenses the finite `8(E^D−6)` inversion test — `hunt/r1b-r2` §3.1). G2R3's re-derived union computes it under uniform wrappers; the exact-`E^D` refinement belongs to U16 | D16, L12; `TssZones.zVirgin` + completeness under L11 premise STATED; fixed-window sharpness `L12_fixedWindowExposureSeven_sharp` PROVEN |
| U22 | **NEW** — TT policy + ProvenFragment persistence | [S exact-state; strict-verifier-gated] | **BUILT**, default-off | Same shared-fragment store as U18 under the monotone contract; warm verified-hard improvements are licensed, but the completed campaigns leave deep saturation NULL and the selected leaf config keeps fragments off | `BUILD_SHARED_FRAGMENTS.md` and `HUNT_REPORT_SHARED_FRAGMENTS.md`, completion `e4ef021f`, consolidation `b45b9bf0`; leaf ruling `5172d42d` |
| U23 | **NEW** — Residual re-attack frontier | [T→H] | **LIVE** | On UNKNOWN, emit the blocking defender reply set as a routing-only field; re-solve only those at deeper caps next pass. Mints nothing. Payoff unlocked by U19 (quiet width now exists to exploit the routing) | none needed (routing only) |
| U24 | **NEW** — Macromove / class-partition defender collapse | [UNPROVEN → needs fresh lemma] | **LIVE** (backlog) | Extends U11: merge distinct defenses sharing one winning continuation (RZOP T2 ancestry); sound only where absorbed fillers are frontier-inert, which caps the win (honest impact medium/low). Hostile-review the lemma before implementation | DOMINATION.md Lemma 7 (frontier-inertness); Wu & Lin Lemma 12 as architecture template only |
| U25 | **NEW — Lazy frontier admission** | [S refinement; cap-aware] | **BUILT**, default-off | Discovered by NQ4 (`f30e3fb1`), built at `86a6418c`; official ruling is 1 GiB + `TSS_LAZY_FRONTIER=1` (`5f836b70`), with the filtered 512 MiB bottleneck 8.4x faster (8.38x exact). Component of the winning leaf config (`5172d42d`) | `HUNT_REPORT_TURN_QUOTIENT.md`; `PROOF_LAZY_FRONTIER.md`; `HUNT_REPORT_LAZY_FRONTIER.md`; `HUNT_REPORT_LAZY_MEMORY_WALL.md`; `HUNT_REPORT_LEAF_SURFACE.md` |
| U26 | **NEW — Interior census gate** | [S bounded-WIN dismissal] | **BUILT**, default-off | DTW census gate built at `90f559be`; inert at the unbounded profile, live 79–93% on horizon-bounded forcing solves (78.9–93.4% exact), and a component of the leaf winner at `5172d42d` | `HUNT_REPORT_PN_INIT.md`; `BUILD_INTERIOR_GATE.md`; `HUNT_REPORT_LEAF_SURFACE.md` |
| U27 | **NEW — R-FIX1 verifier-exact zone clock** | [S correctness] | **ALWAYS-ON FIX** | Materializer stamps the exact D14 local budget and assembled horizon; verifier unchanged. This is the program's only unconditional production behavior change (`2454fa91`, consolidated `b45b9bf0`) | `HUNT_REPORT_HORIZON_LADDER.md`; `FIX_ZONE_CLOCK.md`; `MERGE_RESOLUTION.md` |
| U28 | **NEW — K_reply urgent SecondStone kernel/consume** | [S under five-clause trigger] | **CLOSED**, default-off | Kernel proven and shadow-validated across 220,160 fires at `b8b67bf5`; consume is sound but deep economics are negative from trigger/trajectory tax (`c4b496ed`). It is not routed through wide PN, and the narrow leaf probe is about 85x slower than the selected route (`5172d42d`). Reopen only for a justified precomputed-urgency trigger | `PROOF_QUIET_LOCALITY.md` at `833020ed`; `BUILD_K_REPLY_CONSUME.md`; `HUNT_REPORT_LEAF_SURFACE.md` |
| U29 | **NEW — Certificate support hashing** | [T→S redesign] | **REFUTED** for current format | NQ3 found strict unchanged transfer 0/180 and reuse multiplier 1.000x at every tested radius; current RootBinding/ReplayKey are global. The open successor is the `C_rel` redesign conjecture with eight named obligations (`3cd224fe`) | `HUNT_REPORT_CERT_SUPPORT.md` on `hunt/cert-support` at `3cd224fe` |
| U30 | **NEW — D6 search-TT folding** | [T] | **DEAD** | NQ5 measured zero duplicate TT entries and zero duplicate expanded states across every cohort; hot-path canonicalization would buy no state saving (`f30e3fb1`) | `HUNT_REPORT_TURN_QUOTIENT.md`, D6 section, at `f30e3fb1` |
| U31 | **NEW — Broad semantic-horizon laddering** | [H, verifier-gated] | **DEAD** | Every tested schedule lost on both completed forcing cohorts at `23ffc65b`; R-FIX1 repaired the exposed zone-clock bug but explicitly left the economic refutation unchanged (`2454fa91`) | `HUNT_REPORT_HORIZON_LADDER.md`; `FIX_ZONE_CLOCK.md` |

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
| Lazy-frontier discovery | 62.6–67.3% of retained wide entries never expanded | `HUNT_REPORT_TURN_QUOTIENT.md`, commit `f30e3fb1` |
| Lazy official profile / reduced-TT bottleneck | 1 GiB + flag is full-gate recommendation; filtered 512 MiB `0l` row 8.4x faster (8.38x exact) | `HUNT_REPORT_LAZY_MEMORY_WALL.md`, commit `5f836b70`; consolidation rerun `b45b9bf0` |
| Shared fragments | deep saturation NULL; leaf 22/875 hits and 0 added verdicts | `HUNT_REPORT_SHARED_FRAGMENTS.md`, completion `e4ef021f`; `HUNT_REPORT_LEAF_SURFACE.md`, `5172d42d` |
| Interior census gate | 79–93% headline (78.9–93.4% exact) on horizon-bounded forcing cohorts; 0 evaluations unbounded | `BUILD_INTERIOR_GATE.md`, commit `90f559be` |
| K_reply shadow / leaf routing | 220,160 shadow fires; selected wide leaf route about 85x faster than the narrow probe route | shadow `b8b67bf5`; `BUILD_K_REPLY_CONSUME.md` at `c4b496ed`; `HUNT_REPORT_LEAF_SURFACE.md` at `5172d42d` |
| Leaf headline | D h=8: 5.33% at cap 500; D h=16: 13.33% at cap 2,000 | `HUNT_REPORT_LEAF_SURFACE.md`, commit `5172d42d` |
| NQ3 current-format transfer | 0/180 unchanged transfer, 1.000x reuse multiplier; 8 successor obligations | `HUNT_REPORT_CERT_SUPPORT.md` on `hunt/cert-support`, commit `3cd224fe` |
| D6 search-TT folding | 0 duplicate entries and 0 duplicate expanded states across measured cohorts | `HUNT_REPORT_TURN_QUOTIENT.md`, commit `f30e3fb1` |
| Spare-corpus coverage | 2 near-vacuous checked-in NO rows | consolidation commit message `b45b9bf0`; `MERGE_RESOLUTION.md` |
| Lean status | T3/T4/T5/T9 + dismissal corollaries PROVEN; T10 kernel-checked | `E:\tss-lean\` commit `69adffc7`; `BUILD_SHARED_FRAGMENTS.md` |

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
6. **Domination b=2 adjudication**: `EXPERIMENT_DOMINATION_B2.md` §6.4 at
   `af6f777c` says PROOF-ROUND-READY, but that same commit's binding message
   explicitly says the orchestrator ruling supersedes the session reading
   and parks the lane OPEN-COMPUTATION-LIMITED. U11 records the superseding
   ruling and the exact reopening condition; it does not erase the report
   text.

# Change log

- 2026-07-17: Folded the efficiency program through leaf-surface landing
  `5172d42d`: U18/U22 shared fragments, lazy frontier, interior census gate,
  R-FIX1, K_reply, NQ3/NQ5/NQ8, domination b=2, official profiles, and the
  Phase-3 leaf configuration. Documentation-only register update.
- 2026-07-16: Ground-up rewrite for the wide-engine era (this document).
  Supersedes FINAL (R3 PASS) 07-14. Old doc stubbed in place at
  `hexfield-eq-main-review` worktree with a SUPERSEDED banner.
