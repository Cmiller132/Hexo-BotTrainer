# Plan — Forced-Tree Search & Learn-from-Search for hexfield_eq (TSS v2)

> **SUPERSEDED (2026-07-17).** This v2 plan is superseded as plan-of-record by
> **`docs/PLAN_TSS_MCTS_INTEGRATION.md` (v3)** — the primary and normative
> integration doc. The §2 soundness contract is carried into v3 verbatim and
> remains binding; this file is kept as the design history of the built
> Stage 0–4 stack (`claude/tss-v2-build`). Do not plan new work from this doc.

> **Provenance.** v2, 2026-07-13. Supersedes the v1 plan (27-agent design workflow +
> Codex ultra, reconciled). This revision follows: (a) a line-by-line source
> re-verification of v1 (~20 file:line claims checked, **all accurate**); (b) an
> independent re-derivation of λ¹ soundness (**sound post-opening**, confirmed);
> (c) a second adversarial Codex `gpt-5.6-sol` ultra pass targeted at the points
> where this redesign diverges from v1 (verdicts folded in below); (d) **owner
> direction**: the program is prioritized, the organizing principle is the
> forced-tree thesis (§0), training integration goes **targets-first**, and the
> search must remain free to work through *all* moves of a hitting set — proofs
> collapse only what is decidable, never genuine choice.
> Static analysis only; nothing here has run on the GPU. Design doc, not a build order.

Goals (owner): during live threats the defender's replies are *forced* (hitting
sets), so everything else can be dropped and search can go deep inside the limited
move tree — the current implementation does not take full advantage of this. The
changes must (1) allow that deep search, and (2) feed it back into **training**, so
the model genuinely learns from the search and both training and play strength
improve. Closed loop: the net internalizes the forced tree → the prior itself
prunes → freed budget goes deeper.

---

## 0. Organizing principle — the forcedness boundary

Let `B` = placements remaining this turn (2 at FirstStone, 1 at Opening/SecondStone)
and `k` = `min_hitting_set` over the opponent's live ≥4 windows
(`threats_shared.rs:97`, exhaustive for k ≤ 2). At a node with **verdict None**
(no own win-now, defense feasible):

| Condition | Status | Sound move set |
|---|---|---|
| `k == B` (defense consumes the whole turn) | **Fully forced** | The hitting-cell universe (every empty of every opponent ≥4 window). Every non-hitting move is *provably lost by a one-ply λ¹ lemma*: it leaves the windows untouched with insufficient remaining budget → opponent completes next turn. (Verdict None ⇒ no own count-4/5 exists ⇒ no dropped move can secretly be a win.) |
| `k < B` (a spare stone exists) | **Half-forced** | No sound pruning. A quiet developing move, or the classic tempo play — a counter-threat with the spare stone — can be strictly better than a redundant hitting stone. |

Two rules follow, and everything in this plan is an application of one of them:

1. **Prune only with a proof stapled to every dropped move.** At `k == B` the λ¹
   lemma supplies that proof in O(window-scan) — no search needed. This is the
   difference between this design and `vcf.rs`, which dropped moves by assumption
   (and truncated the defender list at 24 — the concrete false-WIN defect).
2. **Division of labor: MCTS explores wherever genuine choice exists; proofs
   collapse only what is decidable.** All hitting cells stay live children — the
   neural search values and discriminates among them (some hitting cells also
   develop or counter-threaten; that ranking is the net's job, not the solver's).
   The solver/guard removes only branches that carry a refutation.

The half-forced layer is where the owner's thesis pays off *indirectly*: when the
MCTS discovers a new threat deep in the tree, the forced continuation *below* it is
collapsed by the guard + solver (fan-out collapse one ply down, verdict overrides,
eval elision), so exploring the branchy half-forced choice itself becomes cheap.
MCTS chooses among the genuine options; proofs eat the forced aftermath.

---

## 1. Verified current state (carried from v1; every seam re-checked 2026-07-13)

TSS today is **λ¹ only**, single-sourced in
`packages/hexo_models/rust/src/threats_shared.rs` (`analyze`/`verdict:81`,
`min_hitting_set:97`, `tactical_cells:192`), `#[path]`-included via
`hexfield_eq/rust/src/lib.rs:20-21`. Three hooks, all gated by `tss_enabled`:

1. **Injection** — root force-include beyond Gumbel top-m (`tree.rs:836-857`);
   leaf `split_tactical` forced edges (`tree.rs:1765-1801`, consumed at
   `add_node_from_eval` `tree.rs:1006-1022`). *Widens only — never narrows.*
2. **Leaf value override** — `search.rs:1939-1945` (lockstep) / `:2018-2024`
   (continuous): childless edge + λ¹ verdict → hard ±1 via `backup_virtual`, **no
   node created, no GPU eval**. Edges are consistently verdict-backed (re-derived
   per visit — cheap at λ¹); `value_sq_sum` stays single-population, LCB σ clean.
3. **Root guard** — `tactical_guard_weights` (`search.rs:3600-3631`);
   `classify_root_move` (`:3573-3598`) is a sound ~1.5-ply prover (child outcome +
   child verdict mapped back **by player identity**). Selection-time only;
   **recorded targets are built pre-guard** — see Lever 1.

Load-bearing facts, re-verified:

- **λ¹ `verdict()` is sound post-opening.** The count-4-facing-B=1 hole cannot
  arise (a completed turn always hands the opponent FirstStone B=2; the Opening
  turn cannot contain a ≥4 window).
- **The tree is the poison channel.** `backup_virtual` (`tree.rs:1561-1593`) feeds
  `edge.value()` → `gumbel_completed_q` (`tree.rs:2319`) → π′
  (`search.rs:3334-3384`), `export_q` (cell_q), and `root.value()` (stvalue)
  (`search.rs:3025-3204`). Any unsound hard value poisons three training targets
  with zero head involvement. This is why the whole plan is soundness-first.
- **The main 65-bin value head takes pure `hard_z`** (`soft_z_lambda=0`,
  `samples.py:159-211`, `selfplay.py:456`).
- **Effective main_2 loss weights: `soft_policy=0.5`, `moves_left=0.2`**
  (`configs/hexfield_eq_main_2.toml` overrides `losses.py` defaults). But note
  (Codex, v2 correction): under `policy_target="gumbel"` the **main policy loss at
  weight 1.0 consumes π′** — target-level signal flows through the heaviest
  channel after all. This is a core argument for targets-first (§4).
- **`vcf.rs` is unwired and unsound** (defender set restricted *and* capped at 24
  → false WINs; its `false` means "not proven", never LOSS). Reference only.
- **No solver cost data has ever been collected.** Every cost claim below is a
  guess until Stage 0 measures it (quotient-reps lesson: budget for 0.6× of
  projections).

---

## 2. Soundness contract (v2 — tightened per the adversarial review)

| Signal | Search use | Training use |
|---|---|---|
| λ¹ proof (`verdict()` ±1) | hard ±1 backup; guard; interior pruning at `k == B` | Lever 1 target sharpening; Lever 2 label (rarely disagrees at λ¹) |
| Deep proof, **certificate verified pre-backup** | hard ±1 backup + eval elision (containment ladder §10) | Lever 2 label + per-action outcomes (deferred) |
| UNKNOWN / capped / heuristic (pn-ratios etc.) | move *ordering* + unforced injection only — never a value | excluded from hard labels |

Rules (each traces to a confirmed failure mode):

1. **Typed seam.** `ProofStatus ∈ {Win, Loss, Unknown}`; only two `HardValue`
   producers (λ¹; verified deep certificate). The type prevents accidental
   routing; it does **not** prove semantics (v1 overstated this) — semantics are
   carried by the verifier + harness, the type by construction.
2. **Verify every deep hard result before backup — including cache hits.**
   Write-time spot-checks are too late: a false value contaminates Q/π′/visits
   the moment it backs up. The solver emits a compact replayable certificate; an
   independent verifier replays it; verification failure → downgrade to Unknown +
   a fatal telemetry counter (must stay 0).
3. **A hard LOSS requires the dual certificate** — a proven *opponent winning
   strategy* whose universal nodes exhaust **our** legal moves. Same machinery,
   seats swapped. "My attack failed" is UNKNOWN, never LOSS. (v1 stated the
   principle but Stage 4 never specified the dual; fixed here.)
4. **UNKNOWN never collapses to a scalar.** Cap/budget exhaustion poisons the
   parent AND/OR result to UNKNOWN. `hit_limit` is telemetry only.
5. **Full-key equality on every value-bearing cache hit.** The TT compares the
   full canonical position (occupancy/owners, side to move, exact phase incl.
   SecondStone witness), never a 64-bit hash alone. **This includes the D6 outer
   cache** — v1's "degrades to a miss" is false under a canonical collision, which
   returns a wrong *hard hit*. The neural `StateHash` (history-bearing u64) is
   never used for proofs.
6. **Determinism where targets are made.** Self-play hard paths use node caps
   only. Wall-clock is allowed at serve *for scheduling* — a completed, verified
   certificate found under a deadline is still sound; only converting a timeout
   into a verdict is unsound (v1's blanket ban, relaxed to the true rule).
7. **Soft signals bias, they don't poison — and we say so honestly.** Ordering /
   unforced injection shifts visits and therefore shifts targets (v1's
   "export-invisible SoftHint" was self-contradictory). That is the intended
   mechanism and is safe *because* a heuristic can misdirect search but cannot
   inject a false hard label.
8. **Pruning = proof-stapled dropping only** (§0 rule 1). `firstK`-style
   truncation of any forced set is forbidden everywhere (recorded targets,
   solver defender sets, interior guard children). Geometric sets are uncapped
   and D6-covariant.

---

## 3. Lever 0 — interior forced-move guard (first search increment)

**What.** At interior node expansion, when `verdict == None`, opponent threats are
live, and `min_hitting_set == Some(B)`: the children set becomes exactly the
hitting-cell universe (= `tactical_cells(state)`, which at verdict-None nodes is
precisely the opponent-window empties). Everything else is dropped, each dropped
move carrying its λ¹ refutation (§0). At `k < B`: current behavior (inject-widen,
no narrowing). Root expansion: **unchanged in this increment** (root candidate
narrowing would change recorded-target support; it becomes consistent only
together with Lever 1, and is optional even then).

**Why it's the owner's thesis, literally.** Today a fully-forced node still fans
out across the whole widening nucleus; visits smear over provably-lost children
(the override punishes them, but only after spending selections). Narrowing the
children to the forced set concentrates every visit down the forced lines — the
neural search itself penetrates the limited tree, before any deep solver exists.

**Multiple hitting cells are all kept** — the universe, not one minimal set. The
search works through them and ranks them by value (a hitting cell that also
develops or counter-threatens wins the ranking). Proofs never adjudicate genuine
choice.

**Soundness.** Dropped ⇒ non-hitting at `k == B` ⇒ child verdict −1 by the λ¹
lemma (parent verdict None ⇒ no own count-4/5 ⇒ a dropped move cannot create a
count-5; at B=2, k=2 a non-hitting first stone leaves `k=2 > B_rem=1` ⇒ forced
loss; at B=1, k=1 the opponent completes at B=2). No new theory; pure λ¹.

**Seams.** `split_tactical` (`tree.rs:1765`): when the condition holds, return
`(forced=tactical, rest=∅, cap=|tactical|)`. Condition data (`analyze()`) is
already computed at expansion for injection. Config-gated
(`tss_interior_guard`), off-default until its A/B.

**Effects to measure.** Fan-out at pruned nodes (before/after), forced-line depth
histogram, pos/s (fewer wasted evals), h2h. Changes self-play data distribution →
ships only as a fork A/B (§9), never toggled mid-run.

---

## 4. Lever 1 — guard-consistent policy targets (λ¹, no solver needed)

**The asymmetry it fixes.** Play follows the guard-sharpened distribution;
recorded policy targets are built pre-guard — the net is trained on the
distribution the games do *not* follow. Distilling the guard into the policy head
is the most direct learn-from-TSS lever, and it flows through the **weight-1.0
main policy loss** (π′ under `policy_target="gumbel"`), not the 0.5 soft channel.

**Mechanism.** Per recorded move, build one classification map with
`classify_root_move` over the **union** of the supports involved (visit-weight
export, π′ support, selection support — they differ). Then:

- any class = +1 exists → mask visit weights **and** π′ to the class-+1 set,
  renormalizing by original mass. All proven winners keep their relative mass —
  a *set*, never a single PV (avoids encoding traversal order; owner requirement).
- else → zero class = −1 moves, renormalize; if that empties the target, restore
  the original (mirrors the guard's all-zero fallback — a zero-mass visit target
  hard-fails expansion, `samples.py:281`).

**Implementation conditions (Codex-verified, all confirmed real):**

- Export **raw** visits + a per-action class column from Rust; apply masking in
  the Python writer. This preserves pre-mask visits for `policy_surprise` — else
  sharpening silently reweights rows by up to 8× (`policy_surprise_max_weight`),
  and it makes shadow-measurement free (class column present, masking off).
- Zero weights, never delete actions — `export_q`/cell_q are parallel arrays
  (`selfplay.py:274`).
- A singleton proven set yields a one-hot target (the soft-policy T=2 softening
  acts only on positive support) — acceptable: it is a proven win.
- `opp_policy` copies the next opponent row's π′ (`samples.py:91`); concentration
  can project off the earlier row's legal support → monitor `opp_coverage`.
- Zeroing proven-losing moves concentrates mass on *unproved* moves, which in a
  lost position are merely not-yet-refuted. That is a conservative preference,
  not an exact game-theoretic distribution — acceptable for a policy target.
- Tag rows with a `target_regime` version (schema §8) — never introduce a target
  semantics change invisibly into the ~210k-row rolling buffer.

**Gate: shadow-measure first.** λ¹ hard Q already flows into completedQ → π′ may
already be near proof-sharp. Metrics: fraction of rows with a class-+1 move,
retained proof-set mass, KL(raw‖masked). If the mask moves essentially no mass,
skip this lever as a strength experiment (infrastructure stays for deep proofs,
where the mass will be new).

---

## 5. Lever 2 — proof-corrected value targets

**What.** Where a row's position carries a **sound proof** whose value disagrees
with the game outcome `hard_z`, train on the proven ±1. This is a deliberate
semantic choice — the value head moves toward *game-theoretic value where known*,
behavioral return elsewhere — which is the right estimand for a head whose
consumer is a search that treats values as bounds.

**Conditions (Codex-verified):**

- Typed per-row proof plumbed through the payload — **never inferred from
  `root_value`** (a backup average, not a solved status).
- **Persist both labels** (raw `hard_z`, proof value, proof kind λ¹/deep + cert
  version, disagreement flag). The disagreement stream is the best production
  alarm the program gets; overwriting it destroys the signal.
- Player-identity perspective conversion (never sign-flip-per-ply — FirstStone
  keeps the player).
- Proof validity is independent of the truncation mask: a proof-labeled row in a
  truncated game must train (today `truncated` zeroes `value_mask`; the schema
  gains value-valid-by-proof).
- Ships together with Lever 1 semantics (a +1 value row must not coexist with a
  policy target that abandons the win) — but is **A/B'd separately** (§9).
- 65-bin head: ±1 maps to the endpoint bins exactly like ordinary hard outcomes;
  no special handling needed.

**Honest expectation.** At λ¹, disagreements are rare (the guard already forces
wins through), so this lever is *infrastructure* whose payoff arrives with deep
proofs — every deeply-solved position becomes a perfect training example, exactly
the "model truly learns from search" goal.

---

## 6. Lever 3 — the deep forced-tree solver

**What.** A df-pn solver that searches *only* the forced tree, to depths the
neural MCTS never reaches (the branching factor inside forcing sequences is
1–4):

- **OR (attacker) nodes:** threat-creating moves only. Safe by direction —
  omitting attacker options can only miss wins, never fabricate them.
- **AND (defender) nodes:** *exhaustive-with-instant-dispatch*. Every legal
  defender move is either (a) refuted instantly by the λ¹ lemma (non-hitting under
  live threats with insufficient spare budget — the O(1) staple), or (b) enters
  the search set. At `k == B` the search set is exactly the hitting universe (the
  owner's pruning, now with proofs attached). At `k < B` the spare-stone
  alternatives (counter-threats, quiet tries) genuinely enter — this is where
  UNKNOWNs will concentrate under caps, and honestly so. `H∪C` is a move
  *ordering*, never a generator.
- **LOSS = the dual proof** (§2.3), produced by the same machinery with seats
  swapped.
- Engine: make/unmake via `apply_with_delta`/`undo` (`state.rs:289/361`);
  3-valued results; order-independent Zobrist keys folding stm + phase +
  placements_remaining; full-key equality on every hit (§2.5); D6-canonical outer
  cache with full-representation equality; per-search solve memo + **solved-node
  markers** so a deep solve runs once, never per re-selection (today's λ¹
  re-derivation per visit is fine; a re-run df-pn solve per visit would be
  catastrophic).

**Assurance stack (replaces v1's phase-class build artifact):**

1. Sound by construction (directions above + UNKNOWN propagation).
2. **Per-result certificate verification before backup** (§2.2) — the load-bearing
   runtime gate.
3. Differential harness vs a **common-mode-independent reference**: independent
   legal-move enumeration, a dumb direct six-in-line scanner (not the production
   window store), player-*identity* perspective (naive per-ply negamax is wrong
   for this game's turn structure). Random + curated positions; bounded exhaustive
   sweeps.
4. Property tests: TT-on/off equality, forced-collision tests, make/unmake
   integrity, D6 tests that **replay certificates across orientations** (scalar
   result equality alone misses action-transform bugs).
5. Production shadow soak (§10) before any consumption.

v1's exhaustive-rollout-per-phase-class artifact is dropped as a *gate* (a finite
phase sample proves nothing about unseen states) but its spirit survives as CI
defense-in-depth if ever wanted, tied to solver/verifier/rules hashes.

**Where it runs.** Self-play leaves on a measured fraction (subsampled — the
selection loop is the hot path); the root every move (cheap, highest value);
serve root optionally (§12 Q4). Node caps sized from Stage-0 histograms;
concentrate budget where threats are live (`has_threats()` gate, as today).

---

## 7. Lever 4 — harvesting proofs into depth and throughput

Proofs are facts; this lever converts facts into speed, in four independent
pieces (each its own A/B):

1. **Eval elision.** A proven leaf backs up ±1 with **no GPU eval** — already true
   for λ¹ (`search.rs:1939/2018`); deep proofs extend it up the tree. The refund
   is largest exactly where evals are dearest: the S²-attention endgame batch
   collapse (~13×) is where threats live.
2. **Solve once.** Solved-node markers + per-search memo + cross-move TT (§6). A
   forcing subtree is solved once per game, not once per visit.
3. **Overlap.** Solver CPU runs in the GPU's shadow. `HEXFIELD_PIPELINE_DEPTH2`
   (`search.rs:1130-1166`, requires `HEXFIELD_ASYNC_EVAL`) ships OFF; enable as
   its own VRAM-checked A/B (keep `MAX_GROUP_ROWS=260`, do not raise
   `PAIR_CEILING` — the ATTN2 regression note). Add the select-phase wall timer
   first — do not assume the shadow absorbs the solver.
4. **Root adjudication (last, plumbing-heavy).** When the root is proven, the
   game's outcome is known — stop playing it; biggest saving. Requirements
   (Codex-confirmed breakage list): a **distinct adjudicated status** (not
   `truncated`, which masks value/STV/cell_q/moves_left) with per-head masks —
   value valid by proof, `moves_left` masked or proof-depth-derived, STV/opp_policy
   handled at the cutoff; `.hxr` record status; winner-balance/game-length stats
   corrected; ε ≈ 10–25% of proven games still play out. **The ε playouts are an
   alarm, not a falsifier**: an ordinary playout cannot falsify an existential WIN
   proof (the winner may deviate). Real audits **force the proving side to follow
   its certificate** and verify certificate coverage of the opponent's replies.
   Adjudicate only after measuring counterfactual saved eval-ms (it may be small:
   proofs near natural termination save only a few moves).

Every visit and eval refunded by 1–4 is spent elsewhere, deeper. "Deeper search"
here means *never spending budget on anything already proven* — not a per-move
visit-count jump, and it is delivered mostly in self-play, not at serve.

---

## 8. Data plumbing (schema v5) and deferred auxiliary heads

**Schema v4 → v5** (`shards.py:35-44`; old shards load with empty proof columns):

```
target_regime: u16                    # Lever-1/2 semantics version; never silent
tss_class_off: u64[N+1]  tss_class: i8[K]     # per-action λ¹/deep class, aligned to pol_act
tss_proof_outcome: i8    tss_proof_kind: u8   # 0/±1; none|λ¹|deep
tss_cert_version: u16    tss_cert_id: u64     # deep only; verifier provenance
raw_hard_z kept in `value`; proof label + disagreement derived at expand
(later, Head B) tss_action_offsets/ids/outcomes CSR — certificate-verified only
```

Six seams as in v1 (payload `search.rs:2907-2930` → writer `selfplay.py:272-326` →
dataclass `samples.py` → expand **both** `samples.py:304-312` *and*
`replay_expand.rs:636-645` (`expand_backend="rust"` is live) → collate
`batching.py:158-171` → losses). A claimed-proven action missing from the legal
projection is a hard data error.

**Auxiliary heads are deferred** until Levers 1–2 metrics stall (fire-rate and
proven-row agreement unmoved): they shape the trunk indirectly, targets feed the
serving heads directly. When/if built:

- **Corrected must-play label** (v1's Head A label was *not derivable from
  `analyze()`* — membership in one feasible hitting pair is not individually
  mandatory): must-play = intersection over all minimal hitting sets (needs a new
  all-minimal-hitting-sets enumeration; trivial at k ≤ 2), plus a
  proven-winning-completion channel.
- Mechanics kept verbatim from v1: additive zero-init D6 per-cell head, built
  **after** `_init_weights()` (`model.py:1766`), train-only emission (zero serve
  cost), `initialize_from` fork.
- Head B (per-action proven outcomes + 0.10 value bootstrap on the 65-bin head)
  once deep certificates exist.

---

## 9. Metrics and A/B protocol

**Shadow metrics (Stage 0, before any behavior change):**
solver cost histograms keyed by `stones_on_board` (nodes, µs, W/L/U rates);
select-phase wall timer (shadow absorption); Lever-1 mask preview (rows with a
proven winner, retained mass, KL raw‖masked); proof-vs-outcome disagreement
count; injection fire-rate (`tree.rs:836-857`, fraction of roots where a forced
cell falls outside top-m); `opp_coverage`; interior-guard fan-out preview.

**Success metrics per lever:** L0 — pos/s + forced-line depth + h2h; L1 —
fire-rate drop + proven-row top-1 agreement + h2h; L2 — disagreement-row value
calibration; L3 — proven-leaf rate, verify-failure counter (**must be 0**),
UNKNOWN rate under production caps (the "vise" kill-criterion input); L4 —
evals-elided/s, saved eval-ms, VRAM headroom.

**Deployment strategy (owner decision 2026-07-13): strap the full stack onto
main_3 once built and well tested** — no ablation gate, no mandatory fork A/Bs.
"Well tested" is defined as: harness green (§6) + differential/property suites +
a **shadow soak on main_3 itself** (solve/classify/log, consume nothing) before
each consumption rung. Rungs attach to the live run **one at a time, in the §10
ladder order, each behind its own config flag** with main_3's regular eval
cadence (pool + Strix + SealBot h2h) as the health gate; any rung reverts by flag
+ checkpoint. Fork A/B (twin `initialize_from` forks, treatment = one scalar,
MCTS h2h ≥ 150–200 paired openings) is demoted to an *optional instrument* for a
rung whose effect looks ambiguous on the live metrics. **One lever per
deployment step** — in particular never attach Levers 1+2 in the same step
(attribution). Target-semantics changes (L1/L2) enter the rolling buffer tagged
with `target_regime`; expect ~4 epoch-equivalents of mixed-regime rows.

---

## 10. Staging and the hard-value containment ladder

- **Stage 0 — typed refactor + shadow instrumentation.** `ProofStatus`/typed
  verdict wrapper (verbatim `analyze().verdict()`), rewire the three hooks,
  differential-tested bit-identical; add the shadow metrics + raw-visits/class
  export (masking off). Zero behavior change. Buys every number the plan
  currently guesses.
- **Stage 1 — Lever 0 (interior forced-move guard)** as a fork A/B. First
  behavior change; search-only; the owner's thesis inside the neural tree.
- **Stage 2 — Lever 1 A/B**, contingent on the Stage-0 mask-mass metric.
- **Stage 3 — solver core + verifier + harness, offline.** No production
  consumption until the harness is green and forced-collision/D6 replay tests
  pass. Schema v5 lands here (columns written, consumers tolerant).
  **Delegated to Codex (owner decision)** with a proof-carrying spec: the
  deliverable includes a written soundness argument for the design (move-set
  completeness lemmas per node type incl. the instant-dispatch boundary, dual
  LOSS construction, UNKNOWN propagation, cache-identity guarantees) mapped to
  the code and test suite; ambitious optimization is welcome **only where the
  proof survives it** — any speedup that cannot be argued correct is rejected.
  Verifier and solver are built as independently as practical (shared engine
  primitives only), so a solver bug is not silently mirrored in its checker.
- **Stage 4 — production consumption ladder** (each rung its own flag + soak):
  1. **Shadow soak**: solve + verify + log at PCR-fraction leaves; consume nothing.
  2. **Ordering/injection tier** (UNKNOWN-safe): pn-ratio ordering + unforced
     injection.
  3. **Hard LOSS canary**, leaf eval-elision only — no target changes. False
     LOSSes are the *silent* failure (the bot just avoids lines), so the soak
     must actively probe avoided lines in the harness, not wait for alarms.
  4. **Hard WIN canary** with certificate-forced audits. Loss-first is a
     blast-radius ordering, not correctness evidence; once the stack is green,
     WIN-side is justified (false WINs are the *loud* failure).
  5. **Lever 2 deep labels** (target change, own deployment step).
  6. **Serve-time deep root guard (owner: include).** Solver at the serve root;
     proven ±1 classes feed the existing `tactical_guard_weights` /
     `classify_root_move` machinery in place of λ¹-only classes. CPU-only,
     zero training contact (serve games produce no shards), reverts by flag.
     Enable after rung 4 (WIN-side certs must be trusted before they may force
     serve moves).
- **Stage 5 — Lever 4.3/4.4** (`PIPELINE_DEPTH2` A/B; adjudication with the §7.4
  plumbing, pending the owner's keep/cut call) **+ §8 heads** if metrics say the
  trunk hasn't internalized.

**Parallel track note.** Maturity remains the #1 strength lever (Strix-gap
finding); main_3 keeps training in parallel. **Owner decision: no
`GROUP_ORDER=1` ablation gate** — the program proceeds on thorough testing +
judgment (its λ¹-grounded components are logically sound independent of the
fiber question; the ablation stays on the general backlog, unrelated to TSS).

---

## 11. Risks, kill-criteria, walls

- **The vise (kill Stages 3–4).** If Stage-0 histograms show a deterministic
  node cap small enough for throughput is too small to prove λ²⁺ in the
  threat-dense endgame, the deep solver delivers ~nothing.
- **Lever-1 no-op (skip Lever 1 as strength work).** Shadow mask moves ~no mass
  → π′ is already proof-sharp; keep the plumbing for deep proofs only.
- **Verify-failure counter > 0** — hard stop on deep-value consumption; the
  certificate path has a bug; fall back to λ¹-only overnight.
- **Nonstationarity.** More ±1 mass into cell_q/stvalue/policy targets mid-run on
  a rolling buffer mixes populations; `policy_surprise` amplifies shifted rows up
  to 8×. Mitigations: `target_regime` tag, fork-based A/Bs, calibration tracking.
  The LCB purity firewall protects search, not training targets.
- **Host memory (a real run-killer, unrelated to soundness).** The training host
  is a 29 GB WSL VM with an earlyoom backstop and a history of memory incidents
  (legacy expansion transient; graph-key ladder kills at 26.7 GB). The solver's
  TT, D6 outer cache, and certificate buffers must be **hard-capped and
  accounted** (bounded TT with replacement, per-search memo freed at move end,
  cert buffers streamed not accumulated); an unbounded cache would get the run
  killed by earlyoom regardless of how correct the proofs are. Memory ceiling is
  a Stage-3 acceptance criterion, not an afterthought.
- **Walls (respect, don't fight):** S² endgame batch collapse (~13×),
  `SUPPORT_RADIUS=4`, CUDA-graph VRAM ceiling (ATTN2 regression precedent). Deep
  search runs PCR-fraction in self-play, not per serve move. Budget for 0.6× of
  projected wins (quotient-reps precedent).

---

## 12. Decisions log (owner, 2026-07-13)

> **Build status (2026-07-13, branch `claude/tss-v2-build`):** Stages 0–4 are
> BUILT and tested, all flags default-off — Stage-0 typed core + shadow
> metrics (bit-identity golden-proven), Lever 0, Lever 1, the Codex-built
> proof-carrying solver + independent verifier (docs/TSS_SOLVER_PROOF.md),
> and the Stage-4 ladder (shadow / verified LOSS / verified WIN / deep root
> guard with play override). The Lever-2 train-read label swap builds at its
> rung (§10 rung 5) once `proof_disagreements` justify it — both labels are
> already captured per row. Deployment: **docs/TSS_RUNBOOK.md**.

1. **Target run: main_3.** The full stack attaches to the live main_3 run once
   built and well tested, rung-by-rung per §9/§10 (flags + shadow soaks + eval
   cadence as the gate; fork A/B optional instrument only).
2. **No `GROUP_ORDER=1` gate.** Thorough testing + judgment instead; the
   λ¹-grounded components are logically sound independent of the ablation.
3. **Adjudication: CUT** (owner 2026-07-13): self-play games always play out —
   the model must learn from those positions, and forced lines make them quick
   anyway. §7.4 is void; eval-elision (§7.1) is unaffected (it skips GPU evals
   inside search and drops no training rows).
4. **Serve-time deep root guard: include** (§10 rung 6) — after WIN-side
   certificates are trusted.
5. **Stage 3 delegated to Codex** with the proof-carrying spec (§10): ambitious
   optimization permitted only where the soundness argument survives it.

---

## Appendix — key file:line seams (re-verified 2026-07-13)

| Concern | Seam |
|---|---|
| λ¹ threat model | `packages/hexo_models/rust/src/threats_shared.rs` (`analyze:139`, `verdict:81`, `min_hitting_set:97`, `tactical_cells:192`) |
| Include into hexfield_eq | `packages/hexfield_eq/rust/src/lib.rs:20-21` |
| Injection | `tree.rs:836-857` (root), `add_node_from_eval:1006-1022` + `split_tactical:1765-1801` (leaf; Lever-0 seam) |
| Leaf value override | `search.rs:1939-1945` (lockstep), `:2018-2024` (continuous) |
| Root guard / classifier | `search.rs:3600-3631` / `:3573-3598` (Lever-1 class source) |
| Backup (poison channel) | `tree.rs:1561-1593`; `edge.value():242`; `gumbel_completed_q:2319` |
| Export payload (targets) | `search.rs:2907-2930` (native), `:3025-3204` (build), π′ `:3334-3384` |
| Value-target purity | `samples.py:159-211` (`soft_z_lambda=0`), `selfplay.py:456`; truncation masks `samples.py:362`, zero-mass fail `:281` |
| opp_policy coupling | `samples.py:91` (`_future_opponent_policy`) |
| policy_surprise | `selfplay.py:274` → `batching.py:114` (8× cap `main_2.toml`) |
| Loss weights | `configs/hexfield_eq_main_2.toml` (`soft_policy=0.5`; `policy_target="gumbel"`); defaults `losses.py:20-28`; cell_q wiring `losses.py:319-326` |
| Warm-start / forks | `checkpoints.py:104-135`, `:167-169` (`initialize_from`/`warm_start_into`) |
| Head init gotcha (deferred heads) | `model.py:1766` (`_init_weights`), `1768-1776` (register lane), `2412` (cell_q archetype) |
| Shard schema | `shards.py:35-44` (v4 → v5) |
| Depth/throughput | `search.rs:1130-1166` (`PIPELINE_DEPTH2`); make/unmake `state.rs:289/361` |
| Unsound prototype (reference only) | `packages/hexgnn/rust/src/vcf.rs` (defender restrict + `cap=24:126`) |
| Eval fork mechanics | `eval_driver.py`, `eval_stats.py` (paired openings) |
