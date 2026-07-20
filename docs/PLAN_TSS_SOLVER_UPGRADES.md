# PLAN: TSS Solver Upgrades from the Defender-Zone Proof Program

> **SUPERSEDED (2026-07-16).** This plan's anchor engine — the narrow DFS
> prover of branch `claude/tss-v2-build` — is no longer the normative
> solver. The normative solver is `WidePnSearch` (round-9b, gate-verified),
> and the living unified master plan is now
> **`docs/PLAN_TSS_SOLVER_UPGRADES.md` on branch `claude/tss-vcf-width`**
> (worktree `.claude/worktrees/tss-vcf-width`). That rewrite dispositions
> every U-item below (U1–U18) in its Part III status ledger and carries
> forward the soundness classes, design invariants, and every per-item
> soundness analysis that survives the engine change. This copy is retained
> unchanged below as the historical record those carried-forward analyses
> cite.

Status: **FINAL (R3 PASS)** — hostile Codex review R1 (ultra) → repairs →
confirmation R2 (FAIL, nine residual defects) → repairs → narrow
confirmation R3 (**PASS**, all nine ruled APPLIED-CORRECTLY, numbering and
review log consistent). See §6 review log.
Author: Claude (Fable), from the proof program of
`docs/PROOF_TSS_DEFENDER_ZONES.md`.

Normative sources:
- **Proof document** `docs/PROOF_TSS_DEFENDER_ZONES.md` (T1–T8, L1–L10,
  D9–D13, (Z1)(Z2)(Z4)(Z5), §10 ES layer, §11 domination). Theorem tags
  below refer to it. Full derivations for P1–P3:
  `docs/proof_parts/DOMINATION.md`; ES layer:
  `docs/proof_parts/ES_POTENTIAL.md`.
- **Survey** `docs/PLAN_TSS_MOVESET_ZONES.md` (§9 experiments, §10 adopted
  generator, counterexample families G1/G2/G3).
- **Solver as built** on branch `claude/tss-v2-build`:
  `packages/hexfield_eq/rust/src/{tss_core,tss_solver,tss_verify,tss_reference}.rs`,
  integration in `tree.rs`/`search.rs`, specs `docs/TSS_SOLVER_SPEC.md`,
  `docs/TSS_SOLVER_OPT_SPEC.md`, `docs/TSS_SOLVER_PROOF.md`, profile
  `docs/TSS_SOLVER_PROFILE.md`.
- Measured set sizes (this worktree, scratch probe 2026-07-13, 200 random
  midgame positions): full legal mean 302.8; r3 ∪ A-touched-window empties
  145.1; + stale-area filter 135.5; + count≥2 threshold 114.1; closure
  R(P,D) at D=1/2/3/4/5 → 98.4/99.3/104.5/120.7/158.2; hitting-only 2.1.

Soundness classes:
- **[S]** proven — value-preserving by a theorem of the proof doc (with its
  stated caveats).
- **[H]** heuristic — affects only *completeness* (which proofs are found
  within budget), never soundness, because the verifier gate
  (`hard_value_from_verified`, `tss_core.rs:159`) re-checks every claim.
  Note (review R1): [H] items can still change *which* verified values get
  minted and hence search/training trajectories — "heuristic" means no
  false value can be minted, not that behaviour is unchanged.
- **[T]** tooling/measurement — no behavioural change.

Design invariants (unchanged from the build spec; every upgrade must
preserve them):
1. **Single mint.** Values enter search/training only through
   `hard_value_from_verified`. No upgrade may add a second path.
2. **No silent caps.** The defender list is never truncated by count — the
   `vcf.rs` failure mode. Restrictions must be *predicates* (set
   definitions the verifier re-derives), never budgets.
3. **Failure degrades to Unknown.** Any verify failure ⇒ Unknown + fatal
   counter, never a value.
4. **Opening excluded.** λ¹ and every zone/theorem argument here is
   post-opening; zone omission, dispatch, and futility arms must all reject
   Opening-phase nodes (the verifier already does for dispatch,
   `tss_verify.rs:403-405`).

---

## 1. The solver as built (summary)

Single-pass, statically-ordered AND/OR depth-first proof constructor
(`tss_solver.rs`; not df-pn — no live pn/dn counters, no iterative
deepening). Fixed claimant (`prove_for`); OR nodes take threat-creating
moves only (window-store-driven, `threat_creating_moves` :630 — already the
T2-optimal generator, and its move *ordering* already uses WindowStore
static features without child replay, correctness argument at
`docs/TSS_SOLVER_PROOF.md:392-422`). AND nodes dispatch on λ¹:
- `k == B` (`implicit_dispatch`, :450): explicit set = hitting universe;
  omitted moves carry an implicit λ¹ refutation **re-checked per move by
  the verifier** (`tss_verify.rs:372-397`: apply + analyze every omitted
  legal move).
- `k < B` or no threats: explicit set = **all legal moves** (:463).

Certificates are strategy DAGs (`TssCertificate`, `tss_verify.rs:82`) with
exact root binding, acyclicity + reachability checks, full engine replay;
generic `Terminal`/`Lambda1` leaves only — no ply clock, no zone data, no
named witness windows. Two-tier exact-key TT (solve-local `BoundedTt` +
persistent `SharedProofCache` importing fragments directly into the arena,
`tss_solver.rs:520-592`). Integration: leaf trigger (threats + hash
subsample, node_cap 2000), root guard (`SolveGoal::Both` + forced-move
override), interior λ¹ forced-move guard (default off). Profile: cost is
move-gen/analyze-bound; the historical 91.4% hotspot (per-omitted-move λ¹
staple sweep) was moved out of search by `implicit_dispatch` but **still
exists inside the verifier**.

What the proof program adds, in one line: the searched/omitted split at AND
nodes stops being "hitting vs everything" and becomes a *certified zone*
(T3/T4), the per-move staples become *set-membership dismissals* (the U3
lemma), the budget knobs become a *ply clock* (D9), and the pair level of
spare turns halves by a proven equivalence (P3) — with the verifier
remaining the sole authority.

---

## 2. Upgrade catalog

### Tier A — high impact

#### U1. Zone generator at `k < B` AND nodes — kill the full-legal sweep [H search-side; consumable only with U2]

**Proof basis.** T3/T4 (zone soundness), D13/T7 (closure R(P,D) as
candidate generator + certificate extras), T5 (static coverage), L10 (and
its explicit failure boundary at the 4th future attacker placement).

**Change.** In `prove_universal` (`tss_solver.rs:443`), replace the
`write_legal_moves` explicit set at non-dispatch, post-opening nodes with
the layered generator, ordered hitting-first as today:

1. hitting(P) — empties of live opponent-threat windows (T1: ⊆ r2);
2. 𝒜(P) — empties of A-touched alive windows (cnt_A ≥ 1, cnt_D = 0);
3. ℬ(P, D) — empties of D-alive windows with cnt_D ≥ 6 − D (completion
   guard), D from the ply clock (U4); **D ≥ 6 ⇒ full legal** (D13
   fallback);
4. core-so-far ∩ Legal and the (Z5) band, maintained by a **monotone
   closure loop** (review R1): the current `prove_universal` fixes its move
   vector before proving any child (:459-493), but core grows as children
   are proven — a 4th-or-later future attacker setup cell can enter final
   core while lying outside the 𝒜 term (L10's boundary). After building
   children, union child cores, add newly required core∩Legal cells and
   newly induced band cells, prove the added replies, and repeat until
   stable. Compute the common-case band short-circuit
   (Prot ⊆ Legal ∪ Stones ⇒ band empty) first. Note the exact (Z5) band is
   a ball-of-radius-8·D computation around not-yet-legal protected cells —
   a bounded geometric scan, not a window-store pass (R1 nitpick).
5. optional trims, flag-gated, both **[H]**:
   - *stale-area filter*: skip cells all of whose 18 incident windows have
     ≥ 1 stone of each colour (two-coloured = dead by L4). **This is
     deliberately NOT proof-doc P1** (review R1): a cell whose incident
     windows are merely *empty* is not dead (it seeds 18 virgin windows and
     extends the frontier — the (-4,-4) distance-8 example), and true P1
     additionally requires a certificate-named substitute with a
     frontier-inertness obligation. The all-windows-two-coloured version
     used here never fires on virgin-window cells by construction, but it
     ships as [H] regardless; a true P1 domination arm is future work
     (§2 U11).
   - *count≥2 threshold* on the 𝒜 term (passed 52/52 divergence probe;
     unproven).

**Consumption gate (review R1, phasing).** Until U2's zone verifier is
complete, `k < B` nodes with omissions are **non-consumable**: the solver
may search with the zone (finding proofs faster) but any certificate
containing a non-dispatch Universal with omitted legal moves must be
rejected by the verifier by construction — i.e. ship the generator behind
the same flag as U2, or restrict its use to refutation-side work that never
mints values.

**Expected effect.** Attacks the documented UNKNOWN concentration
(PLAN_TSS_DEEPENING §6): spare-budget nodes go from ~303 children to
~98–160 (D ≤ 5); with U5 the b=2/b=1 spare turn drops ~4.4× in placement
pairs.

**Non-goals / honesty.** D ≥ 6 stays full legal by default (see §5 for the
optional theorem-permitted refinement and why it's deferred). No count caps
(invariant 2). Opening excluded (invariant 4).

**Amended by §7:** U12 supersedes this zone formula with T4's ranked `Z_dir ∪ Z_seed ∪ Z_touch ∪ Z_virgin`; hitting becomes ordering-only.

#### U2. Zone-carrying certificates + full D9 verification [S — the keystone, and the hardest item]

**Proof basis.** T3's R4-ruled caveat verbatim: *proven for valid
zone-carrying certificates obeying the D9 grammar and satisfying (Z1),
(Z2), (Z4), (Z5), with exact D_N and the full defender-placement budget.*
Review R1's central ruling: a verifier that checks only the set inclusions
is **not** covered by T3 — the D9 grammar items are load-bearing, and
omitting them admits a concrete exploit (a zero-edge `Universal` at a
heavily blocked node where hitting, 𝒜, ℬ, core, and band are all empty
passes every set inclusion vacuously and becomes a fake proof). The
checklist below is therefore normative and complete; each item cites its
D9/D11 clause.

**Schema.** `CertNode` gains typed leaves and zone data (a real redesign,
not a field bolt-on — current `Terminal`/`Lambda1` are generic,
`tss_verify.rs:85-99`):
- `Choice{mv, child}` unchanged; a Choice whose placement completes becomes
  a typed **OR-COMPLETION leaf** naming its witness window and completion
  ply (D9).
- **WIN leaf**: names witness window(s), count/phase evidence (count-5 any
  b; count-4 with b = 2), resolution ply.
- **LOSS leaf**: names the witness family 𝒯 (window identities), with the
  adaptive contract's declared resolution ply = leaf-ply + b + 2.
- `Universal{edges, dispatch: bool, zone: Option<ZoneInfo>}` where
  `ZoneInfo` carries the node's D value (recomputed by the verifier — never
  trusted).
- Window identities are stable geometric keys `{start_cell, axis}` with
  validation, explicit caps, and D6 remapping support (the current
  remapper handles only root and move coordinates, `tss_verify.rs:647-672`
  — witness keys must be added to `d6_remap_certificate`).

**Verifier obligations (complete list — review R1).** For every
certificate containing any zone-omission node:
1. Root binding equality AND **nonterminal root** (D9).
2. Path-derived **ply clock** recomputed by the verifier; every resolution
   label ≤ T; the verifier itself establishes T = max resolution ply over
   the tree (never trusts a stored T).
3. **Typed maximal nodes**: every leaf is one of the typed leaves; every
   internal AND node has **S(N) ≠ ∅** (kills the zero-edge exploit); every
   internal node's children are the exact D4-successors of its placements;
   every placement legal at its node.
4. **No defender-terminal edges** — explicit rejection, not just the
   Terminal-winner side effect.
5. **¬own_win_now for the defender at every AND node** and again at every
   LOSS leaf (D9 requires it at both).
6. **OR-COMPLETION leaves** (R2 fatal-defect repair — D9 line 195): the
   leaf's mover is the claimant; the designated placement is **replayed**
   and re-derived to complete: the named witness window contains the
   placement cell and becomes claimant-complete in the replayed successor;
   the recorded completion ply **equals** the path-derived ply of that
   placement.
7. WIN leaves: re-derive own_win_now from the replayed position and the
   named window (count/phase match), with **exact derived resolution
   semantics** (R2): count-5 witness ⇒ resolution = leaf-ply + 1; count-4
   witness (requires b = 2) ⇒ resolution = leaf-ply + 2. Labels merely
   "within T" are insufficient — the verifier recomputes and compares
   equality.
8. LOSS leaves: re-derive that every named 𝒯 member is a live
   claimant-threat window at the replayed position; hitting number of the
   *named family* > b (exhaustive, b ≤ 2); declared resolution **equals**
   leaf-ply + b + 2 (exact, R2), and ≤ T. The contract itself is a theorem
   given these checks (D9/T3 leaf transfer) — nothing further to
   enumerate.
9. **(Z1)** S(N) ⊇ hitting ∩ Legal; **(Z2)** S(N) ⊇ Prot(N) ∩ Legal with
   core(𝒞, N) computed **bottom-up from the final reachable DAG** in a
   post-order pass (this is what makes U1's partial-core-during-search
   safe: any dismissal made against a smaller core than the final Prot(N)
   fails here ⇒ Unknown); **(Z5)** the horizon-scaled band, with the
   Prot ⊆ Legal ∪ Stones short-circuit; **(Z4)** every attacker placement
   within distance 8 of an attacker/root stone of its predecessor — checked
   directly, including the WIN/LOSS-leaf *continuation* placements (derive
   from the named witness windows: their empties are within distance 2 of
   attacker stones, L1/T1).
10. **D ≥ 6 fallback**: a zone node whose recomputed D ≥ 6 must have
    S(N) ⊇ Legal(P_N) — i.e. no omissions (the touched-window store cannot
    enumerate qualifying all-empty windows; full-legal is the only
    verifiable form of the guard there).
11. **Opening**: any zone-omission node in Opening phase ⇒ reject
    (invariant 4).
12. Existing checks preserved: claimant binding, duplicate/illegal edge
    rejection, acyclicity + full reachability (`tss_verify.rs:461-520`),
    ReplayMemo semantics.

**Verify the minimal obligation, not the search heuristic (review R1
missed-opportunity, adopted).** The verifier checks T4's zone — hitting ∪
Prot ∪ band — NOT the D13 𝒜-superset the *search* uses. Search may
over-generate freely; certificates are judged against the smaller certified
zone, so heuristic search sets can never cause spurious rejections.

**Cost note.** Set checks are per-node window-store passes plus the bounded
(Z5) geometry; certificate growth (typed leaves + window keys + D bytes)
needs `SharedProofCache` admission re-measurement before fragment promotion
re-enables for zone certificates (see U4 interaction — promotion of
zone-bearing fragments stays **disabled** until U4's composition rule
lands).

**Amended by §7:** U12 replaces the Z1/band/core checklist rows with compressed live roles and ranked (Z2)/(Z4)/(Z5′) verification.

#### U3. Staple-by-theorem at dispatch nodes — delete the per-omitted-move replay [S — ruled SOUND in R1]

**Proof basis.** L3 (channel confinement), T1, T6, λ¹ definitions D5–D6.
The lemma (wording per review R1):

> At a post-opening AND node with verifier-checked ¬own_win_now and
> min_hitting_set = b, let m be any omitted legal defender move outside the
> hitting universe. Then the child after m is λ¹-lost for the defender,
> with no engine replay needed:
> (i) m is empty and non-hitting, so it lies in no live **attacker**-threat
> window (else it would be one of that window's empties, hence in the
> hitting union — exactly the set the code builds, `tss_verify.rs:415-426`,
> `tss_solver.rs:793-805`); by L3/C2 every **attacker** threat mask — not
> every window mask; defender windows may change — is unchanged, so the
> child's hitting number is still b > b − 1 = child budget (b = 2), or the
> child is the attacker to move with an intact count≥4 threat and budget 2
> ⇒ own_win_now (b = 1).
> (ii) No defender own_win_now at the child: at b = 2, ¬own_win_now bounds
> every D-alive window at count ≤ 3; one placement reaches ≤ 4 < the
> count-5 threshold at b' = 1. At b = 1 the mover flips to the attacker,
> whose own_win_now is the proof. (Own-win precedence:
> `threats_shared.rs:79-89`.)
> (iii) m cannot complete a defender window: one-placement completion needs
> a pre-move alive count-5 = own_win_now at any b — excluded. (Engine win
> check precedes phase advance, `state.rs:309-337`; a count-5 window
> containing a defender stone is two-coloured, not alive,
> `tactics.rs:171-202`.)
> (iv) Frontier changes (L3/C3) don't affect λ¹ verdicts (mask functions,
> D5–D6); and the surviving threat's empties stay legal — within distance 2
> of permanent attacker stones (T1).

**Change.** In `verify_universal`'s omitted-move loop
(`tss_verify.rs:372-397`): `omitted ∧ dispatch ∧ m ∉ hitting_universe ⇒
accepted` by the lemma; node premises already checked by
`dispatch_boundary` (:403-426). **Adopted R1 extension:** don't build the
omitted complement at all — verify the node premises, require and replay
every hitting cell explicitly, and theorem-dismiss *the rest of the legal
set without enumerating it* (the only per-move work left is confirming
explicit edges ⊆ legal, already done). Keep the old per-move staple behind
a debug flag as a paired oracle (U10).

**Expected effect.** The largest single verifier win: O(|legal|)
apply+analyze per dispatch node (~300 engine replays — the direct
descendant of the profiled 91.4% hotspot) becomes O(|hitting|) replays.
Verification stops being the deep-solve bottleneck; `tss_solver_mode ≥ 2`
consumes more proofs per second at the same CPU budget.

#### U4. Path-derived ply clock + horizon semantics [S core; cache composition NEEDS the rule below — R1]

**Proof basis.** D9 (path-derived clock), T3's horizon parameterization,
L7. Two R1 corrections are normative here:
- *Citation fix*: "Prot monotonicity" (D11) concerns descent at **fixed**
  T. The statement actually used is elementary and separate: for a
  **completed** subtree with fixed core, reducing T reduces every D_N, and
  both the ℬ guard and the 8·D band are monotone in D_N — so
  verify-with-exact-(smaller)-D succeeds whenever search-with-guessed-
  (larger)-D generated the searched sets. Final core growth during the
  closure loop (U1) can independently widen obligations; the verifier's
  final-DAG recomputation (U2 item 9) adjudicates.
- *Cache compositionality counterexample* (R1, verbatim keep): a cached
  fragment rooted at defender SecondStone with a D-alive count-4 window,
  proven quickly with local D_N = 1 (4 + 1 < 6 ⇒ empty omitted), imported
  into a flattened certificate whose slower sibling raises global T so the
  node's D_N = 2 (4 + 2 ≥ 6) — the omitted empty is now protected and the
  flattened certificate is **invalid** under its global T. Semantic
  monotonicity ("WIN by T₁ holds for horizon ≥ T₁") does not transfer to
  the *syntactic* D9 object.

**Change.**
- Thread the absolute ply index through `prove` (root ply + depth); typed
  leaves record resolution plies; `compact_certificate` computes a
  candidate T; **the verifier establishes T** (U2 item 2).
- Search guesses a target T (e.g. root ply + 2 × configured max turns) for
  its ℬ/D terms; verification uses exact T. **Preflight vs fatal counter
  (R2 defect-4 repair):** the horizon check on the solver's own candidate
  certificate is a *solver-side preflight*, run before submission to the
  minting verifier — a preflight failure triggers the **single** retry
  (with T taken from the failed attempt, only on a diagnosed horizon-zone
  failure, then stop) and increments a separate `horizon_retry` counter.
  `deep_verify_failed` remains the post-submission fatal counter and must
  stay 0 in steady state (invariant 3 unamended): a certificate that
  reaches the minting verifier and fails there is a bug, not an expected
  retry. `SolveCaps` (`tss_core.rs:97-103`) gains a semantic horizon field
  — prerequisite for U9 as well.
- **Cache rule (mandatory before zone-fragment promotion re-enables):**
  store with each fragment both its `resolution_T` and the `zone_build_T`
  its zone terms were generated against. **Composite stamps (R2 defect-3
  repair):** for a fragment containing imported subfragments,
  `resolution_T` = max over all contained resolutions and `zone_build_T` =
  min over all zone-bearing components' admissible build horizons; and the
  import condition — enclosing global T ≤ `zone_build_T` — must be
  **re-checked against the final candidate global T** after every slower
  sibling is added (the preflight recomputes it on the assembled
  certificate), not only at online import time; otherwise the R1
  slow-sibling failure recurs inside a promoted composite fragment.
  Alternative designs — sealed, independently verified subcertificate
  leaves with local deadlines, or re-closing imported fragments against
  the enclosing T — are future work (§5); the two-stamp rule is the
  minimal sound gate. Non-zone (dispatch-only) fragments are unaffected:
  they carry no D-dependent omissions.

**Amended by §7:** U13 replaces unprincipled fragment-local deadlines with D14/L11 hereditary local budgets and narrows the cache restriction accordingly.

### Tier B — medium impact

#### U5. P3 same-turn commutation: pair-canonical generation + verifier arm [S under the full side-condition set — R1 conditions incorporated]

**Proof basis.** P3 (`docs/proof_parts/DOMINATION.md:677-727`), including
its metadata caveat: P3 equates *outcome* (winner/ply-count), not
`PositionKey` identity — a joint-second-win pair (four stones in an alive
window with two empties p, q: neither singleton wins, either order wins on
the second placement) yields terminal states whose keys differ in the
`SecondStone{first}` witness. The plan's earlier "same PositionKey" claim
was false (R1); the dedup below never relies on key identity for terminal
pairs.

**Change (search).** At the b = 1 child of a post-opening b = 2 AND node
reached via first placement d₁, with the parent nonterminal and d₁'s
singleton successor nonterminal: generate second placements d₂ with
ord(d₂) > ord(d₁) under a **strict total coordinate order frozen at the
b = 2 parent** (do NOT recompute `canonical_frame` per SecondStone child —
the frame includes the first-placement witness, `tss_solver.rs:812-844`),
plus exactly the exception class: d₂ ≤ d₁ that were NOT legal at turn
start (newly legalized by d₁). P3's premises exclude only
**singleton-terminal prefixes** — covered because singleton-win prefixes
never reach a Universal (`prove` strips own_win_now nodes, :384-395).
**Joint-second-win pairs** (both orders win on the second placement, same
winner and ply) are expressly *allowed* by P3 and may be
outcome-canonicalized (R2 defect-5 correction); what they can NOT be is
deduplicated by terminal `PositionKey`, whose `SecondStone{first}` witness
differs between the orders — the dedup compares outcomes, never terminal
keys. Attacker OR turns need nothing (existential).

**Change (verifier).** Accept an omitted d₂ < d₁ at such a node iff ALL of
(R1's condition list, adopted verbatim):
- the b = 2 parent occurrence is nonterminal FirstStone, both cells legal
  at the parent (replayed), both singleton successors nonterminal
  (replayed);
- the **mirror is state-bound**: the certificate contains, under the SAME
  parent occurrence, the d₂-first Universal with an explicit d₁ edge,
  independently verified (a global arena search for "some Universal with a
  d₁ edge" is insufficient);
- the mirror edge is materialized — a first move omitted by U1/U3 zones, or
  a mid-turn node collapsed to a λ¹ leaf, cannot serve as the mirror;
- commutation references participate in acyclicity, reachability, and
  memory accounting;
- circularity is excluded structurally: under the parent-frozen order,
  d₂ < d₁ implies the mirrored branch needs its d₁ > d₂ edge explicit, so
  that edge cannot itself be U5-omitted — well-founded by construction.

**Expected effect.** ≈2× on the pair level of spare AND turns — scoped
(R1) to turn-start-legal, nonterminal, materialized pair branches;
multiplicative with U1. The `PositionKey` grandchild dedup already handles
the subtrees; U5 removes the mid-turn enumeration overhead.

#### U6. Interior forced-move guard: default-on with a proof; λ¹ policy mask [S for game-theoretic outcome; wording per R1]

**Proof basis.** The U3 lemma (unconditional: every non-hitting reply at a
¬own_win_now ∧ mhs = b node is lost — D9's adaptive LOSS argument + T6's
count bound; R1 re-derived it independently and ruled no λ¹-to-game-theory
gap). T1 (kept children ⊆ r2, complete).

**Change.** `tss_interior_guard` (`tree.rs:1286-1327`, `:2114-2138`)
currently ships default-off as a "risky prune". The lemma upgrades its
claim to: **exact game-theoretic outcome preservation** — every dropped
child is a proven loss, so no non-losing move is ever removed. It is NOT
"value-exact for finite MCTS" (R1): dropping known-losing actions changes
visit counts, Q averages, and possibly moves-left preferences; and when
*all* moves lose, omitted moves may be policy-optimal under a loss-delay
objective. Conditions to enforce before default-on (R1 list): reachable,
nonterminal, post-opening states only; b ≤ 2; own-win precedence intact;
the guard's kept set is the **full engine-legal hitting universe** — no
crop, no widening omission, no count cap (at ¬own_win_now,
`tactical_cells` is exactly the hitting universe — keep it that way under
refactors). Run the shadow rung first per the runbook even with the proof
in hand (R1 phasing).

**Policy mask.** The same predicate feeds the Stage-2 policy-target lever
as a **"non-losing-support safe"** mask (R1 wording): at fully-forced
defender nodes, masking targets to the hitting universe provably never
excludes a non-losing move. It is not an exact target distribution claim —
loss-delay preferences among all-losing children are flattened; acceptable
for the value head, a recorded caveat for the policy head.

**Caveat to record.** The premise is verifier-grade only because `analyze`
is exhaustive for k ≤ 2 = B; if B ever changes, re-audit
(threats_shared.rs computes mhs ≤ 2 exactly).

#### ~~U7. OR-node ordering without child replay~~ — ALREADY IMPLEMENTED (R1)

Struck: review R1 established the branch already computes OR ordering from
WindowStore static features without child replay
(`tss_solver.rs:627-720`; correctness argument
`docs/TSS_SOLVER_PROOF.md:392-422` — ordering can never affect soundness,
only discovery-vs-Unknown under a cap; the profile's eager-ordering hotspot
was the pre-optimization baseline). Retained residue → U10: add a DEEP_WIN
ordering-regression telemetry bucket so future ordering changes are
A/B-able. Two of the draft's wording errors are withdrawn with it (a
"count-4→win" mislabel — at FirstStone a count-4 completion creates count 5
and a same-turn λ¹ finish, not an immediate placement win — and a backwards
acceptance inequality).

### Tier C — narrow / optional

#### U8. Trigger + regime detector [H — with R1's correctness-safety conditions]

The leaf trigger (`tss_deep_leaf`, `tree.rs:949-991`) gates on
`has_threats` + hash subsample. Add λ¹-informed gating: prioritize
mover-side mhs = b (the forced regime — where the *dismissal complement*
is provably cheap; R1 correction: mhs = b does NOT make the hitting
children themselves cheap to prove), deprioritize D ≥ 6 no-forcing-
structure roots; |hitting| + |𝒜| as a one-pass width predictor. R1
correction adopted: this **does** have a value-affecting path — the trigger
decides whether `tss_solve_verified` runs at all, so discovery, backups,
play, and targets can change. It is correctness-safe iff: trigger
quantities schedule budget and never mint values; skipped attempts take the
existing neural/Unknown fallback; every consumed result still passes
`hard_value_from_verified`; scheduling stays deterministic with existing
caps.

#### U9. ES-potential futility (Cor. 2 integer check) [S under the exact guard — R1 conditions]

**Proof basis.** ES Theorem 2 + Corollary 2
(`docs/proof_parts/ES_POTENTIAL.md:553-569`, `:907-924`), trust boundary
(`:997-1011`).

**Change.** Requires U4's semantic horizon in `SolveCaps` (the depth cap is
not a horizon). Exact guard (R1, adopted verbatim): at a nonterminal
universal node with

  `current_player != claimant && phase == FirstStone &&
   T − placements_made ≤ 4`

and Corollary 2's strict integer test over **all** claimant-touched,
opponent-free count-1…5 windows (a = n₁+3n₃+9n₅, b = n₂+3n₄; Φ < 1 ⇔
b ≤ 8 ∧ a² < 3(9−b)², wide arithmetic), return internal `None` — "no
claimant proof by this deadline". It must never mint Loss, draw, defender
win, or any unbounded-horizon result (the source forbids it explicitly).
Maintain Φ bins incrementally: ±18 window updates per placement (R1
suggestion). A 6-ply variant (safety through the following defender pair —
only the mover can complete a line) and Theorem 1/3/4 longer cutoffs are
possible future extensions once their closure/reserve premises can be
certified; out of scope now. Ship last; kill if < 1% of horizon-bounded
leaves fire.

#### U10. Adversarial fixtures + differential harness [T — gates redefined per R1]

Port the proof program's adversaries to Rust golden tests, with R1's
corrections to what each can actually assert:
- **G1 junction** (capped 4-arm + single-window pin): the fixture is
  *unrefuted at bounded depth*, not proven globally non-WIN — so the gate
  is a **matched-horizon differential** against `tss_reference.rs`, not
  "must never claim WIN". Also assert the junction cell is generated at
  the critical node.
- **G3 counterfork**: same matched-horizon contract; assert the fork cell
  is generated.
- **Verifier rejection tests are certificate mutations, not solver runs**
  (R1): a negative-control *solver* finding no proof demonstrates nothing
  about the gate. Supply explicitly malformed certificates and assert
  rejection: zero-edge AND node; terminal root; AND-node defender
  own_win_now; corrupt/nonlive witness family; **adaptive-LOSS resolution
  corruption** — declared_resolution ≠ leaf-ply + b + 2, and separately a
  resolution exceeding an externally supplied semantic horizon (R2: the
  earlier "resolution > T" mutation was tautological, since the verifier
  defines T as the max resolution label); **forged OR-COMPLETION leaf**
  (placement doesn't complete / named window doesn't contain the placement
  / completion ply ≠ path ply); **corrupted WIN witness** (count/phase
  mismatch, wrong resolution arithmetic); late-core-growth dismissal (a
  cell in final core dismissed at an ancestor); D ≥ 6 node with omissions
  in a D-alive virgin window; dropped (Z5) band cell ((Z5) corridor
  construction);
  U5: singleton-win pair, joint-second-win pair, newly-legal second cell,
  unfrozen-order mirror, absent mirror, circular mirror reference.
- **Cache composition test**: the U4 quick-fragment/slow-sibling
  counterexample as an executable case — import must be refused by the
  two-stamp rule.
- **Paired oracles**: U3 staple-by-theorem vs per-move staple on the full
  corpus (0 divergences required); zone solver vs `tss_reference.rs` on
  the random corpus at matched horizon. The differential is **one-sided**
  (R2): every hard WIN/LOSS the optimized solver claims must agree with
  the exhaustive reference at the same semantic horizon; the optimized
  solver returning Unknown where the reference finds a value is
  *legitimate* (restricted search trades completeness for speed) and is
  tracked as a yield metric, not a failure. 0 one-sided divergences is
  evidence, not the soundness gate; the mutation tests are the gate.
- DEEP_WIN ordering-regression bucket (from struck U7).

#### U11. True domination arms — P1/P2 in the verifier [S for P1/P2; sub-hitting item is [UNPROVEN]]

R1 missed-opportunity, recorded as a real backlog item: implement proof-doc
P1 (dead-cell dismissal with exact dead-mask check on all 18 incident
windows + a certificate-named searched substitute *a* that **either wins
immediately or** satisfies the frontier-inertness obligation
B₈(a) ⊆ Λ(P) — both theorem branches, R2 defect-9) and P2 (dead-spoke
interchangeable hitting cells) as verifier arms with certificate support.
P2 is otherwise entirely unused by this plan.

Separately, **[UNPROVEN]** (R2 defect-8 labelling): the sub-hitting
dispatch refinement suggested in R1 — at a post-opening defender AND node
with ¬own_win_now and mhs = b: at b = 1 only *common* hits of the threat
family can survive, and at b = 2 only first-hits extendable to a two-cover
need search, with hitting-set algebra stapling the rest. This is a
conjecture with stated premises, not a theorem of the proof doc; derive it
as a lemma and put it through the same hostile-review treatment before any
implementation.

---

## 3. Phasing (maps onto the TSS_RUNBOOK rung ladder; R1 phasing rulings incorporated)

| Phase | Contents | Gate |
|---|---|---|
| P0 | U4 clock plumbing (typed-leaf resolution semantics land WITH U2's schema, not before — R1); U10 fixture scaffolding | golden tests green |
| P1 | U3 (staple-by-theorem + no-complement-enumeration) | paired-oracle differential, 0 divergences; verify-time budget on threat-dense buckets |
| P2 | U2 (typed schema + full D9 verifier) then U1 (zone generator, consumable) | full certificate-mutation suite green; matched-horizon G1/G3 + reference differentials; UNKNOWN-rate at fixed node_cap on spare-node buckets |
| P3 | U4 cache two-stamp rule (zone-fragment promotion stays OFF until then); U5 (P3 pairs incl. verifier arm + its six fixtures) | cache composition test; pair telemetry |
| P4 | U6 (shadow rung first, then default-on + policy mask), U8, U9 | shadow counters; trigger yield; futility hit-rate |

Every phase keeps invariants 1–4; every flag ships default-off and rides
the shadow → consume ladder; U1 omissions are non-consumable before U2
(hard ordering, R1).

## 4. Measurement plan

- Re-run `TSS_SOLVER_PROFILE` buckets per phase: nodes/solve, solves/s,
  verify µs/node, UNKNOWN rate at node_cap ∈ {2k, 10k, 50k}, proof-depth
  distribution, SharedProofCache hit-rate + bytes (re-measure admission
  constants after U2's fatter certificates).
- New buckets: SPARE_WIN (k < B at root turn), DEEP_SPARE (spare node ≥ 4
  plies deep) — cap-out concentration today, acceptance metric after P2;
  DEEP_WIN ordering-regression (from struck U7).
- Training-facing: consumption counters per mode tier;
  `deep_verify_failed` must stay 0 in steady state (any nonzero = search
  bug, by construction never a value error — horizon guesses are caught by
  the solver-side preflight, not the minting verifier); `horizon_retry`
  rate (U4 preflight-diagnosed horizon failures — expected nonzero, cheap).

## 5. Risks, non-goals, deferred refinements

- **G2 / D ≥ 6 honesty**: default fallback is full legal. The
  theorem-permitted refinement (dismiss cells in no D-alive window AND
  outside hitting/core/band even at D ≥ 6 — proof doc lines 262–270) is
  real but deferred: it requires enumerating D-alive-window *absence* over
  virgin windows, which the touched-window store cannot do; revisit only
  with a dedicated legal-complement pass and its own fixtures.
- **Sharpened budget (F + H_W) and band shrinking are OPEN** (proof doc
  §12): U1/U2 use exact D_N and the full band. Any sharpening is a new
  proof obligation.
- **Sealed subcertificates** (branch-local deadlines solving cache
  composition more elegantly than the two-stamp rule, and enabling tighter
  per-branch zones) — designed future work, after P3.
- **Certificate size**: zone data + typed leaves grow certificates;
  re-measure `SharedProofCache` admission before re-enabling promotion.
- **No defender-count caps, ever** (invariant 2).
- **Lean formalization** of the U2 checker remains the long-term hardening
  path (proof doc §12.4); the U10 mutation suite is the interim substitute.

## 6. Adversarial review log

- **R1 (Codex ultra, 2026-07-13) — full-plan hostile review.** Verdicts:
  U3, U7 SOUND (U7 additionally *already implemented* on the branch —
  struck); U2 FLAWED-fatal (incomplete D9 checklist; zero-edge exploit) —
  repaired with the full obligation list; U1 FLAWED (missing producer-side
  monotone closure loop; the "P1" trim was not P1 — renamed stale-area
  filter, downgraded [H]; Opening exclusion added); U4 FLAWED (cache
  compositionality counterexample — two-stamp import rule added; wrong
  monotonicity citation fixed; retry loop bounded + diagnosis-gated);
  U5/U6/U8/U9/U10 NEEDS-CONDITION — all conditions adopted verbatim
  (§2 items). Missed opportunities adopted: U3 no-complement-enumeration,
  minimal-T4-zone verification, U11 (P1/P2 arms + sub-hitting dispatch
  algebra), ES incremental maintenance + 6-ply variant (deferred), D ≥ 6
  refinement (deferred), sealed subcertificates (deferred). Nitpicks
  incorporated (Z5 is a geometric scan; WindowKey identities + D6
  remapping; verifier establishes T; scoped U5 speedup claim).
- **R2 (Codex, 2026-07-13) — confirmation pass, ruled FAIL with nine
  residual defects; all repaired in place.** APPLIED-CORRECTLY: U1, U3,
  U6, U7, U8, U9; APPLIED-BUT: U2 (fatal — OR-COMPLETION leaves were in
  the schema but absent from the verifier obligations, admitting a forged
  maximal leaf; now obligation 6, with exact derived resolution semantics
  for all leaf types in obligations 7–8), U4 (composite two-stamp
  aggregation + final-T recheck added; preflight/fatal-counter conflict
  resolved via the solver-side `horizon_retry` preflight), U5
  (joint-second-win pairs correctly stated as outcome-canonicalizable but
  not terminal-PositionKey-deduplicable), U10 (tautological resolution>T
  mutation replaced with exact-arithmetic corruptions; OR-COMPLETION and
  WIN-witness mutations added; reference differential made explicitly
  one-sided), U11 (sub-hitting refinement labelled [UNPROVEN] with
  premises; P1's immediate-winning-substitute branch restored). Phasing
  ruled correctly represented; no R1 verdict silently dropped.
- **R3 (Codex, 2026-07-13) — narrow confirmation on the nine R2 repairs:
  overall PASS.** All nine ruled APPLIED-CORRECTLY (OR-COMPLETION
  obligation; exact leaf-resolution semantics; composite two-stamp rule;
  preflight/fatal-counter separation; joint-second-win canonicalization;
  certificate mutations; one-sided differential; [UNPROVEN] labelling;
  P1 substitute branch), plus obligation numbering (1–12) and review-log
  consistency confirmed. The document is final.

---

## 7. Round-5/6 ranked-zone extension (U12–U18)

**Provenance.** Round-5/6 revision, 2026-07-14: normative mathematics and
adoption history are in `docs/PROOF_TSS_DEFENDER_ZONES.md` §13; repair
rationale is in `docs/_T3_TIGHTENINGS_REVIEW_ROUND1.md`; confirmation is in
`docs/_T3PLUS_ROUND6_CONFIRMATION.md`. This section controls the three
amendment lines above where the earlier U1/U2/U4 plan text differs.

### Tier A — revised verifier core

#### U12. Ranked zone generator and verifier zone [S — revised T3/T4, round-6 confirmed]

**Proof basis.** D10–D16, L9′, L11–L12, T3/T4/T7. The ordinary certified
zone at an internal AND node is now exactly

  `Z_dir ∪ Z_seed ∪ Z_touch ∪ Z_virgin`,

under the live verifier labels (Z2), (Z4), and (Z5′). `S(N)` must still be
independently nonempty; if the union is empty, one arbitrary legal fallback
edge is mandatory. The former Z1 hitting requirement is gone. Current
hitting cells may be generated and ordered first, but are a search heuristic
and never a reason for verifier rejection outside the separate U15 kernel.

**Where in code.** Generator: `tss_solver.rs::prove_universal` and its
WindowStore candidate helpers. Certificate data and bottom-up role/exposure
pass: `tss_verify.rs::{CertNode, TssCertificate}`. Authority check:
`tss_verify.rs::verify_universal`. Extend `d6_remap_certificate` for every
new coordinate-bearing role/window label.

**Change (builder).** Replace U1's `hitting ∪ 𝒜 ∪ ℬ ∪ core ∪ band`
formula with the four ranked components:

1. `Z_dir = Prot(N) ∩ Legal(P_N)`;
2. `Z_seed`: for every protected nonlegal, nonstone cell `y` with a live
   role rank `r_N(y) ≥ 1`, legal seeds within radius `8(r_N(y)−1)`;
3. `Z_touch`: every empty of a D-alive window already containing a D-stone
   when `cnt_D(W) + E_N^D(W) ≥ 6`;
4. `Z_virgin`: legal cells within distance `8(E_N^D(W)−6)` of an
   all-empty window with `E_N^D(W) ≥ 6`.

The first implementation uses U13's uniform `B(N)` fallback for all live
role ranks and window exposures. Exact clocks are U16. The old `D ≥ 6 ⇒
full legal` verifier fallback is superseded: the finite `Z_virgin` test is
performed by inverting from each legal candidate, at every admissible local
budget. Search may retain hitting, `𝒜`, and `r3` as over-generation and
ordering terms; none belongs to `R_cert`.

**Change (U2 verifier checklist, exact row replacements).**

- Delete the Z1 row. Retain only the independent `S(N) ≠ ∅` fallback.
- Replace the old horizon-scaled Prot band by (Z5′): per live role, use
  radius `8(r−1)`, take the maximum rank when several roles share a cell,
  form bands only at internal AND nodes with `r ≥ 1`, and discharge a role
  only after its deadline check. Uniform `r = B(N)` is valid.
- Replace full-window `core`/Prot construction with D10's compressed roles:
  every future designated certificate attacker placement (including
  OR-COMPLETION), plus `(leaf, witness window, empty cell)` roles for every
  WIN/LOSS witness empty at leaf entry. Full witness-window cells are not
  protected merely because they lie in a named window.
- Split the former completion/core row into `Z_dir`, `Z_touch`, and
  `Z_virgin`, re-derived from the replayed node position and checked window
  exposure. Keep (Z4) for every designated attacker placement and every
  leaf-contract continuation.
- Remove the old `D ≥ 6` full-legal checklist row; high-budget virgin
  danger is checked by `Z_virgin`, while touched danger is checked by
  `Z_touch`.

Per node the verifier enumerates all reachable live roles, validates their
deadlines and takes cellwise maximum ranks; enumerates D-alive touched
windows and their exposures; and, for each legal candidate, enumerates
nearby all-empty windows admitted by the exposure radius. It then recomputes
all four components rather than trusting serialized zone membership.

**Testing traps.** Mutations must cover an empty ranked union without a
fallback child; a dropped direct role; a shared cell whose shorter role is
used instead of the maximum; an `r = 0` negative-radius band; a missing
OR-COMPLETION role; a touched completion cell; the sharp virgin `E = 7`,
radius-8 seed; and a certificate that omits a current hitting cell but is
otherwise valid (it must be accepted). Retain late-descendant-role and
(Z4) continuation fixtures. Differential tests remain one-sided at matched
resolution: restricted-search Unknown is not a failure.

#### U13. Local budget labelling [S core; cache representation needs-derivation]

**Proof basis.** D14 and L11. Every certificate node carries a nonnegative
defender-placement budget `B(N)`. Exact labels satisfy `B = 0` at
OR-COMPLETION/WIN, `B = b` at LOSS, copy the child at OR, and use
`1 + max(child)` at AND. The verifier may accept conservative labels if it
checks `B(LOSS) ≥ b`, `B(parent) ≥ B(child)` on OR edges, and
`B(parent) ≥ 1 + B(child)` on AND edges. These inequalities cover every
selected path and LOSS remainder and imply hereditary decrease.

**Where in code.** Store labels on the `CertNode`/zone schema in
`tss_verify.rs`; compute exact or conservative labels during
`tss_solver.rs::compact_certificate`; check them in the verifier's
post-order certificate pass. Carry the reuse metadata through
`tss_solver.rs::{BoundedTt, SharedProofCache}` and its fragment-import path.

**Change.** Certificates carry and the verifier checks local `B` labels;
zone generation uses those labels rather than deriving one flattened global
`D_N` from the slowest sibling's `T`. U4's R1 counterexample does not show
that local budgets are unsound: it refutes *unprincipled* local fragment
deadlines whose omitted cells were later judged under a larger flattened
global horizon. D14 labels plus L11 edge inequalities are the sound
replacement because a sibling cannot increase a verified descendant
label.

**Cache interaction.** The two cache stamps may now be understood as
`resolution_T` plus a verified local-budget/zone-build admissibility stamp.
At a reuse site, recheck every connecting D14 inequality and require the
budget needed by each reused zone component to be no larger than the budget
under which that component was built. The comparison direction is
mandatory: `required_B_at_reuse ≤ zone_build_B`; reversing it recreates the
omitted-fragment defect. Composite fragments retain the maximum resolution
stamp and the most restrictive zone-build admissibility across their
components.

L11 directly licenses the hereditary budget reasoning, but not a particular
scalar stamp aggregation, coordinate remap, fragment sealing scheme, or
incremental-import algorithm. Replacing U4's final assembled-certificate
preflight solely with an online cache lookup is therefore
**[needs-derivation]**. Until that derivation and mutation suite land, final
assembly must recheck all D14 inequalities, all zone coverage, and the stamp
direction; a failed check is Unknown.

**Testing traps.** Re-run U4's quick-fragment/slow-sibling case: the locally
labelled fragment is accepted only when all connecting inequalities and
zone-build comparisons hold. Add reverse-inequality, omitted LOSS-remainder,
AND-edge off-by-one, conservative-overlabel, and composite-minimum-stamp
mutations. An exact local label and a conservative larger label must verify
to the same value, with only zone size allowed to differ.

#### U14. Sparse LOSS witnesses [S — L13 constructive bound]

**Proof basis.** L13. A rank-two threat-empty family with transversal number
greater than one has a witnessing subfamily of at most three sets; one with
transversal number greater than two has one of at most six. The bounds count
windows in the named family and `b` counts defender placements.

**Where in code.** Add the constructive selector to LOSS-leaf creation in
`tss_solver.rs`; enforce caps, window replay, and the named-family mhs check
in `tss_verify.rs`'s typed LOSS-leaf arm. Witness keys continue through
`compact_certificate`, cache import, and `d6_remap_certificate`.

**Change.** When constructing a LOSS leaf, choose L13's constructive
subfamily and emit at most three witness windows for `b = 1` or six for
`b = 2`. The verifier rejects an over-cap family, replays every named
window, derives its one- or two-cell empty set, and exhaustively re-derives
`τ(named family) > b`; it never trusts the builder's choice and never uses
unnamed threats to rescue a bad witness.

**Payoff.** LOSS leaves and their D10 witness-empty roles become bounded.
That cuts certificate bytes and, downstream of U12, bounds the leaf-derived
contribution to `Prot`, `Z_dir`, and seed-band work. It does not bound future
designated attacker-move roles elsewhere in the subtree.

**Testing traps.** Include sharp triangle (`b = 1`) and six-edge `K₄`
(`b = 2`) fixtures, duplicate window keys, a family valid only when an
unnamed threat is included, an oversized but otherwise valid family, and a
family whose replayed empty set changed. Compute τ on the named family after
deduplication; count placements, not turns.

#### U15. Kernel T6 at forced nodes [S — T6, round-6 confirmed]

**Proof basis.** T6's extendable-hit kernel. For current attacker-threat
empty-set family `ℱ_N`, define

  `K_b(N) = { d ∈ Legal(P_N) : τ(ℱ_N ∖ d) ≤ b−1 }`.

**Where in code.** Build `K_b` beside `implicit_dispatch` in
`tss_solver.rs::prove_universal`, using the exact `b ≤ 2` threat/mhs
machinery in `threats_shared.rs`. Re-derive the boundary and exact kernel in
`tss_verify.rs::verify_universal`; keep the existing U3 dispatch arm as the
non-kernel path.

A kernel region starts only where real and certificate positions are equal.
At every governed internal AND node the verifier re-derives
`mhs(P_N) ≤ b`, explicitly checks `¬own_win_now`, and requires the searched
set to be exactly `K_b`. If `mhs < b`, `K_b` is all legal moves and performs
no pruning; if `mhs = b`, it weakly refines the hitting set and may be
strictly smaller. No original core/Prot term is required.

**Hard guard.** If `mhs > b`, `K_b` is empty. Such a node must be a valid
LOSS leaf or retain/handoff to its existing nonempty sound searched subtree;
it cannot remain kernel-governed. A later kernel region needs a fresh
equal-position entry.

**Interaction with U3.** Inside a valid kernel region, T6 replaces U3's
per-omitted-move staple obligation: the verifier re-derives the current
threat family, `mhs`, and exact `K_b` once at the node, and the residual
transversal argument dismisses every legal move outside the kernel without
child replay or original-core obligations. Outside that region, including
the `mhs > b` handoff subtree, U3's existing dispatch theorem and its
premises remain unchanged.

**Testing traps.** Exercise `mhs < b` (kernel must equal full Legal),
`mhs = b` with equality and strict reduction, and `mhs > b` (kernel empty,
so internal-kernel form must reject). Mutate `¬own_win_now`, enter a region
after a non-equal substitution, omit one true kernel member, add a
non-kernel member, and attempt a core-based rescue of an invalid kernel.
Pair node-level kernel verification against the old U3 per-move staple on
the exhaustive `b ≤ 2` corpus.

### Tier C — backlog and note

#### U16 (backlog). Exact ranks and exposures [S when D15/D16 checks pass]

**Where in code.** Extend the same `CertNode` labels and verifier post-order
pass introduced by U12/U13; compute optional exact labels during
`tss_solver.rs::compact_certificate`. The generator consumes them only in
`prove_universal`, behind a dedicated flag.

The default U12 rollout uses uniform `B(N)` for every live role and relevant
window. A later flag may compute exact per-role `r_N(ρ)` by the D15
recurrence, collapse roles sharing cell `y` with `max`, and compute exact
per-window `E_N^D(W)` by D16, stopping when the attacker first enters that
specific window.

The cost is bottom-up per-node bookkeeping over every reachable live role
and every relevant overlapping window, plus larger certificate labels and
more verifier work. The payoff is smaller `8(r−1)` seed bands, fewer
`cnt_D + E^D ≥ 6` touched guards, and smaller `8(E^D−6)` virgin guards.
Recommended default: ship uniform `B` first; put exact clocks behind a flag
and retain uniform-B recomputation as a differential oracle.

**Testing traps.** Multiple roles on one cell require the maximum live rank;
LOSS roles end at leaf entry; OR-COMPLETION is a role; `r = 0` forms no
band; attacker entry stops only that window's exposure; overlapping windows
keep separate clocks; every AND edge contributes one and every ordinary OR
edge zero.

#### U17 (backlog). Branch-indexed substitution envelopes [S theorem; highest verifier complexity]

**Proof basis.** D17/T9. A dismissal may name a searched substitute child
and validate obligations, ranks, and exposures over all descendants of that
child, with an independently nonempty fallback `F(N)`. This can shrink a
whole-subtree union to the selected branch envelope, but it adds per-dismissal
annotations, nested-envelope reasoning, A2/A3-specific transition rules,
and the largest verifier surface in this catalog.

**Where in code.** Add dismissal-to-substitute annotations to the Universal
node schema in `tss_verify.rs`; generate them in
`tss_solver.rs::prove_universal`; validate transition-inclusive envelopes in
`tss_verify.rs::verify_universal`. Include annotations in compaction, D6
remapping, reachability/acyclicity, and certificate memory accounting.

Both transition-inclusive `+1` charges are mandatory. The C3
counterexample has child rank one and a dismissed cell distance 8 from a
future ghost-illegal obligation: radius `8(r−1) = 0` admits it, while the
parent-inclusive radius `8r = 8` rejects it. The C2 counterexample starts a
D-alive window at count two with `B(child) = 3`: child-only `2+3 = 5`
misses the real current fill, while `2+1+3 = 6` catches completion. Use
`Ê = 1 + E_child` for exact exposures for the same reason.

Gate U17 default-off after U12/U13 stabilize. Build it only if profiles show
that whole-subtree `Prot`/completion unions dominate verified zone width and
that repeated candidate substitutes usually select substantially smaller
descendant envelopes. Required evidence is a shadow verifier with zero
disagreement against ordinary U12 coverage, both `+1` mutation fixtures,
non-circular fallback tests, bounded certificate growth, and a material
fixed-cap yield or verification-time win. The non-circular fallback is
`F(N) ⊆ S(N)` chosen independently; `S(N)` may not be defined as only the
replies lacking a safe substitute.

#### U18 (note). Certificate DAGs [S — D18/T10 unfolding]

**Where in code.** The sharing boundary is the `TssCertificate` arena and
its acyclicity/reachability checks in `tss_verify.rs`; emission and reuse
touch `tss_solver.rs::{compact_certificate, BoundedTt, SharedProofCache}`.
`d6_remap_certificate` must preserve shared identities rather than clone
inconsistently labelled copies.

A finite acyclic certificate may share a node only when every incoming path
agrees on its exact position, mover/budget, action or searched-successor map,
leaf witness data, and path clock. Local budget/rank/exposure inequalities
must hold on every outgoing edge; obligations are unions over reachable
descendants, while coupling histories remain path-local. Verification may
unfold conceptually as in T10 rather than materializing the tree.

DAG sharing lets the transposition/cache layer store and verify one repeated
subproof instead of duplicating it, reducing certificate bytes and improving
cache admission. Its cost is exact-label and consistent-clock checking at
every merge, plus reachable-descendant unions that cannot depend on which
incoming path happened to arrive. The current tree-shaped builder loses no
soundness and no proof expressiveness today: it only gives up this sharing,
so repeated subtrees cost memory and verification work. DAG emission is an
optimization note, not a prerequisite for U12–U17.

**Testing traps.** Reject cycles, unreachable shared nodes, unequal clocks,
same position with different witness/action labels, and a merge whose local
edge inequalities hold on one parent path but not another. Compare the DAG
and a fully unfolded tree for identical verified value and resolution.

### Recommended extension phasing

Treat the already-implemented P0–P3 flags as the compatibility baseline; do
not reinterpret their stored zone certificates in place. Land U14 first
(small bounded schema/verifier change that immediately shrinks LOSS data),
then U13 (the D14/L11 budget foundation and cache mutations). Land U12 next
with uniform `B`, replacing U1/U2 consumption only after the complete ranked
mutation suite passes. Land U15 after that boundary is stable so kernel and
ordinary-zone nodes have an explicit, testable handoff. Keep U16 and U17
default-off backlog flags, and treat U18 as an independent representation
optimization. Every new consuming flag repeats the shadow → verify → consume
ladder; cached pre-extension zone certificates are not promoted across the
new verifier contract without re-verification.
