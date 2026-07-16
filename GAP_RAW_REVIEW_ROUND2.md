# GAP-RAW Round-2 Hostile Review

**Reviewed worktree:** `hunt-gap-raw`  
**Requested review commit:** `74026b7c`  
**Document under review:** `GAP_RAW_PROOF_ROUND2.md`  
**Review standard:** statement fidelity; any material PROVEN/VERIFIED overclaim is rejection-grade.  
**Overall verdict:** **ACCEPT-WITH-EDITS**

## 1. Executive verdict

Round 2 survives the rejection-grade attacks. The document now binds one strategy's actual sequential reply, withdraws the false renewal/pricing/K3 assembly claims, proves the complete one-axis and K2 geometry it newly labels PROVEN, and scopes L2/L3 to universes that the code really checks. I found **no PROVEN or VERIFIED claim that needs downgrading** and no silent downstream use of the withdrawn global L3 table.

The disposition is nevertheless **ACCEPT-WITH-EDITS**, not unconditional acceptance. Two proof-text defects are real:

1. R7.4₂ is stated only at an Attacker handoff but is invoked at positions in different phases in K2 and K3. Its proof is phase-free and proves the needed stronger fact, so this is a scope repair, not a counterexample.
2. K2's final sentence says its chosen spare belongs to an Attacker-alive window "in every case." That is false in the no-critical-core filler branch. A concrete counterexample is given below. L1.2's max-`q` filler argument independently supplies legality and distinctness, so K2 itself remains valid.

There are also provenance and withdrawal-ledger edits: the artifact is reviewed at `74026b7c`, while §13 records only its parent/input `159c75f4` and says no commit was created; and the old A′ should be named explicitly alongside old Theorem A in §9.

## 2. Per-claim verdict table

This follows the 30 PROVEN/VERIFIED rows in the authoritative round-2 status table. Grouped rows remain grouped so the tally is comparable to the document's own 19/11 inventory.

| Claim | Round-2 label | Review verdict | Basis |
|---|---|---|---|
| L1.1 Attacker-`FirstStone` completion criterion | PROVEN | **CONFIRMED** | Both directions respect phase, nonterminality, and sequential legality. |
| L1.2 Defender-epoch servicing criterion | PROVEN | **CONFIRMED** | Exact hitting-set equivalence; finite/nonempty max-`q` fillers are valid. |
| Theorems A₂/A₂′ actual-action reformulations | PROVEN | **CONFIRMED** | One fixed S, all attacker continuations, every S-reached epoch, S's actual ordered reply. |
| L4a one-Attacker pencil bound | PROVEN | **CONFIRMED** | Immediate global stone-count argument. |
| L4b one-turn maturation, six named roots | VERIFIED | **CONFIRMED** | Scope matches the inherited exhaustive table; straight-four service is repaired separately. |
| L5a per-root `ΔΦ` ceilings | VERIFIED | **CONFIRMED** | Exact six-root scope and values match the inherited enumeration. |
| L5b general `ΔΦ` bound | PROVEN | **CONFIRMED** | Promotion plus at-most-18 virgin entries, iterated sequentially. |
| L6₂ touched-window legality and kill multiplicity | PROVEN | **CONFIRMED** | Attacker touch gives a stone within axis distance five; one D cell kills all incident labels. |
| L7.1 membership and L7.2 decomposition | PROVEN | **CONFIRMED** | Count cases and asymmetric trigger exchange are complete. |
| L7.3₂ per-cluster defusal | PROVEN | **CONFIRMED** | Claims only the one trigger cluster; whole-witness claim is withdrawn. |
| L7.4 original mass floors | PROVEN | **CONFIRMED** | Distinct-label count and pre-count weights are correct. |
| R7.4₂ / `β≈0.4147` floor | PROVEN | **CONFIRMED** | Complete one-axis proof and phase-free over-approximate regression; hypothesis wording needs widening. |
| L2₂ global seven-stone fork floor | VERIFIED | **CONFIRMED** | Same-axis anchor reduction is complete; all 902 nonterminal configurations include singleton residuals and hard-assert `τ≤2`. |
| L3 four columns, edge-connected `n=4..12` | VERIFIED | **CONFIRMED** | Every free polyhex is generated; A000228 counts and all advertised vectors are asserted. |
| L3 four columns, unrestricted `[0,6]²`, `n=4..6` | VERIFIED | **CONFIRMED** | Every subset is enumerated; all four columns and variable-residual fork flag are asserted. |
| L7.5 five-prestone witness floor | VERIFIED | **CONFIRMED** | Sound deduction from L2₂; removing up to two triggers leaves at least five. |
| Theorem C₂ lower horizon | VERIFIED | **CONFIRMED** | Correct `t*≥0` indexing and no-attainability wording. |
| Normative-boundary Theorem 2 | PROVEN | **CONFIRMED** | Source proof covers five placements; sharpness is now limited to its specified strategy/filler. |
| L8.1₂ direct clean-escape zero injection | PROVEN | **CONFIRMED** | Thirty-six distinct count-one births contribute zero to Θ₂. |
| L8.2₂ local Θ₂ update and adjacent benchmark | PROVEN | **CONFIRMED** | Exact promotion identity and five-window adjacent-pair calculation. |
| Same-axis L8.3.2 counterexample | VERIFIED | **CONFIRMED** | Exact pre-counts/residuals and full-family `τ≥3` are asserted. |
| Mixed cross-axis L8.3.2 counterexample | VERIFIED | **CONFIRMED** | Omitted mixed branch is exact and hard-gated. |
| L8.3.1 two-trigger locality and L8.3.3 per-cell killing | PROVEN | **CONFIRMED** | These are precisely the retained local/per-cluster statements. |
| K1, every two-cell cover unripe | PROVEN | **CONFIRMED** | Label-disjoint `2/3+1/3` contradiction works for every cover. |
| K2, any one-cell cover has a good spare | PROVEN | **CONFIRMED** | Critical-core completeness and every inventory case close; proof wording needs the two edits above. |
| K3 matching-number-at-most-two mass statement | PROVEN | **CONFIRMED** | Three label-disjoint cores cost `3β>1`. |
| Exact path/cycle critical-core cases | PROVEN | **CONFIRMED** | Shared-label actions cover paths/cycles; C5 mass double count is sound; add trivial P1/P2 prose. |
| Theorem D₂, `J⇒GAP-RAW` | PROVEN | **CONFIRMED** | J.2 is exactly `Service(S,P₀)`; A₂ applies directly. |
| Three stored R1b break-line `τ` audits | VERIFIED | **CONFIRMED** | Every epoch is measured; all three exact ply-56 `τ=3` records are asserted. |
| Six-root no-forced-pileup/six through six plies | VERIFIED | **CONFIRMED** | Inherited completed minimax is scoped to exactly six roots and already used variable count-four/count-five residuals. |

**Claim tally:** 30 **CONFIRMED**, 0 **BROKEN-with-refutation**. By source label: 19 PROVEN rows confirmed and 11 VERIFIED rows confirmed. The refuted K2 sentence is not a separately labeled claim and is non-load-bearing; it is nevertheless a required proof edit.

## 3. Findings ordered by severity

### Medium — R7.4₂ is invoked outside its literal phase hypothesis

R7.4₂ is stated at `GAP_RAW_PROOF_ROUND2.md:616` as applying "at a handoff," which §1.1 defines as a nonterminal Attacker-`FirstStone` position after Defender's full ordered reply. K2 invokes it at `R=P+D@x` (`:667`, `:693`), where Defender is at `SecondStone`; K3 invokes it at the Defender epoch P (`:729`, `:733`). Those are not handoffs under the document's definitions.

This does not refute the mathematical content. The proof at `:617-652` uses only the board, `I=∅`, and an abstract one-/two-cell trigger set; it deliberately drops trigger legality and never uses the side or placement phase. The Rust regression likewise enumerates the larger phase-free line universe. Restating R7.4₂ for any finite nonterminal board position with `I=∅` validates both applications verbatim. Until that sentence is widened, however, the named-lemma citations are formally out of scope.

### Medium — K2's universal touched-window justification is false

The last K2 paragraph says, at `GAP_RAW_PROOF_ROUND2.md:719-721`, that "in every case" the selected `y` belongs to a touched alive window, is legal by L6₂, and differs from `x` because its core labels survived `D@x`. That is true in the critical-core cases but false in the no-core branch at `:693-696`, where `y` is an arbitrary max-`q` filler supplied by L1.2.

Concrete refutation of that sentence: at a Defender-`FirstStone` epoch put

```text
A = {(0,0),(1,0),(2,0),(3,0)}
D = {(-1,0),(100,0)}.
```

The only surviving count-at-least-two Q-axis labels start at 0, 1, and 2, with counts 4, 3, and 2, so

`Θ₂ = 1/3 + 1/(3√3) + 1/9 < 1`.

The imminent family consists only of the start-0 window with residual `{(4,0),(5,0)}`. Choose its one-cell cover `x=(4,0)`. This kills every surviving count-two/count-three label, so the critical-core universe is empty. The max-`q` filler is `y=(101,0)`, legal because it is adjacent to `D@(100,0)`, but it belongs to no Attacker-alive window. Thus the quoted universal statement is false.

K2's conclusion is unaffected: in the no-core branch, L1.2 already proves that the max-`q` filler is empty, legal, and distinct from `x`; in every core branch, L6₂ and core survival give the stated touched-window proof. The repair is to split the final justification by branch.

### Low — exact path proof omits the trivial one-/two-vertex cases

The named path/cycle proof at `GAP_RAW_PROOF_ROUND2.md:757-782` explicitly treats paths on 3, 4, and at least 5 vertices, plus cycles. If "exact path" includes one- and two-vertex paths, those cases are unstated. They close immediately: kill any label of a single minimal core, or kill one shared label of the two adjacent cores. This is a completeness edit, not a status defect.

### Low — provenance and withdrawal-ledger wording

- The reviewed artifact is commit `74026b7c` with parent `159c75f4`; §13 calls only the parent the input commit and then says "No commit was created in this repair round." That sentence is false of the committed artifact now under review. Record both base/input and output/review commits, or say that the authoring pass itself made no commit and the artifact was committed afterward.
- Round 1's authoritative inventory grouped "Theorem A / A′." Round 2's withdrawal row at `GAP_RAW_PROOF_ROUND2.md:1029` names only old Theorem A, although it says A₂/A₂′ replace it. Name old A′ explicitly for literal ledger completeness.
- §8.5's observation that earlier three-/four-window rows sometimes had `τ=1` or `2` is visible in the trace but is not a dedicated hard assertion. The load-bearing ply-56 records are hard-gated, so this is evidence-prose hardening only.

### No high-severity finding

I found no false theorem label, no residual per-epoch existential in A₂/A₂′, no circularity in D₂/J, no incomplete L2 normalization, and no relabeling of restricted L3 data as global.

## 4. Round-1 repair compliance

| # | Review judgment | Round-2 execution |
|---:|---|---|
| 1 | **EXECUTED** | Finite, nonempty, nonterminal blanket positions, side/phase, strict thresholds, and nonempty imminent residuals are explicit. |
| 2 | **EXECUTED** | A₂/A₂′ quantify one strategy's actual sequential service action on every reached epoch. |
| 3 | **EXECUTED** | O1′ is withdrawn; the exact three-gadget regression is present; successor renewal is S-reachable and retains `n₁/9`. |
| 4 | **EXECUTED** | W3′+O1′ is replaced by one strategy obligation J. D₂ uses only that one S. |
| 5 | **EXECUTED** | Both L8.3.2 counterexamples, the mixed-branch split, and a pre-trigger charge interval are explicit. |
| 6 | **EXECUTED BY HONEST SHRINKAGE** | The requested general K3 closure was not proved, but the false pairwise reduction/count-three-pool language is withdrawn, exact path/cycle cases are proved, and the remaining matching-number-two problem is OPEN. This repairs statement fidelity without pretending to solve K3. |
| 7 | **EXECUTED** | L7.3₂ is per-cluster; two heavy halves and the global suppression set-cover are explicit. |
| 8 | **EXECUTED** | L2 has a complete normalized universe. L3 has hard assertions only in the two exact sub-universes and global L3 is OPEN. |
| 9 | **EXECUTED** | Straight-four residuals and cover are exact; L4/L5a are restricted to named roots. |
| 10 | **EXECUTED** | Variable-residual `τ` is computed at every stored epoch and the three claimed break records are hard-asserted. |
| 11 | **PARTIAL — EDIT REQUIRED** | Evidence prose, Theorem 2 sharpness, and direct clean-escape scope are narrowed correctly. Final output/review provenance is not: §13 names only the parent/input and says no commit was created, while the artifact is commit `74026b7c`. |
| 12 | **EXECUTED** | C₂ is nonterminal, uses `t*` from zero, says only "no earlier than," and claims no attainability. |

Thus every repair was substantively addressed, and the one unsolved mathematical request (#6) was handled by withdrawal/OPEN rather than relabeling. Eleven are complete as written; #11 needs the provenance edit above.

## 5. Withdrawal-ledger completeness

The round-1 inventory was diffed against both §9 and the authoritative round-2 table.

- Retained claims—L5b, L7.1/L7.2, the original L7.4 floor, and K1—have explicit round-2 statuses.
- Repaired claims—L1.1/L1.2, A/A′, L4/L5a scope, L6, L7.3, L2/L7.5/C, R7.4, K2, and the R1b traces—are restated with their corrected premises or evidence.
- False load-bearing claims—O1′, L8.3.2, price-based O2′, the K3 pairwise/count-three-pool reduction, W3′, Corollary B′, and old Theorem D—are expressly withdrawn or left OPEN.
- Global L3, universal evidence-root prose, broad Cor-2 neutralization, universal `5/9` minting, broad Theorem 2 sharpness, and attainability-flavored C wording are explicitly withdrawn/narrowed.

No withdrawn round-1 proposition is consumed downstream at its old strength. The only ledger-completeness defect is naming: the row at round-2 line 1029 should say old **Theorem A/A′**, not only old Theorem A. The replacement prose makes the intended withdrawal clear, so this is not a hidden reuse.

## 6. Named attack surfaces A–I

| Surface | Judgment |
|---|---|
| **A. A₂/A₂′ quantifiers** | **SURVIVES.** `Service(S,P₀)` has `∃ one S ∀ continuations ∀ reached epochs`, and names S's actual sequential pair. A₂′ retains that same S. No per-epoch action existential remains. |
| **B. D₂** | **SURVIVES.** Clause 2 alone is literally `Service(S,P₀)` and A₂ closes the implication. Clauses 1/3/4 are redundant as stated. `Hist` is defined before J and does not depend on account renewal. |
| **C. J well-posedness** | **SURVIVES.** `B₂=Θ₂`; equation (9) is the exact Defender kill / Attacker promotion identity including `n₁(c)/9`. J.2 leaves `I=∅`, so every legal two-stone attacker response is nonterminal and the sequential updates are total. §4.2 is included if S allows it; the document does not exclude it circularly. |
| **D. K2 closure** | **SURVIVES WITH PROOF EDITS.** The relevant-trigger over-approximation, minimal-core implication, and inventories are sound. Widen R7.4₂'s phase scope and split the no-core filler justification. |
| **E. One-axis lemma / floor** | **SURVIVES.** Noncollinear/distant triggers yield no low label; internal empties give a common residual; `d=3` has the explicit two-cover; `d=1,2` are forest/matching cases. The machine universe is a safe complete over-approximation. |
| **F. L2 re-verification** | **SURVIVES.** The proof forces all labels onto the anchor line; the code includes disconnected co-window configurations, count-five singletons, nonterminal filtering, and a hard assertion for each of 902 cases. No weaker relabeling was found. |
| **G. L3 relabeling** | **SURVIVES.** Edge-connected `n≤12` and bounded unrestricted `n≤6` are kept separate; global maxima remain OPEN; no downstream theorem consumes them globally. |
| **H. Withdrawal ledger** | **SUBSTANTIVELY COMPLETE.** No round-1 overclaim is silently reused. Add old A′ by name and correct final provenance. |
| **I. Straight-four / R7.4** | **SURVIVES.** Residuals are exactly `{-2,-1}`, `{-1,4}`, `{4,5}` with cover `{-1,4}`; the universal `β=(2+√3)/9≈0.4147` floor follows from the complete all-low exclusion. |

## 7. Required repairs

1. Restate R7.4₂ for **any finite nonterminal position with `I=∅`**, independent of side/phase, or insert an explicit phase-erasure board-geometry corollary. Then cite that version in K2 and K3.
2. Split K2's final legality paragraph: use L1.2's max-`q` construction in the no-core branch; use touched core labels and L6₂ only when a core exists. Delete the false "in every case" sentence.
3. Add the one- and two-vertex exact-path cases to the named path/cycle proof.
4. Change §9's withdrawal entry to old "Theorem A/A′" so the round-1 grouped claim is named completely.
5. Correct §13 provenance to record parent/input `159c75f4` and reviewed output `74026b7c`, and qualify or remove "No commit was created."

No claim-status downgrade and no new machine run are required by these repairs.

## 8. Independent judgment on obligation J

J is a coherent, well-posed **conservative specification**, but it is not a mathematical reduction of GAP-RAW. Clause 2 already is A₂'s exact right-hand side, so D₂ is sound but tautological: the renewal and unripeness clauses add construction discipline, not leverage toward proving the target. The document says this honestly. Calling J "the right residue" is justified only in the engineering sense that one causal strategy must make service and account choices together; it is strictly stronger than necessary and could be false even if GAP-RAW is true.

The most promising attack on J is not to replay the three-gadget state as an arbitrary start. It is to force an analogous dormant count-one stock **from a genuine `Φ<1` root against every causal servicing strategy**. A useful finite-horizon attack game would:

1. start at a normative root;
2. require Defender's actual pair to service `I`;
3. let Attacker accumulate separated non-axis support pairs that expose several promotion centers;
4. prove a set-cover lower bound showing that any two-cell defense leaves a legal center c with enough latent `n₁(P,c)/9` (plus any `(√3-1)S₂(P,c)`) to force `Θ₂≥1` at the next epoch.

That would refute J without refuting GAP-RAW, which is exactly why the distinction matters. Conversely, a proof of J needs a strategy-reachable invariant controlling the distribution of count-one promotion capacity, not merely the current value of Θ₂.

## 9. Machine work and files

I ran the one coordinated gate at reviewed commit `74026b7c`, after confirming no Cargo process and `9.690 GiB` free physical RAM:

```powershell
$env:CARGO_TARGET_DIR = '.target-hunt'
cargo test -p hexfield_eq --lib --release `
  'gap_raw_hunt::tests::round2_' -- `
  --ignored --nocapture --test-threads=1
```

Result: **7 passed, 0 failed**, test time `169.71 s`, command wall time `170.2 s`.

Predicate-to-prose audit:

| Test | Review result |
|---|---|
| `round2_o1_prime_three_gadget_refutation` | Exact engine history/profile, focal-union quotient, exact `10/9`, and next epoch are hard-gated. |
| `round2_l832_same_axis_counterexample` | Exact focal and full-family hitting numbers are gated. |
| `round2_l832_cross_axis_counterexample` | Exact mixed pre-count branch and full-family hitting number are gated. |
| `round2_straight_four_explicit_cover` | Exact residual family, cover, `τ=2`, and both count-five singletons are gated. |
| `round2_r74_collinear_all_count2_max_tau` | Complete flank masks for separations 1–5 and `[2,2,2,1,1]` are gated. |
| `round2_birth_ledger_geometry_complete_and_scoped` | L2's 902 cases and both scoped L3 tables are hard-gated; no connected-to-global inference remains. |
| `round2_trace_r1b_breaks` | Every epoch uses exact variable residuals; all three claimed break records are hard-gated. |

No adversarial source check was added: the two review counterexamples concern proof wording/phase scope and do not challenge the computed predicates. I did not edit `GAP_RAW_PROOF_ROUND2.md` or the Rust harness and created no commit. The only authored deliverable is this review.
