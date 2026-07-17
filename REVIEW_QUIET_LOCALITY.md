# NQ2 hostile review - quiet locality

Review target: commit `833020edbb6942514c4c1df5e9f7ec274aaa9587`.

## Overall verdict

**ACCEPT-WITH-EDITS.** The frozen witness is a sound counterexample to the
position-computable, per-Choice `join_live` / `join_adj2` / `join_adj1`
restriction that the hunt measured and proposed for production. The required
move is legal, quiet under the production post-SecondStone predicate, outside
all five audited locality/adjacency universes, and uniquely winning among all
538 legal completions. The 537 negative classifications are explicit immediate
defender wins, not solver misses. The positive continuation is accepted at the
post-move root and again after rebinding the exact parent root.

Q8's one-turn `K_reply` theorem is also sound. I sign it off as a future
completeness-preserving production pre-filter only under the exact phase and
recomputation contract below. The edits requested here tighten that contract,
qualify two imprecise phrases, and make the regression artifacts freeze what
the prose claims; none changes the REFUTED conclusion or the Q8 proof.

Ledger tally: **13 CONFIRMED, 0 BROKEN**. This confirms all 8 document
refutations, both `VERIFIED-EXHAUSTIVE` entries, and all 3 `PROVEN` entries.
`CONFIRMED` below means that the document's adjudication of the named claim
survived review; for example, Q0 is confirmed as correctly refuted.

## Per-claim ledger

| Claim | Review verdict | Basis |
|---|---|---|
| Q0 | **CONFIRMED** | In the named `tss-vcf-width` ground truth, ordinary SecondStone candidates are tested after placement with both `turn_created_claimant_threat` and `turn_forces_small_defender_reply`; after ordinary failure, consume calls `write_legal_moves` without consulting that predicate. The shadow census alone calls an edge quiet when the pre-edge phase was SecondStone and the post-edge predicate is false. See `tss_solver.rs:3511-3516`, `3540-3553`, `4000-4012`, and `5095-5111`. |
| Q1 | **CONFIRMED** | The seven non-D6 committed records all store `join_adj1.hit=true`; independent replay reproduced live-window membership and `d_stone=1` for all seven. This is coverage, not completeness. |
| Q2a | **CONFIRMED** | If legal `c` is in no old defender-free attacker window with positive attacker count, it changes no old live window. Every newly live defender-free window through `c` changes from count zero to count one, hence delta five. The statement is correctly limited to window-level progress. |
| Q2b | **CONFIRMED** | The family fixture has 44 old live windows and 16 born count-one windows. The born vertical window `(5,0)..(5,5)` overlaps an old live window, so it joins an old overlap component rather than forming only a new family. |
| Q2c | **CONFIRMED** | The remote defensive block realizes the omitted opponent-window channel. Every local substitute loses immediately, while the remote continuation has a verifier-accepted general-branching certificate. No branchwise or principal-variation swap survives this position. |
| Q3 | **CONFIRMED** | Independent census gives `join_live=141`, excluding `(6,-6)`. All 141 are among the 537 explicitly losing alternatives. |
| Q4 | **CONFIRMED** | Independent census gives `join_adj2=75`, excluding `(6,-6)`. All 75 lose explicitly. |
| Q5 | **CONFIRMED** | Independent census gives `join_adj1=38`, excluding `(6,-6)`. All 38 lose explicitly. |
| Q5a | **CONFIRMED** | `d_stone(6,-6)=6`; `adj_stone_k1=39` and `adj_stone_k2=93`, and neither contains the required move. |
| Q6 | **CONFIRMED** | No text in the proof licenses unrestricted absence from a restricted miss. Sections 5.2-5.3 explicitly require `UNKNOWN`/no-certificate semantics and list the no-cutoff/no-Unknown obligations for any future negative artifact. The witness itself demonstrates the false negative. |
| Q7 | **CONFIRMED** | All 538 root moves are classified: `(6,-6)` has a positive certificate and every other move has an explicit legal immediate-loss reply. The 537 losses do not use a cap or horizon. The rerun returned hard `Loss`, not `Unknown`, at the post-move root; the ordinary verifier and dispatch oracle both accepted that certificate and the prepended exact-root `Win`. |
| Q8 | **CONFIRMED** | The two-case proof is complete: immediate attacker wins are in `Win1_A`; otherwise any move outside `BlockAll_D` leaves at least one active defender count-4/5 window unchanged, whose one or two empties the defender legally fills in its next two-stone turn. The phase-matrix test covers a two-empty count-four, overlapping threats, disjoint threats plus Win1, and a FirstStone counterexample to phase widening. |
| Q9 | **CONFIRMED** | `C_full(P)=Legal(P)` is the identity universe. The proof correctly makes no claim that an unstudied smaller unconditional tier is impossible. |

## Q8 production pre-filter sign-off

**SIGNED OFF, with an exact contract.** A future implementation may replace the
attacker's full legal Choice set by `K_reply(P)` only when all of the following
are true:

1. `P` is nonterminal, `P.current_player()==A`, and
   `P.phase()==SecondStone { first }`. The stored coordinate must be part of the
   state/root binding. Do not apply this rule at Opening or FirstStone.
2. Recompute from the current board, after the stored first placement:
   `D=A.other()` and
   `T_D(P)={W: active_player(W)==Some(D) and count_D(W) in {4,5}}`.
   Define `E_P(W)` explicitly as the current empty cells of `W`; "urgent" means
   `T_D(P)` is nonempty.
3. Compute `Win1_A(P)` from the same full legal set and retain every placement
   whose application immediately returns terminal winner `A`.
4. Compute
   `BlockAll_D(P)={c in Legal(P): for every W in T_D(P), c in E_P(W)}` and retain
   `Win1_A(P) union BlockAll_D(P)`.
5. If `T_D(P)` is empty, return all of `Legal(P)`. Do not substitute a locality
   tier for this vacuous case.

Why this is complete: for a legal `c`, missing `E_P(W)` means `c` is outside
that active window. If `c` is not already an attacker win, applying it completes
the attacker's SecondStone and passes control to defender FirstStone. A
nonterminal active defender count-five has one empty and a count-four has two.
Each empty is at distance at most five from an existing defender stone, inside
the engine's radius-eight legal frontier; legality only grows after the first
defender placement. The defender therefore completes six before `A` moves
again. Multiple disjoint or overlapping threats need no extra case: a move
outside the intersection misses at least one of them.

An attacker's first placement cannot create an active defender window: windows
not incident to it are unchanged, while incident windows acquire an attacker
bit and can only cease to be defender-active. It can block old threats and can
create an attacker win-now, so both `T_D` and `Win1_A` must still be recomputed
after that first placement.

The phase guard is production-critical. The added FirstStone fixture has a
winning pair whose first move `(4,0)` is outside the raw kernel and whose second
move `(5,0)` wins before the defender moves. Applying Q8 one placement too early
would prune that win.

## Findings by severity

### Critical / high

None in the proof as scoped. The FirstStone counterexample above is a high-risk
deployment guard, not a defect in Q8: the document correctly says SecondStone.

### Medium

1. **Q8 is mathematically clear but not yet copy/paste-safe as a production
   specification.** `E_P(W)` is used without an explicit local definition, and
   "urgent" is not formally equated to `T_D(P) != empty`. The proof should also
   say in the theorem statement, not only surrounding prose, that the census is
   recomputed after the stored first coordinate. A phase or stale-census error
   is unsound, as the FirstStone fixture demonstrates.
2. **The committed required-remote test was existential over a five-case
   catalog, not frozen to the named witness.** It could skip an `Unknown` or
   hard non-`Loss` first case and pass on a later witness while the proof's exact
   replay, 538/537 tally, census, or horizon drifted. Review-time ignored checks
   now pin the ID, replay, phase, counts, singleton kernel, hard status, both
   verifier paths, and all 12 D6 images. Those changes should be retained.

### Low / editorial

3. **Q0's "unconditional" needs its control-flow qualifier.** The fallback is
   unconditional with respect to the quiet predicate, not absolutely: an
   earlier hit-limit return prevents reaching it, and a non-null `PairContext`
   filters the written legal set. Neither applies at the frozen root, so the
   refutation is unaffected.
4. **Section 2.5 asserts the D6 covariance instead of deriving or citing it.**
   The claim is true: the verifier has coordinate, window, stored-phase, Choice,
   Universal, and commutation remapping, and the added all-12 exact-witness test
   accepted every replay and certificate. The proof should cite that mechanism
   or present the short covariance corollary.
5. **"Strongest tier" is order-ambiguous.** Full legal is the largest universe
   and therefore the weakest restriction. "Only unconditional universe proved
   here" expresses the intended result safely.
6. **Stale harness comments contradicted Q0/Q6.** The helper comment said the
   consume machinery "fires" exactly on a false quiet predicate, and the
   two-stage comment equated a capped VCF non-Win with no pure-forcing win.
   Review-time edits now say the fallback is ungated and a VCF miss is not an
   absence proof. The executable predicates were correct throughout.

## Required repairs

1. In Q8, define `E_P(W)`, define `urgent <=> T_D(P) != empty`, and put the exact
   nonterminal/current-player/`SecondStone { first }`/post-first recomputation
   guard in the theorem or a normative production contract.
2. Qualify Q0's "unconditional" fallback as ungated by quietness after clean
   ordinary failure, with `PairContext` filtering where applicable.
3. Make section 2.5 a stated covariance corollary with a citation to
   `d6_transform_coord` / `d6_remap_certificate`, or retain and cite the added
   all-12 exact-witness regression.
4. Carry forward the review-time ignored-test hardening in
   `tss_quiet_locality_hunt.rs`: exact frozen-witness assertions, dispatch-oracle
   verification, Q8 phase-matrix fixtures, exact family tallies, and all-12 D6
   replay/certificate checks. If the proof's machine-check inventory is updated,
   distinguish the original two proof tests from this third review test.
5. Replace "strongest tier" with "only unconditional universe established" (or
   "weakest restriction") and clarify that the repository state line names the
   pre-proof base while this review targets commit `833020ed`.

## Attack-surface results

### A. Frozen witness

Independent replay of the literal 36 coordinates was legal and nonterminal at
every prefix. Ownership is 18/18, the root is Player 0
`SecondStone { first: (6,0) }`, and there are exactly 538 legal moves. The same
length-six census used by the hunt gives:

| Universe | Size | Contains `(6,-6)` |
|---|---:|---|
| full legal | 538 | yes |
| `join_live` | 141 | no |
| `join_adj2` | 75 | no |
| `join_adj1` | 38 | no |
| `adj_stone_k2` | 93 | no |
| `adj_stone_k1` | 39 | no |

The nearest Player-0 stone is distance 6. For every other legal completion, the
attacker does not win, `(6,-6)` remains legal, and Player 1 wins immediately by
playing it. This exhausts 537 alternatives without invoking the solver.

After Player 0 plays `(6,-6)`, the move is nonterminal. The defender-to-move
analysis sees two Player-0 count-four windows with the same empties
`{(4,0),(5,0)}`; the minimum hitting number is 1 while the defender budget is 2,
so the production forcing predicate is false. This is loose-quiet under the
same post-SecondStone census used by the hunt.

The rerun then produced `Loss` for the post-move current player at absolute
horizon 66 using 4,957 search nodes and 3,857 child certificate nodes. The
ordinary verifier and its per-move dispatch oracle accepted the child. After
prepending `Choice { mv:(6,-6) }` and rebinding the exact root, both accepted the
3,858-node `Win`. Generator caps therefore affect discovery cost only, not the
accepted positive claim.

### B. Q8 adversarial shapes

The review test exercised:

- one defender count-four with two distinct empties;
- four overlapping count-four/count-five windows with a singleton common block;
- disjoint defender threats with empty `BlockAll` plus an attacker immediate
  win retained only by `Win1`; and
- a FirstStone state where the raw kernel omits a winning first placement.

For every omitted legal completion in each urgent SecondStone fixture, the test
selected a missed window and replayed the defender's one- or two-placement
terminal win. On the frozen root, defender urgent windows have empty sets
`{(6,-6)}` and `{(6,-6),(7,-7)}`, so `Win1` is empty and `K_reply` is exactly
`{(6,-6)}`.

### C. Restricted exhaustion

The proof consistently treats restricted exhaustion as no found certificate,
not an unrestricted loss. Its unique-win statement instead combines 537
explicit counterplays with one verified positive certificate. The historical
hunt report contains capped-search overstatements, but the proof explicitly
identifies and disowns them; stale duplicates in harness comments were corrected
during review.

### D. Scope honesty and production semantics

The refutation is not a FirstStone strawman. The hunt measured candidate sets at
the current per-placement state, included five specimens entered at SecondStone
with the first coordinate pre-root, and proposed shrinking the full legal
completion fallback used by Group 2. The frozen state is exactly such a
SecondStone Choice state. The proof also honestly leaves a whole-turn
FirstStone ordered-pair normal form open.

The read-only `tss-vcf-width` worktree still routes round-3 consume into the same
`prove_choice` mechanics described by Q0. Engine legality/phase sources and the
verifier are unchanged between the reviewed commit and that worktree; the local
harness uses the real `HexoState`, `apply_placement`, production solver, and
production verifier rather than a reference-game substitute.

### E. Counterexample family

As written, section 2.5 gives a covariance assertion rather than a standalone
derivation. The underlying claim checks out. D6 preserves axial distance,
length-six windows, ownership masks, radius-eight legality, phase and stored
first coordinate, and the forcing census. The verifier's remapper covers every
coordinate-bearing certificate field. The added regression transformed the
exact replay and full certificate through all 12 symmetries; every image kept
538/537, all five census sizes, `d_stone=6`, singleton `K_reply`, quietness, and
a verifier-accepted root Win.

## Machine work

Before cargo: 13.59 GiB free RAM, zero `cargo` processes, zero `rustc`
processes. One serialized invocation was used:

```powershell
$env:CARGO_TARGET_DIR='.target-hunt'
$env:QL_ADV_CAP='200000'
$env:QL_ADV_HORIZON_SLACK='30'
$env:QL_TT_BYTES='268435456'
cargo test --release -p hexfield_eq quiet_locality_adversarial -- --ignored --test-threads=1 --nocapture
```

Result: **3 passed, 0 failed**; test body 5.28 s, release compile 6.27 s,
11.8 s wall time. Setup reported 12.88 GiB free RAM. The three tests were the
family counterexample, the added Q8 phase matrix, and the pinned required-remote
witness.

## Files

- `REVIEW_QUIET_LOCALITY.md` - this review.
- `packages/hexfield_eq/rust/src/tss_quiet_locality_hunt.rs` - review-time,
  ignored-only adversarial/pinning/D6 checks and corrected comments.
- `PROOF_QUIET_LOCALITY.md` - reviewed and intentionally not edited.

No commit was made.
