# Round 6 hostile confirmation of the round-5 T3+ revision

## Scope and ruling convention

This pass read, in full, `PROOF_TSS_DEFENDER_ZONES.md`,
`_PROOF_PRE_R5_BACKUP.md`, `_T3_TIGHTENINGS_REVIEW_ROUND1.md`,
`_T3_TIGHTENINGS_REVIEW_CLAIMS.md`, and `_SPEC_T3PLUS_REVISION.md`.
The round-5 report controls wherever the claims file differs. A malformed
normative sentence or an out-of-scope file change is sufficient for `DEFECT`
or final `FAIL` even when the underlying mathematics is recoverable.

## A. Verdict table: twelve adopted items and four repair installations

| ID | Adopted item / repair | Verdict | Exact location and ruling |
| --- | --- | --- | --- |
| 1 | Remove Z1 from T3/T4 | **APPLIED-CORRECTLY** | D11, lines 358–360, expressly makes current hitting cells optional and retains the T6 threat-family use; T4, lines 565–567, has no hitting term. D9's independently nonempty searched set remains. No live proof uses Z1. |
| 2 | Replace exact global \(D_N\) by admissible local \(B(N)\) | **APPLIED-CORRECTLY** | D14, lines 218–229, gives the exact recurrence, placement unit, admissible inequalities, and the old \(\mathfrak D(P_N,T)\) special case. L11, lines 296–320, proves hereditary decrease, completion monotonicity, and selected-path anchor coverage. |
| 3 | Replace the \(8D_N\) band by \(8(B-1)\) / ranked radii | **APPLIED-CORRECTLY** | L9′, lines 365–399, installs the required occupation-only lemma and sharpness example. D11, lines 350–356, gives the ranked and uniform bands. The stronger old L9 conclusion is not used live. |
| 4 | Compress full witness windows to live obligation roles and witness empties | **APPLIED-CORRECTLY** | D10, lines 231–248, includes all designated attacker placements, OR-COMPLETION, and WIN/LOSS witness empties. T3, lines 486–492 and 521–545, supplies the OR-completion and compressed-mask transfer arguments. MI remains a universal identity rather than a full-window premise. |
| 5 | Per-window exposure with touched/virgin split | **APPLIED-CORRECTLY** | D16, lines 268–282, gives the clock-correct recurrence and attacker-entry stop. The zones are at lines 329–338; L12, lines 420–461, proves the touched, virgin, mixed-chain, post-LOSS, companion-stop, and \(E=7\) sharpness cases. |
| 6 | Cell-specific deadlines | **DEFECT** | The five formal repairs are mathematically present in D10/D15 and the zone formula, but T4 line 563 is malformed: `Exact cell ranks and can only reduce...`. Replace lines 563–564 with: `Exact cell ranks can only reduce that uniform obligation band; exact window exposures likewise reduce the corresponding B-clock completion guards.` |
| 7 | Branch-indexed substitution | **APPLIED-CORRECTLY** | D17, lines 688–730, contains all eight transition-inclusive requirements and the repaired default-child form. T9, lines 732–751, proves the synchronized, nested-envelope transfer. Both mandatory `+1` counterexamples occur at lines 753–762. |
| 8 | T6 extendable-hit kernel with no original core | **APPLIED-CORRECTLY** | T6, lines 606–652, has fresh equal-position entry, \(\operatorname{mhs}\le b\) at every governed internal AND node, explicit `¬own_win_now`, exact \(K_b\), no original core, weak-refinement wording, the \(\operatorname{mhs}>b\) exit, and the repaired same-\(T\) argument. |
| 9 | Sparse LOSS witnesses, at most 3/6 | **APPLIED-CORRECTLY** | D9, lines 198–205, permits the bounds and counts placements; L13, lines 569–588, proves both transversal bounds and gives the triangle / \(K_4\) sharp examples. |
| 10 | Pathwise T3 conclusion | **APPLIED-CORRECTLY** | T3, lines 467–475 and 547–550, states and proves: earlier real win or a finite mapped certificate path resolving by that path's declared ply; global \(T\) is only the maximum. |
| 11 | Internal-AND `¬own_win_now` redundancy under completion zoning | **APPLIED-CORRECTLY** | D9, lines 210–213, retains the diagnostic and the LOSS/T6 exceptions. L14, lines 764–779, correctly excludes count-5 and count-4/\(b=2\) internal nodes using the completion zone and defender-terminal-edge ban. |
| 12 | Finite acyclic certificate DAGs | **APPLIED-CORRECTLY** | D18, lines 781–787, requires one exact label, consistent clock, edge inequalities, reachable-descendant obligations, and path-local histories. T10, lines 789–801, gives the finite unfolding proof. |
| R1 | L9′ first-protected-occupation repair | **APPLIED-CORRECTLY** | Lines 365–399 match report §4: both hypotheses, the weaker conclusion, first-violation/backward-seed proof, \(8(p-1)\), \(B/r\) anchor, and sharpness. D12, lines 400–418, retains exactly invariants (i)–(iii); invariant (iv) is gone. T3 A3 uses only occupation avoidance. |
| R2 | T6 scope/no-core/same-\(T\) repair | **APPLIED-CORRECTLY** | Lines 611–624 state the full scope and handoff; lines 626–652 prove nonemptiness, residual LOSS/WIN refutations, defender-first exclusion, followable minimum-transversal line, killed-count-5 step, same horizon, and kernel emptiness when \(\operatorname{mhs}>b\). |
| R3 | D17/T9 transition-inclusive repair | **APPLIED-CORRECTLY** | D17 points 1–8 are at lines 699–725; the independently nonempty fallback and anti-circular rule are at lines 694–697 and 716–717; the default-child repair is at lines 727–730; both counterexamples are at lines 753–762. |
| R4 | D15 five-part deadline repair | **APPLIED-CORRECTLY** | Role/occurrence identity and OR-COMPLETION are at lines 231–246; maximum rank over live roles, leaf-entry deadlines, `r=0` discharge, internal-AND-only bands, and separate completion clocks are at lines 250–266 and 326–334. LOSS witness protection correctly ends at leaf entry. |

## B. Proof-soundness walk

### Definitions D9–D18

- **D9:** the finite exact-successor grammar, typed leaves, path clocks,
  defender-terminal-edge ban, nonempty searched sets, adaptive LOSS remainder,
  and global maximum are mutually consistent. The LOSS budget is placements.
- **D10:** reachable live roles are nested. Protecting shared future attacker
  cells and leaf empty roles is sufficient: ghost attacker stones are shared,
  an A-alive witness has no ghost defender/Y cell, and every remaining cell is
  a protected leaf empty. OR-COMPLETION needs only its designated empty.
- **D11:** the `(Z2)/(Z4)/(Z5′)` mapping is explicit at lines 352–356 and the
  four finite zone components implement those three labels consistently.
- **D12:** the canonical X/Y updates in A1–A3 and the filler subroutine preserve
  ply, mover, budget, shared attacker stones, disjoint canonical differences,
  and \(X\cap\operatorname{Prot}=\varnothing\). MI has the correct sign.
- **D13:** `R_cert` is exactly the four ordinary mandatory components;
  `R_search` adds only named heuristics. Its finiteness argument is valid.
- **D14:** the edge inequalities imply the needed pathwise hereditary and
  anchor properties, including every LOSS remainder.
- **D15:** ranks attach to roles, not bare cells; the cell maximum and deadline
  rules close multiple-role/branch collisions and the rank-zero off-by-one.
- **D16:** the recurrence counts one per AND edge, zero per ordinary OR edge,
  and \(b\) at LOSS, and stops on permanent non-D-aliveness. Overlapping
  windows correctly retain separate clocks.
- **D17:** every selected child envelope includes the current transition and
  every reachable descendant. Earlier envelopes remain valid because later
  selected subtrees are nested; A2 creates no X; a ghost-illegal A3 traces to
  an earlier tested seed.
- **D18:** consistent labels/clocks make every unfolded copy a valid D9 node;
  edge-local clock inequalities and reachable obligations survive unfolding.

### Lemmas and theorems

- **L9′:** sound. The first violating protected occupation either is
  ghost-legal and searched or traces through finitely many earlier X-stones to
  a first ghost-legal seed. Protection nesting and monotone ghost illegality
  put the target in that seed's guarded set; \(p\le B\) or \(p\le r\) yields
  the exact \(8(p-1)\) contradiction.
- **L11:** sound. In particular, the two dependencies made explicit by the
  authoring pass are correct: \(E_N^D(W)\le B(N)\) follows pointwise from the
  D16/D14 recurrences and leaf bounds; an exact role rank
  \(r_N^*(\rho)\le B(N)\) because its deadline occurs no later, in defender
  placements, than the selected path's declared resolution. Conservative
  verifier ranks may be larger, as the text says.
- **L12:** sound. The no-dismissal branch follows from MI; the first dismissed
  W-fill is either touched/ghost-legal, virgin/ghost-legal, or supported by a
  ghost-illegal causal chain. In the last case, \(j\) approach links plus all
  six virgin-window fills require at least \(j+6\) exposure placements.
  Independent seeds and interleavings only consume more exposure. D10 keeps
  the attacker-in-W stop playable, and D16's LOSS value includes the leaf's
  \(b\) placements without extending the coupling circularly.
- **L13:** sound. The singleton/two-set and triangle cases prove the \(b=1\)
  bound. The maximal-disjoint-subfamily dichotomy and at most four cross-pair
  obstructions prove the \(b=2\) bound.
- **L14:** sound under its stated completion-zone scope; it does not remove the
  LOSS check or T6 premise.
- **T3:** sound under the compressed obligation set. Step O obtains real
  emptiness from protection and real legality from a shared `(Z4)` witness.
  A1–A3 close C1/C3; L12 closes defender-completion C2. WIN and LOSS masks
  agree at entry using only leaf empties, and the adaptive LOSS continuation
  is direct real-game reasoning after role discharge. The selected path is
  finite and its local clock supports the pathwise conclusion.
- **T4:** its mathematical zone implication is sound, subject to correcting
  the malformed sentence identified in table item 6.
- **T6:** sound. Fresh equal real/ghost positions at region entry are necessary
  and sufficient: equality persists through searched kernel edges until the
  first out-of-kernel reply, where the proof abandons the original subtree and
  uses the residual threat family directly. The repaired minimum-transversal
  path proves that the original horizon reaches the auxiliary second attacker
  placement.
- **T9:** sound. The current `+1` is present in the rank and completion
  channels, and all later choices remain within the selected reachable
  subtree. LOSS masks and completion/own-win exclusion are covered through
  the correct, distinct deadlines.
- **T10:** sound by finite path unfolding and T3/T9.
- The revised **L10, T5, and T7** coverage statements were also checked; their
  joint hypotheses and conclusions are consistent with the ranked zone.

**New mathematical proof gaps found:** none.

## C. Untouched-section integrity and out-of-scope diff

A content-normalized line diff gives the following complete result.

| Region | Result |
| --- | --- |
| D1–D8 and §§2–4 | No content difference. |
| §8, T8/T8.1 | No content difference. |
| §10, ES | One deliberate cross-reference update only: revised lines 942–944 replace “this document's closure set” by “D13's A-touched heuristic/candidate term.” No other content difference. |
| §11, domination | No content difference. |
| Historical R1–R3 | No content difference. |
| Historical R4 | Text is identical, but revised lines 1119–1125 use LF terminators whereas the backup's corresponding seven lines use CRLF. This is an out-of-scope byte-level hunk inside a log entry required to remain unchanged. Restore CRLF terminators on those seven lines without changing their text. |

There is no other unauthorized substantive or line-ending hunk in the
protected regions. The revised file is otherwise intentionally mixed-EOL at
the round-5 edit hunks; only the R4 range above changes an old protected entry.

## D. Internal consistency

- Every live D/L/T and section cross-reference resolves.
- The live verifier labels are stated once as `(Z2)`, `(Z4)`, `(Z5′)` and are
  used consistently. `Z_dir`, `Z_touch`, and `Z_virgin` implement `(Z2)`;
  `Z_seed` implements `(Z5′)`.
- No live requirement remains for Z1, exact global \(D_N\), full witness
  windows, old L9's never-legal conclusion, D12 invariant (iv), or an
  \(8D_N\) band. Remaining positive occurrences are either definitions,
  explicit negations/remarks, or preserved historical log text.
- Pre-existing numbering is stable: D1–D13, L1–L10 with L9 replaced by L9′,
  and T1–T8/T8.1 retain their numbers. D14–D18, L11–L14, and T9–T10 are fresh.
- Every revised/new provable lemma or theorem carries `[PROVEN]` and the exact
  round-6-pending caveat. D9–D18 remain untagged definitions, following the
  document's established D-label convention; they are not propositions
  requiring proof-status tags.
- The only live prose inconsistency is T4 line 563, repaired verbatim in
  table item 6.

## E. Sections 9 and 12

- Section 9, lines 856–888, describes the ranked ordinary zone, kernel scope,
  sparse LOSS witnesses, pathwise/DAG clocks, and historical predecessor
  measurements consistently. It no longer presents the old \(D_N\) zone as
  normative and correctly labels the old mechanized experiment as evidence
  for the predecessor heuristic rather than a proof of revised T7.
- Section 12.1, lines 1005–1010, honestly leaves \(F+H_W\) open. Local \(B\),
  per-window \(E^D\), and branch substitution do not prove that compulsory
  threat hits are unavailable for filling a chosen window; the stated need
  for branchwise worst-case accounting is correct.
- Section 12.2, lines 1011–1013, correctly marks the former band problem
  resolved by the ranked obligation and touched/virgin exposure accounting.
- The pairing problem remains unchanged in substance, and the formalization
  list at lines 1016–1019 names the new labels, clocks, envelopes, and DAG
  checks.

## F. Final ruling

**FAIL.** The round-6-pending caveat may not be dropped yet. No mathematical
repair is required, but both of the following edits are mandatory:

1. At normative T4 lines 563–564, replace the malformed sentence with exactly:
   `Exact cell ranks can only reduce that uniform obligation band; exact window exposures likewise reduce the corresponding B-clock completion guards.`
2. Restore CRLF line terminators on revised lines 1119–1125 (the unchanged
   historical Round-4 entry), preserving every character of the entry.

After those two mechanical repairs, the twelve substantive adoptions and all
four controlling repair installations are confirmed at PROVEN quality; no
further proof change is prescribed by this pass.
