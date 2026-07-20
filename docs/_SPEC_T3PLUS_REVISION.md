# Spec — revise PROOF_TSS_DEFENDER_ZONES.md to adopt the round-5 tightenings

## Goal

Rewrite `docs/PROOF_TSS_DEFENDER_ZONES.md` in place so that the tightened
theorems become the normative statements. All twelve reviewed tightenings are
adopted. The four items ruled CONFIRMED-WITH-REPAIR are adopted **with their
repairs installed exactly as prescribed** in
`docs/_T3_TIGHTENINGS_REVIEW_ROUND1.md` (the round-1 hostile review report).
Where the claims file (`docs/_T3_TIGHTENINGS_REVIEW_CLAIMS.md`) and the
report disagree, **the report wins**.

Read order: the current normative doc in full, then the report in full, then
the claims file for statement material.

## The twelve items to integrate

From the report's verdict table:

1. Remove Z1 as a hypothesis of T3 and as a mandatory term of T4's zone.
   Record in a remark that hitting cells remain a sensible search heuristic
   and that T6's kernel regime still uses the current threat family.
2. Replace exact global D_N by an admissible local budget B(N): the
   recurrence (OR-COMPLETION/WIN = 0; LOSS = b; OR = child; AND = 1 + max
   over children), the three properties actually used (hereditary decrease,
   completion monotonicity, anchor coverage), and the verifier rule that any
   integral upper bound satisfying the inequalities is admissible. The old
   global 𝔇(P_N, T) becomes a remark: the special case that is always
   admissible.
3. Replace the 8·D_N band by 8(B(N)−1) — WITH REPAIR: current L9 must NOT
   merely have its radius edited (that would make L9(a)/(b) false). Replace
   L9 and D12 invariant (iv) with the **first protected-occupation lemma**
   exactly as stated and proved in report §4: hypotheses (direct legal
   protected cells searched; every ghost-legal dismissal outside radius
   8(B(N)−1) of Prot(N) ∖ (Legal ∪ Stones)), conclusion (no defender
   placement ever creates a real-only stone in the current protected set),
   proof by first violation + backward trace to the first ghost-legal seed.
   D12 keeps invariants (i)–(iii) and drops (iv); Step A3 invokes the new
   lemma for ghost-illegal dismissals. Record the sharpness example
   (x₀ followed by B−1 successive distance-8 placements, target last).
4. Compress the obligation set: protected named-witness material = every
   future certificate attacker placement (including OR-COMPLETION designated
   moves and leaf continuation placements) plus E(W, P_L) for every named
   WIN/LOSS witness at its leaf. Full witness windows disappear from the
   protection requirement; report §5 carries the mask-agreement argument
   (shared attacker stones; no Y-cell in an A-alive window; MI is a
   canonical identity needing no full-window agreement).
5. Per-window exposure budgets and the split zone — report §8: E^D_N(W)
   recurrence (clock-correct per D9; "attacker enters W" means permanently
   non-D-alive, which is weaker than D5-dead and sufficient by L4), the
   hereditary inequality, Z_touch (touched windows: all empties legal, no
   frontier term, monotone guard cnt_D + E^D ≥ 6), Z_virgin (seed radius
   8(E^D−6), with the mixed-chain proof from the report: first-ever
   dismissed fill anchor, ghost-legal fill cases, ghost-illegal chain case
   with the j + 6 count), the companion condition that the certificate's
   attacker-in-W stopping move stays playable (covered by the obligation
   zone), and the sharpness example at E = 7.
6. Cell-specific deadlines r_N(y) — WITH the five formal repairs of report
   §7: ranks attach to live obligation occurrences/roles with the maximum
   over live roles; every designated attacker move includes OR-COMPLETION
   moves; WIN/LOSS continuation cells are treated through their
   witness-empty role with deadline = leaf entry; the band applies only at
   internal AND nodes while r ≥ 1 (no negative radius at r = 0, occurrence
   dropped at deadline); defender-completion windows stay on their separate
   B or E^D clocks. Include the ruling that LOSS witness empties need
   protection only through leaf entry (the adaptive contract handles the
   remainder).
7. Branch-indexed substitution — WITH the eight-point transition-inclusive
   repair of report §11, verbatim in substance: transition budget
   B̂(N,d,s) = 1 + B(C_s); obligations = union over ALL reachable
   descendants of C_s; d itself avoids those obligations and
   transition-dangerous completion empties; parent seed radius 8·r_{C_s}(y);
   completion test cnt_D(W,P_N) + 1 + B(C_s) ≥ 6 (and the analogous
   1 + E^D_{C_s}(W)); independently nonempty searched fallback S(N) (the
   "search only replies with no safe substitute" rule is otherwise
   circular); explicit transition rules (A3 uses φ_N(d); A2 may use any
   searched filler; ghost-illegal A3 inherits the earlier envelope); the
   selected envelope protects LOSS witness empties through leaf entry and
   counts the leaf's b placements for completion/own-win exclusion. Include
   BOTH of the report's counterexamples to the child-only version (the C3
   radius-0 failure and the C2 budget-5 failure) as remarks establishing
   why the +1 is mandatory. The simpler default-child variant f(N) gets the
   same transition-inclusive treatment.
8. T6 → kernel form — WITH REPAIR per report §9: K_b defined via
   τ(F ∖ d) ≤ b − 1; scope condition **mhs ≤ b at every kernel-governed
   internal AND node** (at mhs > b, K_b = ∅ violates D9; such nodes must be
   valid LOSS leaves or keep an existing nonempty subtree); the original
   core term removed (proof per the report: X = Y = ∅ before the first
   dismissal; the auxiliary refutation uses the residual threat family
   directly); "weakly refines and can strictly reduce" (not "strictly
   refines"); T6/T6⁺ RETAINS the explicit internal ¬own_win_now hypothesis
   (the kernel has no completion guard); the same-T argument as repaired
   (min-hitting line followable through K_b; defender stones create no new
   A-threats; the killed count-5 argument).
9. LOSS-witness sparsification — report §10: D9 may additionally require
   |𝒯| ≤ 3 at b = 1 and |𝒯| ≤ 6 at b = 2; the rank-two transversal
   argument (singleton case two sets; triangle sharp; maximal disjoint
   subfamily size dichotomy at b = 2; K₄ sharp); b counts placements.
10. Pathwise conclusion — restate T3's conclusion pathwise (per report §6):
    for every real play, either the real attacker wins strictly earlier, or
    the play maps to a certificate path and completes by that path's
    declared resolution; global T is the maximum over paths.
11. own_win_now redundancy at internal AND nodes — report §13: under the
    completion-zone requirement and the ban on defender-terminal edges,
    count-5 and count-4/b=2 internal nodes are impossible; the check stays
    as a diagnostic, stays logically necessary at LOSS leaves, and stays an
    explicit premise of the T6 kernel regime.
12. Finite acyclic certificate DAGs — report §14: one exact D9 label and
    one consistent clock per shared node; finite unfolding to a D9 tree;
    reachable-descendant obligation unions preserve protection
    monotonicity.

## Structural rules

- **Stable numbering.** Existing tags D1–D13, L1–L10, T1–T8, (Z1)…(Z5) are
  referenced by external documents. Do not renumber. Revised objects keep
  their numbers (D9, D11, D12, T3, T4, T6, L9→replacement). The L9 slot is
  taken by the first protected-occupation lemma, labelled **L9′** with a
  one-line note that it replaces the former L9 and why. New objects get
  fresh numbers (D14+, L11+, T9+ or lettered variants) — choose a clean
  scheme and use it consistently.
- Z1's clause in D11 is deleted as a requirement; keep a short remark where
  it stood. Zone labels used by the verifier become (Z2), (Z4), (Z5′)
  (the 8(B−1)/ranked band) — state the label mapping explicitly once.
- D1–D8 (the game model) are untouched. §8 (T8 pairing), §10 (ES), §11
  (domination) are untouched except where they cross-reference revised
  statements. §9 (quantitative) may need its zone description refreshed.
- §12 open problems: the band-sharpening problem (§12.2) is resolved by the
  exposure/virgin accounting — replace it with a short "resolved" note
  pointing at the new statements, or fold it into a remaining sharper
  question if one honestly remains. §12.1 (F + H_W) stays open — the report
  confirms none of the tightenings proves it. Keep pairing-threshold and
  formalization items; update the zone list the checker would verify.
- §13 review log: append a **Round 5** entry in the established style:
  external-model claims document, independent Claude review, Codex ultra
  hostile review (verdicts: 8 confirmed, 4 confirmed-with-repair, 0
  refuted, no error in the prior normative statements), pointers to the two
  underscore files. Then a **Round 6** placeholder line: "confirmation pass
  of this revision — pending."
- **Tags.** Revised/new statements carry [PROVEN] with a caveat line in the
  established style: "(R5-adopted; round-6 confirmation of this revision
  pending — tags provisional until §13 records Round 6.)" Do not invent new
  legend entries.
- The provenance block at the top gets one added sentence recording the
  round-5 revision and its date (2026-07-14).
- Prose register: match the existing document — dry, precise, no rhetorical
  flourishes. Proof completeness over brevity: every revised statement
  carries a complete proof or an explicit named dependency, C1/C2/C3
  channel discipline observed (§3).
- Consistency sweep before finishing: grep the revised doc for `8·D_N`,
  `8\*D_N`, `Z1`, `𝔇(P`, `L9(`, `invariant (iv)`, `belt and braces` and
  confirm every remaining occurrence is intentional (historical §13 entries
  are exempt and should remain verbatim).

## Hard boundaries

- Modify ONLY `docs/PROOF_TSS_DEFENDER_ZONES.md`. No other file. No git
  commands. Do not touch the PLAN_* docs, proof_parts/, or the underscore
  review files.
- Every mathematical statement you write must be one you can defend against
  a hostile reviewer — a confirmation round follows. Where the report
  prescribed exact statements or proofs, deviate only to fix outright
  typos, and note any such deviation in your final message.

## Definition of done

- The revised document is internally consistent end to end; all twelve
  items integrated; the four repairs installed; consistency sweep clean.
- Final message: a numbered list (1–12) with one line each stating where
  the item landed (section/label), plus any deviations from the report and
  the results of the consistency sweep.
