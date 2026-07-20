# SPEC: Attack open problem 1 — forced-hit budget debit (F + H_W)

You are attempting to settle the FIRST open problem in
`docs/PROOF_TSS_DEFENDER_ZONES.md` §12 (the normative, round-6-confirmed
revision). Work at full rigor: this is a proof attempt, not a survey.

## Required reading (in this worktree)

- `docs/PROOF_TSS_DEFENDER_ZONES.md` — the whole document, with close
  attention to: D7 (plies/horizon), D9 (certificate grammar, LOSS-leaf
  adaptive contracts τ(𝒯) > b), D10 (compressed obligations), D11 (ranked
  zones Z_dir/Z_seed/Z_touch/Z_virgin), D12 (ghost coupling, X/Y, the (MI)
  inequality), D14 (local budget B(N)), D15 (role ranks r_N(y)), D16
  (per-window exposure E^D(W)), L9′, L11–L13, T3, T4, §9, §12.1, §13
  (review log — this shows the grade of scrutiny your output will face).

## The problem, precisely

B(N) and E^D(W) currently count EVERY defender placement before the local
resolution / exposure stop, as if all of them were available to occupy
protected cells or fill a chosen window W. But along certificate branches
the defender is under pressure: attacker threat windows carry hitting
obligations, and a defender who ignores them loses faster. Intuition says
some placements are therefore unavailable for budget-relevant harm, so the
budgets should admit a debit: quiet placements F plus per-window forced-hit
capacity H_W, with the zone radii and exposure tests computed from the
debited quantities.

## Known obstructions (do not rediscover these; start past them)

1. **Dual-purpose placements.** A hitting cell may itself lie inside the
   chosen window W, inside Prot(N), or inside a seed ball — a "forced" hit
   can simultaneously be the harmful placement. Naive subtraction is
   unsound.
2. **b = 2 splitting.** One placement of a turn can hit while the other
   builds. Any per-turn argument must handle the pair.
3. **"Forced" needs a certificate-checkable definition.** In the real
   game the defender is never literally forced: he can ignore a threat.
   The trade is that ignoring it lets the attacker resolve the branch
   earlier (smaller residual clock). A sound debit almost certainly takes
   the form of a branchwise dichotomy/race: at each defender opportunity,
   EITHER the placement answers the threat structure (debit it from the
   harm budget) OR the branch's residual resolution length shrinks. The
   bookkeeping must be worst-case over the defender's choice and must be
   verifiable from the certificate plus the position — no unbounded
   quantification the verifier cannot check.

## Candidate routes (suggestions, not mandates — you may find better)

- (a) **Disjointness-conditioned debit:** debit E^D(W) only for plies whose
  entire available hitting universe is disjoint from W (respectively from
  the relevant guard ball for seed radii). Weaker but plausibly provable.
- (b) **Race form via τ contracts:** formalize "ignore ⇒ faster
  resolution" through the LOSS-leaf adaptive contracts and the local-budget
  recurrence; the debited budget becomes min over the dichotomy branches.
- (c) **(MI)-style counting:** an inequality tying |X ∩ W| to the number of
  non-hitting real defender placements, anchored the way (MI) anchors
  cnt_D.

## Definition of done — the report

Write `docs/_OPEN_FHW_REPORT.md` with:

1. A `VERDICT:` line — one of PROVEN / REFUTED / PARTIAL.
2. If PROVEN: complete formal text in the house style of the proof doc
   (new definitions D19+, lemmas L15+, theorem T11+ — numbering
   suggestions only), with every step justified, ready to install; plus
   integration notes: exactly which zone terms shrink, how the verifier
   checks the debit from certificate + position, and interaction with
   D14 nesting, D16 stop conditions, T9 envelopes, and T10 DAGs.
3. If the natural statement is false: an explicit counterexample family,
   verified move-by-move (positions, placements, window counts), plus the
   STRONGEST weakening you can prove in full, with proof.
4. Whatever the verdict: a sharpness example for any debit you prove
   (a position where the debited budget is exactly achieved by a real
   defender line), or an explicit statement that sharpness is open.
5. Mark any unproven step GAP — an honest PARTIAL with clean lemmas beats
   an overclaimed PROVEN, which will be caught in hostile review and
   discarded.

## Hard constraints

- Do NOT edit `docs/PROOF_TSS_DEFENDER_ZONES.md` or any other existing
  file. Your only output is the report (plus, if useful, one scratch
  script `scripts/_fhw_check.py` for verifying counterexamples — offline,
  no network).
- Do NOT run git commands that mutate state (no commits, no stash).
- Prose style: dry, plain, no filler, no rhetorical flourishes.
