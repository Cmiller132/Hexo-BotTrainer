# SPEC: Attack the ES-potential global claim (Φ < 1 forever-blocking)

You are attempting to settle the open claim in
`docs/PROOF_TSS_DEFENDER_ZONES.md` §10: whether Φ < 1 at a nonterminal
defender-FirstStone position guarantees the defender a strategy that blocks
the unrestricted infinite game forever.

## Required reading

- `docs/proof_parts/ES_POTENTIAL.md` — the full potential layer (λ = √3,
  Φ over attacker-touched alive windows, Theorems 1–4, the proven
  counterexamples to Beck's induction, Corollary 2's integer check).
- `docs/PROOF_TSS_DEFENDER_ZONES.md` §10 for the corrected summary and the
  honest-boundary items.

## The problem, precisely

Known already (do not redo):
- Theorem 1: fixed finite family F, Ψ_F < 1 ⇒ greedy blocks F forever.
- Theorem 3: uniform bound over ALL attacker strategies on enrolled birth
  mass B_∞ with Φ(P₀) + B_∞ < 1 ⇒ greedy blocks forever.
- Cex 1: births break the naive induction — two far double-placements
  birth 36 one-stone windows and push Φ to 4/√3 > 1. The "≤ 2 empties"
  repair fails. Φ = 1 does not suffice.

The OPEN question: the raw claim itself. The counterexamples kill the
PROOF STRATEGY (potential-decrease induction), not the claim — after the
births, the attacker's new windows are far from mature, and it is unknown
whether he can ever convert. Settle it either way, or prove the strongest
new partial result.

## Routes worth trying (suggestions)

- **Refutation:** exhibit a position with Φ < 1 from which the attacker
  wins against EVERY defender strategy (not just greedy). Note the bar:
  a full attacker winning strategy. A weaker but still valuable result:
  Φ < 1 position where every GREEDY defense (any tie-breaking) loses —
  that refutes greedy-sufficiency and forces any future proof to use a
  non-greedy defense.
- **Proof:** a modified potential that survives births — e.g. damped
  enrollment (birth mass weighted by distance or maturity), a two-tier
  potential with a separate account for all-empty windows, or an amortized
  argument where the attacker's birth-generating placements provably cost
  him tempo elsewhere (connect to the L7 completion-counting style).
- **Reduction:** show the global claim equivalent to (or implied by) a
  concrete finite statement, even if that statement stays open.

## Definition of done — the report

Write `docs/_OPEN_ES_GLOBAL_REPORT.md` with:

1. `VERDICT:` PROVEN / REFUTED / GREEDY-REFUTED / PARTIAL.
2. Full proofs for whatever you claim, in the house style of
   ES_POTENTIAL.md, every step justified; counterexamples verified
   move-by-move with window counts and potential values computed exactly
   (surds kept exact, no floating point in the proof text).
3. If PARTIAL: the strongest new theorem with proof, plus a precise
   statement of the remaining obstruction and why each tried route fails
   (with the failing example).
4. Mark unproven steps GAP. An honest PARTIAL beats an overclaim; a
   hostile review round follows.

## Hard constraints

- Do NOT edit any existing file. Outputs: the report + optionally one
  scratch script `scripts/_es_global_check.py` for verifying examples.
- No git state mutation. No network.
- Prose style: dry, plain, no filler.
