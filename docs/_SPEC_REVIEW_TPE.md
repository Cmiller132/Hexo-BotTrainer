# SPEC: Hostile review of the tightness, pairing, and ES reports (round 1)

You are a hostile reviewer. Break what can be broken in these three
documents; confirm what survives. All three include machine checks that
already pass — your primary targets are therefore the PROSE PROOFS and the
question of whether the scripts actually assert what the prose claims
(vacuous or mis-aimed checks are findings).

## Documents under review (all in this worktree)

1. `docs/_TIGHTNESS_FRONTIER_REPORT.md` (+ `scripts/_tightness_check.py`)
2. `docs/_OPEN_PAIRING7_REPORT.md` (+ `scripts/_pairing7_search.py`)
3. `docs/_OPEN_ES_GLOBAL_REPORT.md` (+ `scripts/_es_global_check.py`)

Verify against the normative `docs/PROOF_TSS_DEFENDER_ZONES.md`
(round-6-confirmed) and `docs/proof_parts/ES_POTENTIAL.md`.

## Priority targets

### Tightness report (highest stakes — its results will edit the doc)

- **L13+ (3/5 witness caps)**: re-derive the whole proof. Check: the
  singleton case really gives ≤4 (and that ≤4 ≤ 5 is what's claimed);
  the |G|=6 ⇒ K₄ forcing (distinctness of the four cross-pair-missing
  members; can a member miss a cross-pair AND be one of E1/E2?); the
  hex-geometric K₄ exclusion (the three-axes contradiction AND the
  common-line case — is "any window containing p1,p3 contains p2" valid
  for ALL axis configurations, including p1..p4 not consecutive?); and
  both sharpness constructions (does the C5 position's threat family
  really have NO other threats — trust but verify the script's window
  enumeration is complete, including all three axes and both colours).
- **Every absolute pin** (R5b false-WIN certificate, R6, R10 leaf-entry
  and OR-COMPLETION gadgets, R11, R15): re-derive each; check the claimed
  weakened certificate really satisfies every OTHER clause of the
  grammar (a "false WIN" that also violates a second clause is not a pin).
- **§2.1 rank trace**: check it against L9′ and D15 exactly (ranks, turn
  boundaries, Z4 legality witnesses, the claim that no transition is
  terminal). Check the generalization to arbitrary r.
- **§3.1/§3.2**: the reclassification of the doc's existing sharpness
  sentences (virgin radius "sharp at E=7", L9′ chain sharpness) — confirm
  or refute that the existing doc sentences overclaim relative to what a
  full-zone pin requires. This drives edits to the normative doc, so be
  precise about exactly which sentence needs which weakening.
- **§8 T5/L10 pins**: verify coordinates and the "not in r3 / not
  A-touched" claims.

### Pairing report

- The equality-forcing argument (3N ≤ Σ(7−δ) ≤ 6P ≤ 3N) — check each
  inequality and the conclusions drawn (every cell matched, unit pairs,
  exactly one pair per window).
- The rigidity derivation x_s = x_{s+6} and the claim it justifies the
  finite search space (does the search space actually cover all pairings
  periodic under Λ, or only phase-per-line-cycle ones — and is that
  restriction justified by the rigidity?).
- The minimality proof (6 | o for all three axes; the Z_6 order argument).
- The window-covering proof's independence from the script (re-derive by
  hand for one line of each axis).

### ES report

- Lemma 1 (clean escape): the h-monotone argument — check window h-ranges
  for all three axes and the shared-window exclusion for (4,4).
- Lemma 2: the algebra (λ−1)(1+λ)S = 2S and the S ≤ min{δ, X′−δ} step —
  is "a cell danger is a subsum of the remaining potential" correct when
  windows through the cell overlap?
- Theorem 2's three-case window exhaustion and the sharpness example
  (six placements — check it doesn't accidentally refute Theorem 2 itself,
  i.e., the example's Φ conditions).
- Theorem 3: the case analysis (occupied x+v; W* handling; the (3/2)²
  compounding — does Lemma 2 apply twice legitimately, i.e., is the
  second application's "arbitrary placement" the W*-kill and the greedy
  placement F-greedy?). Check the threshold arithmetic.
- Theorem 4: the König argument (finite branching justified?), the 32h
  radius arithmetic, and (30).
- Propositions 2 and 3: the counting; (25)→(26); the w-account chain (28).
- Script audit: confirm the exhaustive greedy enumeration really quantifies
  over ALL tie-breakings (state dedup argument), the exact ℚ(√3)
  comparison logic, and that the engine-reachable check re-verifies the
  full tree rather than sampling.

## Report format

Write `docs/_REVIEW_TPE_ROUND1.md`: verdict table per item (CONFIRMED /
REFUTED / REPAIR with exact repair text), script-audit findings, and a
final per-document line: INSTALLABLE-AS-IS / INSTALLABLE-WITH-REPAIRS /
NOT-SOUND. If you need your own check script, write
`scripts/_tpe_review_check.py`. Dry, plain prose.

## Hard constraints

- Do NOT edit any existing file. Outputs: the report + optional script.
- No git state mutation. No network.
