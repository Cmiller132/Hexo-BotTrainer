# SPEC: Tightness frontier — pin every constant or improve it

Goal: for EVERY quantitative parameter in the revised results of
`docs/PROOF_TSS_DEFENDER_ZONES.md` (the round-6-confirmed T3⁺ revision),
establish exactly one of:

- **IMPROVED** — a strictly better value, with a complete proof; or
- **PINNED** — a matching counterexample proving the current value cannot
  be improved within the stated framework (a concrete position/certificate
  where the smaller value admits a false WIN or breaks the cited lemma),
  verified move-by-move; or
- **OPEN** — neither achieved; state precisely what a pin or an
  improvement would require.

The output is the paper's "limit map": when every row is IMPROVED-to-PINNED
or PINNED, the theorem is at the provable limit of its framework.

## Required reading

`docs/PROOF_TSS_DEFENDER_ZONES.md` in full — D9–D18, L9′, L11–L14, T3, T4,
T6, T9, T10, §9, §12, §13. Several sharpness facts are already recorded;
your first job is to inventory what is already pinned (cite the location)
so you only attack genuine gaps.

## Parameter inventory (attack each; add any I missed)

1. **Role/band radius 8(r−1), uniformly 8(B−1)** (D15/L9′/T4). Is the −1
   the end? Produce a position where a dismissed defender cell at distance
   exactly 8(B−1) from a protected cell breaks soundness under radius
   8(B−2)·(or 8(B−1)−1) — i.e. pin the radius — or prove a further
   reduction (e.g. 8(B−1)−c for some c ≥ 1, or a non-uniform per-role
   radius below 8(r−1)).
2. **Virgin seed radius 8(E^D−6)** (D16/T4). §12.2 records sharpness at
   E^D = 7. Pin or improve for general E^D (is the linear coefficient 8
   necessary for large E^D, or does the defender's need to place 6 stones
   IN the window allow a smaller slope, e.g. because walk stones and fill
   stones interleave?).
3. **Touched-window guard condition cnt_D + E^D(W) ≥ 6** (D16/Z_touch).
   Can the threshold account for b = 2 turn structure or window overlap
   (e.g. parity), or is there a counterexample at exactly
   cnt_D + E^D = 6?
4. **LOSS witness caps 3 (b=1) / 6 (b=2)** (L13). The doc cites sharpness
   via triangle and K₄ — verify both directions are actually recorded
   (upper bound proof AND matching example for each b); fill whichever is
   missing.
5. **T6 kernel scope mhs ≤ b** and the explicit ¬own_win_now premise.
   Pin: counterexample at mhs = b+1 (kernel pruning unsound there), and
   an example showing ¬own_win_now cannot be dropped. Improve: any sound
   relaxation of the scope.
6. **LOSS contract bound τ(𝒯) > b** (D9). Pin or improve.
7. **Substitution envelope B̂ = 1 + B(C_s)** (D17/T9). The +1 is pinned by
   the C2/C3 counterexamples if they are recorded in the doc — verify and
   cite; otherwise reconstruct and record them.
8. **Local budget recurrence** (D14: OR-COMPLETION/WIN = 0, LOSS = b,
   OR = child, AND = 1 + max). Is `1 + max` over children improvable
   (e.g. to a path-dependent value below the max) or pinned?
9. **Legality radius 8 as it enters the accounting** (D4 is a game rule,
   not attackable — but check no derived bound wastes a factor over it,
   e.g. chain-step distance in L9′).
10. **Sparse witness deadline/rank constants in D15** (rank-0, OR-COMPLETION
    handling) — any slack?

## Definition of done — the report

Write `docs/_TIGHTNESS_FRONTIER_REPORT.md` with:

1. A frontier table: Parameter | Current | Result (IMPROVED/PINNED/OPEN) |
   Where proven/pinned (section of this report or existing doc cite).
2. For each IMPROVED row: the full proof, house style, install-ready.
3. For each PINNED row: the counterexample, verified move-by-move
   (positions, placements, counts, which clause of which lemma/theorem
   fails at the smaller value).
4. For each OPEN row: what a resolution requires.
5. A closing paragraph: which rows, if any, could change under a DIFFERENT
   proof framework (i.e. pins are framework-relative — say for each pin
   whether the counterexample breaks the THEOREM (absolute pin: smaller
   value admits a false WIN) or only the current PROOF (relative pin)).
   This distinction is the report's most important deliverable.

## Hard constraints

- Do NOT edit any existing file. Outputs: the report + optionally one
  scratch script `scripts/_tightness_check.py`.
- No git state mutation. No network.
- Prose style: dry, plain, no filler.
