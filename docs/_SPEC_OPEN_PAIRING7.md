# SPEC: Attack open problem 3 — pairing existence at the threshold (k=7, g=3)

You are attempting to settle the THIRD open problem in
`docs/PROOF_TSS_DEFENDER_ZONES.md` §12: does a pairing strategy exist for
7-in-a-row on the hex grid?

## Required reading

- `docs/PROOF_TSS_DEFENDER_ZONES.md` §8 (T8 and Corollary T8.1) — the
  density argument. For k-in-a-row with g = 3 axis directions, a pairing
  needs per-line matched density 2/(k−1) per axis; at k = 7 the total
  required density is exactly 1: every cell matched, zero slack.
- §1–2 for the board model (hex lattice, three axes, windows = k
  consecutive cells on a line).

## The problem, precisely

A pairing is a partial matching M on cells (each cell in at most one pair)
such that every length-7 window on every axis contains both cells of some
pair. T8.1 shows this forces k ≥ 2g+1 = 7 — the density bound is met with
equality at k = 7, so existence is neither implied nor excluded. The
classical analogy: on the square grid (g = 4) the threshold k = 9 DOES have
the well-known pairing draw for 9-in-a-row, so threshold existence is
plausible; but equality forces extreme rigidity (every cell matched, every
line's pairs spaced with zero slack), which may instead support a
non-existence proof.

## Suggested attack

1. Derive the rigidity constraints at equality: on each line, a pair at
   axis-distance δ covers exactly 7−δ window starts; zero slack forces the
   covering to partition the starts exactly — derive what δ values and
   spacings are possible, and how the three axes' matchings must interleave
   on the shared cells (each cell belongs to one pair on ONE axis, but lies
   on lines of all three axes — the other two axes must cover that cell's
   windows using other cells).
2. Search: translation-invariant (periodic) pairings with a period lattice
   can be enumerated exhaustively over fundamental domains (direct
   backtracking or SAT-style encoding). Write a scratch script
   `scripts/_pairing7_search.py` (offline; use the python available on
   PATH; if none is usable, say so in the report and proceed with theory
   only). Verify any found construction exhaustively over a domain large
   enough to certify all window positions by periodicity.
3. If the search is empty for all small periods, convert the rigidity
   constraints into a non-existence proof if you can (the T8-style density
   argument with second-order/boundary terms, or a parity/coloring
   obstruction on the interleaving). A periodic-case-only impossibility is
   a valuable PARTIAL: state exactly what remains (non-periodic pairings).

## Definition of done — the report

Write `docs/_OPEN_PAIRING7_REPORT.md` with:

1. `VERDICT:` EXISTS / NOT-EXISTS / OPEN.
2. If EXISTS: the explicit construction (period lattice, pair list for the
   fundamental domain), the exhaustive verification result (what was
   checked, counts), and the periodicity argument that finitely many
   checks certify all windows.
3. If NOT-EXISTS: a complete proof at the rigor grade of T8.
4. If OPEN: the proven rigidity lemmas with proofs, the exact search space
   exhausted (period shapes, domain sizes, encoding), and a precise
   statement of what a resolution would require.
5. Note: this problem is of independent combinatorial interest and does
   not affect Hexo (k = 6); say nothing about Hexo consequences beyond
   that.

## Hard constraints

- Do NOT edit any existing file. Outputs: the report + the scratch script
  only.
- No git state mutation. No network.
- Prose style: dry, plain, no filler.
