# SPEC: Fold the open-problems campaign into the normative proof doc

You are installing the round-1-confirmed results of four proof reports into
`docs/PROOF_TSS_DEFENDER_ZONES.md` (the normative document), applying the
two hostile reviews' repairs VERBATIM. You are the author of record for this
revision; a fresh confirmation round will check your work, so precision
beats speed.

## Inputs (read all)

- `docs/PROOF_TSS_DEFENDER_ZONES.md` — the document you are revising.
- `docs/_OPEN_FHW_REPORT.md` + `docs/_REVIEW_FHW_ROUND1.md` (repairs R1–R9).
- `docs/_TIGHTNESS_FRONTIER_REPORT.md` + `docs/_OPEN_PAIRING7_REPORT.md` +
  `docs/_OPEN_ES_GLOBAL_REPORT.md` + `docs/_REVIEW_TPE_ROUND1.md`.

Where a review prescribes exact replacement text, use it verbatim. Where it
prescribes a scope downgrade (R5b, R6, R15, R3, R13), install ONLY the
downgraded claim.

## Edits to `docs/PROOF_TSS_DEFENDER_ZONES.md`

1. **New section: the forced-hit gate calculus.** Install D19, D20, D20a,
   D21, L15, L16, L17, T11, T11.1 from the FHW report with ALL of R1–R9
   applied (scope sentence, node taxonomy, F terminology, exact-labels
   requirement replacing the larger-label allowance, which-clocks-debit
   clause, D13/T7 augmented clause, L16(2) as the defined statistic, L17
   repaired sentences, completed T11.1 premise/proof, terminology fixes).
   Include: the §1.1 race counterexample (it is load-bearing — it explains
   why the debit is conditional), a compressed statement of §1.2–1.4 with
   pointers to the report for full traces, the §4.1/§4.2 sharpness results,
   and the verifier procedure. Place after §6 as a new numbered section or
   as §6-adjacent — your call; renumber nothing that exists.
2. **L13 → 3/5.** Replace L13 by the L13+ statement and proof (tightness
   report §4.1, with the review §1.2 explicit singleton inference added);
   change D9's LOSS witness cap to `|T| <= 5 for b = 2`; update §9's
   sentence to `at most 3 witness windows at b = 1 and 5 at b = 2`.
3. **Overclaim corrections** (review §1.3–1.4, exact text): replace the
   sharpness sentence after L9′; replace the sharpness sentence after L12;
   in §12 item 2 replace `including the sharp virgin radius` by
   `including the fixed-window virgin radius` and note full-union sharpness
   remains open.
4. **Pairing threshold.** In §8, add a new theorem (suggest `T8.2
   (threshold pairing at k = 7)`) : existence via the explicit index-12
   construction (period lattice, the six pairs, the ℤ₂×ℤ₆ endpoint
   bijection, one-phase-per-line coverage), the rigidity lemma with the
   review's repaired wording (periodicity yields exact coverage; recurrence
   is then pointwise), and index-minimality. Cite
   `docs/_OPEN_PAIRING7_REPORT.md` and the search script for the exhaustive
   verification. Update §8's closing "Consequence for the program" to note
   the threshold is now exactly characterized: k ≥ 2g+1 is tight (hex k=7
   pairing exists; k=6 has none).
5. **ES layer update.** Create `docs/proof_parts/ES_GLOBAL_BOUNDARY.md`
   containing the full ES report content with the review's repairs applied
   (the two wording replacements in §3.2/§3.3 of the review, the
   `GREEDY-REFUTED` scope statements). Then update §10 of the main doc:
   replace the "Honest boundary" bullet with the strengthened boundary —
   greedy-refutation (all tie-breakings, exhaustively verified,
   engine-reachable), universal clean escape (no Φ<1 renewal possible),
   static pairing and static damping no-gos, the extended horizons
   (Theorem 2 five placements; Theorem 3 thresholds 1, 2/3, 4/9), the
   finite König reduction, and the five named GAPs. Keep §10's
   bullet-summary style; point to the new proof_parts file as the full
   text.
6. **§12 open problems — rewrite.** Item 1 → the R5 "Partially resolved --
   protected exact-copy F+H_W" text verbatim. Item 3 → resolved: pairing
   exists at the threshold (state it, cite T8.2). Keep item 4
   (formalization). Add the two remaining tightness frontiers as explicit
   open items (uniform 8(B−1) wrapper; full-union virgin radius), and the
   ES GAP-RAW as an open item. 
7. **New section: the limit map.** Add a compact section (suggest §12.5 or
   a new §12a "Tightness frontier") containing the frontier table from the
   tightness report WITH the review's corrected rows (R5b combined
   enforcement; R6 combined contract; R15 relative/syntactic; R3/R13
   arithmetic-attained-only) and the absolute-vs-relative pin distinction
   paragraph. This is the document's record of which constants are final.
8. **§13 review log.** Append a Round 7 entry: the four-report campaign,
   the two hostile reviews, verdicts, and the one refuted classification.
9. **§9 quantitative summary**: apply the R5 replacement bullet for the
   augmented-certificate case; update the witness-cap sentence (item 2
   above).

## Edits to the report and script files

10. Apply the reviews' prescribed edits to the four report files themselves
    (they are the source documents of record): FHW R1–R9 text repairs in
    `_OPEN_FHW_REPORT.md`; the tightness report's heading/paragraph/table
    replacements from review §§1.3–1.5 and the frontier-row changes; the
    pairing report lines 39–41 replacement; the ES report wording
    replacements.
11. Apply the checker repairs: the `exact_family` strengthening +
    `len(attacker)==20` + `__debug__` guard in `_tightness_check.py`;
    `assert states == 419` + `__debug__` guard in `_pairing7_search.py`;
    the incident-count asserts, translated state-set bijection, and
    `__debug__` guard in `_es_global_check.py` (exact code in the review).
    Rerun all three scripts plus `_fhw_review_check.py` and confirm exit 0.

## Constraints

- The normative doc uses CRLF line endings — preserve them (edit in place;
  do not rewrite the file with LF).
- Do not renumber or delete any existing definition/lemma/theorem tag.
- Stable tags for new material: D19–D21, L15–L17, T8.2, T11, T11.1.
- Prose style: dry, plain, no filler; match the document's voice.
- Do NOT commit; leave everything as working-tree edits.
- When done, print a manifest: every file touched and a one-line summary of
  each edit block.
