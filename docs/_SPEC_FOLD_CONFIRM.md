# SPEC: Confirmation round for the campaign fold (Round 8)

Fresh-eyes confirmation that `docs/_SPEC_FOLD_CAMPAIGN.md` was executed
correctly on `docs/PROOF_TSS_DEFENDER_ZONES.md` and the report/script files.
You did not perform the fold; treat the folder as a fallible author.

## Inputs

- `docs/_SPEC_FOLD_CAMPAIGN.md` — what was supposed to happen (11 numbered
  edit blocks).
- `docs/_REVIEW_FHW_ROUND1.md` (repairs R1–R9) and
  `docs/_REVIEW_TPE_ROUND1.md` — the prescribed repair texts.
- The four campaign reports and `docs/proof_parts/ES_GLOBAL_BOUNDARY.md`.
- `docs/PROOF_TSS_DEFENDER_ZONES.md` — the revised normative document.

## Checks

1. For each of the 11 edit blocks in the fold spec: APPLIED-CORRECTLY or
   DEFECT (with the exact fix). Where the reviews prescribed verbatim text
   (R1–R9; the TPE replacement sentences, headings, and table rows),
   compare word-for-word.
2. Scope downgrades: confirm the installed claims are the DOWNGRADED ones
   (R5b combined enforcement; R6 combined contract; R15 relative/syntactic;
   R3/R13 arithmetic-attained-only; L9′ and L12 sharpness sentences; §12
   item 2 fixed-window wording).
3. Internal consistency of the enlarged doc: every tag referenced anywhere
   (D1–D21, L1–L17, T1–T11, F1, MI, Z-labels) is defined exactly once; no
   pre-existing tag was deleted or renumbered; §6a's cross-references into
   D9–D18 and T3/T4/T6/T9/T10 are accurate; §12/§12a/§13 agree with each
   other and with §6a/§8/§10 (no contradictory sharpness or open/resolved
   claims anywhere in the doc).
4. Collateral damage: confirm §§0–5, 6, 7, 9, 11 still contain their
   pre-fold content (the fold should only have touched the places its spec
   names; flag any unexplained change or truncation).
5. `docs/proof_parts/ES_GLOBAL_BOUNDARY.md` matches the repaired ES report.
6. Rerun all four checkers (`scripts/_tightness_check.py`,
   `_pairing7_search.py`, `_es_global_check.py`, `_fhw_review_check.py`);
   all must exit 0, and the three assertion-based ones must refuse
   `python -O`.
7. Line endings: the normative doc must be pure CRLF, UTF-8 without BOM.

## Report

Write `docs/_CAMPAIGN_FOLD_CONFIRMATION.md`: per-check verdicts, any
defects with exact fixes, and a final line PASS or FAIL. If you find only
mechanical defects (wording, line endings), fix them directly and note the
fixes in the report; for anything substantive, report only.

## Constraints

- You may edit files ONLY to fix mechanical defects found in check 1–7.
- No git state mutation. No network. Dry, plain prose.
