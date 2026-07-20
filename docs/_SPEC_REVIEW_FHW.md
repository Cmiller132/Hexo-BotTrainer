# SPEC: Hostile review of the F+H_W report (round 1)

You are a hostile reviewer. Your job is to BREAK `docs/_OPEN_FHW_REPORT.md`
— a proposed conditional extension (D19–D21, L15–L17, T11, T11.1) of the
normative `docs/PROOF_TSS_DEFENDER_ZONES.md` (round-6-confirmed revision).
Anything you cannot break after genuine effort, confirm. Prior rounds of
this program have found fatal holes in polished-looking drafts; assume this
one has some too until proven otherwise.

## Required reading

- `docs/PROOF_TSS_DEFENDER_ZONES.md` in full — you must verify the new
  material against the EXACT existing machinery (D4 terminal ordering, D9
  grammar and adaptive LOSS contracts, D10 roles, D11 zones, D12 coupling
  and (MI), D14–D18, L9′, L11–L14, T3/T4/T6/T9/T10, §13 review log).
- `docs/_OPEN_FHW_REPORT.md` — the document under review.
- `docs/_SPEC_OPEN_FHW.md` — what it was asked to do.

## Mandatory work items

1. **Machine-check every coordinate construction.** The report shipped NO
   verification script — all its positions are hand-built. Write
   `scripts/_fhw_review_check.py` (offline, stdlib only) that verifies:
   - §1.1: full history legality (radius-8 at each placement), no
     premature completion by either colour at every prefix, the exact
     threat-empty family F and tau(F)=2, kernel {a,b}, disjointness from
     W, and the defender's legal two-fill win.
   - §1.2: the fragment's window counts, the two ghost threats, kernel
     disjointness, the real defender's win, and the LOSS-leaf family's
     pairwise-disjoint empties with tau=3.
   - §1.3 and §1.4: the mask/count and distance claims (real-legal vs
     ghost-illegal at exactly 8 vs 9).
   - §4.1: no complete window, D-alive max count 3, exact singleton threat,
     old exposure 3 vs new Q=2 on the displayed certificate, and the real
     line attaining count 5 in W.
   - §4.2: Q(W')=3 attained, and the false value 2 without the
     dual-purpose indicator.
   Any mismatch is a finding. Print one line per check.

2. **Attack the proofs.** Specific attack surfaces (add your own):
   - L15: the dichotomy's case 2 — does "A completes by p(Q)+b+2" survive
     the defender using his b-1 remaining placements to BLOCK differently
     than assumed, or to win elsewhere? Is the L1/T1 legality citation for
     the surviving window's empties airtight when the empties were never
     protected before the gate?
   - D19 two-phase Prot at gates: checkpoint roles discharge after the
     entry mask check — can a real-only stone slip into a checkpoint cell
     BETWEEN the last ordinary check and gate entry? Trace the coupling
     step order precisely against T3's A1–A3.
   - The claim "X ∩ Prot⁻(Q) = ∅ is maintained and proved in L17" — is the
     L17 argument actually complete for checkpoint carriers, including
     carriers that are ghost-LEGAL (not just ghost-illegal chains)?
   - D20 gate clause `max over copied kernel children` with NO +1: at a
     gate the defender DOES place a stone (the copied hit). For role ranks
     f this is claimed safe because the stone is shared (no X). Verify no
     lemma downstream uses f as a bound on DEFENDER PLACEMENTS rather than
     X-opportunities (the report itself warns about this — check the
     warning is actually sufficient everywhere f is used).
   - D20a gate clause: the child-by-child max vs the report's own remark
     about branch-combining; verify the recurrence is well-founded on DAGs
     (T10 unfolding with gate labels).
   - Escape deadline p(Q)+b+2 folded into the global horizon: does T11's
     induction correctly handle a gate whose escape deadline EXCEEDS the
     old declared resolution of its own subtree? Interaction with D9's
     path clock and with T9 envelopes.
   - L17's completion argument: the case split (no dismissal / off-kernel
     first fill / ordinary first fill; touched vs virgin ghost W) — is it
     exhaustive? In particular a first real-only fill that occurs AT a
     gate via a copied hit in W (charged by H_W): confirm the counting
     actually covers it in every case.
   - T11.1: is retaining full undebited D17 tests really sufficient — or
     can a substitution INSIDE a subtree containing gates interact with
     checkpoint roles in a way D17.2's union misses?
   - The §1.x counterexamples: are they actually counterexamples to the
     precise rules they claim to refute (not straw men)?

3. **Consistency with the normative doc.** Numbering collisions (D19–D21,
   L15–L17, T11 — check nothing with those tags exists), terminology
   drift, and whether the claimed "ready to install" text actually
   composes with D13/T7's search-superset story and §9's quantitative
   claims.

## Report format

Write `docs/_REVIEW_FHW_ROUND1.md`:
- A verdict table: every definition/lemma/theorem/counterexample/sharpness
  item → CONFIRMED / REFUTED (with the break, concretely) / REPAIR
  (with the exact minimal repair text, install-ready).
- Machine-check results pasted verbatim.
- A final line: INSTALLABLE-AS-IS / INSTALLABLE-WITH-REPAIRS / NOT-SOUND.
- Dry, plain prose. Mark anything you could not decide as UNDECIDED with
  the sticking point stated precisely.

## Hard constraints

- Do NOT edit any existing file. Outputs: the report + the check script.
- No git state mutation. No network.
