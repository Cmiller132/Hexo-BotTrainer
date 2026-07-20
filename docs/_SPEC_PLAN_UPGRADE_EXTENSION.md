# Spec — extend PLAN_TSS_SOLVER_UPGRADES.md with the round-5/6 tightenings

## Goal

Append a new catalog section to `docs/PLAN_TSS_SOLVER_UPGRADES.md` mapping
the round-5/6 revision of `docs/PROOF_TSS_DEFENDER_ZONES.md` onto the built
solver, in the plan's established per-item style (what / where in code /
soundness status tags / phasing / testing traps). Plan only — no
implementation, no code edits.

Read first: the current plan doc in full (its conventions, existing U1–U11,
status tags, and §9-style testing traps are binding); the revised proof doc
(the round-5 form: D9–D18, L9′, L11–L14, T3/T4/T6/T9/T10, §13 rounds 5–6);
`docs/_T3_TIGHTENINGS_REVIEW_ROUND1.md` for the repair rationale (the +1
transition counterexamples, the mhs ≤ b scope defect, the L9′ necessity).

## New catalog items (number them U12 onward)

- **U12 — ranked zone generator and verifier zone.** Replace the U1/U2 zone
  formula with T4's ranked zone Z_dir ∪ Z_seed ∪ Z_touch ∪ Z_virgin under
  labels (Z2)/(Z4)/(Z5′). The mandatory hitting term is gone — hitting
  cells demote to move-ordering heuristic. Spell out exactly which U2
  verifier-checklist rows change (the Z1 row, the band-radius row, the
  Prot/core rows under compressed obligations) and what the verifier now
  enumerates per node (roles, ranks, exposures). Soundness: revised T3/T4,
  round-6 confirmed.
- **U13 — local budget labelling.** B(N) node labels with the D14 verifier
  inequalities; certificates carry local budgets instead of deriving one
  global D_N. State the U4 implication precisely: R1's counterexample
  (local-D_N fragment omissions invalid under a larger flattened global T)
  was against *unprincipled* local budgets; D14/L11 hereditary budgets are
  the sound replacement. Specify what the two-stamp cache rule can now key
  on (hereditary B admissibility at the reuse site) and what it still must
  guard (the stamp comparison direction). If any part of the cache-rule
  relaxation is not directly licensed by L11, tag that part [H] or
  [needs-derivation] per plan convention — do not overclaim.
- **U14 — sparse LOSS witnesses.** Builder emits ≤ 3 / ≤ 6 witness families
  (L13 constructive choice); verifier enforces the size cap and re-derives
  τ > b. Note the certificate-size and Prot-size payoff downstream of U12.
- **U15 — kernel T6 at forced nodes.** K_b via residual transversal number
  at internal AND nodes with verifier-checked mhs ≤ b and ¬own_win_now; no
  core term. Note the mhs > b guard (node must be a LOSS leaf or keep its
  searched subtree — K_b would be empty). Interaction with U3
  staple-by-theorem: state whether the kernel changes the staple's
  per-omitted-move obligations.
- **U16 (backlog) — exact ranks and exposures.** Per-role r_N(y) and
  per-window E^D_N(W) exact recurrences vs the uniform B(N) fallback:
  costs (per-node bookkeeping over roles/windows), payoffs (smaller seed
  bands, smaller completion guards), and a recommended default (uniform B
  first, exact clocks behind a flag).
- **U17 (backlog) — branch-indexed substitution envelopes (D17/T9).**
  Largest verifier complexity; transition-inclusive +1 budgets are
  mandatory (cite the two counterexamples); non-circular fallback S(N).
  Recommend gating and what evidence would justify building it.
- **U18 (note) — certificate DAGs (D18/T10).** What DAG sharing would buy
  the cache/transposition layer and what the label/clock consistency check
  costs; note whether the current tree-only builder loses anything today.

## Amendments to existing items

Mark, in place and clearly attributed to this extension (one-line inserts
of the form "Amended by §<new section>: …"), the items whose text is now
partially superseded: U1 (zone formula), U2 (checklist rows), U4 (the
constraint note about local budgets). Do not rewrite their bodies; do not
touch any other existing text.

## Boundaries and definition of done

- Modify ONLY `docs/PLAN_TSS_SOLVER_UPGRADES.md`. No git. No code.
- The new section carries a provenance line (round-5/6 revision, date
  2026-07-14, pointer to the proof doc §13 and the two underscore review
  files).
- Recommended phasing at the end: which of U12–U15 land first and why,
  relative to the already-implemented P0–P3 flags.
- Final message: one line per new U-item, the list of amended existing
  items, and your phasing recommendation.
