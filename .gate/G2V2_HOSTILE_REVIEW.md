# G2 v2 — FhwGateV1 verifier ACCEPT path: independent hostile review

Reviewer: independent hostile lane (Claude, Opus). Did NOT author the code.
Target: commit `fdf2b97d` on `claude/g2-cert`
(`tss_verify_group2.rs` +~2000-line accept path; `tss_verify.rs` D6 gate re-sort).
Authorities: `DESIGN_G2_CERT_EXTENSION.md` (§2.2/§2.3/§2.4/§3.3/§3.5) +
`DESIGN_AMENDMENT_R1_R2.md` + `HOSTILE_REVIEW_1.md`; soundness bar
`PROOF_TSS_DEFENDER_ZONES.md` / FHW-T3-R.
Date: 2026-07-21.

## VERDICT: **SOUND** (Exact/FrontierCovered accept class as implemented)

- CRITICAL findings (false-accept demonstrated or plausible): **0**
- REQUIRED findings (unenforced reachable hypothesis): **0**
- ADVISORY findings: **3** (all enablement-scoping / provenance; none is a
  soundness defect in the shipped accept class)
- Tests: baseline lib unittest binary **241/0/37** → **243/0/37** with the two
  added hostile tests (both REJECT as required). Full serialized invocation is
  the same lib binary that carries the entire gate accept path; other binaries
  do not touch it and were untouched by this change.

I could not construct any certificate accepted by this verifier at a position
not won by the claimant. The design itself was already hostile-reviewed to
SOUND-WITH-REQUIRED-CHANGES (R1, R2); this review confirms **both required
changes are implemented in the code** and that the code faithfully realizes the
reviewed spec for the Exact/FC subset, with the `NonFrontierCovered` branch
fail-closed before any grant.

---

## 1. Theorem license for the shipped accept class (attack program §5)

The accept path grants only when every `d in K` maps to a representative `s`
with edge class **Exact** (`d==s`) or **FrontierCovered**
(`B_8(d) ⊆ Lambda(P_Q+s)`). Omitting `Legal\K` and omitting `K\R` is licensed
by FHW-T3-R's reductive case in which the **legality-frontier charges (RC/WC)
are vacuous (`epsilon=0`)**, so the reduction rests only on the C1 occupation
and C2 window-completion channels, both re-derived verifier-side. Every
hypothesis of that case is enforced by code:

| FHW-T3-R hypothesis (Exact/FC case) | enforced at |
|---|---|
| post-opening, defender-to-move, nonterminal, not own_win_now(P_Q) | `reconstruct_gate` 795-802 (+ root post-opening 1068; own_win_now over-approx **and** analyzer, both rejectors) |
| `b∈{1,2}`; `H_Q` canonical, count-bounded, each a real A-threat (A-alive, A-count≥4, 1–2 empties) | 806-846 |
| **exact** `transversal_number(F_Q)==b` | 848 (`transversal_exact`, cap 2 exact for b≤2) |
| `K={d∈Legal: transversal(F_Q\d)≤b-1}`, nonempty; every applied `d∈K` nonterminal | 852-875 |
| `R⊆K`, canonical, one nonterminal exact child `C_s` each, recursively verified | 884-904; `replay_node` 1444-1451 |
| map domain `==K` exactly; `phi(d)∈R`; `phi(s)=s` as an Exact self-edge | 909-964 |
| edge class **recomputed** from geometry; stored must match; **NonFC rejects** | 929-947 |
| Exact/FC ⇒ `epsilon=0`, and carrier avoidance `d≠y` mandatory | `classify_role` 595-601 |
| every charging window (`d∈W`) is in demands and its completion guard holds | direct-18 seed 1927-1940; Cartesian `K×demands` 2243-2251; `classify_window` guard 711-720, failure ⇒ None |
| `B(Q)=1+max_s B(C_s) ≥ b`; checkpoint roles clock-0 at gate; paired `f_cut` before max | 1767-1820 |
| escape `= p(Q)+b+2`, byte-equal, and `≤ semantic_horizon` (R1); folded into derived T | 966-973, 1054, 1061 |

**Completeness of the C2 channel for Exact/FC (the load-bearing subtlety).**
For an Exact/FC edge, `kappa≠0` only on the `d∈W` row (`ExactOrFcDirect`). Every
window with `d∈W` is one of the 18 length-6 windows through `d`, which are
exactly the `SOURCE_DIRECT18` seeds pushed for every `d∈K` (1927-1940). Hence
every window that can charge is in `demands(Q)`, and the Cartesian exactness
check forces a recomputed, guard-checked row for each. A window `d∉W` has
`kappa=0`/guard `NotApplicable`, so a demand miss there cannot hide a charge.
Independently, the load-bearing clock values come from `window_clock`
(recomputed by recursion, memoized), **not** from the stored rows or the demand
set — so demand-completeness is a belt-and-suspenders auditing layer, not the
sole soundness support. This closes the author's flagged attack #1 for the
Exact/FC class.

## 2. Attacks attempted and why they FAILED (credibility ledger)

1. **NonFC reaching accept via any branch.** Closed: `reconstruct_gate`
   recomputes the class (929-941); `recomputed != stored ⇒ None` (942); and
   `recomputed == NonFrontierCovered ⇒ None` (945-947). No `unwrap_or`, default
   enum arm, or ordering path bypasses it. The RC/WC/charged classifiers exist
   and are unit-tested but are never on a granting path (ghost passed as `None`
   on accept). Confirmed by `gate_with_nonfrontiercovered_edge_rejects` and my
   added `hostile_omitted_kernel_reply_bare_fc_label_rejects`.
2. **Geometrically non-FC classified FC.** `frontier_covered` is a faithful
   217-cell `B_8(d) ⊆ Lambda` test; any out-of-range ball cell fails FC
   (conservative); `Lambda` is built from `occupied_cells()` of the exact ghost
   `P_Q+s`. My added test asserts the sparse-board coupling `(5,1)→(4,1)`
   recomputes **non-FC** and rejects, exercising the only end-to-end `d≠s` path.
3. **Cartesian omission / duplicate / reorder.** Row-set is forced to
   `K×demands` by length + membership + `seen_*` de-dup (2217-2251);
   reconstruction rejects noncanonical/duplicate threats, reps, and map rows
   (826-829, 886-888, 918-920). `window_domain_short`/`duplicate_map_entry`
   mutations reject.
4. **(Q_cut,E_full) split at a non-gate node.** The pair coincides on any node
   with no gate descendant; a gate's asymmetry propagates upward correctly
   (Universal takes `(max_q+1,max_e+1)`). `Q_cut ≤ E_full ≤ B` is enforced on
   every evaluated pair (1876). Zones consume `Q_cut` per design.
5. **D6-remapped cert changing which checks bind.** The verifier re-derives
   every semantic quantity from the transformed board; the canonical re-sort is
   a remap-correctness fix, and any drift shows as a digest mismatch across the
   12 transforms (fail-closed). All 12 images verify
   (`gate_certificate_is_d6_invariant`).
6. **Independence (v1 review point / attack §4).** The accept path imports no
   `tss_solver` symbol (the only such import is in the `#[cfg(test)]` module),
   and `verify_group2_impl` never calls `finder_fill_gate_rows` (test/finder
   builder only). The verifier's `derive_gate_*` helpers are shared with the
   finder's row-filler, but the verify path recomputes them from the replayed
   board; the digest detects tamper, and my hand-audit of those helpers against
   §3.3 discharges the correlated-bug risk for the Exact/FC class (see A2).
7. **R1 / R2.** Per-gate `escape_resolution_ply ≤ semantic_horizon` over ALL
   nodes (1059-1064, conservative) plus derived-T fold; Opening root rejects
   (1068). Both amendment tests present and green.
8. **Reply landing on a carrier / kernel spoofing.** `classify_role` returns
   `None` on `d==y`; a legal-but-non-kernel map reply rejects at the
   `K`-membership check — my added `hostile_map_reply_outside_kernel_rejects`.

## 3. ADVISORY findings (enablement-scoping; not accept-class defects)

- **A1 — the FC (`d≠s`) accept branch has no end-to-end POSITIVE fixture.** The
  shipped positive fixture is an all-Exact (`R==K`) gate; the FC predicate and
  classifier are unit-tested and the reconstruction recomputes FC, but no cert
  is accepted through a genuine `B_8(d)⊆Lambda` coupling in the suite (a dense
  board is needed; the report concedes this in §7). My added test drives the
  `d≠s` path only on the reject side. **Discharge for enablement:** before the
  solver is allowed to emit FC (`R⊊K`) gates consumed by this verifier, add one
  FC positive fixture (dense board, `R⊊K`, at least one non-self Exact-or-FC
  edge accepted) confirming the coupling accepts and its D6 images verify; OR
  restrict emission to Exact edges (`R==K`) until that fixture exists. Reason:
  "never ship an accept branch with no positive test" — the Exact branch is
  fixture-tested end-to-end; the FC branch is not.

- **A2 — shared derivation (`finder_fill_gate_rows` ↔ verifier).** Both use
  `derive_gate_role_row`/`derive_gate_window_row`/`window_clock`. The Merkle
  digest catches drift/tamper, not a correlated bug in the shared helper. This
  review re-derived the Exact/FC rows by hand against §3.3 (§1 table above), so
  the risk is discharged for the shipped class; it re-opens for the non-FC
  extension and should be re-reviewed there. No change required now.

- **A3 — `authority.matches_compiled()` is provenance, not protection**
  (echoes design O3, and design N3: `E_full`'s gate `b`-floor is a gratuitous
  no-op on the upper-bound side that slightly weakens the `Q_cut≤E_full` sanity
  margin). Neither is a false-accept vector; recorded so nobody later cites the
  six-field compare or the digest as the soundness guarantee — the semantic
  re-derivation is.

## 4. Tests added (both REJECT / fail as required; suite 243/0/37)

Both in `tss_verify_group2::tests`:
- `hostile_omitted_kernel_reply_bare_fc_label_rejects` — a real kernel reply
  `(5,1)` omitted from `R` and coupled to `(4,1)` with a bare `FrontierCovered`
  (and, in a variant, `Exact`) label; asserts the coupling is geometrically
  non-FC, that `finder_fill_gate_rows` cannot reconstruct it, and that the
  verifier rejects both labels. Exercises the otherwise-untested `d≠s`
  edge-class recompute end-to-end.
- `hostile_map_reply_outside_kernel_rejects` — substitutes a legal move outside
  `K` for a genuine kernel reply (map length unchanged, move legal); asserts the
  `K`-membership check rejects.

## 5. What I did NOT fully close (honest scope)

- The **proof-level** soundness of FHW-T3-R itself (that `B_8(d)⊆Lambda(P_Q+s)`
  licenses the coupling, and that A-count≥4 / `tau==b` force `Legal\K`) is taken
  from the pinned authority + `HOSTILE_REVIEW_1` (which found it SOUND under one
  proof round; note its O1 provenance caveat that the FHW companion has one
  hostile round, not six). I verified the CODE enforces every hypothesis that
  review mapped; I did not re-run a proof-level pass over FHW-T3-R.
- A live **FC positive** false-accept construction (A1) — not built here because
  it requires a dense board; flagged as the enablement gate.
- I ran the library unittest binary (which carries the whole accept path and my
  tests): 243/0/37. I did not separately re-run the unrelated integration
  binaries; this change touches only the `tss_verify_group2.rs` test module.

## 6. ENABLE RECOMMENDATION

**Enable the solver's Exact-gate emission consumed by this verifier: YES**, once
the report's harness A/B (selector-off cross-verification) is run, because the
Exact accept class is end-to-end fixture-tested, independently re-derived, and
fail-closed on every boundary I attacked.

**Enable FrontierCovered (`R⊊K`) emission: NOT YET** — gate it on A1 (one FC
positive fixture + its 12 D6 images, or restrict emission to Exact until then).
The FC classifier is sound as coded, but shipping an accept branch with no
positive end-to-end test violates the campaign's own "never accept on an
untested branch" law. `NonFrontierCovered` emission stays out (fail-closed).
