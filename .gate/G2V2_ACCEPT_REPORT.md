# G2 v2 — FhwGateV1 verifier ACCEPT path: build report

Lane: `claude/g2-cert`, worktree `.claude/worktrees/g2-cert` (from `07600433`).
Author: engineering lane (Claude). Date: 2026-07-21.
Status: **DELIVERED — the largest SOUND, fully-tested accept subset; suite green;
reject boundary documented.**

Headline: the wholesale `FhwGateV1` reject at `verify_group2_impl` is **removed**
and replaced with real §3.3 gate reconstruction and acceptance for the
**Exact / FrontierCovered gate class**. A constructed positive gate certificate
now VERIFIES end-to-end; the full mutation battery rejects; the 12 D6 images
verify with invariant digests. Gates containing any `NonFrontierCovered` edge
stay **fail-closed rejected** (documented boundary — this cleanly disposes both
known design defects).

Suite: `CARGO_TARGET_DIR=E:/cargo-targets/g2-cert cargo test --features python
--target x86_64-pc-windows-msvc -- --test-threads=1` → main lib unittest binary
**241 passed / 0 failed / 37 ignored** (baseline 229; +12 new gate tests);
**0 failed across every test binary**. Library `--lib` build: clean (0 errors).

---

## 1. What ships (accept coverage)

ACCEPT (end-to-end, positive-fixture-tested):
- **Exact / FrontierCovered forcing gates** — every real reply `d in K` maps to a
  representative `s = phi(d)` with edge class `Exact` (`d==s`) or
  `FrontierCovered` (`B_8(d) ⊆ Lambda(P_Q+s)`). This is the reductive class the
  finder-closure lane measured as common (525/704 closed gates reductive; the 12
  captured examples were all Exact self-edges + FrontierCovered edges, zero
  NonFC). It is sound because for Exact/FC edges every FHW-T3-R legality-frontier
  charge (RC/WC) is vacuous (`epsilon = 0`), so the reduction rests only on the
  C1/C2 occupation/window channels.

REJECT (fail-closed, documented §6):
- Any gate with a `NonFrontierCovered` edge (RC/WC/charged-role/N-virgin/WcFail
  and the gate-local WC demand enumeration). Reason: a sound non-FC end-to-end
  gate additionally needs the gate-local WC demand set tied to a **proven
  `B(C_s) >= 6` representative subtree**, which cannot be positively
  fixture-tested in this lane (the same subtree-provenance boundary the
  finder-closure lane documented). The RC/WC + charged-role classifiers are
  nonetheless implemented and unit-tested verifier-side (they are what a later
  non-FC extension recomputes, and they realize both design defects as
  rejections).

## 2. Pass-by-pass map (design clause → code → test)

All code in `packages/hexfield_eq/rust/src/tss_verify_group2.rs` unless noted.

| pass / clause | design | code | test |
|---|---|---|---|
| Top-level: delete wholesale gate reject, route gates, keep R1 loop | §3.1, R1 | `verify_group2_impl` (removed `:443` reject; added `check_gate_nodes` call) | `gate_certificate_reconstructs_and_verifies` |
| Preflight gate arm (schema, authority, reps/threats/map canonical sorted-unique, indegree) | §2.3, §5.2 | `preflight_structure` `FhwGateV1` arm `:605` | mutation `duplicate_map_entry`, `noncanonical` via D6 |
| Gate reconstruction: post-opening, defender-to-move, not own_win_now, `b∈{1,2}`, H_Q validity, exact `transversal==b`, `K`/`R`/`phi`, map domain `==K`, `phi(s)=s`, edge classes, escape `p+b+2` | §3.3 `:753-864`, R1/R2 | `reconstruct_gate` `:789`; `transversal_exact` `:752` | `transversal_exact_small_families`; mutations `threat_window`, `map_domain_short`, `representative_move`, `edge_class`, `escape_ply` |
| FC/GI/RC/WC predicates over B_8 ball | §3.3 `:797-850` | `ball` `:508`, `VGhost` `:523`, `frontier_covered` `:535`, `rc_pass` `:556`, `wc_pass` `:569` | `fc_and_gi_predicates`, `window_rows_non_fc_and_wc` |
| Role rows (3 classes, byte-compare) | §3.3 `:781-822` | `classify_role` `:588`; `derive_gate_role_row` `:1010`; check in `check_gate_nodes` `:2192` | `role_rows_exact_fc_and_rc_zero`; mutations `role_epsilon`, `role_child_f` |
| Window rows (9 kappa leaves, incidence bits, mandatory guards REJECT even if finder wrote Pass) | §3.3 table `:824-863` | `classify_window` `:692`; `derive_gate_window_row` `:993` | `window_rows_exact_fc_paths`, `window_rows_non_fc_and_wc`; mutations `window_kappa`, `window_child_q` |
| Demand fixed point: incoming ∪ direct-18; Cartesian `K(Q)×demands(Q)` exact; `I_FHW` restricted to ordinary origin | §2.2 `:259-261`, §3.2 | `derive_window_demands` gate seeding `:1908` + propagation `:1... (FhwGate arm)`; Cartesian check in `check_gate_nodes` `:2192` | `gate_certificate_reconstructs_and_verifies` (Cartesian assert); mutation `window_domain_short` |
| Budget/role postorder: `B(Q)>=b`, checkpoint roles, PAIRED `f_cut` (no separate marginal maxima), `escape == p+b+2 <= horizon` | §3.3 `:854-864`, R1 | `derive_budgets_and_roles` `FhwGateV1` arm `:1747` (checkpoint roles + paired `f_cut`, `r_full` full-charge) | `gate_certificate_reconstructs_and_verifies`; mutation `semantic_horizon` |
| Window-clock PAIR split `(Q_cut, E_full)`, all callers updated, non-gate pair stays equal | §3.2/§3.3 `:1070` clock | `window_clock` `:1830` (returns `(u32,u32)`), `gate_window_clock` `:1887`; callers updated | all 229 gate-free tests stay green (pair equal off gates) |
| Both digest encoders' gate branches | §2.4 `:376-487` | `enc_semantic_local` gate arm `:2488`; `gate_derived_class_payload` `:2566`; `build_digest_tables` gate arm `:3173`; `enc_role_key` `Checkpoint` tag `:... ` | `gate_certificate_is_d6_invariant` |
| D6 remap gate canonical re-sort (threats/map/roles/windows) | §2.4 canonical order | `tss_verify.rs` `d6_remap_certificate` `FhwGateV1` sort arm | `gate_certificate_is_d6_invariant` |

Supporting refactor (kept gate-free behaviour bit-identical): the role clock map
became `(r_full, f_cut)` pairs and `window_clock` became a `(Q_cut, E_full)`
pair; off gates both members coincide, so the 229 gate-free tests and their
digests are unchanged.

## 3. `tss_verify.rs` core

Untouched acceptance semantics of the legacy path; `LegacyOnly` still rejects
every extension variant (`legacy_policy_rejects_extension_certificates`,
`legacy_certificates_verify_identically_under_both_policies`). The only edit is
`d6_remap_certificate`'s gate arm gaining the canonical re-sort of the
transformed gate's `threats`/`map`/`roles`/`windows` — a remap-correctness fix
(the extension class mandates canonical order), not an acceptance-semantics
change. Record types at `:184-272` were already present and are used unchanged.

## 4. The two design-defect dispositions (fail-closed)

Both are triangle-inequality contradictions between a charged row's trigger and
its mandatory radius guard (from `.gate/G2_FINDER_CLOSURE_REPORT.md §3`). Handled
FAIL-CLOSED and demonstrated as verifier-side rejections:

1. **`NonFcEmptyNonIncidentWcFail` unrealizable with a passing guard.** WC fail ⇒
   `dist(d,W) <= 8(q-5)`, contradicting the N-virgin guard `dist(d,W) > 8(q-5)`.
   `classify_window` returns `WcPass` or `None` (reject) — never a passing
   WcFail row. Test: `defect_wc_fail_leaf_is_unrealizable_with_passing_guard`.
2. **Charged role via a ghost-illegal RC-fail carrier unrealizable.** RC fail ⇒
   `dist(d,y) <= 8k`, contradicting the D22-N guard `dist(d,y) > 8k`.
   `classify_role` rejects (`None`). Test:
   `defect_charged_via_ghost_illegal_rc_fail_is_unrealizable`.

An unrealizable claim (a cert asserting one of these combinations) is a malformed
cert and is rejected; no guard was relaxed. On the shipped Exact/FC accept path
these branches are unreachable a fortiori (non-FC edges reject at
reconstruction), so the defects cannot license an accept.

## 5. Deviations from the spec

- **Accept-path narrowing to Exact/FC edges.** The spec describes the full
  non-FC machinery; this lane ships accept for Exact/FC gates only and rejects
  non-FC gates. The non-FC classifiers (RC/WC/charged/N-virgin) ARE implemented
  and unit-tested; they are simply never on a granting path here. Rationale: a
  sound non-FC end-to-end accept requires a proven `B(C_s)>=6` subtree +
  gate-local WC demand set that is not positively fixture-testable in-lane, and
  "never accept on a gap" forbids shipping an untested accept branch.
- **Positive fixture provenance.** The end-to-end positive fixture uses hand-built
  `Win`-leaf representative subtrees (double-threat board; `finder_fill_gate_rows`
  fills the redundant role/window rows from the shared derivation, exactly as a
  finder would). The committed `structural_gates.txt` fixtures remain
  structural-only; per the DoD their passes are exercised by the direct
  reconstruction/classifier unit tests.
- **`r_full` at gates.** Tracked as the full `1+child` charge distinct from
  `f_cut = child + epsilon`; off gates they coincide (byte-identical gate-free
  digests).

## 6. What a hostile reviewer should attack first

1. **Cartesian window-demand completeness at the gate** (`check_gate_nodes` +
   `derive_window_demands` gate seeding): is `demands(Q)` exactly incoming ∪
   direct-18, and is every `(d,W)` row present and recomputed? Try omitting /
   adding a direct-18 window, or a d-specific window.
2. **Paired-clock soundness** (`window_clock`/`gate_window_clock`): confirm
   `Q_cut = max{b, max_d(kappa+Q_cut(C_s))}`, `E_full = max{b, max_d(1+E_full)}`,
   `Q_cut<=E_full<=B`, and that non-gate pairs stay equal.
3. **`f_cut` pairing before the max** (`derive_budgets_and_roles` gate arm):
   verify no separate marginal maxima; `f_cut(Q,rho)=f_cut(C_s,rho)+max_d eps`.
4. **Shared-derivation risk**: `finder_fill_gate_rows` and the verifier share the
   row derivations; the digest detects drift/tamper, not correlated bugs. The
   independent hostile review should re-derive one gate by hand.
5. **The Exact/FC soundness claim itself**: is omitting `Legal\K` sound for an
   Exact/FC gate given only the reconstructed premises (exact `tau==b`, exact
   `K`, proven representative children, checkpoint roles live in ancestors,
   escape within horizon)? This is the load-bearing FHW-T3-R invocation.

## 7. Remaining accept-coverage gaps (still rejected, why)

- **NonFrontierCovered gates** (RC/WC/charged-role/N-virgin/WcFail, gate-local WC
  demand, genuine-non-FC root gate): rejected. Need proven `B(C_s)>=6` subtrees
  for positive fixtures (out-of-lane native-PN closure).
- **Reductive FC gates from production nodes**: the accept path handles them, but
  the positive fixture is an all-Exact (`R==K`) double-threat gate (reduction 0);
  a genuinely reductive FC positive fixture (`R⊊K`) would need a board dense
  enough that `B_8(d) ⊆ Lambda(P_Q+s)` — constructible but not built here. The FC
  edge class and predicate are unit-tested (`fc_and_gi_predicates`).
- **Nested / non-root gates and gate-under-Group2**: the passes support them
  (demands propagate through gates; digests hash gate subtrees under a parent
  Group2's child-plan), but the shipped positive fixture is a root gate. The
  gate-under-Group2 digest path is exercised only structurally.

## 8. Honest stops

- Non-FC accept: **not shipped** (fail-closed reject retained + documented).
- Full non-FC positive fixtures + genuine-non-FC root gate: **not built** (proven
  subtree provenance boundary).
- Not committed: per instructions, the orchestrator gates and commits.
