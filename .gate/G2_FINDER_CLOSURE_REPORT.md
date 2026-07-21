# G2 finder-side FHW closure — build report

Lane: `claude/g2-cert`, worktree `.claude/worktrees/g2-cert` (from `7d6e5f1b`).
Author: engineering lane (Claude). Date: 2026-07-20.
Assignment: DESIGN §5.3 — the finder-side FHW closure that lets the Group-2
selector construct reductive gates at implicit-dispatch nodes, plus the positive
fixture set and the first firing measurement at production-shaped nodes. The
verifier accept path was explicitly OUT OF SCOPE and is untouched.

Status: **DELIVERED — the largest sound subset, suite green, honest boundary
documented.** No verifier-accept-path code was improvised; the wholesale
`FhwGateV1` reject stays intact.

Headline finding (the 40.5%-ceiling question): **reductive gates exist and are
common at production-shaped nodes.** Over the 19-position forcing corpus the
structural closure closed **704 gates on 918 eligible implicit-dispatch defender
nodes, 525 of them genuinely reductive (R ⊊ K)**, with average |K|=2.12,
|R|=1.29, |Legal|=580.9. Every closed gate passed an independent structural
self-check.

---

## 1. What was built (file / line map)

All new code is isolated to a single **`#[cfg(test)]`-only** module, so the
production binary is byte-for-byte unaffected (flag-off/golden-digest bit
identity holds by construction — there is no new production code path).

| item | location |
|---|---|
| module registration (test-only) | `packages/hexfield_eq/rust/src/lib.rs` (`#[cfg(test)] mod tss_g2_gate_finder;`) |
| closure builder + classifiers + measurement | `packages/hexfield_eq/rust/src/tss_g2_gate_finder.rs` (947 lines) |
| test/fixture/measurement harness | `packages/hexfield_eq/rust/src/tss_g2_gate_finder/tests.rs` (687 lines) |
| emitted fixtures + manifests | `packages/hexfield_eq/tests/fixtures/g2_gates/` |

Key symbols in `tss_g2_gate_finder.rs`:

- Geometry: `ball` (B_8 / 217-cell axial ball), `Ghost` (`G = P_Q + s`,
  materialized `Lambda(G)`, `GI(G)` predicate), `frontier_covered` (`:116`),
  `rc_pass` (`:125`), `wc_pass` (`:142`).
- Threats/transversal: `attacker_threats` (`:181`, H_Q as `WindowKey`s +
  F_Q empties), `transversal_number` (`:207`, EXACT bounded transversal — exact
  for τ≤2 and returns `cap+1` above, all the builder needs since b≤2; this is
  the `== b` and `≤ b-1` predicate the report notes `family_hitting_exceeds`
  (`>b` only) could not provide).
- Classifiers (faithful to design §3.3 tables): `classify_role` (`:249`, 3
  `FhwRoleRowV1` leaves), `classify_window` (`:322`, 9 `FhwKappaRowV1` leaves).
- Structural closure: `try_build_gate` (`:442`) → `GateBuild` (`:414`);
  failure taxonomy `ClosureFail` (`:389`).
- Self-check: `self_check_structural` (`:617`).
- Measurement: `measure_position` (`:793`) + `walk` (`:815`), bounded
  forcing-tree shadow walk; `FiringStats`, `GateExample`.

The module imports only `hexo_engine`, `crate::threats_shared`, and the
`FhwEdgeClassV1/FhwKappaRowV1/FhwRoleRowV1/GuardResultV1` record TYPES from
`tss_verify`. It never imports or calls the verifier accept path.

Constraint compliance (verified): `git diff --stat` on `tss_verify.rs`,
`tss_verify_group2.rs`, and `tss_solver.rs` is **empty** — zero edits to the
verify path or the solver. Only `lib.rs` (one `mod` line) and new files.

---

## 2. The sound subset, and why the boundary is where it is

The design's full gate cert carries, per `(d,s)` pair, role rows and window rows
whose scalars `child_f = f_cut(C_s,ρ)` and `child_q = Q_cut(C_s,W)` are the
CLOCKS OF A PROVEN REPRESENTATIVE SUBTREE `C_s`. Producing those subtrees for a
production node is exactly the design §5 native-PN Open/Closed closure — the
prior lane's "largest single unknown," 450–800 lines that make trainer-wide PN
consume the selector. That is not deliverable to the soundness bar from this
single lane.

Everything that depends ONLY on the gate position `P_Q` and a single ghost
placement `G = P_Q + s` IS deliverable and self-checkable in-lane:

- **the reductive core** — H_Q, F_Q, exact transversal `== b`, kernel
  `K = {d : τ(F_Q\d) ≤ b-1}`, representatives R, retraction φ, and every edge's
  class (Exact / FrontierCovered / NonFrontierCovered). This is precisely the
  |K| vs |Legal| and |R| vs |K| reduction that answers whether reductive gates
  exist.
- **the row classifiers** — pure functions of ghost geometry + the two clock
  inputs. They MIRROR design §3.3 exactly, so feeding them real geometry plus a
  specified `q`/`k` is a positive fixture that is derivably true modulo the clock
  provenance.

So the delivered gate is the **structural gate** (H,F,K,R,φ,edge-classes,
self-checked) plus the classifier truth for the rows. The one thing NOT emitted
for corpus nodes is the fully-rowed cert, because its `q`/`k` scalars need a
subtree this lane cannot prove. This is the "smaller true thing beats a bigger
uncertain one" call the DoD authorizes.

---

## 3. Self-check design and why it is faithful to the theorem

`self_check_structural` (`:617`) is run on every emitted gate and rejects (→ no
emit; fail-closed at the finder) unless ALL of the following independently
recomputed conditions hold. In the absence of the accept path these are the
correctness evidence.

1. **R2 + class rules** (design §2.3 rule 4/5, amendment R2): post-opening,
   defender-to-move (`current_player == claimant.other()`), nonterminal, not
   own-win-now, `b ∈ {1,2}`.
2. **Threat family**: every emitted `H_Q` window is re-verified as a real
   attacker-alive ≥4 window with a nonempty empty-set (matches design §3.3
   "validate each key as a current A-threat").
3. **Exact transversal `== b`** recomputed from the empties (§3.3
   `transversal_number(F_Q) == b`).
4. **Kernel equality**: K is recomputed independently as
   `{d ∈ Legal : τ(F_Q\d) ≤ b-1}` and must equal the emitted kernel EXACTLY
   (§3.3 `K` derivation; no finder `min_hitting_set` is trusted). Each kernel
   reply applied is required nonterminal.
5. **φ/R structure**: R ⊆ K; the emitted map domain equals K exactly (one edge
   per real reply); every representative appears as an `Exact` self-edge
   (`φ(s)=s`); every edge's class is recomputed from geometry (`d==s` ⇒ Exact,
   else FC iff `B_8(d) ⊆ Λ(P_Q+s)`, else NonFC) and must match.
6. **R1 escape deadline** (amendment R1): `escape_resolution_ply =
   p(Q)+b+2` recomputed and required `≤ semantic_horizon`.

The classifiers `classify_role`/`classify_window` reproduce the design §3.3
ordered mutually-exclusive tables verbatim, including the mandatory retained
guards (`cnt_D+1+q<6`, `1+q<6`, WC ball, N-virgin, D22-N radius). A failed
mandatory guard returns `None` (reject), exactly matching "a failed mandatory
guard rejects even if the finder wrote Pass" and the `GuardResultV1` no-`Fail`
invariant.

### Two soundness FINDINGS surfaced by faithfully implementing the guards

Both are triangle-inequality contradictions between a charged row's TRIGGER and
its mandatory RADIUS guard, so the charged leaf is only ever reachable as a
REJECTION (or via a different, uncharged path). They are asserted as tests and
are useful signals for the accept-path build / hostile review:

- **`NonFcEmptyNonIncidentWcFail` is unrealizable with a passing guard.** WC
  fails ⇒ ∃ ghost-illegal `z` with `dist(z,d)≤8` and `dist(z,W)≤8(q-6)` ⇒
  `dist(d,W) ≤ 8 + 8(q-6) = 8(q-5)`, which contradicts the mandatory N-virgin
  guard `dist(d,W) > 8(1+q-6) = 8(q-5)`. So whenever WC fails the guard also
  fails and the gate rejects; the WcFail leaf is never emitted with a passing
  guard. (test: `kappa_leaf_..._wc_fail_is_geometrically_unrealizable...`)
- **The charged role row via a ghost-illegal RC-fail carrier is unrealizable.**
  RC fails ⇒ `dist(d,y) ≤ 8k`, contradicting the mandatory D22-N guard
  `dist(d,y) > 8k`. The charged role row is reachable only via the ghost-LEGAL
  carrier path (where design §2.2 does NOT require D22-N). (test:
  `role_charged_via_ghost_illegal_rc_fail_is_geometrically_unrealizable`)

These do not weaken the deliverable; they refine the realizable-leaf set and are
flagged for the reviewer (the guards may be intentionally conservative
rejection triggers, or the design radii may want a re-look).

---

## 4. Fixture inventory (realized vs unrealized, with reasons)

Fixtures live in `packages/hexfield_eq/tests/fixtures/g2_gates/`. All are
regenerated deterministically by the test suite.

### 4.1 Real structural gates from production-shaped nodes — `structural_gates.txt`
12 captured real closed gates (exact `P_Q` occupancy + H_Q + K + R + per-edge
`d→s:class`), each self-check-passing. Includes genuinely reductive gates, e.g.
`id=0` (corpus `8is963b`): b=2, |K|=3, |R|=1 — a single representative
FrontierCovered-covers the whole kernel (2 FC edges + 1 Exact self-edge),
escape_ply 111 within horizon. Captured edge classes: 12 Exact self-edges + 15
FrontierCovered.

### 4.2 Classifier leaf / role / incidence fixtures — `classifier_fixtures.txt`
The positive rows the accept path will recompute, each asserted by a test:

| class | leaf | realized | note |
|---|---|---|---|
| kappa | NonDAlive | ✅ | window carries a claimant stone |
| kappa | ExactOrFcNonIncident | ✅ | exact/FC, d∉W |
| kappa | ExactOrFcDirect | ✅ | both touched (`cnt_D+1+q<6`) and all-empty (`1+q<6`) guard paths |
| kappa | NonFcTouchedNonIncident | ✅ | |
| kappa | NonFcTouchedDirect | ✅ | |
| kappa | NonFcEmptyDirect | ✅ | |
| kappa | NonFcEmptyNonIncidentQlt6 | ✅ | |
| kappa | NonFcEmptyNonIncidentWcPass | ✅ | real ghost, q=6, WC ball empty |
| kappa | NonFcEmptyNonIncidentWcFail | ❌ | geometrically unrealizable with a passing guard (§3 finding); realized as the REJECTION |
| role | ExactOrFcZero | ✅ | exact and FC |
| role | NonFcRcZero | ✅ | ghost-illegal carrier, k=0 ⇒ RC passes, ε=0 |
| role | NonFcCharged | ✅ | ghost-legal carrier, ε=1 (the D22-N-charged path is the §3 unrealizable one; realized via the ghost-legal charged path) |
| incidence | (d∈W,s∈W) all 4 pairs | ✅ | `all_four_incidence_pairs_are_constructible` |

Realized: **8 of 9 kappa leaves; all 3 role rows; all 4 incidence pairs; Exact,
FC, RC-zero, WC(pass)-zero, and a charged row.** The one unrealized leaf
(WcFail) and the D22-N-charged role path are documented as geometrically
contradictory, not skipped.

Genuine-non-FC ROOT gate: not realized as a full end-to-end cert — a root gate
with a dangerous all-empty/nonincident WC window requires the gate-local WC
demand enumeration tied to a proven `B(C_s)≥6` subtree, which is the same
subtree-provenance boundary (§2). The NonFC edge class itself is exercised by
the classifiers, and NonFC reductions are present in the corpus population.

### 4.3 Firing measurement — `firing_measurement.txt`
Ground-truth corpus numbers (see §5), regenerated by the measurement test.

---

## 5. Firing / closure measurement (19-position forcing corpus)

Method: a bounded forcing-tree shadow walk (`measure_position`) from each corpus
root — attacker nodes follow the tactical/forcing set, defender nodes recurse
through hitting-set replies — invoking `try_build_gate` at every
implicit-dispatch-eligible defender node. It reaches real forcing defender nodes
without importing or perturbing `tss_solver` (per-position node cap 6000, depth
cap 10).

```
defender_nodes_seen = 8272
eligible_nodes      = 918     (implicit-dispatch: opp threats, τ==b, not own-win, post-opening)
gates_closed        = 704     (76.7% of eligible; all structural self-checks passed)
reductive_gates     = 525     (57.2% of eligible, 74.6% of closed — R ⊊ K)
avg |K| = 2.12   avg |R| = 1.29   avg |Legal| = 580.85
best |K|/|Legal| = 0.0015   best |R|/|K| = 0.25
closure-failure histogram (eligible-but-not-closed):
    ThreatCountOutOfRange : 214   (H_Q outside the v1 compact bound: b=2 with >3 named threats, or b=1 with >1)
```

Interpretation: **the reductive prize is real at production-shaped nodes.** A
typical eligible node has ~581 legal replies collapsing to a kernel of ~2 and a
representative set of ~1.3 — the structural reduction the FHW gate is designed to
license. The only closure failure is the v1 grammar's own compactness bound
(`|H_Q|≤1` at b=1, `≤3` at b=2), not a soundness gap.

Honest limitation of the measurement: the bounded forcing heuristic surfaced
eligible nodes in **2 of 19** corpus roots (`8is963b`, `dy3dg99`), because the
forcing-move heuristic (`tactical_cells`) advances existing threats but does not
build fresh attacker threats from quiet roots the way a full PN search does. The
918-node / 704-gate sample is nonetheless substantial and consists of genuine
forcing-line defender nodes with real threat structure. Fuller corpus reach
needs the production search integration (the §2 out-of-lane native-PN closure);
a shadow hook at `tss_solver.rs:8199` (the `prove_universal` dispatch site,
behind a default-off telemetry flag) is the natural next step but was avoided
here to guarantee flag-off bit identity.

---

## 6. Suite status

Full independent serialized suite (baseline invocation):
`CARGO_TARGET_DIR=E:/cargo-targets/g2-cert cargo test --features python
--target x86_64-pc-windows-msvc -- --test-threads=1`
→ **229 passed / 0 failed / 37 ignored** in 87.6 s (baseline 209/0/37 + 20 new
tests). The `--no-run` compile is clean apart from the 7 pre-existing pyo3
deprecation warnings.

Flag-off bit identity: preserved by construction (the new module is
`#[cfg(test)]`; the production binary contains no new code path). The existing
determinism/identity tests (`flag_off_solver_is_deterministically_identical...`,
etc.) remain green inside the 229.

---

## 7. Honest stops

- **Verifier accept path**: untouched, as required. The wholesale `FhwGateV1`
  reject in `verify_group2_impl` is intact; the emitted structural gates will
  NOT verify yet — expected.
- **Fully-rowed production gate certs**: NOT emitted for corpus nodes. Their
  role/window scalars (`f_cut(C_s,ρ)`, `Q_cut(C_s,W)`) require a proven
  representative subtree that only the native-PN Open/Closed closure (design §5)
  can produce for a production node — out of this lane's scope. Fixtures carry
  those scalars as documented inputs.
- **Genuine-non-FC root gate as an end-to-end cert**: NOT realized (same
  subtree-provenance boundary + the gate-local WC demand enumeration). The NonFC
  class and the WC/N-virgin machinery are exercised at the classifier level.
- **`NonFcEmptyNonIncidentWcFail` leaf** and the **ghost-illegal D22-N-charged
  role row**: geometrically unrealizable with a passing guard (§3 findings);
  realized as rejections. Flagged for the accept-path hostile review.
- **Corpus reach**: eligible nodes surfaced in 2/19 roots via the bounded
  forcing walk (§5); full reach needs the solver integration.
- **Not committed**: per instructions, the orchestrator gates and commits.

## 8. Recommended next steps

1. Native-PN Open/Closed closure (design §5) so representative subtrees are
   proven and `f_cut`/`Q_cut` become derivable → fully-rowed production gate
   certs and the genuine-non-FC root-gate fixture.
2. A default-off telemetry hook at the `prove_universal` dispatch site
   (`tss_solver.rs:8199`) to run this closure builder in true production shadow
   for full-corpus firing (guard bit identity with the existing golden-digest
   discipline).
3. Route the WcFail / D22-N-charged contradictions (§3) to the accept-path
   hostile-review round; decide whether the guards are intended rejection
   triggers or the radii need adjustment.
