# G2 v2 — FhwGateV1 certificate verification: gate report

Lane: `claude/g2-cert` worktree `.claude/worktrees/g2-cert`, from v1 tip `2a6fc0db`.
Author: engineering lane (Claude). Date: 2026-07-20.
Status: **BLOCKED — no code shipped; v1 baseline preserved (sound, green).**
Verdict in one line: the FhwGateV1 accept path is fully specified but cannot be
delivered to the campaign's soundness bar from this single lane, because (a) its
positive fixtures require a finder-side FHW closure that does not exist yet, and
(b) the design itself (§6.1) mandates an independent adversarial review of the
accept path that this lane cannot supply. Shipping an undertested strict-verifier
accept path would **regress** the currently-sound state (gates reject wholesale).

This report is the actionable build spec for the next round, derived from the
real code, so the accept-path build can be routed correctly.

---

## 1. Baseline confirmed (truthful counts)

- Full independent serialized suite (the v1-recorded invocation, reproduced):
  `CARGO_TARGET_DIR=E:/cargo-targets/g2-cert cargo test --features python
  --target x86_64-pc-windows-msvc -- --test-threads=1`
  → **209 passed / 0 failed / 37 ignored** in 67.4 s. Matches v1's record exactly.
  (`--features python` is required so `mod search`/`mod tree` are not silently
  skipped — the documented cargo trap; confirmed present in this invocation.)
- `--no-run` compile: clean (18.4 s warm), 7 pre-existing pyo3 deprecation
  warnings only.
- Host free RAM at build time: 15.11 GB (> 10 GB law). Cache dir reused:
  `E:/cargo-targets/g2-cert`.
- v1 state is SOUND: `verify_group2_impl` rejects any certificate containing an
  `FhwGateV1` node wholesale (`tss_verify_group2.rs:443-449`), after enforcing R1
  per-gate `escape_resolution_ply <= semantic_horizon` (`:434-440`) and folding
  escape deadlines into derived T. Removing that wholesale reject is exactly what
  v2 must replace with real verification — and until real verification is
  present and reviewed, the reject IS the fail-closed correct behavior.

## 2. Why this is BLOCKED rather than built (the load-bearing argument)

The gate accept path is not an isolated add; it threads a gate arm through
**every** pass (each currently `FhwGateV1 => return None`): replay/`build_context`
(`:762`), `derive_budgets_and_roles` (`:1056`), `window_clock` (`:1102`),
`derive_window_demands` (`:1230`), `check_group2_nodes` (skips non-G2), and both
Merkle encoders (`enc_semantic_local :1542`, `build_digest_tables :1642`).

Two hard, non-negotiable blockers:

1. **Positive fixtures for the reductive rows are not constructible without a
   finder FHW closure.** §6.2 mandates positive fixtures for Exact, FC, RC-zero,
   WC-zero, and fully-charged D22-N rows, all nine `FhwKappaRowV1` leaves, all
   four `(d in W, s in W)` incidence pairs, and a genuine-non-FC **root gate**.
   A valid reductive gate (`R ⊊ K`, real `phi`, non-Exact edge classes) can only
   be produced by the FHW closure builder the design estimates at 350–600 lines
   (§5, `NarrowCompatSearch::prove_universal`) plus native PN Open/Closed closure
   (450–800 lines) — the design's own hardest, least-specified piece ("The
   grammar change alone does not make trainer wide PN consume the selector",
   §5). Hand-constructing a certificate that passes the verifier's recomputed
   FC/RC/WC/kappa rows **and** the 12-transform D6 Merkle digests is not
   feasible. Without positive fixtures, the reductive accept path is untestable,
   so shipping it would violate "any gap you cannot verify per the design must
   reject, never accept."

2. **The design mandates an independent adversarial review of the accept path
   before it is trusted (§6.1.1).** A single lane cannot self-certify a
   strict-verifier accept path. Per owner policy this profile — "long, grinding
   execution with a clear goal and a checkable definition of done" plus
   "high-stakes correctness verdict" — is prime Codex territory; this subagent's
   instructions do not authorize launching Codex (for the build **or** for
   review), so no independent cross-check is available in-lane.

The "Exact-only sub-class" stepping stone (accept only `R == K`, `phi = id`,
every `edge_class == Exact`) was evaluated and **rejected as a deliverable**: it
yields ~0 fanout reduction (Exact `R = K` is the implicit-dispatch kernel
re-expressed), so it takes on the FULL soundness risk of the gate accept core —
transversal `K`-derivation, Cartesian window-demand completeness, the paired
`f_cut`/`Q_cut` recurrences, and checkpoint-role liveness/deadlines, all of which
are load-bearing even for Exact gates because they are what license omitting
`Legal \ K` — for **zero** measured prize. That risk/reward is wrong against a
fail-closed campaign law and a sound baseline.

## 3. Constraint compliance (line-item)

- Strict verifier core `tss_verify.rs` acceptance semantics: **UNTOUCHED**
  (no edits this lane).
- All v2 logic isolated to `tss_verify_group2.rs`: N/A (no v2 logic shipped);
  the isolation boundary and `no tss_solver import` rule remain intact.
- Fail-closed everywhere: **PRESERVED and strengthened as the deliberate
  outcome** — every gate rejects; the sound baseline is unchanged.
- Verifier recomputes everything / no finder-derived trust: N/A (no accept path).
- Never-decides-less: **PRESERVED** (selector-off is the only path that can
  decide gates; nothing regresses).
- Hostile counterexamples + v1 mutation battery: **still all rejecting** (209/0/37).
- R1 (`escape_resolution_ply <= semantic_horizon`) and R2 (post-opening root):
  **enforced** on the extension path (v1 code, unchanged).

## 4. Firing measurement (DoD #3), truthful

With v1 (gates reject), the selector fires only on the constructed gate-free
fixture (19 explicit edges vs 886 legal, ~46× local reduction, verified). It does
**not** fire on the wide-profile forcing corpus
(`packages/hexfield_eq/rust/corpus/forcing_corpus_moves.txt`, present in this
worktree, 19-position corpus): real wide proofs at forcing nodes use
implicit-dispatch, which the v1 class excludes; gates (the class's replacement
for dispatch) are the narrowed-out piece. This re-confirms v1's finding. The
40.5%-of-solve-wall prize (unforced-defender generation, residue `e11c393d`) is
unlocked only by **reductive** gates (`R ⊊ K` with FC/RC/WC), which require
blocker (1) above — Exact gates would fire with ~0 reduction and are not shipped.

## 5. Complete build spec for the accept path (so the next round is turnkey)

### 5.1 Record types (already present, unused on accept)
`tss_verify.rs:184-272` — `FhwEdgeClassV1`, `RoleKeyV1` (incl. `Checkpoint{gate,
threat, cell}`), `FhwRoleRowV1{ExactOrFcZero,NonFcRcZero,NonFcCharged}`,
`FhwKappaRowV1` (9 leaves), `GuardResultV1{NotApplicable,Pass}`,
`FhwRoleClaimV1`, `FhwWindowClaimV1`, `FhwMapV1{real_reply,representative,
edge_class,roles,windows}`, `FhwGateProofV1{schema_version,authority,threats,
escape_resolution_ply,map}`, `FhwGateNodeV1{representatives,proof}`. Frozen tags
in §2.4. D6 remap for gates already implemented (`d6_remap_certificate`,
`d6_transform_role_key` `:1866`).

### 5.2 Per-pass gate arms to implement in `tss_verify_group2.rs`

- **`preflight_structure` (`:529`)**: schema==1, authority.matches_compiled,
  representatives nonempty + canonical sorted-unique by move, threats canonical
  sorted-unique by `window_sort_key`, indegree++ per representative child, path
  lengths within `MAX_AUTHORITY_PATH`.
- **`replay_node` (`:762`)** — gate reconstruction (§3.3, lines 753-779):
  post-opening, defender-to-move, nonterminal, `not own_win_now` (reuse
  `direct_own_win_now_upper` + `threats_shared::analyze` as the two rejectors),
  `b = placements_remaining ∈ {1,2}`; validate each `H_Q` key as a live claimant
  threat (real window, claimant-alive, claimant count ≥4, exact empties);
  `b=1 ⇒ |H_Q|=1`, `b=2 ⇒ |H_Q|∈1..=3`; `F_Q` = empties; **exact
  `transversal_number(F_Q) == b`** (extend `family_hitting_exceeds` `:943` to an
  exact transversal predicate; note the LOSS path only needs `> b`, the gate
  needs `== b` AND the `K` membership `transversal_number(F_Q\d) <= b-1`);
  `K = {d ∈ Legal : transversal(F_Q\d) <= b-1}`, nonempty; every `d ∈ K` applied
  is nonterminal; `R` = representative moves ⊆ `K`, one nonterminal child each;
  emitted map domain == `K` exactly; `s = phi(d)` ∈ `R`, `phi(s)=s`. Store a
  `GateInfo{b, k_domain, phi, fc_per_pair, escape_ply}` for later passes.
- **FC/GI/RC/WC predicates** (§3.3 lines 797-850) over the inclusive radius-8
  ball (217 cells; enumerate via the bounding-box+`hex_distance<=r` pattern
  already in the virgin seeder `:1149-1161`). `G = P_Q + s`; `Lambda(G)` = union
  of `B_8(x)` over occupied x in G; `FC ⇔ d==s || B_8(d) ⊆ Lambda(G)`;
  `GI(G)(z) ⇔ z not occupied and not legal in G`; `RC ⇔ GI(G) ∩ B_8(d) ∩
  B_{8(k-1)}(y)` empty (`k=f_cut(C_s,rho)`, empty ball when `k=0`);
  `WC ⇔ GI(G) ∩ B_8(d) ∩ B_{8(q-6)}(W)` empty. All membership uses the
  materialized `Lambda(G)` set; charge `217*(A+M+RC+WC)` per §3.5.
- **Role rows** (§3.3 lines 781-822): per `(d,s)` union all live roles reachable
  below `C_s`; require `d` avoids every carrier; classify each ghost-illegal role
  → `ExactOrFcZero` (Exact/FC, ε=0) | `NonFcRcZero` (non-FC, ghost-illegal, RC
  pass, ε=0) | `NonFcCharged` (ε=1 + mandatory D22-N `dist(d,y) > 8k`); compare
  the cert's `FhwRoleClaimV1{role,child_f,row,epsilon}` byte-for-byte to the
  derivation.
- **Window rows** (§3.3 table lines 824-863): recompute all four `(d∈W,s∈W)`
  incidence bits and select the mutually-exclusive `FhwKappaRowV1` leaf; evaluate
  the retained guard (`cnt_D(W)+1+q<6` touched, `1+q<6` all-empty, N-virgin
  `dist(d,W)>8(1+q-6)`); a failed mandatory guard **rejects even if the finder
  wrote Pass**. Direct-incidence row is terminal (no `q<6`/WC overwrite).
- **Demand fixed point (`derive_window_demands :1230`)** — gate arm: seed
  `demands(Q)` with incoming ordinary keys ∪ 18 windows through each `d ∈ K` ∪
  (for genuine non-FC `(d,s)`) the bounded gate-local all-empty/nonincident WC
  enumeration `B(C_s)>=6, d∉W, B_8(d)∩B_{8(B(C_s)-6)}(W)≠∅` (§3.2 lines 722-743;
  the root-gate soundness fix). Require **exactly** the Cartesian
  `K(Q) × demands(Q)` rows in every `map[d].windows` — no missing/duplicate/
  unrequested row (§2.2 lines 259-261). Keep `I_FHW` (the measurement index)
  restricted to ordinary-origin `0x01/0x02` keys — do NOT let direct-18 or
  gate-local rows enter it.
- **Budget/role postorder (`derive_budgets_and_roles :1056`)** — gate arm:
  `B(Q)=1+max_{s∈R} B(C_s)`, require `B(Q)>=b`; add checkpoint roles
  (`RoleKeyV1::Checkpoint`, clock 0 at the gate, discharged there — live in
  strict ancestors); `f_cut(Q,rho)=max_{d∈K} branch_f(d,rho)`,
  `branch_f = 0` if rho unreachable in `C_phi(d)` else `epsilon_cut(d,rho)+
  f_cut(C_phi(d),rho)` — **paired before the max; separate marginal maxima are
  forbidden** (§3.3 line 864). `t_sub(Q) = max(escape_resolution_ply, child
  t_sub)`; require `escape_resolution_ply == placements_made(P_Q)+b+2` and
  `<= semantic_horizon` (R1).
- **Window clock split (`window_clock :1070`)** — THE gate-free hazard:
  `Q_cut != E_full` at gates, so the single-value memo must become a **pair**
  `(Q_cut, E_full)`. Gate clauses: `E_full(Q,W)=max{b, max_d(1+E_full(C_phi(d),
  W))}`; `Q_cut(Q,W)=max{b, max_d(kappa(d,W)+Q_cut(C_phi(d),W))}`; check
  `Q_cut<=E_full<=B` per pair. Update all three callers (demand loop `:1206`,
  `check_group2_nodes` Z_touch/Z_virgin uses `Q_cut`, digests use both). Guard
  the gate-free tests: for non-gate nodes the pair must stay equal (regression).
- **Digests (`enc_semantic_local :1542`, `build_digest_tables :1642`)** — gate
  branches per §2.4 lines 476-487 (derived record: `b`, H, K, R, escape,
  per-map derived bits `child_reachable/child_f/carrier_ghost_legal/rc_evaluated/
  rc_pass/d22n_pass/derived_role_row/epsilon` and `child_q/d_alive/all_empty/
  d_in_window/s_in_window/wc_evaluated/wc_pass/derived_kappa_row/kappa/
  retained_guard`) and the semantic-local gate payload
  (`schema_version,authority,threats,escape_resolution_ply,map`).
- **Top-level (`verify_group2_impl :443`)**: delete the wholesale gate reject;
  route gates through reconstruction; keep the R1 per-gate horizon loop.

### 5.3 Finder (for positive fixtures + firing with reduction)
`tss_solver.rs:8172-8202` — at an implicit-dispatch node the finder must run the
FHW closure (choose `H_Q`, kernel `K`, representatives `R`, `phi`, verify
coverage via the theorem's C1/C2/C3 channels) and emit `FhwGateV1`, mirroring
`prove_universal_group2`/`finder_finalize_group2` (`tss_verify_group2.rs:8334`,
`:1986`). This is the design's §5 native-PN-closure piece and is required both
to fire with reduction on the forcing corpus AND to build the §6.2 positive
fixtures. It is the largest single unknown.

### 5.4 Test battery to add (§6.2)
Positive: Exact, FC, RC-zero, WC-zero, charged-D22-N; all 9 kappa leaves; all 4
incidence pairs; the R-Z11 `q=5` reject / `q=4` pass; genuine-non-FC root gate
(delete gate-local row ⇒ reject); FHW-O1 escape-floor deletion (hostile C2);
role-splice reject (C4); 12 D6 images identical verdict + invariant digests.
Mutation: schema/authority/H/tau/K-domain/R/`phi(s)=s`/rep-child/FC/every
role+window row/`q`/`f`/charge/RC-WC input/retained-guard/checkpoint-role/B/
LOSS-base/escape-horizon/plan+summary-digest/required-coord/legal-move/terminal
defender edge/duplicate/order/work-cap/arithmetic — each rejects.

## 6. Remaining shared derivation helpers (v1 review point, unchanged)
Finder and verifier still share the role/clock/zone derivations and digest
encoders in `tss_verify_group2.rs` (v1 deviation 3). The digest comparison
detects drift/tampering, not correlated implementation bugs. The gate accept path
must, per the task and design, reimplement the verifier-side gate derivation
independently of any finder helper; the shared-helper risk grows with gates and
is a first-order review item.

## 7. Honest stops
- FhwGateV1 accept path: **not built** (blockers §2). v1 fail-closed reject
  retained. No acceptance semantics were improvised.
- Reductive prize (40.5% wall): **not unlocked** — gated on the finder FHW
  closure (§5.3).
- Recommendation: route the accept-path build to a dedicated Codex session
  (owner-preferred tier for this profile) with the §6.1 independent hostile
  review, using this document as the spec; keep the sound v1 reject until that
  candidate passes review.
