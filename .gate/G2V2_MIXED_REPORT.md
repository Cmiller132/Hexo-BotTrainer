# G2 v2 — MIXED certificate amendment (gate + legacy compact) + adoption A/B

Lane: `claude/g2-cert`, worktree `.claude/worktrees/g2-cert` (from `4f87b893`).
Author: G2 verifier-amendment lane (Claude, Opus). Date: 2026-07-21.
Scope: the OWNER-AUTHORIZED trusted-base amendment named in
`G2V2_WIDEPN_REPORT.md §5` — ADMIT MIXED CERTIFICATES so that a certificate may
carry `FhwGateV1` gates (verified by the unchanged, hostile-reviewed gate accept
path) alongside non-gateable forced nodes verified by the legacy full-coverage
path, in the same certificate. The amendment is COMPOSITION, not relaxation of
either channel.

**Verdict up front: BUILT + SOUND; firing is now NON-ZERO (0 → 36–106 gate
positions across two runs, 6,462 positions each); but DO-NOT-ADOPT at the
production config** — coverage is identical (1,217 = 1,217) and nodes/decision
is byte-identical ON vs OFF, so the owner's adoption bar (coverage up, or equal
coverage at materially lower nodes/decision) is not met. This is now an
ECONOMICS DO-NOT-ADOPT (the reduction is real in the certificate but is not
consumed by the search to prune), not a boundary/soundness one. The previous
lane's firing-zero blocker (the no-mixing class rule) is resolved.

Suite: hexfield_eq lib unittest **250/0/37 → 251/0/37** (+1 mixed-cert seam
mutation battery), plus `hexfield 68 / hexo_engine 23 / hexo_models 57 /
hexo_utils 8`, **0 failed across every binary**. Command (from worktree root):
`CARGO_TARGET_DIR=E:/cargo-targets/g2-cert cargo test --features python
--target x86_64-pc-windows-msvc -- --test-threads=1`.

**Frozen-file law honored where it must be.** `git diff` on `tss_verify.rs`
(the legacy acceptance core) is **EMPTY** — byte-identical. The gate
reconstruction internals (`reconstruct_gate`, `classify_role`,
`classify_window`, `frontier_covered`, all row/clock derivations) are UNTOUCHED.
The amendment is confined to the composition seam in `tss_verify_group2.rs`
(+147 lines: preflight relax, per-node dispatch, kernel-staple reconstruction,
unfold) and the emission side in `tss_solver.rs` (+469 lines: mixed nesting).

---

## 1. The seam and why it is the whole problem

A wide-PN forcing proof is a tree of forced defender AND nodes. Three kinds
occur:

1. **Unforced (k<b)** → `UniversalGroup2V1` (FHW reduced zone; §3.4 coverage).
2. **Forced, gateable** (`b∈{1,2}`, ≤3 named threats, `tau=b`) → `FhwGateV1`.
3. **Forced, NOT gateable** (`tau=b` but the threat family is outside the
   compact `H_Q` bound — the measured 214/918 `ThreatCountOutOfRange` nodes).

Kind 3 had no admissible representation: the wide builder emits it as a
**compact `implicit_dispatch=true` Universal** (the forced replies only, omitted
replies covered by the legacy T6 argument), and the FROZEN no-mixing preflight
rejected any `implicit_dispatch` node in a new-class cert. So a single kind-3
node made the whole cert reject → clean fall back to the flag-off cert → firing
zero. The amendment admits kind-3 nodes via the legacy full-coverage path,
composed with the gate channel.

**The seam invariant (the load-bearing claim).** A compact forced node's ENTIRE
local obligation is the legacy **T6 kernel-staple**: reconstruct the exact
`tau==b` dispatch kernel `K` from the replayed board and require the explicit
forced replies to cover it (`explicit ⊇ K`). Every omitted (non-kernel) reply
fails to block a live attacker threat, so it is a dead defender move (the
attacker completes the threat next) — **no window/role/zone coverage is owed at
this node.** For every OTHER quantity (budget `B`, live-role clocks
`r_full/f_cut`, window clocks `Q_cut/E_full`, and demand propagation) a compact
forced node is an ORDINARY defender AND and composes exactly as the design's
ordinary-AND clause already dictates — which is precisely what the existing
derivation passes already do for a legacy `Universal`. The two channels are
therefore non-interfering at a shared parent: a gate's `f_cut(Q,ρ)` /
`Q_cut(Q,W)` over a representative child `C_s` reads `roles[C_s]` and
`window_clock(C_s)` uniformly whether `C_s` is a gate or a compact legacy node
(the gate row derivations `derive_gate_role_row` / `derive_gate_window_row`
consume only the child's derived `child_f` / `child_q`, never the child's node
kind); and a compact legacy parent's `B` / clocks fold in a gate child's derived
values with the same `1+max` ordinary-AND recurrence.

## 2. What the no-mixing rule protected, and how each protection is preserved

The design (§2.3 rules 2/3) chose no-mixing as a completeness/performance
restriction, but it was standing in for real soundness protections. Each is
preserved BY OTHER MEANS, not dropped:

| the no-mixing rule forbade | why (what it protected) | how preserved under mixing |
|---|---|---|
| a legacy `Universal` unless it was the FULL legal set | an omitting legacy node whose omitted replies were never checked would false-accept | a compact node's omitted replies ARE checked — the T6 kernel-staple (`explicit ⊇ K`) is reconstructed from the board and is the complete omitted-reply obligation for a `tau=b` forced node (`tss_verify_group2.rs:1384` + `reconstruct_dispatch_kernel:1553`). An UNFORCED node cannot masquerade as compact: the reconstruction REQUIRES `tau==b`, so `k<b` rejects. |
| same-turn commutation anywhere | a commuted omitted-reply envelope could splice a cheaper sibling label into a gate's ancestor envelope | commutations stay **fail-closed rejected** in the preflight (`preflight_structure:1104`) and in the unfold (`copy_subtree:3224`). The solver never needs them: `build_defender_pair_gate` emits a **commutation-free** nested tree (both turn orders proven explicitly). |
| legacy zoned nodes | a stored zone is a finder oracle; the new class re-derives everything | legacy `zone.is_some()` stays **fail-closed rejected** in preflight and unfold. |
| implicit-dispatch/T6/D17/SR mixing | unproven compositions | only the kind-3 compact forced Universal is admitted; T6/D17/SR remain rejected. The compact node's kernel-staple is the SAME obligation the legacy `verify_universal` implicit-dispatch arm discharges, re-derived in-module with the gate path's own threat-family / transversal primitives (so the two forced-node channels are consistent). |

## 3. Seam design — clause → code → test

All verifier code in `packages/hexfield_eq/rust/src/tss_verify_group2.rs`.

| seam clause | code | test |
|---|---|---|
| Preflight ADMITS `implicit_dispatch` (either value); still REJECTS `zone`/`commutations`; canonical order + exact-tree indegree unchanged | `preflight_structure` `:1104` (Universal arm) | `mixed_cert_seam_mutation_battery` M1 (commutation), M2 (zone) |
| Per-node dispatch at replay: compact ⇒ kernel-staple; full-set ⇒ `edges==legal`; both replay explicit children | `build_context`/`replay_node` Universal arm `:1384` | M3 (compact≠full-set), M5 (drop forced reply) |
| Exact `tau==b` dispatch-kernel reconstruction (`K = {legal d : tau(F\d) ≤ b-1}`, `F` = every count≥4/def-0 window's empties); `k<b` and `tau≠b` REJECT; empty kernel REJECT | `reconstruct_dispatch_kernel` `:1553` | M5; A/B cross-verification (§4) |
| `explicit ⊇ K` (omitted-reply coverage; the whole local obligation) | `build_context` `:1384` (`set_contains` loop) | M5 (every single-edge drop rejects) |
| Compact node composes as ORDINARY AND for `B`, `r_full/f_cut` (full `1+child` charge) | `derive_budgets_and_roles` `Universal|UniversalGroup2V1` arm `:1727` | positive verify + D6 images |
| Compact node composes as ORDINARY AND for `(Q_cut,E_full)` (`1+max child`); feeds a gate ancestor's paired clock uniformly | `window_clock` `Universal|UniversalGroup2V1` arm `:1858` | positive verify + D6 images |
| Demands: a compact node SEEDS nothing (its coverage is the kernel-staple) but PROPAGATES incoming gate/zone demands to every child | `derive_window_demands` seeding skip + propagation `:2042` | positive verify (gate rows under a compact seam) |
| Gate rows over a legacy child read only the child's derived `child_f`/`child_q` | `derive_gate_role_row`/`derive_gate_window_row` (unchanged) + `check_gate_nodes` | positive verify |
| Digest: semantic encoder binds `implicit_dispatch`; derived record folds the compact subtree; finder + verifier agree | `enc_semantic_local` Universal arm (unchanged) + `build_digest_tables` | D6 images (12/12 verify) |
| Unfold PRESERVES `implicit_dispatch`; still rejects `zone`/`commutations` | `copy_subtree` `:3271` | finalize self-verify (solver) |

Solver (emission), `packages/hexfield_eq/rust/src/tss_solver.rs`:

| emission clause | code |
|---|---|
| Single-stone non-gate forced node ⇒ compact legacy Universal (already emitted; now accepted by the unfold) | `build_universal` fallback |
| Two-stone forced turn: outer FHW gate when it closes, else COMPACT legacy over every first stone (commutation-free) | `build_defender_pair_gate` `:7593` |
| Inner b=1 node: reduced FHW gate when it closes, else COMPACT legacy over every forced second stone (the primary mixed-nesting seam) | `build_pair_inner_node` `:7713` |
| One pair-child edge, with exact final-position identity checked before trust | `build_pair_second_edge` `:7772` |
| Firewall unchanged: dual-materialize + strict in-process `Group2Verifier` self-verify; any failure ⇒ `None` ⇒ bit-identical flag-off fallback | `materialize_group2_wide` |

## 4. Adoption A/B (group2-on vs off, production config)

Instrument: `hexfield_eq_deep_solve_batch` via the freshly built Windows cdylib
staged at `.gate/g2ab/_rust.pyd` (CPU-only, `CUDA_VISIBLE_DEVICES=-1`, serial),
production coverage config (`node_cap=500, goal=both, horizon=0 (unbounded),
wide=true, dual_pass=true, zone=false`) over the dev splits + the 19-position
forcing corpus. Driver `.gate/g2ab/ab_driver.py`; raw `.gate/g2ab/ab_result.json`.
Only nodes/decision is quoted (never wall time).

Table below is the current `ab_result.json` (run 2). A first run gave lower
firing (36 gate positions / 43 gate nodes; human_v1 29/36) — see the
non-determinism note.

| cohort | n | decided off (=on) | nodes/dec off | nodes/dec on | gate positions | gate nodes |
|---|---|---|---|---|---|---|
| selfplay_v1 | 3255 | 257 | 80.9 | 80.9 | 4 | 17 |
| human_v1 | 2720 | 727 | 55.2 | 55.2 | 85 | 193 |
| puzzle_v3 | 468 | 228 | 181.7 | 181.7 | 17 | 45 |
| forcing_corpus | 19 | 5 | 107.8 | 107.8 | 0 | 0 |
| **total** | **6462** | **1217** | — | — | **106** | **255** |

Gates:
- **(a) Verdict parity:** PASS — **0** positions decided OFF but not ON (coverage
  identical per cohort and class, both runs). Never-decides-less holds
  structurally.
- **(b) Verifier failures:** PASS — **0** in both arms, all cohorts, both runs.
- **(c) Cross-verification:** run 2 **106 gate-decided positions checked, 106
  agree, 0 mismatch**; run 1 **36 checked, 36 agree, 0 mismatch**. Every
  gate-decided verdict was re-solved group2-OFF at a 50k-node budget and
  reproduced the identical W/L verdict with zero verify failures. No false accept
  surfaced anywhere across either run.
- **Firing:** **non-zero** — 106 gate positions / 255 gate nodes (run 2), 36 /
  43 (run 1), versus the prior lane's exact zero.

- **Firing count is NON-DETERMINISTIC run-to-run** (36 vs 106 gate positions).
  The wide-PN batch solves positions in parallel, and at a forced node whether
  the FHW closure succeeds or falls back to the compact legacy path depends on
  materialization/hash-ordering, so which nodes emit a gate varies. This affects
  only the COUNT of gates, never soundness or the decision: coverage
  (1,217 = 1,217), nodes/decision (identical to the last digit), parity (0),
  verifier failures (0), and cross-verification (100% agree / 0 mismatch) are
  invariant across both runs. A third confirmatory run was started per protocol
  and INTENTIONALLY ABORTED under orchestrator direction (trainer contention;
  the decision package is already complete and the firing sign / soundness
  invariants are established).

Nodes/decision is byte-identical ON vs OFF because the group2 pass only
re-materializes the SAME proven wide-PN tree — it does not prune the search
(never-decides-less is structural). The certificate-level reduction (gates carry
`|R|` representative subtrees instead of `|K|` forced replies) is real but is not
CONSUMED by the search, so it does not lower the search node count.

## 5. Recommendation: DO-NOT-ADOPT (production config, this build)

Owner's bar: adopt only if the harness score improves — coverage up at equal
soundness, OR equal coverage at materially lower nodes/decision. Measured:
coverage identical (1,217 = 1,217), nodes/decision identical to the last digit,
soundness clean everywhere (0 parity, 0 verify-fail, cross-verify 100% agree /
0 mismatch on all 106 gate-decided positions).
Enabling `tss_solver_group2` at the production config is a strict no-op on the
harness score.

**Net progress vs the prior lane:** the firing-zero boundary has been REMOVED —
the no-mixing class rule is amended, mixed certs fire (0 → 106), verify, and
cross-verify clean. The residual reason for DO-NOT-ADOPT has moved from
"verifier rejects mixed certs" to "the wide-PN search does not consume the gate
reduction to prune." Realizing a harness gain would require a `Consume`-mode
change (design §5.1) that lets the wide search reduce fanout at a proven gate —
a separate, larger build, out of scope for this composition amendment.

## 6. Soundness scope and honest stops

- **The amendment is composition, not relaxation.** Legacy acceptance semantics
  (`tss_verify.rs`) are byte-identical; the gate reconstruction internals are
  untouched; the FHW-T3-R per-gate argument and the legacy T6 per-legacy-node
  argument each apply to their own nodes; the seam is checked explicitly at every
  invariant and REJECTS fail-closed on anything unproven.
- **Commutation mixing stays OUT** (design gap §4.2.4). The solver unfolds
  two-stone turns commutation-free; the verifier rejects any surviving
  commutation. This is a deliberate scope boundary, not a soundness weakening.
- **Empirical firewall.** The in-process self-verify (never emit a rejected
  cert), the outer mint re-verify, and the A/B cross-verification (100% agree,
  0 mismatch across both runs) form three independent checks; no false accept
  appeared in 6,462 positions x 2 runs.
- **Seam battery** (`mixed_cert_seam_mutation_battery`, on a REAL production
  mixed cert): positive verify + 12/12 D6 images + flag-off parity, then every
  boundary mutation REJECTS — commutation smuggle, zone smuggle, compact-claims-
  full-enumeration, duplicate forced-reply edge, every single forced-reply drop,
  and compact-relabeled-as-bogus-gate.
- **Not committed:** per instructions, the orchestrator gates and an independent
  hostile review runs after this lane.
