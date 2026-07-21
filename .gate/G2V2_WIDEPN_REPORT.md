# G2 v2 — WIDE-PN FhwGateV1 emission (native-PN closure) + adoption A/B

Lane: `claude/g2-cert`, worktree `.claude/worktrees/g2-cert` (from `20aee045`).
Author: engineering lane (Claude, Opus). Date: 2026-07-21.
Scope: port the FHW gate closure into the **production wide-PN certificate
builder** so Group-2 gates can fire at the production wide profile, behind the
default-OFF `tss_solver_group2` flag, keeping the never-emit-a-rejected-cert
firewall and flag-off bit identity; then rerun the adoption A/B.

**Verdict up front: DO-NOT-ADOPT at the production config.** Soundness is clean
everywhere (0 parity failures, 0 verifier failures, cross-verification vacuous).
The design §5 native-PN closure — including the DefenderPair→`C_s` grammar
reduction that both prior lanes named the "largest single unknown" — **is now
built and lives inside the wide-PN builder**, and the single-stone wide closure
is proven live end-to-end (fires + self-verifies + never-decides-less). But
firing at the production config is still **zero**, because the hostile-reviewed
**no-mixing class rule** (a verifier preflight, FROZEN this lane) requires EVERY
forced node of a certificate to be a compact-bound gate, and production wide
certificates contain forced nodes outside the v1 compact `H_Q` bound. Lifting
that is a verifier class-rule amendment (mixed legacy+gate certificates), which
is an HONEST STOP with the exact required change documented below (§5).

Suite: baseline hexfield_eq lib unittest **249/0/37** → **250/0/37** (+1 new
wide-PN emission test), plus `hexfield 68 / hexo_engine 23 / hexo_models 57 /
hexo_utils 8`, **0 failed across every binary**. Command:
`CARGO_TARGET_DIR=E:/cargo-targets/g2-cert cargo test --features python
--target x86_64-pc-windows-msvc -- --test-threads=1`.

**Frozen-file law honored.** `git diff` on `tss_verify.rs` and
`tss_verify_group2.rs` is **EMPTY** (byte-identical to `20aee045`). All
production and test changes are confined to `tss_solver.rs` (324 insertions).

---

## 1. What was built (native-PN closure, file / symbol map)

The wide-PN prover proves positions with a group2-agnostic search, then
`WideProofMaterializer` walks the proven tree into a `TssCertificate`. Gate
emission is added to that materializer; the PN search is untouched.

| piece | location (`tss_solver.rs`) |
|---|---|
| `group2` field on the materializer + `materialize(state, root, group2)` param | `WideProofMaterializer`, `WidePnSearch::materialize` |
| Dispatch: forced single-stone AND ⇒ gate; two-stone forced turn ⇒ nested gate; else legacy | `WideProofMaterializer::build_universal` (gate arms ahead of the legacy Universal / DefenderPair build) |
| **Single-stone gate** (b∈{1,2}): finder closure, prove `|R|` representative subtrees among the forced replies, emit reduced `FhwGateV1` | `build_single_stone_gate` |
| **DefenderPair nested gate** (design §5 native-PN closure): a b=2 OUTER gate whose one-stone representative children `C_s = P_Q + s` are **materialized** as nested b=1 gates; the inner gates' second-stone representative subtrees are drawn from the already-proven pair children | `build_defender_pair_gate` |
| Shared `FhwGateV1` node builder (empty rows; finalizer fills them) | `alloc_fhw_gate` |
| Dual-materialize + self-verify firewall + flag-off fallback | `materialize_group2_wide` (free fn) + `prove_for_wide_pn_with_lazy_frontier` |
| `group2` threaded wide-side | `prove_for` → `prove_for_wide_pn_with_lazy_frontier(..., group2)` |

**The DefenderPair reduction (the hard part).** The wide search aggregates the
defender's whole b=2 turn as pair replies with checked commutations, and the
gate grammar (no commutation, one-stone representative children `C_s`) cannot
represent that directly — the exact diagnosis in `G2V2_ADOPTION_REPORT.md §4.3`.
`build_defender_pair_gate` resolves it by **materializing the intermediate
one-stone child state `P_Q + s` for representatives ONLY** (|R_outer| of them,
not |K| or |Legal|), running the finder closure there to get the inner b=1 gate,
and reusing the proven two-stone pair children as the inner gate's second-stone
representative subtrees. This is precisely the "only |R| subtrees needed"
reduction the design points at, realized inside the wide engine.

**Soundness firewall (unchanged contract).** After the group2 materialization,
`materialize_group2_wide` runs `finder_finalize_group2` (canonical strict tree,
derived scalars/rows, both digests) and a strict in-process `Group2Verifier`
self-verify. Only a self-verified, gate-bearing cert is returned; ANY closure
failure, mixed/legacy-only cert, or verifier rejection returns `None` and the
caller falls through to the **bit-identical flag-off materialization**. The
outer mint (`tree.rs` `hard_value_from_verified_group2`) re-verifies the
returned cert under `Group2Verifier` — a second, independent firewall.

**Never-decides-less is structural.** The PN search runs once; the group2 pass
only re-materializes the same proven root. If the gate pass returns `None`, the
result is the exact legacy cert flag-off would produce. So flag-on decides a
position iff flag-off does — by construction, not statistics.

**Flag-off bit identity.** All new work is behind `group2` (default OFF); with
the flag off the `group2.then(...)` guard is never evaluated and the path is the
untouched legacy `materialize(state, root, false)`. Flag-off determinism/golden
tests stay green inside the 250.

## 2. Live validation (single-stone wide closure)

`wide_pn_emits_reductive_gate_on_forced_single_stone_defender`
(`tss_solver.rs` tests): the narrow-emission fixture (P0 defender, SecondStone,
b=1, single named threat, kernel `{(4,1),(5,1)}`, FC coverage stone `(9,1)` ⇒
reductive `R = {(4,1)} ⊊ K`) is solved through the **production leaf profile
(`wide=true`)**. Flag-on emits a reductive `FhwGateV1` (`|K|=2`, `|R|=1`) INSIDE
the wide-PN certificate builder; it survives finalize + the in-process strict
self-verify, re-verifies here under `Group2Verifier`, and flag-off decides the
identical verdict gate-free (extension-free cert asserted). This is a genuine
design §5 native-PN closure fired by the ported wide-path code — the piece
prior lanes documented as structurally out of reach.

The b=2 dispatch is additionally exercised (no crash, never-decides-less) by the
pre-existing `gate_dispatch_never_decides_less_on_b2_node`, which now runs
`solve_b2(true)` through the wide path.

## 3. Adoption A/B (group2-on vs group2-off, production config)

Instrument: `hexfield_eq_deep_solve_batch` via the freshly built Windows cdylib
staged at `.gate/g2ab/_rust.pyd` (CPU-only, `CUDA_VISIBLE_DEVICES=-1`, serial),
production coverage config (`node_cap=500, goal=both, horizon=0 (unbounded),
wide=true, dual_pass=true, zone=false`) over the dev splits + the 19-position
forcing corpus. Driver `.gate/g2ab/ab_driver.py`; raw `.gate/g2ab/ab_result.json`.
Only nodes/decision is quoted (never wall time).

| cohort | n | decided off (W/L) | decided on (W/L) | nodes/dec off | nodes/dec on | gate certs | zone-G2 certs |
|---|---|---|---|---|---|---|---|
| selfplay_v1 | 3255 | 257 (189/68) | 257 (189/68) | 80.9 | 80.9 | 0 | 0 |
| human_v1 | 2720 | 727 (393/334) | 727 (393/334) | 55.2 | 55.2 | 0 | 0 |
| puzzle_v3 | 468 | 228 (118/110) | 228 (118/110) | 181.7 | 181.7 | 0 | 0 |
| forcing_corpus | 19 | 5 (3/2) | 5 (3/2) | 107.8 | 107.8 | 0 | 0 |
| **total** | **6462** | **1217** | **1217** | — | — | **0** | **0** |

Gates:
- **(a) Verdict parity:** PASS — 0 positions decided OFF but not ON (coverage
  identical per cohort and class).
- **(b) Verifier failures:** PASS — 0 in both arms, all cohorts.
- **(c) Cross-verification:** vacuous (no gate-decided positions at production).
- **Firing:** 0 gate certs / 0 gate nodes across 6462 positions.

Nodes/decision is byte-identical ON vs OFF because the group2 materialization
returns `None` every time and the fallback cert is the flag-off cert; the extra
materialization pass is wall-time only (not quoted) and does not touch the
search node count.

## 4. Why firing is zero at production (the precise boundary)

The single-stone closure PROVABLY works (§2), so the boundary is not the
emission machinery. It is structural, and stacks two frozen constraints:

1. **Turn cadence ⇒ b=2 DefenderPair, not single-stone.** Post-opening turns
   always start at FirstStone (b=2), so a turn-start forced defender node is a
   two-stone DefenderPair; standalone b=1 single-stone forced universals occur
   only mid-turn (absorbed into the pair). So production firing requires the
   nested DefenderPair path (§1), which this lane built.

2. **No-mixing class rule + compact `H_Q` bound (verifier preflight, FROZEN).**
   A certificate carrying any new node must have EVERY forced node be a gate
   (`tss_verify_group2.rs` preflight rule 2/3): a single legacy
   implicit-dispatch Universal inside the cert ⇒ REJECT. The v1 gate grammar
   only admits a compact threat family (`|H_Q| = 1` at b=1, `≤ 3` at b=2). The
   prior finder measurement (`G2_FINDER_CLOSURE_REPORT.md §5`) found **214 / 918
   eligible nodes fail `ThreatCountOutOfRange`** — real forcing certificates
   routinely contain forced defender nodes with `>3` named threats. For a whole
   cert to gate, ALL of its forced nodes (across both nesting levels) must close
   inside the compact bound simultaneously; one out-of-bound node makes the cert
   mixed ⇒ self-verify rejects ⇒ clean fall back to the legacy cert. Across
   1217 decided production certs, none met that all-gates bar.

So the reductive prize is real at individual nodes (525/704 reductive closures
measured previously), but it is not harvestable at the certificate level under
the current no-mixing rule. This lane resolved the DefenderPair materialization;
the residual boundary has moved to the verifier's class rule.

## 5. HONEST STOP — the exact verifier change adoption now requires

Firing at production is gated by a FROZEN, hostile-reviewed verifier preflight.
Per the lane's hard law, this is documented, not edited:

- **Required change:** amend the no-mixing rule (design §2.3 rule 2/3;
  `tss_verify_group2.rs` `preflight_structure` + the `implicit_dispatch` legacy
  Universal rejection) to permit **mixed certificates** in which some forced
  nodes are `FhwGateV1` gates and the rest are legacy full-forced-reply
  Universals, with the gate soundness argument (FHW-T3-R omitted-reply coverage)
  applied per-gate and the legacy T3 dismissal applied per-legacy-node. This is
  a genuine soundness extension (the two channels must be shown non-interfering
  at a shared parent), so it needs its own adversarial review — exactly the
  "verifier class-rule amendment would need its own hostile review" the prior
  lane flagged.
- Optionally, raising the v1 compact `H_Q` bound (b=2 beyond 3 named threats)
  would enlarge the gateable-node fraction, but does not by itself fix the
  whole-cert all-gates requirement; the mixing amendment is the load-bearing one.

Neither is improvised here. The wide-PN emission machinery is ready to consume
either the day the verifier admits it.

## 6. Recommendation: DO-NOT-ADOPT (production config, this build)

Owner's bar: adopt only if the harness score improves (coverage up at equal
soundness, or equal coverage at materially lower nodes/decision). Measured:
coverage identical (1217 = 1217), nodes/decision identical to the last digit,
firing zero. Soundness clean everywhere. Enabling `tss_solver_group2` at the
production config is a strict no-op. This is an economics/boundary
DO-NOT-ADOPT, not a soundness one.

**Net progress vs the prior lane:** the wide-PN native-PN closure (the
DefenderPair→`C_s` reduction, the documented "largest single unknown") is now
BUILT inside the production prover and validated live for the reachable
single-stone case; the boundary that keeps firing at zero has moved from
"emission not built" to the verifier no-mixing class rule (§5).

## 7. Tests added (1 net; suite 250/0/37)

- `wide_pn_emits_reductive_gate_on_forced_single_stone_defender`
  (`tss_solver.rs`) — end-to-end wide-PN gate emission: the wide prover emits a
  reductive `FhwGateV1` that survives finalize + strict self-verify and
  re-verifies; flag-off decides identically gate-free. Self-contained fixture
  (frozen verifier test module untouched).

## 8. Honest stops

- **Wide-PN single-stone gate:** BUILT + validated live end-to-end.
- **Wide-PN nested DefenderPair gate (b=2 native-PN closure):** BUILT + safe
  (self-verify firewall + never-decides-less) + structurally exercised (no
  crash, verdict parity on the b=2 dispatch test). NOT positively
  fixture-validated (a hand-built b=2 nested LOSS fixture where the whole cert
  gates was not constructed) and NOT observed firing on the corpus — the
  no-mixing/compact-bound boundary (§4) blocks whole-cert gating before this
  path's self-verify is ever reached at production. Its soundness is guaranteed
  by the firewall regardless.
- **Firing at production:** zero; gated by the frozen no-mixing class rule
  (§5), an honest stop with the exact required verifier amendment documented.
- **Verifier code:** untouched (`tss_verify.rs`, `tss_verify_group2.rs`
  byte-identical to `20aee045`).
- **Not committed:** per instructions, the orchestrator gates and commits.
