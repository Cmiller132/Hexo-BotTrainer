# G2 v2 — FhwGateV1 production emission + adoption A/B

Lane: `claude/g2-cert`, worktree `.claude/worktrees/g2-cert`.
Author: engineering lane (Claude, Opus). Date: 2026-07-21.
Scope: discharge advisory **A1** (end-to-end positive FrontierCovered
certificate), port the structural closure into the production selector path
behind `tss_solver_group2` (default OFF), keep the never-emit-a-rejected-cert
contract, extend the suite, and run the group2-on/off adoption A/B.

Suite: baseline hexfield_eq lib unittest **243/0/37** → post-change
**249/0/37** (+6 new tests), and `hexfield 68 / hexo_engine 23 /
hexo_models 57 / hexo_utils 8`, **0 failed across every binary**. Command:
`CARGO_TARGET_DIR=E:/cargo-targets/g2-cert cargo test --features python
--target x86_64-pc-windows-msvc -- --test-threads=1`.

**Verdict up front: DO-NOT-ADOPT at the production config** (§5) — soundness
is clean everywhere, but the production harness profile is the wide-PN prover,
which structurally never reaches the v1 selector, so enabling the flag changes
nothing there (identical coverage, identical nodes/decision, zero firing). The
emission machinery itself is proven live end-to-end on the narrow path,
including the first solver-emitted, verifier-accepted, cross-verified
reductive FC gate certificate.

---

## 1. A1 — FrontierCovered accept path: DISCHARGED

The FC (`d != s`) accept branch previously had no end-to-end positive fixture
(hostile review advisory A1). Discharged with a genuine reductive FC
certificate that verifies through the accept path.

**Construction** (`tss_verify_group2.rs`, `FC_GATE_MOVES` /
`accepted_fc_gate_cert`). The b=1 Exact fixture board densified by relocating
one inert defender scatter stone to `(9,1)`, a far-`+q`-frontier stone whose
`B_8` ball closes the last 10 uncovered cells of `B_8((5,1))`
(`LEGAL_RADIUS = 8`; the uncovered residue was the q=12..13 arc). This makes
`B_8((5,1)) ⊆ Λ(P_Q + (4,1))`, so `(5,1) -> (4,1)` is a genuine `d != s`
FrontierCovered coupling and one representative covers the whole kernel:
`R = {(4,1)} ⊊ K = {(4,1),(5,1)}`.

**Positive test** `fc_gate_certificate_reductive_reconstructs_and_verifies`:
`|K|=2`, `|R|=1`, exactly one FC coupling (recomputed FC on the board), cert
**verifies** under `Group2Verifier` (rejects under legacy / under `Win`), and
all **12 D6 images verify**.

**Mutation twin** `fc_gate_mutation_twin_rejects`: (a) the same reductive cert
on the sparse base board (coverage stone absent) **rejects** — the verifier
recomputes the coupling as NonFrontierCovered and reconstruction fails;
(b) relabelling the FC edge as Exact or NonFrontierCovered on the dense board
**rejects** (recomputed class ≠ stored byte).

Consequence: FC emission is **enabled** in the production closure
(`GROUP2_FHW_ALLOW_FC = true`); NonFC stays fail-closed (no gate, legacy
fallback).

## 2. Production emission design (file / symbol map)

At an eligible **forced (implicit-dispatch) defender node** on the narrow
selector path, the solver runs the structural FHW closure, proves one
representative subtree per FC-cover class, and emits a reduced `FhwGateV1` in
place of the full forced-reply `Universal` — `|R|` proven children instead of
`|K|` forced replies.

| piece | location |
|---|---|
| Structural closure (reuses the verifier's own geometry — H_Q, exact τ==b, kernel, greedy FC-cover, per-edge class, R1 escape) | `packages/hexfield_eq/rust/src/tss_verify_group2.rs` `finder_build_fhw_gate` + `FhwGateSkeleton` |
| Dispatch at forced nodes (flag-on, `!emitted_dirty`, post-opening) | `packages/hexfield_eq/rust/src/tss_solver.rs` `prove_universal` (gate branch ahead of the zone selector) |
| Emission (prove reps, skeleton map with empty rows, alloc) | `tss_solver.rs` `prove_universal_fhw_gate` |
| Compaction gate arm (remap representative children; refuses already-filled gates) | `tss_solver.rs` `compact_certificate` |
| Finalize row-fill (role/window rows derived from the proven subtrees before the check + digest passes) | `tss_verify_group2.rs` `finder_finalize_group2` (calls `finder_fill_gate_rows`) |
| Firing telemetry (`gate_nodes` per solve row) | `packages/hexfield_eq/rust/src/search.rs` `hexfield_eq_deep_solve_batch` |

**Enabled-class gating.** `finder_build_fhw_gate` returns a skeleton only when
every edge is Exact or (A1-discharged) FrontierCovered; any NonFC edge ⇒
`None` ⇒ the unchanged legacy full-coverage path.

**Soundness firewall (kept intact).** Every group2 attempt passes through
`finder_finalize_group2` + a strict in-process `Group2Verifier` self-verify at
the finalize boundary (`tss_solver.rs` `prove_narrow_compat`) BEFORE
consumption; failure drops the cert and triggers the clean group2-off
re-solve. The solver never emits a cert the strict verifier rejects, and
flag-on never decides fewer positions — structural, not statistical.

**Flag-off bit identity.** All emission is behind `self.group2` (default OFF);
the flag-off determinism/identity tests stay green inside the 249; flag-off
certs remain extension-free (asserted).

**Live end-to-end emission proof** (`solver_emits_reductive_fc_gate_on_narrow_path`,
board `NARROW_EMIT_MOVES`): a single left-blocked threat forces the kernel
`K = {(4,1),(5,1)}`; the coverage stone makes both couplings FC so the greedy
cover emits `R = {(4,1)} ⊊ K`; after the forced hit the claimant's two
count-3 groups extend to τ>3 loss leaves, so the gate is the proof's only
forced node. The solver flag-on returns a certificate CONTAINING the reductive
FC `FhwGateV1`; it survives finalize + self-verify, re-verifies independently,
and flag-off decides the identical verdict gate-free. Via the batch API:
on = loss/15 nodes with `gate_nodes=1`, off = loss/13 nodes, cross-verified
(§4c).

## 3. Tests added (6 net; suite 249/0/37)

- `fc_gate_certificate_reductive_reconstructs_and_verifies` — A1 positive (+12 D6 images).
- `fc_gate_mutation_twin_rejects` — A1 mutation twin (break the coupling ⇒ reject, both ways).
- `production_closure_fires_at_forced_b2_node` — the ported closure fires at a
  forced b=2 two-threat node (`|K|=4`, sparse ⇒ all-Exact `R==K`), FC-off arm
  identical.
- `finalize_handles_reductive_fc_gate_certificate` — the finalize boundary
  (re-)fills gate rows and the result verifies (the exact live-emission path).
- `gate_dispatch_never_decides_less_on_b2_node` — wide-profile flag-on/off
  verdict parity on the forced b=2 board; any returned extension cert must
  verify.
- `solver_emits_reductive_fc_gate_on_narrow_path` — end-to-end solver emission
  (§2), never-decides-less at the emission boundary.

## 4. Adoption A/B (group2-on vs group2-off)

Instrument: the pure-Rust `hexfield_eq._rust.hexfield_eq_deep_solve_batch`
(freshly built Windows cdylib staged at `.gate/g2ab/_rust.pyd`, CPU-only,
`CUDA_VISIBLE_DEVICES=-1`, serial single-process batch) at the production
coverage config (`node_cap=500, goal=both, horizon=0 (unbounded), wide=true,
dual_pass=true, zone=false`) over the dev splits + the 19-position forcing
corpus. Driver `.gate/g2ab/ab_driver.py`; raw JSON `.gate/g2ab/ab_result.json`.
Verdicts are load-robust; only nodes/decision is quoted (never wall time).

### 4.1 Coverage / nodes / firing (production config)

| cohort | n | decided off (W/L) | decided on (W/L) | nodes/dec off | nodes/dec on | gate certs | zone-G2 certs |
|---|---|---|---|---|---|---|---|
| selfplay_v1 | 3255 | 257 (189/68) | 257 (189/68) | 80.9 | 80.9 | 0 | 0 |
| human_v1 | 2720 | 727 (393/334) | 727 (393/334) | 55.2 | 55.2 | 0 | 0 |
| puzzle_v3 | 468 | 228 (118/110) | 228 (118/110) | 181.7 | 181.7 | 0 | 0 |
| forcing_corpus | 19 | 5 (3/2) | 5 (3/2) | 107.8 | 107.8 | 0 | 0 |
| **total** | **6462** | **1217** | **1217** | — | — | **0** | **0** |

### 4.2 Gates

- **(a) Verdict parity:** PASS — 0 positions decided OFF but not ON (and 0
  vice versa; coverage identical per cohort and per class).
- **(b) Verifier failures:** PASS — 0 in both arms across all cohorts.
- **(c) Cross-verification:** vacuous on the production arm (no gate-decided
  positions). On the narrow supplementary arm (§4.3) the one gate-decided
  position (`narrow_emit_fixture`, verdict loss with `gate_nodes=1`)
  re-solves group2-OFF at the 50k budget to the SAME verdict with zero
  verifier failures — **no soundness alarm**.

### 4.3 Why firing is zero at production config (diagnosis, exact)

- The production profile (`wide=true`) runs the **wide PN prover**; its
  certificate builder constructs forced defender AND nodes directly
  (`tss_solver.rs` `build_universal` / `build_defender_pair_universal`,
  `forced_defender_pair_plan`) and never calls
  `NarrowCompatSearch::prove_universal`, which is where ALL v1 Group-2
  emission lives (the pre-existing zone selector too — note `zone-G2 certs = 0`
  in §4.1, same cause; this reconfirms the prior lane's documented boundary
  "Group-2 emission on the WIDE profile is out of scope, design §5").
- Independently, the hostile-reviewed class rule 2/3 (no mixing: any
  `implicit_dispatch` legacy Universal inside an extension cert ⇒ REJECT,
  `tss_verify_group2.rs` preflight) means a gate can only appear in a
  certificate whose EVERY forced node is a gate. Wide-PN certs carry many
  forced nodes (including two-stone `DefenderPair` units that have no
  intermediate one-stone child state, which the gate grammar's representative
  subtree `C_s` requires), so a partial port would be self-verify-rejected
  into fallback anyway.
- Porting emission into the wide-PN builder = design §5's native-PN
  Open/Closed closure + a pair-node grammar question — the explicitly
  out-of-scope "largest single unknown" of both prior lanes, and a verifier
  class-rule amendment would need its own hostile review. Not improvised here.

Narrow-arm supplement (19-corpus + emission fixture, narrow profile, zone on,
cap 2k): 17/19 corpus positions are Unknown in BOTH arms (the corpus needs the
wide engine; the narrow prover is weak there), 2 decide as trivial loss leaves
(1 node, no AND node ⇒ no gate site). The emission fixture fires
(`gate_nodes=1`) and cross-verifies. Parity clean, verify failures 0. One
economics note: on narrow-arm UNKNOWN positions, flag-on costs ~2x nodes
(3022 vs 1508 etc.) — the documented v1 fail-safe re-solve after failed
selector attempts; this cost is absent at the production config (nodes
byte-identical, §4.1) because the selector never runs there.

## 5. Recommendation: **DO-NOT-ADOPT** (production config, this build)

Owner's bar: adopt only if the harness score IMPROVES (coverage up at equal
soundness, or equal coverage at materially lower nodes/decision). Measured:
coverage identical (1217 = 1217), nodes/decision identical to the last digit,
firing zero. Enabling `tss_solver_group2` at the production config is a strict
no-op — there is nothing to buy, and on any narrow-profile deployment it would
cost ~2x nodes on undecided positions. Soundness is clean everywhere
(parity, zero verifier failures, cross-verification agree), so this is an
economics DO-NOT-ADOPT, not a soundness one.

**What adoption would actually require** (the honest path, for the next lane):
port gate emission into the wide-PN certificate builder (design §5 native-PN
closure) so forced nodes there attempt closure, and resolve the
`DefenderPair`-vs-`C_s` grammar mismatch (either build the intermediate
one-stone child states or amend the class rules — the latter needs re-review).
The reductive prize at production-shaped forced nodes is real (525/704
measured reductive closures; avg |K| 2.12 vs |Legal| 580.9), but it is
reachable only from inside the wide engine.

## 6. Honest stops

- Wide-PN emission: NOT built (out-of-scope §5 boundary, documented above).
- NonFC emission: stays fail-closed (unchanged).
- The A/B ran on a debug-profile build (both arms same binary, so coverage and
  nodes comparisons are valid; wall times are meaningless and not quoted).
- Not committed: per instructions, the orchestrator gates and commits.
