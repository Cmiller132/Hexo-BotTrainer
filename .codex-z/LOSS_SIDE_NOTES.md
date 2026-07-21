# R-LOSS-SIDE running notes

Date: 2026-07-20  
Branch/worktree: `claude/loss-side` / `.claude/worktrees/loss-side`  
Base observed: `07bd492c`, stacked on dual-pass implementation `6f044e2d`.

## Constraints and invariants

- No commits; leave all changes in the worktree.
- `packages/hexfield_eq/rust/src/tss_verify.rs` must remain byte-for-byte unchanged.
- New loss policy is config-gated and default-off; the existing `tss_solver_dual_pass`
  behavior remains the default/current behavior.
- Every returned verdict continues through `tss_solve_verified` and the strict verifier.
- Per-position aggregate nodes must never exceed the caller's node cap.
- Frozen harness sets and gates are not modified.

## Running log

- Started by recording the clean tracked state. The only pre-existing untracked file
  was `.codex-z/BRIEF_LOSS_SIDE.md`; it is user-owned and will be preserved.
- Launched parallel read-only audits of verifier theory, Lane C cost data, and the
  existing dual-pass/config/API/manifest plumbing.
- **Pre-implementation prediction checkpoint (no solver code edited yet):** parsed
  all 889 Lane C rows and joined the 116 atlas-loss labels in the frozen puzzle
  dev split to the existing dual-pass standard archive. The current arm proves
  58 and misses 58. Every missed proof costs at least 512 dedicated-loss nodes
  (conventional median 1,064; type-7/inclusive p90 2,856.3; max 19,536), so no
  mere ordering/split policy can recover any of those 58 beneath the unchanged
  aggregate cap of 500. Search efficiency or a higher cap is required.

## Policy predictions made before implementation

All counts below use the recorded `loss_pass.deep_nodes` from
`raws/lanec_labels.jsonl`. A configured first-probe allowance is compared with
that dedicated-pass cost; exact shared-root accounting will be stated in the
design and tests.

| Policy at cap 500 | Predicted coverage | Predicted cost / win risk |
|---|---|---|
| Leftover-only (current dual pass) | Existing standard archive: all 58/116 atlas losses with dedicated cost <=500 are proved; the other 58 are not. It cannot inspect the opponent claim after a cap-bound primal. | Preserves the full primal cap and therefore win coverage. Extra work occurs only when the primal returns early. |
| Loss-first, bounded 32 | Lane C gross recall: 24/192 labeled losses (1/150 atlas, 21/40 human, 2/2 forcing). No incremental recovery is predicted among the 58 missed atlas losses. | Most Lane C `Unknown` loss passes terminate at 2 nodes (402/426; p50/p90 both 2), consistent with the V1 16--22 microsecond probe. Threat-rich misses can consume the full bound and can hide a near-cap win. Cheap proved losses skip the expensive primal. |
| Loss-first, bounded 48 or 64 | Both thresholds have the same Lane C gross recall: 29/192 (1 atlas, 26 human, 2 forcing); +5 human labels over 32. No incremental recovery is predicted among the 58 missed atlas losses. | Same usual 2-node quiet cost, with a 48/64-node worst-case debit before a win grind. Sixty-four is the most useful single experimental bound because it covers the 33–48-node cluster without exceeding 12.8% of cap. |
| Reserved loss floor 64, primal first | At best the same <=64 dedicated-loss class when the primal is cap-bound, while retaining larger leftover loss budgets when the primal exits early. Still zero predicted atlas-miss recovery. | Clips only a cap-bound primal; when the primal exits early, the current dual-pass rule still gives the loss attempt every actual leftover node. It risks near-cap wins but does not duplicate opponent work. |

Initial decision before touching solver code was to prototype one default-zero
bounded loss-first knob. That decision was superseded during the pre-measurement
audit below, before completing or exposing that API.

### Refinement from paired archives (before any harness measurement)

- A preprobe and the post-primal dual are fresh `WidePnSearch` instances; a
  capped preprobe is not resumable. With both enabled, the opponent work is
  duplicated. Exact Lane C accounting predicts puzzle-dev losses 92 -> 88/87/86
  for probe bounds 32/48/64, and quick puzzle losses 10 -> 8 at every bound
  (the 471- and 480-node proofs no longer fit after the discarded probe).
- On the exact quick cap-bound IDs, prior full-budget loss-goal archives show
  human 0/22 and selfplay 0/26 losses; Lane C shows puzzle 0 losses. Therefore
  the predicted incremental quick yield of any allocation policy is exactly
  zero, subject only to engine-version/cache drift.
- A primal-first reserve combined with the existing actual-leftover dual rule
  preserves all 412 current standard dev losses: every dual-added loss has a
  primal cost of 2 nodes, so the opponent still receives 498 nodes. Quick has
  no wins costing above 436, hence reserves 32/48/64 have zero predicted quick
  win loss. Standard near-cap win risks are 11/13/16 respectively.
- The 58 missed atlas losses remain impossible at cap 500: min 512, conventional
  median 1,064, inclusive p90 2,856.3, max 19,536 dedicated-loss nodes.

Revised implementation decision: expose only a default-zero primal-first
`loss_reserve_nodes` floor, retain the current dual-pass actual-leftover block
unchanged, and measure reserve 32 first. The nonresumable preprobe is rejected
before API completion because it is predictably loss-regressive.

The final split clamps a configured reserve to at most one less than the
post-root allowance, so a `Both` solve with any available search work always
runs a nonempty primal. A positive reserve schedules its fixed opponent floor
even with `dual_pass=false`; `dual_pass=true` upgrades it to every actual
leftover node after an undecided primal.

Implementation plumbing is complete under production key
`tss_solver_loss_reserve_nodes` and harness key `loss_reserve_nodes`: Python
config/divergence maps, strict Rust parsing, root/inline/async requests, batch
API, shared-resolver manifest echo, and benchmark overlay/scorecard echo.
Existing harness gates are untouched. Default zero preserves current behavior.
The final runs explicitly declared both loss-budget fields, making the existing
manifest-subset gate check their shared-resolver echoes. Wide `Both` is the
only affected solve shape, and every verdict still goes through
`tss_solve_verified`.

## Measurements

Final matching-build quick archives (native release PyO3 build, coverage gates
only because the required Linux/GPU bench venv was unavailable):

- anchor, `dual_pass=true`, reserve 0:
  `scripts/tss_harness/harness_runs/20260720_234546_loss_reserve0_gated`;
- arm, `dual_pass=true`, reserve 32:
  `scripts/tss_harness/harness_runs/20260720_234604_loss_reserve32_gated`.

Both runs report `GATES: ALL PASS`. The paired arm has no upgrades, downgrades,
or verified contradictions.

| Quick set | Anchor W/L | Reserve-32 W/L | Total nodes 0 -> 32 | Max nodes | Over cap | Verify failed |
|---|---:|---:|---:|---:|---:|---:|
| human_v1 (338) | 48 / 42 | 48 / 42 | 19,068 -> 18,572 (-496) | 500 | 0 | 0 |
| puzzle_v3 (48) | 11 / 10 | 11 / 10 | 17,412 -> 16,916 (-496) | 500 | 0 | 0 |
| selfplay_v1 (343) | 18 / 7 | 18 / 7 | 19,844 -> 19,110 (-734) | 500 | 0 | 0 |
| **Total** | **77 / 59** | **77 / 59** | **56,324 -> 54,598 (-1,726)** | **500** | **0** | **0** |

The prediction was exact: no incremental loss and no lost win. Reserve 32
reduces nodes on 56 quiet unknowns because the opponent attempt exhausts its
restricted width early (55 records save 31 nodes, one saves 21). This is an
economic side effect, not deeper loss coverage.

## Test/build record

- The requested `CARGO_TARGET_DIR=E:/cargo-targets/loss-side` is outside this
  managed workspace and failed with `Access denied`. All Cargo work used the
  isolated workspace target `.codex-z/cargo-target` instead, from Windows Git
  Bash with the MSVC toolchain and `--features python`.
- `cargo test --features python --no-run`: pass.
- Focused final loss-reserve suite: 7 passed, 0 failed.
- Full serialized final suite: 197 passed, 37 ignored, 0 failed.
- A parallel rerun before the final two boundary tests had 194 passed and 37
  ignored; only the known pre-existing
  `tss_cap_resume::cap_resume_discards_on_binding_or_cap_mismatch` failed with
  `UnsupportedProfile` while environment-mutating warmth tests ran. The final
  serialized rerun passed that test, as required.
- Harness selftest: 19 / 19 intentional violations caught (unchanged gates).
- Direct Python config smoke test: default/off and enabled 32 values propagate
  identically through the base and Fast divergence maps.
- Adapter range smoke test: accepts 0 and `u32::MAX`, rejects -1 and
  `u32::MAX + 1` before the PyO3 call.
- The zero-reserve test compares an implicit-default constructor against an
  explicit setter at both dual-pass settings, including status, certificate,
  and stats. Separate tests cover a positive reserve without dual-pass, strict
  total-cap accounting, no-primal-skip clamping, inactive goals/profiles, config
  parsing, and manifest/real-resolver identity.
- This Windows host has no installed WSL distribution, `/root` venv, or usable
  network package install. The final Rust revision was therefore built as
  native release PyO3 extensions and loaded through the runner; both quick
  runs used `--no-bench`. No benchmark claim is made.
- `cargo fmt --all --check` reports broad pre-existing formatter-version diffs
  across the repository; it changed no files. `git diff --check` is clean.
- `packages/hexfield_eq/rust/src/tss_verify.rs` remains untouched.

## Final recommendation

Do **not** promote reserve 32 to a standard-tier run. It preserves quick wins
and losses and saves 3.1% of sampled nodes, but the required loss-coverage gain
is zero. The frozen standard archive predicts 11 near-cap wins at risk under
reserve 32; reserve 48/64 raise that risk to 13/16 without any plausible
cap-500 atlas gain.

Keep `tss_solver_loss_reserve_nodes` experimental and default zero. The 58
unreachable atlas losses have recorded standalone loss costs above cap 500:
27 become reachable by 1,000 nodes, 48 by 2,000, 56 by 4,000, 57 by 5,000, and
all 58 by 20,000. At fixed cap 500, pursue resumable dual work, proof reuse, or
search-efficiency improvements rather than another split/order policy.
