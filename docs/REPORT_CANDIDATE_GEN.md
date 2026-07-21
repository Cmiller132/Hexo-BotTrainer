# Candidate-generation efficiency round

Date: 2026-07-21  
Branch/worktree: `claude/candidate-gen`  
Scope: `packages/hexfield_eq/rust/src/tss_solver.rs`, plus test-profile output in `tss_corpus.rs`

## Disposition

**CODE-FACT.** Four bit-preserving candidate-generation rungs are retained in the uncommitted tree. `tss_verify.rs` was not edited. No sibling-branch code was merged or copied.

**MEASURED.** The final serialized Rust suite with the Python feature is green: 218 passed, 0 failed, 39 ignored. The final 6,443-position production-shape battery is green and has exactly the baseline statuses, nodes, and certificate-representation hashes: 556,452 total nodes, FNV digest `a8c6f3ca3ba55827`, and emitted-row SHA-256 `02CD63718E0D06F83853B523C40F7057626A7A3113264235C3CECB162482CFDB` at baseline, every retained rung, and final.

**MEASURED — BLOCKED GATE.** The Stage-0 Python golden digest could not run in this host. The only installed Python is 3.14 and has neither `pytest`, `numpy`, nor the built `hexfield_eq` extension. The direct gate fails at import with `No module named pytest`; an attempted local dependency install could not reach the package index because network access is restricted. Consequently the brief's full definition of done is not literally green: suite and frozen identity gates pass, but Stage-0 remains unexecuted. The exact failure is in [stage0_golden.log](../.gate/candidate-gen/stage0_golden.log).

## Retained rungs

### 1. Stateless rank-two defender plans

**CODE-FACT.** `forced_defender_pair_plan_direct` directly enumerates the canonical unordered two-covers only when all licensing checks hold: defender `FirstStone`, budget two, nonempty opponent threats, no own immediate win, minimum hitting-set size two, every live threat edge has size one or two, and both kernel and pair counts fit the observed rank-two bound of four. Every other position falls back to the retained dynamic reconstruction.

**CODE-FACT.** Debug builds recompute the historical dynamic plan on every eligible direct call and assert complete plan equality, including kernel order, pair order, priors, and final position keys.

**MEASURED.** At this rung boundary the production battery changed from 51.834 s to 40.892 s (1.268x); solve-only wall changed from 49.956 s to 39.178 s. The full suite was 217/0 and the emitted 6,443-row identity file matched baseline byte for byte.

### 2. First-candidate enumeration residual

**CODE-FACT.** The existing `WindowStore` build now creates the ordered first-candidate vector during the same window scan instead of rescanning the threat analysis. A debug oracle compares the complete candidate vector with the historical generator.

**CODE-FACT.** Count-three incidence is pre-indexed per cell. Exact pair-prefilter counts start from that index and subtract only windows incident to the first move. Defender-only first moves whose maximum count-three incidence is below two are rejected before the remaining window scan; those pairs are provably `None` under the existing gate.

**MEASURED.** Battery wall changed from 40.892 s to 37.741 s (1.084x incremental); solve-only wall changed from 39.178 s to 36.134 s. The full suite was 217/0 and identity remained exact.

### 3. `second_candidates` allocation and hashing churn

**CODE-FACT.** Per-call promoted/fresh vectors became reusable per-node scratch buffers. Membership uses a reusable `HashSet` with a fixed, coordinate-specialized hasher, avoiding randomized `RandomState` initialization and SipHash while retaining insertion/output order. The historical implementation remains test-only as an equality and same-binary A/B oracle.

**MEASURED.** In a same-release-binary A/B, the optimized path took 36.639 s battery / 34.946 s solve, versus 39.748 s / 38.066 s for the forced historical path: 1.085x battery and 1.089x solve.

**MEASURED.** Across 3,109,898 `second_candidates` calls, growth-bearing container events fell from 6,597,406 to 1,208,902, an 81.7% reduction. This counter measures capacity-growth events plus nonempty ephemeral local-vector construction in the reference; it is not a global-allocator call count. The equality oracle ran alongside the optimized path for the instrumented battery and found no divergence.

**MEASURED.** Three alternatives were measured and rejected for wall regressions: linear `Vec::contains` (45.399 s solve), sorted-vector binary membership (45.274 s), and `AHashSet` (44.189 s). The temporary dependency for the last experiment was removed.

### 4. Exact cross-node window-generation memo

**CODE-FACT.** Attacker OR-generation now owns a bounded 512-slot direct-mapped memo. Its exact key is `(WindowKey, player-0 occupancy mask, player-1 occupancy mask)` and its value is a fixed-capacity `CompactEmpties` representation. A placement therefore changes the key only for windows whose position masks changed; collisions replace a slot and cannot create false hits.

**CODE-FACT.** Every debug memo hit recomputes empties from the live `WindowEntry` and asserts equality. A dedicated test proves a repeated unchanged lookup hits and that a placement changing the window masks misses and recomputes.

**MEASURED.** The production battery recorded 16,071,373 lookups and 4,032,020 hits at the rung boundary, a 25.088% hit rate. Repeats ranged from 25.083% to 25.090%. Battery wall changed from rung 3's 36.639 s to 34.189 s (1.072x incremental); solve-only wall changed from 34.946 s to 32.537 s.

**MEASURED.** A first `Arc`/`HashMap` memo achieved roughly 96% hits but regressed solve wall to 42.960 s; removing test-only atomic/fresh-check overhead still left 39.973 s. It was rejected in favor of the compact direct-mapped design.

## Production-shape wall results

All rows use release mode, cap 500, 256 KiB TT, unbounded semantic horizon, wide leaf profile, dual pass, the exact frozen sets `human_v1` + `selfplay_v1` + `puzzle_v3`, and serialized execution.

| Boundary | Battery wall (s) | Solve wall (s) | Incremental battery speedup | Identity |
|---|---:|---:|---:|---|
| Baseline | 51.834 | 49.956 | — | reference |
| Rung 1 | 40.892 | 39.178 | 1.268x | exact |
| Rung 2 | 37.741 | 36.134 | 1.084x | exact |
| Rung 3, optimized A/B arm | 36.639 | 34.946 | 1.030x vs rung 2; 1.085x paired A/B | exact |
| Rung 4, first retained run | 34.189 | 32.537 | 1.072x | exact |

**MEASURED.** The conservative sequential baseline-to-rung-4 comparison is 1.516x for total battery wall and 1.535x for solve wall.

**MEASURED.** Three retained-rung-4 timing runs were 34.189/32.537, 31.154/29.648, and 30.666/29.186 seconds (battery/solve). Their medians are 31.154/29.648 seconds. Compared with the single baseline observation this is 1.664x battery / 1.685x solve, but that comparison has asymmetric sample counts and should not replace the conservative sequential number above. A later final verification run was 30.466/29.007 seconds and is not used in either headline.

## Deep F19 result

F19 used its fixed corpus harness, 256 MiB TT, and the existing per-row caps, including cap 100,000 for the two known unresolved rows.

| Run | Harness wall (s) | Pair gen (ms) | Defender gen (ms) | Regen (ms) | Expand (ms) | Memo hit rate |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 45.97 | 16,081 | 18,677 | 3,835 | 35,148 | — |
| Final, conservative | 32.29 | 11,688 | 9,771 | 1,527 | 21,844 | not printed |
| Final, memo-instrumented repeat | 29.04 | 10,553 | 8,954 | 1,401 | 19,837 | 2,981,411 / 12,743,545 = 23.394% |

**MEASURED.** The conservative F19 wall speedup is 1.424x. The instrumented repeat is 1.583x, reported separately because host-load variation is visible between repeated batteries.

**MEASURED.** Both baseline and final F19 runs reach the same harness assertion for the same two positions, `0l4291i_live` and `lz60mfb`: expected WIN but UNKNOWN at cap 100,000. These fixed-cap failures are pre-existing benchmark outcomes, not optimization mismatches. The exact 6,443-position acceptance battery and serialized full suite are green.

**HYPOTHESIS.** The large pair/defender reductions are consistent with rungs 1–3 removing repeated plan reconstruction, scans, allocation, and hashing; the remaining run-to-run wall spread prevents assigning the full aggregate delta precisely to any one sub-block.

## Gate matrix

| Boundary | Serialized `--features python` suite | 6,443 identity | Stage-0 golden |
|---|---|---|---|
| Rung 1 | MEASURED: 217/0 | MEASURED: exact | BLOCKED: Python stack absent |
| Rung 2 | MEASURED: 217/0 | MEASURED: exact | BLOCKED: Python stack absent |
| Rung 3 | MEASURED: 217/0 | MEASURED: exact | BLOCKED: Python stack absent |
| Rung 4 | MEASURED: 217/0 | MEASURED: exact | BLOCKED: Python stack absent |
| Final source | MEASURED: 218/0 | MEASURED: exact | BLOCKED: Python stack absent |

**CODE-FACT.** The extra final test is the memo mask-delta invalidation regression, hence 218 rather than 217 passing tests.

## Reproduction commands

All Cargo commands were run inside this worktree with:

```powershell
$env:CARGO_TARGET_DIR = (Join-Path (Get-Location) '.cargo-target')
cargo test -p hexfield_eq --features python --target x86_64-pc-windows-msvc -- --test-threads=1

$env:TSS_IDENTITY_OUT = (Join-Path (Get-Location) '.gate/candidate-gen/final.identity.tsv')
cargo test -p hexfield_eq --features python --release --target x86_64-pc-windows-msvc tss_frozen_identity_battery -- --ignored --test-threads=1 --nocapture

$env:TSS_CLOSURE_COUNTERS = '1'
cargo test -p hexfield_eq --features python --release --target x86_64-pc-windows-msvc tss_frozen_identity_battery -- --ignored --test-threads=1 --nocapture

python -m pytest tests/test_hexfield_eq_tss_shadow.py::test_stage0_digest_matches_golden
```

## Raw evidence

- Baseline: [battery](../.gate/candidate-gen/baseline_battery.log), [identity rows](../.gate/candidate-gen/baseline.identity.tsv), [F19](../.gate/candidate-gen/baseline_f19.log)
- Rung 1: [battery](../.gate/candidate-gen/rung1_battery.log), [suite](../.gate/candidate-gen/rung1_full_suite.log), [identity rows](../.gate/candidate-gen/rung1.identity.tsv)
- Rung 2: [battery](../.gate/candidate-gen/rung2_battery.log), [suite](../.gate/candidate-gen/rung2_full_suite.log), [identity rows](../.gate/candidate-gen/rung2.identity.tsv)
- Rung 3: [optimized paired arm](../.gate/candidate-gen/rung3e_new.log), [reference paired arm](../.gate/candidate-gen/rung3e_ref.log), [allocation profile](../.gate/candidate-gen/final_alloc_profile.log), [suite](../.gate/candidate-gen/rung3_full_suite.log), [identity rows](../.gate/candidate-gen/rung3d.identity.tsv)
- Rung 4: [first retained battery](../.gate/candidate-gen/rung4c_battery.log), [repeat 2](../.gate/candidate-gen/final_battery_rep2.log), [repeat 3](../.gate/candidate-gen/final_battery_rep3.log), [suite](../.gate/candidate-gen/rung4_full_suite.log), [identity rows](../.gate/candidate-gen/rung4c.identity.tsv)
- Final: [battery](../.gate/candidate-gen/final_battery.log), [suite](../.gate/candidate-gen/final_full_suite.log), [identity rows](../.gate/candidate-gen/final.identity.tsv), [F19 conservative](../.gate/candidate-gen/final_f19.log), [F19 memo repeat](../.gate/candidate-gen/final_f19_memo.log)
- Blocked Stage-0: [stage0_golden.log](../.gate/candidate-gen/stage0_golden.log)

