# TSS production bottleneck diagnostic: cap 750, J2near off

Date: 2026-07-21  
Code base: `55f3c2b6d4dede69d590b462d0b2f2f0e4198cbf` plus test-only instrumentation  
Production shape: cap 750, `vcf_pair_complete` (J2near off), dual pass, 256 KiB TT, unbounded semantic horizon

## Executive result

**MEASURED.** The old 32.85% unattributed block is principally state
make/unmake plus proof-number control work. At cap 750, exclusive state
`apply_with_delta`/`undo` time is 24.10% and PN selection, recomputation,
backpropagation, stage reopen/refresh, and expansion bookkeeping is 7.71%.
Together they are 31.80% of solve wall. Certificate work (1.31%) and the
exhaustive other/orchestration bucket (1.46%) account for the small remainder.

**MEASURED.** The phase buckets sum to exactly 100.000% of the test-only phase
clock. The cap-750 instrumented battery took 38.7720 s; its bucket sum was
38.7715 s and the begin/take clock gap was 0.00046 s. The uninstrumented
cap-750 battery median in the same run was 37.9989 s, so profiling overhead was
2.03%.

**MEASURED.** Node identity passed twice: the uninstrumented and instrumented
cap-750 batteries both produced exactly **730,143 nodes**, matching
`REPORT_J2NEAR_CAP.md`. Cap 500 likewise matched its production-off anchor at
556,452 nodes. Every row also retained the same status, node count, verifier
result, and endpoint root `pn/dn` across three uninstrumented repetitions and
the instrumented pass. There were zero verifier failures.

**HYPOTHESIS.** The next speed round should attack (1) the combined
attacker/window/second-candidate path, (2) state make/unmake, and (3) guarded
cap-bound abandonment. Direct TT optimization is too small to rank.

## 1. Full phase decomposition at cap 750

**CODE-FACT.** The clock is exclusive and nested: entering a child bucket
stops its parent bucket, and `Other` is active from `begin_cap_profile()` to
`take_cap_profile()`. Thus no time is double-counted and there is no residual
subtraction bucket. All clock types, storage, guards, call sites, snapshots,
and the runner module are behind `#[cfg(test)]`; a non-test release MSVC
`cargo check` passed. `tss_verify.rs` has no diff.

**MEASURED.** Cap-750 phase shares over all 6,443 rows:

| Exclusive phase | Seconds | Share of profiled solve wall |
|---|---:|---:|
| Candidate / attacker generation, excluding window construction | 8.102 | 20.90% |
| Second-candidate regeneration | 2.376 | 6.13% |
| Window analysis, gate build, and maintained-window queries | 8.310 | 21.43% |
| Defender child generation/materialization | 0.287 | 0.74% |
| Defender-pair plan construction | 6.113 | 15.77% |
| PN select/recompute/backprop/stage bookkeeping | 2.988 | 7.71% |
| TT exact-key probe/store | 0.122 | 0.31% |
| Search state make/unmake | 9.342 | 24.10% |
| Certificate extraction, relabel, verification, and compaction | 0.509 | 1.31% |
| Per-attempt setup/root insertion | 0.057 | 0.15% |
| Other solve orchestration | 0.565 | 1.46% |
| **Total** | **38.772** | **100.00%** |

**CODE-FACT.** `Other` contains dual-pass routing, immediate-winner routing,
attempt/result assembly, stats folding, and teardown not owned by a more
specific guard. The PN bucket includes the full proof-number run loop and
stage maintenance, but nested generation, window, TT, and state guards remove
those costs from PN. The defender-plan bucket is specifically
`forced_defender_pair_plan`; materializing its returned children remains in
defender generation. Certificate state replay is charged to certificate work,
not search state make/unmake.

**MEASURED.** The prior profile's headline buckets reconcile with this split.
At cap 750, attacker generation + window work is 42.33%, defender generation +
plan work is 16.51%, second-candidate regeneration is 6.13%, TT is 0.31%, and
setup is 0.15%. Most importantly, state make/unmake + PN bookkeeping is
31.80%, naming essentially all of the prior 32.85% residual.

## 2. Per-outcome and per-row attribution

### Outcome shares

**CODE-FACT.** With the unbounded horizon and this restricted-width solver,
an `Unknown` ending at exactly `nodes == cap` is `UNKNOWN-at-cap`; an `Unknown`
ending below cap is structural width exhaustion/self-termination. Decided rows
are classified by their verified final status.

**MEASURED.** Cap-750 outcome attribution uses uninstrumented solve wall. Wall
seconds and wall shares are medians across the three full-battery
repetitions; nodes are deterministic.

| Final row status | Rows | Median wall s | Wall share | Nodes | Node share |
|---|---:|---:|---:|---:|---:|
| WIN | 757 | 4.926 | 13.01% | 91,854 | 12.58% |
| LOSS | 544 | 3.643 | 9.59% | 64,142 | 8.79% |
| UNKNOWN at cap 750 | 641 | 24.850 | 65.46% | 480,750 | 65.84% |
| Width-exhaust self-termination | 4,501 | 4.584 | 11.93% | 93,397 | 12.79% |
| **Total** | **6,443** | **37.999** | **~100%** | **730,143** | **100%** |

**MEASURED.** Cap-bound waste is therefore 24.850 s, or 65.46% of this run's
battery wall. A perfect early-stop oracle would recover that full amount on
this host. Applying the measured share to the adopted 45.956 s production
median gives a **30.083 s** perfect-oracle upper bound; this is an upper bound,
not an implementable estimate.

**MEASURED.** The cheap stagnation test compared deterministic endpoint root
`pn` at cap 500 and cap 750. Of the 641 cap-750 UNKNOWN-at-cap rows, 259
(40.41%) were also cap-bound at 500 with the same root `pn`, so their root `pn`
was frozen across the added 250 expansions. Those rows consumed 10.106 s
(26.60% of total cap-750 wall). Stopping them at cap 500 would have saved a
matched median 3.447 s on this host, 9.07% of total wall, before accounting for
classifier overhead or false-positive safety.

### Top 20 rows by wall

**MEASURED.** Rows are ranked by median uninstrumented per-row wall across
three repetitions at cap 750.

| Rank | Set | Position | Final status | Nodes | Median wall ms |
|---:|---|---|---|---:|---:|
| 1 | selfplay_v1 | `sp_47_p75` | UNKNOWN-at-cap | 750 | 78.851 |
| 2 | puzzle_v3 | `sp_47_p73` | UNKNOWN-at-cap | 750 | 78.794 |
| 3 | selfplay_v1 | `sp_47_p74` | UNKNOWN-at-cap | 750 | 78.651 |
| 4 | selfplay_v1 | `sp_47_p73` | UNKNOWN-at-cap | 750 | 77.971 |
| 5 | puzzle_v3 | `sp_47_p74` | UNKNOWN-at-cap | 750 | 76.799 |
| 6 | selfplay_v1 | `sp_18_p81` | UNKNOWN-at-cap | 750 | 69.124 |
| 7 | selfplay_v1 | `sp_18_p67` | UNKNOWN-at-cap | 750 | 68.968 |
| 8 | selfplay_v1 | `sp_29_p63` | UNKNOWN-at-cap | 750 | 68.818 |
| 9 | puzzle_v3 | `sp_29_p67` | UNKNOWN-at-cap | 750 | 68.671 |
| 10 | selfplay_v1 | `sp_29_p67` | UNKNOWN-at-cap | 750 | 68.607 |
| 11 | human_v1 | `human_109c7b52881de543_p113` | UNKNOWN-at-cap | 750 | 66.929 |
| 12 | human_v1 | `human_9912dd17e41680d7_p165` | LOSS | 614 | 66.333 |
| 13 | selfplay_v1 | `sp_27_p50` | UNKNOWN-at-cap | 750 | 66.134 |
| 14 | human_v1 | `human_6ec854e63e6c5b03_p104` | UNKNOWN-at-cap | 750 | 65.202 |
| 15 | puzzle_v3 | `sp_29_p63` | UNKNOWN-at-cap | 750 | 64.328 |
| 16 | puzzle_v3 | `sp_8_p82` | UNKNOWN-at-cap | 750 | 63.346 |
| 17 | human_v1 | `human_fb00d63dc6d7ef92_p89` | UNKNOWN-at-cap | 750 | 63.324 |
| 18 | selfplay_v1 | `sp_8_p82` | UNKNOWN-at-cap | 750 | 62.984 |
| 19 | selfplay_v1 | `sp_13_p65` | UNKNOWN-at-cap | 750 | 61.457 |
| 20 | selfplay_v1 | `sp_18_p79` | UNKNOWN-at-cap | 750 | 61.073 |

### Why ns/node changes with cap

**MEASURED.** The same-run uninstrumented medians were 50,665 ns/node at cap
500 and 52,043 ns/node at cap 750, a 2.72% increase. This matched diagnostic
did **not** reproduce the earlier cross-run 62,941 ns/node cap-750 observation.
The earlier 50.7k -> 62.9k comparison therefore cannot be assigned wholesale
to a solver bucket from these data.

**MEASURED.** Phase-share changes in the same instrumented run were:

| Phase | Cap 500 | Cap 750 | Change | Per-node cost ratio, 750/500 |
|---|---:|---:|---:|---:|
| Defender-plan construction | 14.25% | 15.77% | **+1.52 pp** | **1.008x** |
| PN select/backprop | 7.39% | 7.71% | +0.32 pp | 0.950x |
| Window work | 21.29% | 21.43% | +0.15 pp | 0.917x |
| Defender generation | 0.71% | 0.74% | +0.03 pp | 0.946x |
| State make/unmake | 24.27% | 24.10% | -0.17 pp | 0.904x |
| Attacker generation | 22.32% | 20.90% | -1.42 pp | 0.853x |

**MEASURED.** Defender-plan construction is the only major bucket whose share
and per-node cost both grow with cap. It is the measured solver-side cause of
the modest same-run ns/node growth. Attacker generation, state transitions,
window work, second-candidate work, TT, setup, certificate, and other work all
became cheaper per node.

**HYPOTHESIS.** The much larger archived 24% ns/node increase combined deeper
row mix with host/load timing variation. The solver-side component is deeper
search reaching proportionally more expensive defender-pair plans; the
matched data do not support blaming TT, attacker generation, or state
transitions for the archived increase.

## 3. Ranked speed attacks

**HYPOTHESIS.** Battery-second estimates below apply measured cap-750 shares to
the adopted 45.956 s production median. Equivalent cap headroom uses the local
archived cap-640-to-860 slope, `(52.675 - 38.907) / 220 = 0.06258 s/cap`.
These are planning estimates, not measured optimized runs.

1. **Fuse/cut attacker, window, and second-candidate work.** **MEASURED:** the
   combined path is 48.46% of wall (20.90% attacker generation + 21.43%
   window work + 6.13% second-candidate regeneration). **HYPOTHESIS:** a 25%
   reduction in this combined path saves **5.57 s/battery** and buys cap
   **~839** at constant 45.956 s wall. The implementation direction is to
   reuse the already-built window/gate state across first/second candidates
   and sibling expansions, and eliminate repeated candidate materialization;
   this targets both the known candidate cost and the newly separated window
   cost.

2. **Reduce state make/unmake cost.** **MEASURED:** state transitions alone are
   24.10% of wall, the largest single exclusive bucket and the main content of
   the old residual. **HYPOTHESIS:** halving it saves **5.54 s/battery** and
   buys cap **~838**. Candidate designs are lighter reversible deltas for PN
   descent, applying atomic two-stone edges as one window update, or deriving
   more child facts from the stateless gate before touching `HexoState`.

3. **Guarded cap-bound early abandonment.** **MEASURED:** capped UNKNOWN rows
   consume 65.46% of wall; 259 rows have root `pn` frozen over the cap-500 to
   cap-750 increment, and that final increment costs 9.07% of battery wall.
   **HYPOTHESIS:** a safe classifier that stops only this frozen subset at 500
   saves **4.17 production-equivalent seconds** and buys cap **~817**. The
   perfect-oracle ceiling is 30.08 s / cap ~1,231, but root `pn` alone is not a
   safe oracle: the earlier triage showed provable casualties, so this attack
   requires sub-root stagnation evidence and a no-regression verdict gate.

**HYPOTHESIS.** First runner-up: halving defender-plan construction (15.77%)
would save **3.62 s** and buy cap **~808**. Its share growth with cap makes it
more attractive after the three attacks above.

**MEASURED.** TT probe/store is only 0.31% of wall. **HYPOTHESIS:** even deleting
its entire direct cost saves only 0.14 production seconds (cap ~752), so TT
policy work is not a speed attack. Cross-sibling reuse is interesting only if
it avoids window/generation/state work; a 25% reduction of the window bucket
alone would save 2.46 s (cap ~789), which is why window-state reuse is included
in attack 1 rather than ranked as a TT change.

## Reproduction and raw evidence

**CODE-FACT.** The ignored runner uses `make_solver(false)`, which selects
`vcf_pair_complete`, then explicitly enables dual pass. Each solve supplies
`tt_bytes_cap = 256 << 10` and `semantic_horizon = u32::MAX`. The runner asserts
all solver-altering `TSS_*` experiment variables are unset, including
`TSS_VCF_J2NEAR`.

**MEASURED.** Free physical RAM was 14.657 GiB before the measurement build/run
and 13.897 GiB before the non-test release check, both above the 8 GiB gate.
The exact measurement command was:

```powershell
$env:CARGO_TARGET_DIR = (Join-Path (Get-Location) '.cargo-target')
$env:RUST_MIN_STACK = '33554432'
$env:TSS_J2NEAR_PROFILE_CAPS = '500,750'
$env:TSS_BOTTLENECK_REPETITIONS = '3'
$env:TSS_J2NEAR_OUTPUT_DIR = (Join-Path (Get-Location) '.gate/bottleneck-750')
cargo test -p hexfield_eq --lib --release `
  --target x86_64-pc-windows-msvc `
  tss_j2near_ab::tss_j2near_cap_profile -- `
  --ignored --exact --test-threads=1 --nocapture
python .gate/bottleneck-750/analyze_profile.py
```

**MEASURED.** The test completed in 277.38 s: 1 passed, 0 failed. The separate
non-test validation command also passed:

```powershell
$env:CARGO_TARGET_DIR = (Join-Path (Get-Location) '.cargo-target')
cargo check -p hexfield_eq --lib --release `
  --target x86_64-pc-windows-msvc
```

**MEASURED.** Raw artifacts and SHA-256 digests:

| Artifact | SHA-256 |
|---|---|
| `.gate/bottleneck-750/profile.jsonl` | `700A3C7AEA90BEAB0E4BD1D16A6276685479A809DFC885649C53754BE285D94C` |
| `.gate/bottleneck-750/summary.json` | `52AE15FD8B3FF69E5612F11D4A64C312E2119541FD496AFB40E75E740024E49C` |
| `.gate/bottleneck-750/measurement.log` | `40E5CD037070125D86672E7E625B1A6DC88620C8181ED2330DDAEB632CD64041` |
| `.gate/bottleneck-750/measurement_tee_utf16.log` | `0A243B71DC8F47EB12BADEF958D7E2E38E9A7581F75FA1BCB686D3C6465D7AD9` |
| `.gate/bottleneck-750/analyze_profile.py` | `763051DE7F2F5BAE4F7AE2F86B138C1AF7534A7B82661BE83DB529ED68E27809` |

**CODE-FACT.** The UTF-16 artifact is the original PowerShell `Tee-Object`
capture. `measurement.log` is its normalized concise transcript; both are
retained so no raw console evidence was discarded.
