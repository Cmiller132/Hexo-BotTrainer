# R-LF1 — lazy frontier admission

Status: **IMPLEMENTED DEFAULT-OFF; PROOF, EXACT EQUIVALENCE, OFFICIAL GATE,
AND NQ4 MEASUREMENT PASS.** No commit was made.

## Result

`TSS_LAZY_FRONTIER=1` keeps unselected wide defender children out of the PN
arena and exact-position TT. A child edge retains its move/result/prior and
exact future key. A solve-local deferred-key registry preserves the first
eager-equivalent prior/depth and prospective transposition identity without an
arena record. On first selection, the key is realized through `insert_position`
before expansion. The flag is read once by `WidePnSearch::new_with_width` and
is absent/default-off by default.

The refinement proof is `PROOF_LAZY_FRONTIER.md`. Exact identity required two
subtleties found by the harness:

1. a defender thunk must virtually observe a transposition expanded through a
   different parent, including the first eager admission's prior; and
2. attacker future keys are selection-only. Making them visible before
   selection changes the historical attacker-lazy scheduling policy.

## Exact equivalence gate

Command: `lazy_frontier_equivalence_campaign`, release, serialized, 2 GiB TT,
with `TSS_LAZY_FRONTIER_VALIDATE_KEYS=1`.

Coverage was all 19 forcing rows at 10k and 100k, `double_fork_compact`, and
the first 20 roots from the NQ4 deterministic human sample: 59 cold roots and
118 solves. Every returned certificate was independently verified. Result:

- verdict identity: exact;
- canonical certificate-byte identity: exact;
- expanded-node identity: exact;
- every saved future key recomputed equal at realization; and
- no WIN/LOSS flip.

Log: `.codex-hunt/lazy-frontier-equivalence.log` (`LF_EQ_DONE result=PASS
node_identity=exact certificate_bytes=exact`, 243.96 s).

## Headline paired measurement

These are sums across cold roots from the exact off/on equivalence run. The
human timing row is the required first-20 sample. `peak_tt_bytes` is the sum of
per-root peaks, suitable for paired aggregate comparison (not a concurrent
resident peak). Key validation was enabled, so the on-mode wall time includes
an extra test-only key reconstruction at every realization.

| cohort | indexed entries off → on | retained arena off → on | peak_tt_bytes off → on | TT hits off → on | wall off → on |
|---|---:|---:|---:|---:|---:|
| forcing 10k (19) | 267,457 → 100,052 (-62.6%) | 267,457 → 100,052 (-62.6%) | 54,550,575 → 19,140,045 (-64.9%) | 30,667 → 6,425 (-79.1%) | 20.847 s → 19.737 s (-5.3%) |
| forcing 100k (19) | 1,283,238 → 419,151 (-67.3%) | 1,283,238 → 419,151 (-67.3%) | 307,070,115 → 96,957,540 (-68.4%) | 129,266 → 13,645 (-89.4%) | 105.350 s → 89.544 s (-15.0%) |
| double_fork_compact | 258 → 258 | 258 → 258 | 67,177,998 → 67,177,998 | 51 → 51 | 60.790 ms → 59.436 ms |
| human first 20, 10k | 36,826 → 12,977 (-64.8%) | 36,826 → 12,977 (-64.8%) | 9,716,288 → 3,351,786 (-65.5%) | 12,210 → 43 (-99.6%) | 3.336 s → 2.876 s (-13.8%) |

The compact fixture takes the narrow compatibility path, so no lazy-frontier
change is expected there.

## Full NQ4 rerun

The original NQ4 campaign was rerun flag-on at its original 512 MiB TT setting
over 19 forcing rows at both caps, compact, and all 100 human roots. Verdicts,
nodes, and W/L/U cohort counts equal the baseline; `TQ_DONE result=PASS
anomalies=0`.

| cohort | baseline retained/indexed | lazy retained/indexed | baseline never-expanded | lazy never-expanded | TT hits off → on |
|---|---:|---:|---:|---:|---:|
| forcing 10k | 267,457 | 100,052 | 167,405 (62.6%) | 0 / 100,052 (0%) | 30,667 → 6,425 |
| forcing 100k | 1,283,238 | 419,151 | 864,087 (67.3%) | 0 / 419,151 (0%) | 129,266 → 13,645 |
| human 100, 10k | 259,824 | 89,179 | 170,645 (65.7%) | 0 / 89,179 (0%) | 81,336 → 3,623 |
| double_fork_compact | 258 | 258 | narrow; n/a | narrow; n/a | 51 → 51 |

The full campaign wall clock fell from 106.48 s to 89.63 s (-15.8%). The lazy
entry counts equal the unique-expanded denominators in all three wide cohorts,
eliminating the measured retained-never-expanded class at exact search-output
identity.

The TT-hit drop is expected and meaningful: eager mode counted prospective
hits while generating children that were never selected. Lazy mode counts
actual `by_position` hits at realization; matches against deferred frontier
metadata are not TT hits.

## Official gate

With `TSS_BACKWALK_TT_BYTES=2147483648`, `TSS_LAZY_FRONTIER=1`, one release
Cargo process and `--test-threads=1`:

```text
CORPUS_DONE failures=0
test result: ok. 1 passed; 0 failed
finished in 613.13s
```

Log: `.codex-hunt/lazy-frontier-corpus-gate.log`.

## Honest memory/work boundary

`peak_tt_bytes` retains its existing meaning: charged exact-position TT bytes.
As before, it excludes PN arena records and child vectors; it also excludes
edge-owned future keys and the deferred frontier registry. Exact equivalence
therefore still constructs future keys and performs lightweight deferred-map
lookups/inserts. The measured claim is a large cut in arena/TT admission,
charged TT bytes, eager TT-hit bookkeeping, and wall time—not elimination of
all frontier memory or hashing.

After a TT-index cap refusal, eager can create separate unindexed arena records
for equal later keys while lazy may retain one deferred identity. The proof's
cap-aware corollary consequently permits traversal/node-count and capped
UNKNOWN timing differences after that refusal, while certificate validity
remains invariant. The 2 GiB equivalence campaign encountered no refusal and
proved exact node identity on its required sample.

## Regeneration

Check RAM before every Cargo command. Use `.target-hunt` and one test thread.

```powershell
Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
$env:CARGO_TARGET_DIR='.target-hunt'
$env:TSS_LAZY_FRONTIER_TT_BYTES='2147483648'
$env:TSS_LAZY_FRONTIER_VALIDATE_KEYS='1'
cargo test --release -p hexfield_eq lazy_frontier_equivalence_campaign -- --ignored --test-threads=1 --nocapture
```

```powershell
Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
$env:CARGO_TARGET_DIR='.target-hunt'
$env:TSS_BACKWALK_TT_BYTES='2147483648'
$env:TSS_LAZY_FRONTIER='1'
cargo test --release -p hexfield_eq tss_corpus_check -- --ignored --test-threads=1 --nocapture
```

```powershell
Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
$env:CARGO_TARGET_DIR='.target-hunt'
$env:TSS_LAZY_FRONTIER='1'
cargo test --release -p hexfield_eq turn_quotient_campaign -- --ignored --test-threads=1 --nocapture
```
