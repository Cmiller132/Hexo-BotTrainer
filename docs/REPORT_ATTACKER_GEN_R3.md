# Attacker pair generation speed round 3

Date: 2026-07-21  
Branch/worktree: `claude/attacker-gen-r3`  
Production baseline: `2290f4cf`

## Disposition

**CODE-FACT.** The shipped attacker-pair path now uses a fixed-capacity,
stack-backed post-pair threat family and deterministic coordinate hashers in
the construction-only maps and sets. The candidate universe, ordering, gates,
priors, and semantic classifier are unchanged. The historical heap-backed pair
classifier remains under `cfg(test)` as the `TSS_ATTACKER_PAIR_REFERENCE=1`
same-binary oracle. `tss_verify.rs` is unchanged.

**MEASURED.** The cap-500 frozen battery produced 6,443 rows, 556,452 total
nodes, FNV digest `a8c6f3ca3ba55827`, and identity-file SHA-256
`02CD63718E0D06F83853B523C40F7057626A7A3113264235C3CECB162482CFDB`.
The optimized and forced-reference identity files have that same SHA-256.

**MEASURED.** At cap 750, the unmodified-tree and optimized-tree complete-width
off arms matched on all 6,443 rows for set, position ID, cap, TT size, status,
node count, certificate-verification result, and verifier-failure count. Both
sum to 730,143 nodes: 757 wins, 544 losses, and 5,142 unknowns. All 1,301
decided certificates verified.

**MEASURED.** Across three paired cap-750 repetitions with alternating pair
order (old/new, new/old, old/new), median solve wall fell from 38.308 s
[38.069-41.572] to 32.859 s [32.837-38.169]. The median of the three paired
speedup ratios is 1.159x; the ratio of medians is 1.166x. Attacker pair
generation fell from 18.164 s [17.955-19.668] to 13.056 s [13.026-15.356],
1.391x by ratio of medians, and its median wall share fell from 47.31% to
39.76%.

## Measured profile

**MEASURED.** Each row below is the complete-width (`vcf_pair_complete`) off
arm over the full frozen battery at cap 750, dual pass, 256 KiB TT, and
unbounded semantic horizon. Times are summed per-position solve wall and
exclude build/Cargo wall. Pair generation is inside inclusive expansion;
regeneration is inside pair generation.

| Pair | Process order | Old solve / pair (s) | New solve / pair (s) | Paired solve speedup |
|---|---|---:|---:|---:|
| 1 | old then new | 41.572 / 19.668 | 38.169 / 15.356 | 1.089x |
| 2 | new then old | 38.308 / 18.164 | 32.859 / 13.026 | 1.166x |
| 3 | old then new | 38.069 / 17.955 | 32.837 / 13.056 | 1.159x |

**MEASURED.** Median bucket comparison:

| Bucket | Old median | New median | Scope |
|---|---:|---:|---|
| Battery solve wall | 38.308 s | 32.859 s | end to end inside `solve_one` |
| Attacker pair generation | 18.164 s | 13.056 s | inclusive of second-candidate regeneration |
| Attacker pair share | 47.31% | 39.76% | per-run share median |
| Defender generation | 6.395 s | 6.324 s | inside expansion |
| Second-candidate regeneration | 2.194 s | 2.035 s | inside attacker generation |
| Wide expansion | 25.487 s | 20.333 s | inclusive |

**MEASURED.** Memo work did not change: every off arm performed 20,743,752
lookups. Old hit rates were 46.8389%, 46.8452%, and 46.8411%; new hit rates
were 46.8385%, 46.8380%, and 46.8442%. The retained 32,768-slot direct-mapped
memo from round 2 was therefore not changed.

**HYPOTHESIS.** The wall spread, especially the first optimized repetition,
reflects concurrent host load noted in the brief. Stable node counts, stable
memo lookup counts, paired alternation, and the consistent pair-bucket
reduction make the 1.159x paired-median end-to-end result the conservative
headline rather than the best observed arm.

## Implementation and identity argument

**CODE-FACT.** One placement touches exactly 18 length-six windows. A pair can
therefore touch at most 36 distinct windows. A claimant window admitted to the
post-pair family starts with at least two stones and reaches at least four after
the pair, so it has at most two post-pair empty cells. `PairEvaluationScratch`
encodes those exact bounds as 36 family slots and 72 universe cells on the
stack.

**CODE-FACT.** The optimized classifier traverses the same first-cell window
indices, then the same second-cell indices while skipping joint windows, as
the reference. `PairFamilyMember::after_pair` preserves the original empty-cell
order while removing the two placements. The minimum-hitting-set routine builds
the same first-occurrence universe and tests singles and unordered pairs in the
same order. It therefore returns the same `None`/1/2 result and derives the same
threat count and PN/DN prior.

**CODE-FACT.** On the rare over-budget tactical branch, family members are
stable-sorted by the same `WindowKey` order, materialized back into the
historical `Vec<Vec<HexCoord>>` form, and passed to the unchanged L13
`inclusion_minimal_loss_obstruction`. Certificate selection is therefore
unchanged.

**CODE-FACT.** `windows_by_cell`, `weak_windows_by_cell`, first-candidate
membership, count-three incidence, second-candidate membership, and unordered
pair dedup now use the existing coordinate-specialized deterministic hasher.
None of those tables is observed by hash iteration: they are lookup-only, or
their output is sorted by the pre-existing raw-coordinate key. Candidate and
child insertion order is unchanged.

**CODE-FACT.** Release gate construction no longer allocates the inner
`created_threats` vectors merely to count them. It increments the exact count
at the same window encounters. Debug builds still materialize the vectors, so
the pre-existing full `Candidate` equality assertion against the historical
generator remains active.

**MEASURED.** The focused
`compact_pair_family_evaluator_matches_heap_reference` test enumerates the
fixture's pair stream and compares every optimized classification/result/prior
with the heap-backed reference. The full cap-500 optimized-versus-reference
battery then matched byte for byte, including certificate Debug hashes. The
cap-750 unmodified-versus-optimized row comparison reported zero mismatches.

## Gates

**MEASURED.** Final-source serialized suites:

| Gate | Result |
|---|---|
| `cargo test -p hexfield_eq --features python --target x86_64-pc-windows-msvc -- --test-threads=1` | 222 passed, 0 failed, 43 ignored |
| `cargo test -p hexfield_eq --lib --release --target x86_64-pc-windows-msvc -- --test-threads=1` | 137 passed, 0 failed, 42 ignored |
| Frozen cap-500 identity | 6,443 rows; 556,452 nodes; digest `a8c6f3ca3ba55827`; exact SHA-256 |
| Full cap-750 row identity | 6,443 rows; 730,143 nodes; zero field mismatches; all decided certs verified |

**CODE-FACT.** The pass-count increase of one over the stated baselines is the
new compact-versus-heap pair-classifier oracle test.

## Reproduction

**CODE-FACT.** All Cargo commands use the worktree-local target directory,
MSVC target, serialized tests, and the required 32 MiB Rust test stack. RAM
checks in the raw artifacts recorded 14.43-15.40 GiB free before builds and
final suites.

```powershell
$env:CARGO_TARGET_DIR = (Join-Path (Get-Location) '.cargo-target')
$env:RUST_MIN_STACK = '33554432'

cargo test -p hexfield_eq --features python --target x86_64-pc-windows-msvc -- --test-threads=1
cargo test -p hexfield_eq --lib --release --target x86_64-pc-windows-msvc -- --test-threads=1

$env:TSS_IDENTITY_OUT = (Join-Path (Get-Location) '.gate/attacker-gen-r3/final_cap500.identity.tsv')
cargo test -p hexfield_eq --features python --release --target x86_64-pc-windows-msvc tss_frozen_identity_battery -- --ignored --test-threads=1 --nocapture

$env:TSS_ATTACKER_PAIR_REFERENCE = '1'
$env:TSS_IDENTITY_OUT = (Join-Path (Get-Location) '.gate/attacker-gen-r3/reference_cap500.identity.tsv')
cargo test -p hexfield_eq --features python --release --target x86_64-pc-windows-msvc tss_frozen_identity_battery -- --ignored --test-threads=1 --nocapture
Remove-Item Env:TSS_ATTACKER_PAIR_REFERENCE

$env:TSS_J2NEAR_CAPS = '750'
$env:TSS_J2NEAR_REPETITIONS = '1'
$env:TSS_ATTACKER_GEN_R3_SINGLE = '1'
$env:TSS_J2NEAR_OUTPUT_DIR = (Join-Path (Get-Location) '.gate/attacker-gen-r3/reproduction_rows')
cargo test -p hexfield_eq --features python --release --target x86_64-pc-windows-msvc tss_j2near_matched_ab -- --ignored --test-threads=1 --nocapture
```

**CODE-FACT.** The final timing sequence used the archived algorithm-identical
instrumented executables `baseline_instrumented.exe` and `optimized.exe`, with
one harness repetition per process. The process order was baseline/optimized,
optimized/baseline, baseline/optimized. The default harness behavior remains
three repetitions; `TSS_ATTACKER_GEN_R3_SINGLE=1` is test-only and is required
to request a one-repetition source-built run.

## Raw evidence

**CODE-FACT.** Raw logs, emitted identity files, cap-750 JSONL rows, archived
test executables, RAM checks, the machine-computed benchmark summary, and the
cap-750 comparison result live under
[`.gate/attacker-gen-r3`](../.gate/attacker-gen-r3/). SHA-256 for every retained
artifact is recorded in
[`SHA256SUMS.txt`](../.gate/attacker-gen-r3/SHA256SUMS.txt).

**MEASURED.** Key artifacts are `baseline_probe.log`, `old_rep2.log`,
`old_rep3.log`, `new_rep1.log`, `new_rep2.log`, `new_rep3.log`,
`benchmark_summary.log`, `cap750_identity_compare.log`,
`final_cap500.log`, `reference_cap500.log`, `final_full_suite_python.log`, and
`final_full_suite_release_lib.log`.
