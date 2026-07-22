# Triage Phase B — sub-root telemetry for Unknown classification

## Outcome

**MEASURED.** The preregistered interest bar is met, but only for a selective
classifier. At `N=100`, the best candidate using at most three features marks
13/94 cap-bound roots, no exhaust roots, and no provable roots: 100% precision,
13.8% recall, and 0/57 provable casualties. The literal bar (precision at least
90%, at most 2 provable casualties, `N <= 500`) therefore passes.

**MEASURED.** This is not broad early separation. Among candidates meeting the
bar, maximum cap-bound recall through `N=500` is 20.2% (at `N=150` and
`N=500`). Broad candidates reaching 80.9–93.6% recall at these decision points
have only 48.2–52.1% precision and cause 24–30 provable casualties. The early
telemetry can identify a small high-confidence cap-bound pocket, but it cannot
classify most budget sinks safely.

**MEASURED.** High-precision recall first rises above 30% at `N=1000`: 31.9%
at `N=1000` and 33.0% at `N=1500`. This is the first material improvement in
the curve, although two thirds of cap-bound roots remain unclassified.

**HYPOTHESIS.** The stable, low root PN plus low child-DN dispersion seen in
many cap-bound roots represents flat churn below a persistent promising root
child. It is a useful descriptive signal, not yet a reallocation policy. The
thresholds were selected and evaluated on the same frozen 248-root cohort, so
their precision is in-sample and needs independent validation before any use.

## Instrumentation

**CODE-FACT.** Trajectory state exists only behind `cfg(test)` and is absent
from the production `WidePnSearch` layout
(`packages/hexfield_eq/rust/src/tss_solver.rs:4254-4259`). The test-only writer
is constructed only when `TSS_TRACE_TRAJECTORY` is present; it samples every
25 expansions and binds each row to `TSS_TRACE_SOLVE_ID`
(`tss_solver.rs:4277-4312`, `tss_solver.rs:4627-4631`). The expansion wrapper's
only collector call is itself `cfg(test)`
(`tss_solver.rs:6752-6756`). With the env variable unset, the test-only option
is `None`; in non-test builds the option, collector type, and call do not exist.

**CODE-FACT.** A snapshot recursively recomputes an observational PN/DN view
of the arena without mutating search state. This avoids stale ancestor values
while df-pn unwinds (`tss_solver.rs:6562-6644`). Snapshot assembly records root
PN/DN; root child count, sums, extrema, zero counts, and four children ordered
by current PN/DN selection score with generator-order ties; open/cutoff node
counts; maximum depth; arena and TT shape; and distinct/reselected expansion
counts (`tss_solver.rs:6646-6732`). Emission occurs exactly when the expansion
clock reaches the next multiple of 25 (`tss_solver.rs:6734-6749`).

**CODE-FACT.** The ignored battery checks the frozen 57/97/94 class
cardinalities, replays all labeled positions, creates a new leaf-profile solver
per root, and uses WIN goal, node cap 5,000, 256 KiB TT, and unbounded semantic
horizon (`packages/hexfield_eq/rust/src/tss_trajectory_phase_b.rs:98-175`). It
also independently verifies every returned verdict and asserts node and TT
caps (`tss_trajectory_phase_b.rs:177-190`). No changes were made to
`tss_verify.rs`.

### Feature definitions

The raw schema is one JSON object per snapshot:

- Root aggregate: `root_pn`, `root_dn`, and `root_kind`.
- Child dispersion: `root_child_count`; PN/DN sum, min, max, and zero counts;
  `root_top[0:4]` with score rank, generator ordinal, current PN/DN, and prior
  PN/DN. Derived analysis features include mean, range, and rank-1/rank-2 gap.
- Frontier/arena: `open_nodes` means `Unexpanded` arena nodes;
  `cutoff_nodes`, `max_depth`, and `arena_size` describe retained shape.
- Reuse/memory: `tt_entries`, `tt_hits`, and
  `tt_admission_rejections`; `distinct_expanded_nodes`,
  `reselected_expansions`, and `max_node_expansions` describe revisitation.
- Trajectory deltas: changes over the preceding 25 expansions and from the
  expansion-25 snapshot. Ratio features normalize arena, open, indexed, and
  reselection counts.

`PN_INFINITY = 1,000,000,000`; consequently PN range/mean features can contain
large sentinel-driven values. Full definitions are serialized in
`.scratch/triage_b/summary.json`.

## Data collection and integrity

**MEASURED.** The release battery produced 34,616 trajectory rows for all 248
solve IDs. Every expansion value is an exact multiple of 25, spanning 25 to
4,975. The solves consumed 870,297 nodes in 93.443 seconds of battery wall
time. At cap 5,000, 44/57 historically provable roots returned verified WIN;
the other 13 were still Unknown. All 97 exhaust and 94 cap-bound labels were
Unknown at this smaller cap.

**MEASURED.** Live populations at each decision point were:

| N | Cap-bound | Exhaust | Provable | Cap-bound base rate |
|---:|---:|---:|---:|---:|
| 100 | 94 | 97 | 57 | 37.9% |
| 150 | 94 | 97 | 57 | 37.9% |
| 200 | 94 | 97 | 57 | 37.9% |
| 300 | 94 | 97 | 57 | 37.9% |
| 500 | 94 | 97 | 57 | 37.9% |
| 1000 | 94 | 74 | 41 | 45.0% |
| 1500 | 94 | 57 | 30 | 51.9% |

“Live at N” means the solve performed at least N expansions. The analyzer uses
the snapshot nearest at or below N; all requested N values are exact snapshot
points.

## Feature separability

**MEASURED.** The table gives class medians as `cap-bound / exhaust /
provable`. It reports representative root, dispersion, shape, and TT features;
the summary JSON contains the complete median table.

| N | Root PN | Root DN | Child DN range | Max depth | TT admission rejections |
|---:|---:|---:|---:|---:|---:|
| 100 | 34 / 35 / 102 | 26 / 25 / 22 | 3 / 8 / 5 | 10 / 10 / 12 | 0 / 0 / 0 |
| 150 | 34 / 35 / 134 | 27.5 / 25 / 22 | 4 / 7 / 8 | 10 / 12 / 13 | 0 / 0 / 0 |
| 200 | 34 / 35 / 166 | 27.5 / 31 / 25 | 6 / 10 / 7 | 11 / 12 / 14 | 0 / 0 / 0 |
| 300 | 34 / 128 / 35 | 28 / 34 / 37 | 6 / 14 / 9 | 14 / 14 / 15 | 0 / 0 / 0 |
| 500 | 34 / 198 / 68 | 25 / 30 / 42 | 6 / 15 / 16 | 16 / 17 / 18 | 0 / 0 / 0 |
| 1000 | 34 / 329 / 131 | 24.5 / 32 / 62 | 5 / 13.5 / 21 | 20 / 20.5 / 21 | 0 / 0 / 0 |
| 1500 | 34 / 238 / 218 | 23 / 27 / 72.5 | 5 / 9 / 16 | 22 / 24 / 22.5 | 196.5 / 215 / 262.5 |

Root PN stagnation remains visible, but the child-DN range adds a clearer
sub-root distinction: cap-bound medians stay at 3–6 while exhaust and provable
medians generally widen. Max depth and TT rejection counts overlap strongly.
No one median is itself a safe classifier.

## Candidate classifiers

The analyzer considers transparent AND rules with one to three thresholds on
distinct numeric features. Each feature uses a deterministic grid of at most
40 observed values. This is a descriptive candidate search, not ML training.

Confusion is shown as `CB/E/P`: counts from each true class predicted
cap-bound. The corresponding predicted-other count is the live class total
minus that value. “Cas.” is the provable casualty count.

### Best candidates meeting the bar

| N | Conditions | Precision | Recall | CB/E/P | Cas. |
|---:|---|---:|---:|---:|---:|
| 100 | child count >=17 AND child DN sum <=40 AND child DN-zero count <=3 | 100.0% | 13.8% | 13/0/0 | 0 |
| 150 | child count >=17 AND child PN max >=199 AND root PN delta-from-25 <=34 | 90.5% | 20.2% | 19/2/0 | 0 |
| 200 | arena delta-from-25 >=173 AND top PN gap >=5 AND TT entries >=195 | 93.3% | 14.9% | 14/1/0 | 0 |
| 300 | child DN mean <=2 AND child PN range <=227 AND child PN max <=264 | 93.8% | 16.0% | 15/0/1 | 1 |
| 500 | child DN mean <=3 AND child PN max <=292 AND TT entries <=495 | 90.5% | 20.2% | 19/1/1 | 1 |
| 1000 | child DN max <=13 AND root DN delta-from-25 >=-1 AND root PN <=317 | 90.9% | 31.9% | 30/1/2 | 2 |
| 1500 | child DN mean <=2.625 AND child PN mean <=4e8 AND child PN-sum delta-25 <=6 | 91.2% | 33.0% | 31/2/1 | 1 |

### Best single-feature thresholds meeting the bar

| N | Condition | Precision | Recall | CB/E/P | Cas. |
|---:|---|---:|---:|---:|---:|
| 100 | child DN max >=94 | 100.0% | 4.3% | 4/0/0 | 0 |
| 150 | selection-rank-3 DN >=62 | 100.0% | 3.2% | 3/0/0 | 0 |
| 200 | child PN max <=130 | 100.0% | 4.3% | 4/0/0 | 0 |
| 300 | selection-rank-3 DN >=60 | 100.0% | 3.2% | 3/0/0 | 0 |
| 500 | selection-rank-3 DN >=61 | 100.0% | 4.3% | 4/0/0 | 0 |
| 1000 | child DN-sum delta-25 >=48 | 100.0% | 2.1% | 2/0/0 | 0 |
| 1500 | TT entries >=1493 | 100.0% | 3.2% | 3/0/0 | 0 |

The single-feature passes are tiny tail selectors (2–4 cap-bound roots), not
class separation in an operational sense.

### Broad-recall controls

For context, the best F1 candidates in the same three-condition family miss
the precision/casualty bar:

| N | Precision | Recall | CB/E/P | Provable casualties |
|---:|---:|---:|---:|---:|
| 100 | 48.2% | 85.1% | 80/59/27 | 27 |
| 150 | 52.1% | 80.9% | 76/44/26 | 26 |
| 200 | 49.4% | 86.2% | 81/59/24 | 24 |
| 300 | 49.4% | 93.6% | 88/60/30 | 30 |
| 500 | 49.1% | 89.4% | 84/63/24 | 24 |
| 1000 | 56.2% | 95.7% | 90/42/28 | 28 |
| 1500 | 66.1% | 89.4% | 84/26/17 | 17 |

## Where separation starts

**MEASURED.** By the literal preregistered definition, separation starts at
`N=100`: precision 100%, no provable casualties, recall 13.8%. The best recall
subject to the bar follows this curve:

| N | Best bar-qualified recall | Precision | Provable casualties |
|---:|---:|---:|---:|
| 100 | 13.8% | 100.0% | 0 |
| 150 | 20.2% | 90.5% | 0 |
| 200 | 14.9% | 93.3% | 0 |
| 300 | 16.0% | 93.8% | 1 |
| 500 | 20.2% | 90.5% | 1 |
| 1000 | 31.9% | 90.9% | 2 |
| 1500 | 33.0% | 91.2% | 1 |

**MEASURED.** If “starts” requires at least 30% cap-bound recall, it starts at
`N=1000`, not at or below 500. There is no measured point here at which the
bar-qualified rules cover a majority of cap-bound roots.

## Reproducibility and verification

Analyzer:

```text
python .scratch/triage_b/analyze.py
```

**MEASURED.** With `TSS_TRACE_TRAJECTORY` unset, the required serialized Rust
library suite passed under the mandated Windows target and stack:

```text
136 passed; 0 failed; 43 ignored
```

The release Phase-B battery passed `1 passed; 0 failed`, with all 248 rows
processed. Both runs used `CARGO_TARGET_DIR=<worktree>/.cargo-target`, target
`x86_64-pc-windows-msvc`, `RUST_MIN_STACK=33554432`, and
`--test-threads=1`. Available physical memory was measured above 13.9 GiB
before each build/run.

### Artifact hashes (SHA-256)

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `.scratch/triage_b/trajectory.jsonl` | 25,954,869 | `5d9a1135955f370bc1ba70692d046073471167ba5a911c4ec8cefa7c506c0925` |
| `.scratch/triage_b/results.jsonl` | 42,940 | `31c3f50c0623e38b4e3f68a1c195ce300b26a1d9a76e2ddf682324b1b436b867` |
| `.scratch/triage_b/analyze.py` | 18,447 | `f31cfe8336cec6db2d2457c5cc56bda93cecf9269220b2113aa611ba777963c9` |
| `.scratch/triage_b/summary.json` | 79,650 | `2a97f954a5291319f8cd36b5a68d3ec2d466786ac77f6d2bfb99e9bf580df77f` |
| `raws/lanec_labels.jsonl` | 268,724 | `48bd13ab76d477feffd3067fd18bca41f0e9e30707a505bdc437c9dafc6ecb95` |
| `../v1-soak/raws/selfplay_positions.jsonl` | 4,356,868 | `b2dae03d4ad99aa62c160eb39e22cd50aa297ccec5091b0f5e90d43632176577` |

The summary embeds hashes for the raw trace, results, analyzer, and both input
corpora. Its own hash is listed above.
