# R-CREL-5 — C_rel Phase-3 leaf-relevance measurement

Date: 2026-07-18  
Branch: `hunt/cert-support`  
Input HEAD: `f7ac3ce48f523a6a4571c12501123eec780364f9`

## Verdict

**ABORT. Do not conclude RELEVANT or IRRELEVANT-AT-LEAF.** The deterministic
real-tree sibling cohort achieved **0.000% positive zoned-certificate coverage
at h=8 (0/3 admitted pairs)** and **0.000% at h=16 (0/62)**. Both are below
the binding 20% construction floor, so the selected-cell economics arms were
not run. Consequently net gain, its clustered lower bound, non-parent template
hit rate, accounted peak, and paired RSS regression are all **not measured**, not
zero.

The authoritative run records the two failed coverage gates and the required
stop at `CREL_LEAF_COHORT_RAW.log:70-71,134-141`. It also records
`hard_without_strict=0`; no warm result was returned or installed.

The binding bars resolve as follows, quoted verbatim:

- “RELEVANT if the clustered 95% LB of net gain >= 10% at either h AND the
  non-parent template-hit rate >= 25% AND hard_without_strict = 0.” **Not
  evaluated after the construction abort.**
- “IRRELEVANT-AT-LEAF if LB < 10% at both h or non-parent hit rate < 25% (the
  Stage-4 gain would then be a parent-reuse artifact — an honest and final
  answer; paper either way).” **Not evaluated after the construction abort.**
- “ABORT (report, do not conclude) if cohort construction cannot achieve >=
  20% positive zoned-coverage fraction — say so rather than diluting the cohort
  definition.” **Fired at both horizons: 0.000%.**

## Frozen leaf envelope

The campaign copied the leaf-surface verdict rather than selecting a new
profile. The in-tree leaf report defines the corpus shuffle seed and 50-game
sample at `HUNT_REPORT_LEAF_SURFACE.md:12-13`, selects configuration D (wide
pair-complete PN plus lazy frontier and the interior census gate) at
`HUNT_REPORT_LEAF_SURFACE.md:210-212`, and fixes the levers and caps at
`HUNT_REPORT_LEAF_SURFACE.md:223-230`:

```text
width = WidthOptions::vcf_pair_complete()
TSS_LAZY_FRONTIER=1
TSS_INTERIOR_CENSUS_GATE=1
TSS_SHARED_FRAGMENTS=0
TSS_K_REPLY_CONSUME=0
goal = SolveGoal::Win
node_cap = 500
tt_bytes_cap = 262144
relative_horizon = 8 or 16
```

The authoritative raw confirms this envelope, the 8 MiB/fanout-1 selected
cell identity, the standard corpus, 6,902 eligible games, 50 selected games,
300 root solves per horizon, and seed `0x9E3779B97F4A7C15` at
`CREL_LEAF_COHORT_RAW.log:16`. One solver was retained across the six corpus
roots in each game batch; solvers were not shared across horizons.

## Exact deterministic cohort generator

The cfg(test)-only generator is
`packages/hexfield_eq/rust/src/tss_crel_leaf_hunt.rs`. It operates as follows:

1. Parse decisive, legally replayable standard-corpus rows; Fisher-Yates
   shuffle game indices with the continuing xorshift stream seeded by
   `0x9E3779B97F4A7C15`; take the first 50 games; choose the same deterministic
   six-root window per game as the leaf-surface campaign.
2. For every root at h=8 and h=16, run the frozen wide/gated cap-500 profile.
   An opt-in cfg(test) collector clones only the actual wide-PN expansion states
   for this run and records expansion-event parent links. Production builds and
   ordinary telemetry do not retain these states.
3. Treat final `proven_leaf`, `depth_cutoff`, and `refuted` arena entries as
   encountered leaves. For every pair, walk event-parent links and compute its
   lowest common ancestor. Retain only pairs in distinct descendant subtrees
   with equal placement count, equal current player, and equal exact turn phase.
   Equal ply plus a proper divergent ancestor establishes that neither member
   is a direct parent/continuation of the other.
4. Sort positions by a canonical full-state spelling (placement clock, player,
   exact phase, terminal value, sorted occupied cells and owners). Use stable
   FNV-1a identifiers only for trace labels. Keep the lexicographically first
   eligible pair per common ancestor. The common ancestor—not the individual
   leaf—is the planned bootstrap cluster.
5. Independently solve `P_parent-solve` under the same cap-500 leaf profile and
   absolute deadline `source.placements + h`. UNKNOWN sources cannot admit a
   template and are reported as `source_unknown`. Every hard source must carry
   a certificate accepted by the unchanged `TssVerifier`.
6. Put the accepted source certificate in the test warm store before solving
   the distinct sibling `P_reuse`. A pair has positive zoned coverage exactly
   when that store contains a certificate with at least one
   `CertNode::Universal { zone: Some(_) }` at reuse start. No non-zoned pair is
   replaced, supplemented, or reclassified to improve the fraction.

Every admitted pair is printed with corpus/root cluster, common-ancestor event
and hash, full-state source/reuse hashes, ply, root inequality,
`direct_parent=false`, both cold statuses, warm-store size, zoned presence, and
strict-authority telemetry (`CREL_LEAF_COHORT_RAW.log:67-69,72-133`).

## Construction result

| Horizon | Discovery roots / hard / zoned | Leaf events | Common-ancestor candidates | Source UNKNOWN | Admitted pairs | Reuse hard | Positive zoned coverage |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 300 / 16 / 0 | 967 | 67 | 64 | 3 | 0 | **0/3 (0.000%)** |
| 16 | 300 / 39 / 0 | 1,643 | 265 | 203 | 62 | 41 | **0/62 (0.000%)** |

These are copied from `CREL_LEAF_COHORT_RAW.log:70` and
`CREL_LEAF_COHORT_RAW.log:134`. The discovery-root hard counts reproduce the
leaf-surface D/cap-500 results (16 at h=8, 39 at h=16), and none of those 55
root certificates was zoned either.

The zero is also structurally expected under the binding profile. The selected
wide-PN materializer's ordinary Universal builder is at
`packages/hexfield_eq/rust/src/tss_solver.rs:6579-6620`, and each wide
Universal construction writes `zone: None` (also lines 6719 and 6756). The
cohort harness explicitly retains `ZoneSearchCaps::default()` at
`packages/hexfield_eq/rust/src/tss_crel_leaf_hunt.rs:227`. Enabling the separate
zone-producing route would change the decided leaf profile and was therefore
not used to manufacture coverage.

## Measurement disposition

| Horizon | Net gain | Common-ancestor-clustered 95% LB | Non-parent hit rate | Accounted peak | Paired RSS regression |
|---:|---:|---:|---:|---:|---:|
| 8 | Not measured—ABORT | Not measured—ABORT | Not measured—ABORT | Not measured—ABORT | Not measured—ABORT |
| 16 | Not measured—ABORT | Not measured—ABORT | Not measured—ABORT | Not measured—ABORT | Not measured—ABORT |

No 10,000-resample bootstrap was performed because there is no authorized
point estimate after the cohort gate fails. The 8 MiB reservation/fanout-1
cell, exact-fragment-both-arms comparator, C_rel lookup/materialization/mint/
verification/fallback accounting, accounted peak, and paired process-RSS
measurement were therefore never instantiated. This preserves the protocol's
“report, do not conclude” instruction and avoids presenting partial cohort
timings as leaf economics.

## Soundness, default-off, and resource audit

- `packages/hexfield_eq/rust/src/tss_verify.rs` has no diff. Every admitted
  source and every hard reuse result strict-verified; `hard_without_strict=0`.
- All additions are behind `#[cfg(test)]`; the harness test is ignored by
  default. Existing C_rel production flags and authority are unchanged.
- The final run used `CARGO_TARGET_DIR=.target-hunt`, release mode, target
  `x86_64-pc-windows-msvc`, and `--test-threads=1`. It passed 1/1 in 2.01 s
  (`CREL_LEAF_COHORT_RAW.log:1,139-141`).
- Every Cargo launch was preceded by both RAM readings and a host-wide Cargo
  check. Across retained logs, minimum launch available memory was
  10,962,931,712 bytes, minimum free physical memory was 10,940,289,024 bytes,
  and foreign Cargo count was always zero. All exceed the 10 GiB / 5 GiB gates;
  every invocation was far below ten minutes.
- The first no-run build wrapper stopped when PowerShell interpreted Cargo's
  normal stderr progress as an error. It ran for about two seconds and did not
  execute a test. The retained retry passed; both facts remain in
  `CREL_LEAF_BUILD_RAW.log:1-18`.
- The first cohort raw is retained as a truthful non-authoritative attempt: its
  extractor required immediate leaf siblings and found none. The corrected
  lowest-common-ancestor generator is present in attempt 2 and the final raw;
  only `CREL_LEAF_COHORT_RAW.log` is authoritative.

## Residual question

The sharpest residual is now a profile-contract question, not an economics
question: **can the authorized wide/gated Phase-3 leaf route be extended to
emit zoned strict certificates without changing its decided search envelope or
soundness contract?** Until that is separately authorized and demonstrated,
the mandated positive-coverage cohort cannot exist and Stage 4's parent-reuse
gain remains neither confirmed nor refuted at leaf.

## Raw evidence and manifest

- `CREL_LEAF_COHORT_RAW.log` — authoritative final cohort construction and
  gate result.
- `CREL_LEAF_COHORT_ATTEMPT1_RAW.log` — retained immediate-sibling extractor
  attempt, non-authoritative.
- `CREL_LEAF_COHORT_ATTEMPT2_RAW.log` — retained corrected-LCA pre-final run,
  non-authoritative.
- `CREL_LEAF_BUILD_RAW.log` — build gates, wrapper failure, successful retries.
- `CREL_LEAF_FMT_RAW.log` — format invocation and resource gate.
- `CREL_LEAF_ANALYSIS_RAW.log` — compact copy-paste aggregate and disposition.
- `CREL_LEAF_HASHES_RAW.log` — SHA-256 manifest.
