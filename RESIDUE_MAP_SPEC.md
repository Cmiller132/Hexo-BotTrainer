# R-CLOSURE-1 residue-map specification

Status: implementation-ready specification; no code is changed by this round.
Source baseline: `claude/tss-vcf-width` at `6ee9ecfb` (line pointers below are
to that revision).

## 1. Purpose and claim discipline

The future lane must measure where official-profile elapsed time goes and bind
each reducible bucket to a design-space disposition: implemented/verified,
impossible, or open with a measured value estimate. The primary measured unit
is a **job** `(profile, corpus row, cap rung, semantic-horizon rung,
repetition)`. `job_wall_ns` begins immediately before solver/session entry and
ends after strict certificate verification or the explicit Unknown path.

Two different bounds must not be conflated:

- `L = max(open category wall share)` bounds an optimization confined to one
  category.
- `R_open = sum(open reducible category wall shares)` is the Amdahl upper bound
  on an arbitrary future optimization that may affect several open categories.

Therefore “the largest category is X%, so any future optimization is bounded
by X%” is valid only with an explicit one-category-only premise. The general
closure claim must use `R_open`; the report also publishes `L`.

## 2. Exclusive category partition

### 2.1 Categories

| ID | Exclusive measured work | Reducibility interpretation |
|---|---|---|
| `D_FORCED_GEN` | Generate/analyse explicit defender replies at a proved forced boundary `k == b`, including the B=2 pair-plan attempt and its safe fallback. | Defender enumeration already governed by T6/kernel/pair implementation; remaining micro-optimization is category-local. |
| `D_UNFORCED_FHW_ELIGIBLE_GEN` | Generate, close, and classify an unforced Universal (`k < b_current`) for which the frozen shadow class says FHW-eligible. | Open FHW consumption route; zero until the selector/classifier actually exists. |
| `D_UNFORCED_NONFHW_GEN` | Same unforced generation for a completed class verdict of ineligible. | Nearby widening/dominance routes only; FHW itself cannot claim this wall. |
| `D_UNFORCED_UNCLASSIFIED_GEN` | Unforced generation for which the FHW shadow class is absent, errored, digest-mismatched, or not run. | Audit debt, not evidence of ineligibility. Must be zero before an FHW closure claim. |
| `A_OR_GEN` | Mandatory attacker candidate/gate construction that is shared by all children: turn-gate build, first/second candidate enumeration, pair evaluation, dedup, child-vector creation. | Generator/characterization work; existing `ClosureDebtStats` supplies useful subfields. |
| `A_OR_WINNER_PATH` | Exclusive PN/search work charged to the child that ultimately proves a completed Choice node, excluding nested named categories. | Productive prefix; an eliminability ceiling, not automatically waste. |
| `A_OR_ORDERING_MISS` | Exclusive PN/search work charged to other children of a completed winning Choice node before the proof-producing child resolves. | Direct ordering-miss wall and the primary ordering value estimate. |
| `A_OR_UNRESOLVED` | Choice-child work at Unknown/refuted/incomplete Choice nodes, or work whose winning child cannot be identified. | Must not be called ordering waste. Open search-quality residue. |
| `TT_PROBE` | Key creation plus local/shared/fragment TT lookup and compatibility checks. | Cache-probe ceiling. |
| `TT_STORE` | Admission checks, insert/replace, proof compaction done solely for a store, and byte-accounting updates. | Cache-store ceiling. |
| `CENSUS_GATE` | Interior-census feature calculation and dismissal decision. | Existing implemented gate cost; compare with wall it avoids only in a separate A/B. |
| `SEARCH_BOOKKEEPING` | PN selection, threshold arithmetic, apply/undo on descent, recompute/refresh, depth-stage maintenance, and Universal/Choice traversal not charged above. | Scheduler/state-machine residue. |
| `CERT_BUILD` | Materialization, arena compaction, zone-distance/fragment-label rebasing, and horizon preflight. | Certificate construction overhead, distinct from verification. |
| `CERT_VERIFY` | Independent strict-verifier replay, including verifier-local memo and zone reconstruction. | Verification is mandatory; only implementation cost is reducible. |
| `HORIZON_LADDER_OVERHEAD` | Exclusive orchestration between semantic-horizon attempts: cut test, caps/result routing, and retry setup. Work inside each attempt stays in its functional category and is tagged by `horizon_rung`. | Prevents double counting. Total tall-attempt wall is reported by tag, not by adding an overlapping category. |
| `CAP_RESUME_OVERHEAD` | Binding capture/check, re-entry setup, advance bookkeeping, and result packaging outside resumed search/materialization/verification. | Resume mechanism overhead only; continued search remains functional work. |
| `OTHER_MEASURED` | Directly timed root-default intervals not covered above, including result/status plumbing and any missed seam. | Audit debt. Never computed only as `total - named` and never assumed irreducible. |

The partition deliberately refines the proposed “horizon re-solves” and
“cap-resume” buckets. Charging an entire re-solve/resume to those buckets would
overlap defender, attacker, TT, and search work. Instead every functional
interval has one category, while orthogonal dimensions record
`cap_rung={10k,100k,1M,20M}`, `horizon_rung={base,tall,exact_retry,unbounded}`,
and `resume={fresh,resumed}`. Summing by a tag answers “what did tall passes
cost?” without destroying the functional partition.

### 2.2 Non-overlap and completeness invariant

Implement one per-job stack clock. It starts in `OTHER_MEASURED`. Entering a
named scope flushes elapsed time to the current leaf, pushes it, and activates
the new leaf; dropping the guard flushes the leaf and restores its parent.
Thus nested scopes pause parents and every monotonic-clock interval is charged
once. Candidate-specific OR intervals may be held under temporary
`(choice_node, child)` keys and mapped after the final verdict to winner,
ordering-miss, or unresolved; mapping changes labels, never elapsed totals.

For every job:

```text
accounted_ns = sum(all exclusive category ns)
residual_ns  = abs(job_wall_ns - accounted_ns)
allowed_ns   = max(1_000_000 ns, ceil(0.005 * job_wall_ns))
require residual_ns <= allowed_ns
```

The aggregate absolute error must also be at most 0.5% of aggregate wall.
`OTHER_MEASURED` is accumulated by the active default timer. The subtraction
`job_wall_ns - sum(named)` is emitted only as `crosscheck_residual_ns`; it is
never used to populate `OTHER_MEASURED`. A negative signed cross-check, timer
stack imbalance, unmapped temporary key, counter overflow, or missing job-end
flush invalidates the run.

## 3. Counters and exact hook map

### 3.1 Core data model (new)

Add test/measurement-only types, preferably in a new
`packages/hexfield_eq/rust/src/tss_residue.rs` module:

```text
ResidueCategory                 // the exclusive IDs above
ResidueClock                    // stack, active start, category totals
ResidueScopeGuard               // pause/resume RAII guard
ResidueJobKey                   // profile,row,cap,horizon,rung,repetition
ResidueJobReport {
  key, status, nodes, expansions, tt_hits, peak_tt_bytes,
  cert_nodes, cert_edges, strict_verify_result,
  job_wall_ns, category_ns[], other_measured_ns,
  crosscheck_residual_ns, instrumentation_events,
  horizon_cut, horizon_cut_tall, deep_kb_death,
  cap_resume_advances, cap_resume_reentries,
  unforced_nodes[eligible,noneligible,unclassified]
}
```

Use saturating `u64` nanoseconds and counters, but also emit an overflow flag;
an overflow invalidates the job. Compile the clock under `cfg(test)` plus one
explicit measurement feature/env gate. Production/default-off code must not
call `Instant::now()` at hot sites.

### 3.2 Solver and harness hooks

All unqualified Rust filenames in this section are under
`packages/hexfield_eq/rust/src/`.

| Category/data | Existing seam at `6ee9ecfb` | Required hook | Availability now |
|---|---|---|---|
| Job wall and row identity | `tss_corpus.rs:426–495`; current `t0` is at `:454`, while strict verification occurs later at `:507–513` | Move the measurement wrapper outside both solve/session advancement and verification. Keep current `ms` for compatibility, add `job_wall_ns`. Mirror in `tss_spare_corpus.rs:840–899`. | Wall/status/nodes exist; current corpus `ms` excludes verification and is insufficient for the partition. |
| Solve/profile facts | `TssSolver::solve_goal`, `tss_solver.rs:858–1036`; F19 ladder `tss_corpus.rs:394–405,444–494`; runbook `docs/TSS_RUNBOOK.md:10–37` | Open/close one job clock per attempted rung; emit goal, claimant, horizon, TT cap, width flags, and fresh/warm store state. | Existing summaries emit status, nodes, expansions, TT entries/hits, peak bytes, gate counters, and wall at `tss_corpus.rs:522–539`. |
| Forced defender generation | Wide forced classification `tss_solver.rs:5974–5985`; `defender_children` `:6469–6525`; B=2 route/fallback `:6527–6599`; kernel `:8907–8960`. Narrow forced branch `:7812–7843`. | Enter `D_FORCED_GEN` before analysis/kernel/pair-plan generation and leave before recursive child work. Count nodes, legal/kernel sizes, pair-plan success/fallback, generated edges. | Counts are not partitioned. Test-only cumulative `WIDE_GEN_DEFENDER_NANOS` covers only the pair-plan function (`:6540–6543`), so it is only a cross-check. |
| Unforced defender generation | Narrow compatibility route `prove_universal`, `tss_solver.rs:7826–7852,7863–7931`; uniform zone support `zone_initial_candidates` `:9030–9066` and `zone_certificate_extras` `:9094–9148`; post-cert shadow identification `round3_shadow_certificate` `:9278–9418`. | Classify before reading set sizes; enter eligible/noneligible/unclassified scope around initial generation, closure generations, and edge-vector construction. Record `k`, `b_current`, local `B`, legal/set sizes, generation count, class/digest. Recursive proof of each defender child is `SEARCH_BOOKKEEPING`, not enumeration. | Uniform/exact shadow primitives exist. FHW class selection does **not**: `GROUP2_IMPL_REPORT.md` says no implementation, and its **Implementation map** records the audited seams. Therefore all current unforced work is `D_UNFORCED_UNCLASSIFIED_GEN` unless a future reviewed selector returns a frozen class. Required class fields are specified by `DESIGN_GROUP2_NEXT.md:620–629` and `DESIGN_VERIFIER_FHW_EXTENSION.md` §§3–4. |
| Shared attacker generation | `attack_children` `tss_solver.rs:6032–6039`; pair route `:6073–6362`; single route `:6376–6466`; `WideTurnGate::build/second_candidates/evaluate_pair` `:8503–8729`. | Enter `A_OR_GEN`; retain sub-counters for gate build, second candidates, evaluation, dedup, raw/retained counts. Do not classify mandatory full candidate enumeration as ordering miss. | `ClosureDebtStats` already supplies `pair_generation_nanos`, `gate_build_nanos`, `second_candidate_nanos`, `pair_evaluation_nanos`, `dedup_nanos`, and avoidable variants (`tss_core.rs:143–176`); collection is at `tss_solver.rs:6073–6362`, reporting at `tss_corpus.rs:553–601,694–741`. These are test-only and not a complete wall partition. |
| Winner/miss/unresolved OR work | Recursive wide work `tss_solver.rs:4178–4219,4423–5119`; child selection `:5121–5380`; final Choice recomputation `:5469–5531`. Existing ordering records are printed at `tss_corpus.rs:69–134`. | On each selected Choice edge, accumulate exclusive residual work by `(node,child)`. At finalization, proof child -> `A_OR_WINNER_PATH`; other attempted children of a proven Choice -> `A_OR_ORDERING_MISS`; unfinished/refuted Choice work -> `A_OR_UNRESOLVED`. Record ranks/expansions as explanatory counters. Nested TT/generation/census/etc. scopes pause this timer. | Rank and reveal-prefix counterfactual telemetry exists, but no exclusive wall attribution. `reveal_avoidable_*` is a counterfactual sub-estimate, not a replacement for measured miss wall. |
| TT probe | Wide position lookup/insert seam `tss_solver.rs:3949–3999`; fragment lookup `:5890–5917`; narrow lookup path `:7504–7517`; local TT `BoundedTt::lookup` `:9669–9678`; shared/fragment lookups `:10032–10041,:10228–10237`. | Time key creation plus lookup/compatibility under `TT_PROBE`; count local, shared, fragment hit/miss separately. Avoid double scopes when a wrapper calls a primitive. | Hit/entry/byte counts exist in `SolveStats` (`tss_core.rs:393–417`) and corpus summaries; no probe wall. |
| TT store | Wide index insertion `tss_solver.rs:3949–4004`; narrow remember/insert `:7973–8005`; local/shared/fragment insertions `:9684–9718,:10094–10121,:10239–10308`. | Time admission, replacement, store-only compaction, and accounting under `TT_STORE`; count rejection/eviction by store. | Counts exist partly (`tt_evictions`, `tt_admission_rejections`, fragment store sizes); global test-only `WIDE_INSERT_NANOS` is a coarse cross-check, not per-job. |
| Census gate | `evaluate_interior_census_gate`, `tss_solver.rs:164–256`; wide call `:5952–5968`; narrow call `:7547–7563`. | Wrap the whole evaluation in `CENSUS_GATE`; emit evaluated/dismissed and quiet/hot class. | Already timed as `interior_gate_nanos` and exposed in `tss_core.rs:415–417`, `tss_corpus.rs:523–539`. Reuse it and assert it agrees with the new category within timer tolerance. |
| Search bookkeeping | Wide run/stage functions `tss_solver.rs:4178–4269`; `work` `:4423–5119`; `recompute/refresh` `:5469–5818`; narrow recursion `:7466–7588,7591–7947`. | Make this the default inside active search, overridden by named leaf scopes. Keep apply/undo as a subcounter. | `ThresholdScaleStats.descent_nanos` and `state_apply_undo_nanos` (`tss_core.rs:215–220`) already provide test-only exclusive-descent/subset timing and corpus reports (`tss_corpus.rs:603–617`); use as a cross-check, not the category source. |
| Certificate build | Wide materialization `tss_solver.rs:6601–7210`; compacting `compact_certificate` `:10369–10520`; rebase `:1567–1733`; cap-resume build `:1497–1515`; horizon preflight `tss_verify.rs:237–323`. | Wrap materialize/compact/rebase/preflight under `CERT_BUILD`; count cert nodes/edges and preflight retries. | Certificate counts are specified in `DESIGN_GROUP2_NEXT.md:631–638` but not in the current F19 row. `horizon_retry` exists in production counters, not build wall. |
| Strict verification | Mint seam `tss_core.rs:467–493`; `TssVerifier::verify` and `verify_certificate`, `tss_verify.rs:171–227`; production caller `tree.rs:665–703`; corpus caller `tss_corpus.rs:507–513`. | Enter `CERT_VERIFY` exactly once around the strict verifier. Prevent the corpus's second assertion verify from silently becoming an unlabelled duplicate: either include and mark `verification_pass=gate_assertion`, or reuse the already captured result. | Result/failure counters exist; verifier wall does not. |
| Horizon ladder | Current production +8/+12 and exact-T retry in `tree.rs:575–638`; future h16/tall policy and counter definitions in `consolidate-main/docs/PLAN_TSS_MCTS_INTEGRATION.md`, **§5** (`:179–216`) and **§9 V1** (`:313–340`). | Time only routing/setup as `HORIZON_LADDER_OVERHEAD`; tag all attempt categories by rung. Add counters: `horizon_cut` at base, `horizon_cut_tall` at tall, conversion status, and `deep_kb_death` when a tall pass remains Unknown and the terminal live frontier contains an unforced `k < B_local` Universal. Store the witness node/class, not only a scalar. | `horizon_retry`/`horizon_preflight_failed` exist (`tree.rs:447–452,625–649`). `horizon_cut_tall` and `deep_kb_death` are design-required but absent from this source revision—new, not reusable fields. |
| Cap resume | `CapResumeSession::new/advance_to_node_cap`, `tss_solver.rs:1417–1560`; F19 routing/report `tss_corpus.rs:435–500,634–642,775–782`; dedicated tests in `tss_cap_resume.rs:57,167`. | Wrap only binding/re-entry/packaging in `CAP_RESUME_OVERHEAD`; tag resumed search. Emit advances, reentries, incremental and cumulative wall. | Existing `CAP_RESUME_PROFILE`, `resume_wall_ms`, advances/reentries are reusable; they do not partition overhead from continued search. |
| Other | Public solve wrapper `tss_solver.rs:861–1036` and harness/result plumbing not in a named scope. | Leave the stack clock in `OTHER_MEASURED`. Emit direct value plus subtraction cross-check. | New direct measurement. |

### 3.3 Existing telemetry that must be retained

Do not replace these fields; join them to the residue row:

- `DESIGN_GROUP2_NEXT.md` **§5.1** (`:606–645`): Choice counts; unforced
  `b/k/B`, legal/uniform/exact/FHW sizes and eligibility; solve status, nodes,
  wall, TT, peak, horizon, certificate size, verifier result, lambda-order
  counts.
- The promotion/economics ratios in **§6.5** (`:859–998`) and the clean-run
  rules in **§6.7** (`:1019–1066`).
- `ClosureDebtStats`, `ThresholdScaleStats`, `SolveStats`, the `CORPUS`,
  `CLOSURE_ROW`, `REVEAL_ROW`, `THRESHOLD_ROW`, `FRAGMENT_PROFILE`,
  `GEN_PROFILE`, and `CAP_RESUME_PROFILE` summaries cited above.
- `horizon_retry`, `horizon_preflight_failed`, `deep_verify_failed`, zone and
  pair counters in `tree.rs:435–461`. `horizon_cut_tall` and `deep_kb_death`
  must be clearly marked **new** until implemented.

Global `wide_gen_profile()` atomics (`tss_solver.rs:2102–2177`) are cumulative
and test-only. They may detect a gross accounting error but cannot populate a
per-row category.

## 4. Measurement protocol

### 4.1 Frozen inputs and profiles

Run one test thread and record commit, dirty-state hash, compiler/version,
target triple, CPU model/power policy, environment, corpus SHA-256, feature
flags, semantic horizon, node cap, TT cap, repetition, and instrumentation
schema version before inspecting results.

1. **F19 acceptance profile:** all 19 rows from
   `rust/corpus/forcing_corpus_moves.txt` (`tss_corpus.rs:1–17,148–219`) at the
   official 2 GiB cap and `10k -> 100k -> 1M -> 20M` ladder; NO rows stop at
   1M. This is exactly `docs/TSS_RUNBOOK.md:14–32` and
   `tss_corpus.rs:394–405`.
2. **S2 spare profile:** both checked-in rows from
   `rust/corpus/spare_corpus_moves.txt` through
   `tss_spare_corpus_check` (`tss_spare_corpus.rs:771–899`): 512 MiB, 1M
   nodes, and each row's `root placements + reference_plies` horizon. These
   are soundness controls, not a second positive ladder, per
   `docs/TSS_RUNBOOK.md:34–37`.
3. **Trainer-context projection (separate report, never pooled with F19/S2):**
   replay the same 21 roots with the documented 256 KiB per-solve cap and
   2,000-node default (`docs/TSS_RUNBOOK.md:20–24,45–47`). Use the channel's
   frozen horizon arm. This answers production economics but does not alter
   acceptance results.

Primary aggregate percentages are profile-specific. There is no unweighted
“grand average” across 2 GiB acceptance, 512 MiB spare, and 256 KiB trainer
contexts. Within F19, the **protocol-wall** aggregate includes every rung
actually executed; also publish a final-attempt-only view so cap-ladder cost is
visible rather than hidden.

### 4.2 Repetitions and statistics

- Run three clean process repetitions per profile. Baseline and instrumented
  binaries/runs must use the same manifest and no cross-process warm TT.
- Node, expansion, status, certificate digest, certificate node/edge count,
  strict-verifier result, and category event counts must be deterministic
  across repetitions. A mismatch invalidates the row; do not median semantic
  mismatches away.
- Per row/category report the median nanoseconds and median share. Aggregate
  using the campaign convention:
  `median_r(sum_jobs category_ns[r]) / median_r(sum_jobs job_wall_ns[r])`.
  Also report p90/p95/max row shares and all raw three repetitions.
- Preserve row/rung stopping exactly. Never raise a cap after seeing a miss,
  delete a slow row, or use only the winning repetition.

### 4.3 Instrumentation overhead gate

Before accepting residue numbers, perform a matched A/A check with residue
instrumentation disabled versus enabled, using the same release binary when
the feature design permits and alternating order across repetitions. Use at
least seven clean repetitions of the F19 protocol and three of S2.

Required semantic identity: status, nodes/expansions, certificate bytes,
verifier result, TT hit/entry counts, and rung stopping all match. Target
overhead is **<=1.0%** of median aggregate wall; the hard validity budget is
**<=2.0%** for each profile:

```text
overhead = median(total_wall_instrumented) /
           median(total_wall_disabled) - 1
```

Also require no instrumented profile to regress p95 per-row wall by more than
5% unless the absolute median difference is below 1 ms and the aggregate hard
budget passes. Report timer-scope event count and `ns/event` estimate. If the
2% budget fails, optimize or coarsen timer boundaries and rerun; do not
“correct” category times by scaling them after the fact.

## 5. Report format

### 5.1 Per-job table

One machine-readable JSONL row and one generated Markdown row per job:

| profile | row | rep | cap rung | horizon rung | resume | status/verified | nodes/exp | TT hit/peak | cert n/e | total ms | forced D | unforced FHW | unforced other/unclass | A gen | A winner | A miss | A unresolved | TT probe/store | census | search | cert build/verify | horizon/resume overhead | other | cross-check |
|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Every category cell carries both milliseconds and `% job_wall`. Add the §5.1
Group-2 fields as a nested object, plus `horizon_cut`, `horizon_cut_tall`,
`deep_kb_death`, cap-resume advances/reentries, instrumentation events, direct
`other_measured_ns`, and signed/absolute cross-check residual.

### 5.2 Aggregate tables

Publish separately for F19 protocol wall, F19 final attempts, S2, and trainer
projection:

| category | median sum ms | wall % | p95 row % | max row/id | disposition | measured value estimate | estimate method | eliminability upper bound |
|---|---:|---:|---:|---|---|---:|---|---:|

For `A_OR_ORDERING_MISS`, the central estimate is its measured wall and is
cross-reported with existing reveal/rank counterfactuals. For FHW, report both
eligible generation wall and the matched shadow consume simulation estimate
specified by `DESIGN_VERIFIER_FHW_EXTENSION.md` **§9.2**; eligibility count
alone is not a speed estimate. Where only a ceiling is known, publish the
interval `[0, category wall share]` and label the central estimate
`NOT_MEASURED`; such a route is not closure-ready until it gains a measured
counterfactual or is moved to impossible/implemented.

An orthogonal rung table reports:

| horizon/cap tag | attempts | cuts | conversions | wall ms | wall % | `deep_kb_death` | verified yield |
|---|---:|---:|---:|---:|---:|---:|---:|

The tag wall is obtained by summing exclusive categories bearing that tag; it
must equal tagged job wall within the same tolerance.

### 5.3 Closure criteria and paragraph template

A profile may be called **CLOSED** only when:

1. every job and aggregate passes the partition and overhead gates;
2. `D_UNFORCED_UNCLASSIFIED_GEN == 0` for any FHW claim;
3. `OTHER_MEASURED <= 1.0%` aggregate and `<=2.0%` on every row, or every
   excess row has been rerun after adding a named category;
4. every named category has an owner disposition and every OPEN category has
   a measured central estimate plus upper bound; and
5. verdict/certificate identity and strict verification are clean.

Use this exact paragraph shape, filling values from the aggregate table:

> On profile `<manifest digest>`, three clean repetitions of `<corpus>`
> accounted for `<accounted>%` of `<median protocol wall>` ms, with direct
> `OTHER_MEASURED=<other>%`, cross-check error `<error>%`, and instrumentation
> overhead `<overhead>%`. The largest single open reducible category is
> `<category>` at `L=<L>%` of wall; an optimization confined to one category
> is therefore bounded by `L`. Across all open reducible categories the
> Amdahl ceiling is `R_open=<sum>%`, so an arbitrary future multi-category
> optimization is bounded by `R_open`. Its measured central value estimate is
> `<estimate>%` by `<method>`. All remaining categories are
> implemented/verified or linked to impossibility-ledger entries `<IDs>`.

If `OTHER_MEASURED`, unclassified FHW work, or an open category without a
central estimate violates the criteria, replace **CLOSED** with **OPEN** and
name the exact residual. Do not round a failing value below its threshold.

## 6. Estimated future implementation scope

Expected release-quality scope: **650–950 Rust lines** plus generated data,
with no solver-semantic change.

| File | Estimated delta | Work |
|---|---:|---|
| new `packages/hexfield_eq/rust/src/tss_residue.rs` | 220–320 | Category enum, stack clock/guards, OR temporary-key finalization, job/report schema, invariant checks. |
| `packages/hexfield_eq/rust/src/tss_core.rs` | 40–80 | Add residue report handle/fields to test diagnostics without changing production decisions. |
| `packages/hexfield_eq/rust/src/tss_solver.rs` | 220–330 | Scope hooks at defender/attacker/TT/census/search/cert/resume seams; FHW-class adapter is measurement-only unless separately authorized. |
| `packages/hexfield_eq/rust/src/tss_verify.rs` | 10–25 | Verifier scope entry or callback; no acceptance-rule change. |
| `packages/hexfield_eq/rust/src/tss_corpus.rs` | 100–150 | Manifest/repetition/job rows, aggregate checks, existing telemetry join. |
| `packages/hexfield_eq/rust/src/tss_spare_corpus.rs` | 35–60 | S2 job rows and profile assertions. |
| `packages/hexfield_eq/rust/src/lib.rs` | 2–5 | Test/feature-gated module declaration. |

The FHW classifier/strict-verifier extension itself is **outside** this timing
estimate; `GROUP2_IMPL_REPORT.md` confirms it is not currently implemented.
If that separate owner-gated change lands, this spec only consumes its frozen
class result.

### Required tests for the cargo-enabled lane

1. `residue_clock_nested_scopes_partition_exactly` — fake clock or generous
   monotonic tolerance; parent pauses while child runs.
2. `residue_other_is_direct_not_subtracted` — deliberate unscoped work raises
   direct Other and independently satisfies the cross-check.
3. `residue_or_edges_finalize_winner_miss_unresolved_without_changing_sum`.
4. `residue_unforced_missing_or_bad_fhw_class_is_unclassified_not_ineligible`.
5. `residue_forced_pair_fallback_is_counted_once` — B=2 pair plan and ordinary
   fallback never overlap.
6. `residue_tt_probe_store_nested_calls_do_not_double_count` — local, shared,
   and fragment cases.
7. `residue_verification_is_inside_job_wall` — accepted, rejected, and Unknown
   paths.
8. `residue_horizon_tags_partition_base_tall_exact_retry` — functional
   categories remain exclusive; tag sums agree.
9. `residue_cap_resume_overhead_excludes_continued_search` and extend existing
   `cap_resume_discards_on_binding_or_cap_mismatch`.
10. `residue_accounting_rejects_stack_leak_overflow_and_tolerance_failure`.
11. Ignored `tss_residue_f19_gate` — exact official F19 manifest, three clean
    repetitions, semantic/certificate identity, per-job and aggregate
    invariants.
12. Ignored `tss_residue_spare_gate` — both S2 rows and exact canonical
    profile.
13. Ignored `tss_residue_overhead_gate` — disabled/enabled A/A protocol and
    the 1% target/2% hard gate.
14. Snapshot/parser test for JSONL and generated aggregate tables, including
    refusal to emit the closure paragraph when Other, unclassified FHW, an
    unmeasured OPEN category, or cross-check error exceeds its bar.

No test may assert performance by sleeping. Performance gates use the frozen
corpus protocols above; unit tests assert accounting structure and semantic
identity.
