# R-CREL-6 Phase 1 — zone-emitting leaf route

Date: 2026-07-18  
Branch: `hunt/cert-support`  
Input HEAD: `5f5da82a04d14f645fdbf08ea96937a428182cde`

## Disposition

**Semantically active, C_rel-admissible zone emission is IMPOSSIBLE under the
frozen wide/gated search envelope. Phase 2 is skipped.** A literal
`zone: Some` field can strict-verify on some dispatch-only certificates, but
the verifier does not exercise its zone theorem there and C_rel's independent
zone correspondence rejects it. That syntactic loophole is not a usable zoned
certificate for the registered leaf-relevance question.

There is a protocol ambiguity that the cold gater must not miss. The landed
R-CREL-5 generator's `zoned_nodes`/`zoned_present` coverage predicate counts
any literal `zone: Some`, including metadata ignored by dispatch. A verbatim
run of that syntactic counter could therefore count the inert annotations that
strict-verified 23/23 here. The disposition in this report interprets
"ZONED" semantically: the verifier's zone branch must be active and the
certificate must be admissible as a C_rel template. Actual `admit_template ->
extract_interface` performs independent correspondence derivation and would
return `zone_rederivation_disagreement`; that rejection is a deterministic
source-path inference, not a separately measured rejection raw. Under the
purely syntactic interpretation, Phase 1 can emit ignored metadata and the
Phase-2 skip is contestable. `CREL6_ZONE_PROTOCOL_AUDIT_RAW.log` preserves both
readings explicitly.

The standard-root identity run reproduced the binding hard counts, 16/300 at
h=8 and 39/300 at h=16, with zero status, structural-statistic, root-PN/DN,
expansion-order, or non-zone certificate mismatches. However, it found no
Universal on which the unchanged verifier's zone theorem can run. At h=8 the
hard certificates contained no Universals. At h=16 they contained 319
Universals, all 319 marked `implicit_dispatch=true`; independent zone
rederivation returned zero and the default-off annotator emitted zero zones
(`CREL6_ZONE_IDENTITY_RERUN_RAW.log:36,110`).

This is not a missing-label problem. Exact replay states, local defender budget
`d`, `build_horizon`, protected cells, and the required uniform zone can all be
derived after materialization. The missing information is a proven child edge
family at a non-dispatch defender state. Producing it requires a second search
and changes certificate topology beyond zone fields, so it cannot be added as
a zone-field-only annotation. Such an outside recomputation need not perturb
the already-finished wide search's metrics; certificate identity is the
decisive blocker.

## Exhaustive route argument

The completeness claim was made only after enumerating the frozen route:

1. Immediate winners return leaf-only certificates and contain no Universal.
2. With fragments off, every non-leaf proof body comes from the local
   `WidePnSearch` arena.
3. Wide PN retains a defender node only when the position is post-opening,
   has an opponent threat, has no own immediate win, and
   `min_hitting_set == Some(b)`. Any other defender node is immediately
   `Refuted` (`tss_solver.rs:5922-5932`).
4. The exhaustive wide materializer reaches the ordinary Universal builder or
   the defender-pair Universal builder. The ordinary path propagates the arena
   flag; both pair forms hard-code `implicit_dispatch=true`; all write
   `zone: None` (`tss_solver.rs:6718-6898`).
5. The unchanged verifier treats dispatch and zone as alternative modes. It
   takes dispatch first for `implicit_dispatch=true`; only the following
   `else if zone` calls `verify_zone_node` (`tss_verify.rs:827-911`). That zone
   check rejects `min_hitting_set >= b` (`tss_verify.rs:1216-1232`).

Therefore the wide eligibility set (`k == b`) and the genuine strict-zone set
(`k < b` or no hitting set) are disjoint. The separate narrow route has the
missing `zone_initial_candidates` and recursive `zone_certificate_extras`
work (`tss_solver.rs:7646-7743,8864,8928`), but invoking it would change the
decided engine and envelope.

## Default-off diagnostic seam

The Phase-1 patch adds only cfg(test)-reachable machinery:

- `TSS_LEAF_ZONED_EMIT` is absent/off by default and is inspected only after
  wide search, materialization, compaction, and ordinary zone-distance rebasing.
- It replays the completed certificate, considers only non-dispatch Universals
  satisfying the verifier's eligibility predicate, rebases `d`, and submits
  every candidate to the unchanged `TssVerifier`. A rejected annotation is
  never emitted.
- The ignored identity harness runs paired persistent solvers across the same
  300 standard roots per horizon and compares statuses, every retained
  `SolveStats` field after normalizing only elapsed-nanosecond fields, final root PN/DN, all captured
  expansion events in order, and certificates after clearing only zone fields.

No candidate reached annotation: eligibility, accepted, rebase-rejected, and
strict-rejected counts were all zero. Thus the strict-rejection instruction is
not applicable; no zoned certificate was emitted. Every ordinary hard
certificate still strict-verified and `hard_without_strict=0`.

## Identity results

| Horizon | Roots | Hard | Expansion events compared | Universals | Dispatch Universals | Genuine-zone Universals | Zoned certs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 300 | **16** | 1,852 | 0 | 0 | 0 | **0/16** |
| 16 | 300 | **39** | 6,649 | 319 | 319 | 0 | **0/39** |

For both horizons: status mismatches 0, structural-statistic mismatches 0,
root-number mismatches 0, event-order mismatches 0, certificate non-zone
mismatches 0, and hard certificates without strict acceptance 0. The
authoritative cold-review rerun passed 1/1 in 1.43 s
(`CREL6_ZONE_IDENTITY_RERUN_RAW.log:114-116`).

## Hostile self-review

Four attempts tried to refute the headline before Phase 1 closed:

1. **“The existing edges are enough; only `d` is absent.”** Refuted. The run
   found 319 Universals, all on the `k == b` dispatch boundary, and zero
   independently rederived zones. The verifier rejects that boundary before
   zone coverage is considered.
2. **“A literal `zone: Some` is sufficient.”** This exposed a syntactic
   loophole, not a solution. A deliberately bogus `d=u32::MAX` annotation
   strict-verified on all 23 h=16 certificates that contained Universals,
   because dispatch ran first and ignored `ZoneInfo`; independent zone
   rederivation remained empty. C_rel requires exact agreement with that
   independent rederivation and would return `zone_rederivation_disagreement`
   (`cert_support_hunt.rs:1360-1395`). Counting this inert field would
   manufacture coverage without exercising the zone contract.
3. **“Use the authorized bounded post-hoc recomputation to rebuild children.”**
   Refuted by the certificate-identity constraint. Candidate enumeration plus
   recursive proof construction may leave the already-finished wide metrics
   untouched, but it adds/replaces edges and children, so the certificate is
   no longer equal modulo zone fields; C_rel admission also still requires the
   independent zone rederivation to agree.
4. **“The unqualified IMPOSSIBLE headline survives the inert mutation.”**
   Refuted in cold review. Literal strict-verifying `Some` emission is possible;
   the corrected claim is specifically about semantically active,
   C_rel-admissible zones. This does not reopen Phase 2 because inert dispatch
   metadata cannot populate an admissible C_rel warm template.

The copy-paste version of these attempts and outcomes is in
`CREL6_ZONE_ANALYSIS_RAW.log`, including the cold-review correction.

A fifth post-seal attack compared this semantic criterion to the R-CREL-5
generator's literal counter. It exposed the protocol ambiguity above but did
not refute the semantic impossibility result
(`CREL6_ZONE_PROTOCOL_AUDIT_RAW.log`).

## Phase-2 consequence

R-CREL-6 says to skip Phase 2 if envelope-identical emission is impossible.
Under the semantic interpretation above, accordingly
`HUNT_REPORT_CREL_LEAF2.md` and `CREL_LEAF2_*` do not exist, and no
RELEVANT / IRRELEVANT-AT-LEAF / ABORT economics verdict is claimed. Per-h net
gain, clustered lower bound, non-parent hit rate, accounted peak, and paired
RSS remain **not measured**, not zero. The prior R-CREL-5 ABORT remains the
last leaf-economics disposition; this phase sharpens its cause.

## Resource, verifier, and worktree audit

Post-run corpus binding: the manifest now pins the external leaf corpus printed
by the authoritative run (`CREL6_ZONE_IDENTITY_RERUN_RAW.log:8`) to 3,696,030
bytes / 6,902 JSONL rows and SHA-256
`54FAE7AEBCEF2A9D19D13C1946FAE36C0565E21BC726C25E2E4E230CFB42A5B7`. The
raw emitted the same path and `eligible_games=6902`, but no corpus byte hash;
this makes future reruns byte-exact and is supporting post-run evidence, not a
cryptographic measurement-time precommit.

- `tss_verify.rs` has no diff and SHA-256
  `9990D38618DA2204351E328CA0143BE2AEF98BB3001E4A0462CF346B707F2CE8`.
- All Cargo launches used `.target-hunt`; tests used release,
  `x86_64-pc-windows-msvc`, and `--test-threads=1`.
- One build check correctly blocked without launching Cargo at
  10,125,586,432 available bytes. The minimum actual-launch readings were
  10,887,544,832 available and 10,885,431,296 free bytes, with zero foreign
  Cargo processes (`CREL6_ZONE_BUILD_RAW.log:1-5`). Every invocation was far
  below ten minutes.
- The repository-wide formatter touched unrelated files. The immediate diff
  audit identified and reverse-applied only those 34 formatter diffs; the
  Phase-1 checkpoint diff contains only `tss_solver.rs` and
  `tss_crel_leaf_hunt.rs` (`CREL6_ZONE_FMT_RAW.log`).
- No commit was created.

## Sharpest residual

A genuine semantically active zone-emitting wide leaf proof needs a separately authorized search
change: retain non-dispatch defender states, enumerate a sound zone edge set,
and prove its missing children. That necessarily changes the frozen leaf
envelope. Post-hoc labels alone cannot supply the absent proof body.

## Artifact set

- `CREL6_ZONE_PREREG_RAW.log` — binding semantic-zone and identity contract,
  written before measurement.
- `CREL6_ZONE_IDENTITY_RERUN_RAW.log` — authoritative full paired run after
  expanding identity to all non-time `SolveStats` fields.
- `CREL6_ZONE_IDENTITY_RAW.log` — superseded pre-cold-review full run.
- `CREL6_ZONE_ATTEMPT1_SUMMARY_RAW.log` — truthful non-authoritative first run.
- `CREL6_ZONE_BUILD_RAW.log` — blocked gate plus successful release build.
- `CREL6_ZONE_FMT_RAW.log` — format gate and scoped recovery audit.
- `CREL6_ZONE_ANALYSIS_RAW.log` — copy-paste aggregates, route enumeration,
  hostile review, resource audit, and disposition.
- `CREL6_ZONE_COLD_REVIEW_RAW.log` — independent checkpoint review and folded
  corrections.
- `CREL6_ZONE_PROTOCOL_AUDIT_RAW.log` — post-seal hostile audit of the
  syntactic R-CREL-5 coverage predicate versus C_rel admission semantics.
- `CREL6_ZONE_CODE_DIFF_RERUN_RAW.patch` — authoritative Phase-1 code snapshot.
- `CREL6_ZONE_CODE_DIFF_RAW.patch` — superseded pre-cold-review snapshot.
- `CREL6_ZONE_HASHES_RAW.log` — SHA-256 manifest.
