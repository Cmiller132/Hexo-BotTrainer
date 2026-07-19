# R-CREL-6 Phase 3B - FirstStone pair-normal-form

Date: 2026-07-18  
Input branch/HEAD: `hunt/cert-support` / `5f5da82a04d14f645fdbf08ea96937a428182cde`  
Disposition: **NULL - OPEN REMOTE-REMOTE FALLBACK**  
Deployment state: no consuming solver change; census is cfg(test), ignored, exact-env opt-in  
Verifier: unchanged

## Headline

The aggressive turn-start `join_live` normal form has a large measured pair-domain
ceiling but is not soundly deployable. Across all 142 eligible unresolved
FirstStone states in the frozen 300-root leaf cohort, its both-turn-start-legal
subdomain retains 7,851,331 of 17,042,354 unordered pairs (46.069522%), a
53.930478% reduction or 2.170632x smaller domain. The entire reduction is the
9,191,023 remote-remote pair class. Second cells that become game-legal only
after the first placement are outside this census.

The missing theorem is decisive: a remote-remote completed turn creates no new
claimant count-4 window, but it may still be the necessary double block,
occupancy tempo, or legality-frontier seed in a winning strategy. The exact
12-root unresolved prefix contains 981,062 such nonterminal pairs and all are
strict-quiet; the measurement classifies them but cannot discard them.

The safe endpoint normal form retains remote-remote pairs and only chooses one
orientation per unordered endpoint. In the measured both-turn-start-legal
subdomain that maps 34,084,708 directed sequences to 17,042,354 endpoints
(50%). The selected wide route deduplicates accepted endpoint children with
`seen_pairs`, but only after directed classification, so a safe generator-level
opportunity remains unmeasured. No consuming A/B was authorized, and the
binding >=3% wall bar is unevaluated. The result is therefore the preregistered
**NULL**, not PASS and not KILL.

## Pre-registration and chronology disclosure

`FIRSTSTONE_PAIR_NF_PREREG_V1_RAW.log` is the binding cold-gater. It was created
at 11:36:05 EDT, before the first resource-gated Cargo attempt at 11:43:18, and
already fixes these outcomes:

- PASS requires a complete fallback-pair theorem, >=20% retained-pair
  improvement, all identity/coverage gates, and an authorized >=3% wall gain
  with no cohort regression above 3%.
- KILL covers required-path omission, a scoped-lemma or commutation
  counterexample, legality/verifier/hard-verdict conflict, or a proved winning
  remote-remote counterexample.
- NULL covers attractive count shrink without a theorem for
  strict-quiet/non-new-threat fallbacks, already-landed order symmetry, >80%
  retention, or a later sub-3% timing result.

The file named `FIRSTSTONE_PAIR_NF_PREREG_FROZEN_RAW.log` was created at
11:56:10, after the 11:53:20 authoritative census. It reconstructs the amended
bounded-scope working text but is not timestamp evidence. Calling it the
binding pre-run artifact would be false; the full correction is recorded in
`FIRSTSTONE_PAIR_NF_CHRONOLOGY_RAW.log`. The bars did not change.

There is a separate protocol-scope deviation. V1 describes real phase-machine
exact classification and a standard-cohort exhaustive census. Execution used
exact snapshot geometry only for NQ2 plus 12 eligible roots and real
phase-machine checks only for a 512-pair NQ2 sample. The post-run reconstruction
cannot establish that narrowing as a pre-run amendment. This blocks PASS and
leaves full endpoint-state commutation unevaluated; it does not change V1's
binding NULL condition for attractive shrink with an open fallback theorem.

The census also excluded eight nonterminal FirstStone roots already resolved
by `own_win_now` or `forced_loss`, although V1 did not preregister that extra
exclusion. Including their start-state combinatorics would yield 8,254,042 of
17,825,283 pairs retained (46.305251%, a 53.694749% ceiling) rather than
46.069522%. This sensitivity does not close the remote-remote theorem or
change NULL; the eight-row arithmetic and immutable handoff provenance are in
`FIRSTSTONE_PAIR_NF_SCOPE_AUDIT_RAW.log`.

## Candidate and enumeration architecture

At a claimant FirstStone state P, seal the turn-start legal set L(P) and let
J(P) contain legal cells belonging to an active claimant length-six window.
The aggressive candidate keeps exactly unordered pairs intersecting J:

`C(|L|,2) - C(|L|-|J|,2)`.

The cfg(test) harness performs:

1. Exact combinatorial L/J and J-J/J-remote/remote-remote counts for every
   eligible unresolved FirstStone state among all 300 deterministic roots.
2. Exact turn-start WindowStore snapshot classification for the canonical NQ2
   fixture and first 12 eligible unresolved roots, enumerating every unordered
   pair once.
3. A scoped lemma check that every nonterminal pair creating a new claimant
   count-4-or-stronger window intersects J.
4. Real phase-machine replay of 512 evenly spaced NQ2 unordered pairs in both
   orientations. The sample is not guaranteed to include the historical pair
   and checks recorded legality/outcome/classification invariants, not equality
   of the two final endpoint states.

This architecture supports complete both-turn-start-legal cohort pair counts
and a bounded exact geometry claim. It does not cover newly legal second cells,
enumerate all reachable states or winning strategies, or replay the engine
beyond the stated NQ2 sample. The output states those scopes explicitly and
binds the deterministic census fingerprint `DFCB3847A41CE280`.

The machinery is a 585-line ignored test module behind exact environment value
`TSS_FIRSTSTONE_PAIR_NF_CENSUS=1`, plus a cfg(test)-only seam exposing the
frozen standard leaf roots. It changes neither solver production behavior nor
the strict verifier.

## Results

All Cargo invocations used `.target-hunt`, release,
`x86_64-pc-windows-msvc`, and serial tests. Every launched invocation recorded
at least 10 GiB available, at least 5 GiB free, and zero foreign Cargo
processes. Three low-memory checks correctly launched no Cargo.

Post-run corpus binding: the manifest now pins the external corpus used to build
the 300 roots and printed by the authoritative census
(`FIRSTSTONE_PAIR_NF_CENSUS_RERUN_RAW.log:11`) to 3,696,030 bytes / 6,902 JSONL
rows and SHA-256
`54FAE7AEBCEF2A9D19D13C1946FAE36C0565E21BC726C25E2E4E230CFB42A5B7`. The
raw emitted the same path and `eligible_games=6902`, but no corpus byte hash;
this makes future reruns byte-exact and is supporting post-run evidence, not a
cryptographic measurement-time precommit.

| Gate | Result | Authoritative raw |
|---|---:|---|
| release no-run rerun | passed in 11.03 s | `FIRSTSTONE_PAIR_NF_BUILD_RERUN_RAW.log` |
| focused pair rerun | 8 passed, 1 ignored | `FIRSTSTONE_PAIR_NF_FOCUSED_RERUN_RAW.log` |
| compact census rerun | 1 passed in 0.11 s | `FIRSTSTONE_PAIR_NF_CENSUS_RERUN_RAW.log` |
| NQ2 sampled replay | 1,024 orientation invariant checks passed | `FIRSTSTONE_PAIR_NF_CENSUS_RERUN_RAW.log:10` |

The authoritative both-turn-start-legal cohort totals are:

| Quantity | Count | Fraction of unordered domain |
|---|---:|---:|
| unordered pairs | 17,042,354 | 100.000000% |
| J-J | 1,316,649 | 7.725746% |
| J-remote | 6,534,682 | 38.343776% |
| retained aggressive domain | 7,851,331 | 46.069522% |
| remote-remote / omitted aggressive domain | 9,191,023 | 53.930478% |

The bounded exact scope checked 16,720 new-count-4 pairs with zero scoped-lemma
misses. That is evidence for the local threat-creation lemma only. It is not a
global completeness percentage because remote-remote pairs lie outside the
lemma antecedent.

At the historical NQ2 parent, |L|=537 and |J|=136: 63,716 of 143,916 pairs are
retained (44.273048%), a 55.726952% reduction or 2.258711x smaller domain. The
required `(6,0),(6,-6)` pair is retained through `(6,0) in J`, and all 1,024
sampled orientation checks passed. The required pair is not guaranteed to be
in that sample, and endpoint-state equality was not compared. This parent has
`own_win_now=true` and is resolved by the solver precheck, so it is a geometry
fixture rather than an unresolved quiet-search root.

The first census raw passed and preserved summary totals, but its console
capture contains a literal truncation marker and only 89 of 142 eligible leaf
ROOT records. It is retained as non-authoritative. The compact rerun contains
the promised 12 leaf ROOT/EXACT records, all 8 resolved SKIP records, the NQ2
records, category totals, and fingerprint in 99 complete lines.

## Why no timing A/B exists

Binding V1 forbids timing the aggressive deletion filter before its theorem
closes. It did not close. A separate >=20% actual-duplicate condition appears
only in the post-run working-text reconstruction, not binding V1, and this
phase did not measure that quantity. Source review instead shows that wide
`seen_pairs` runs after directed classification. Historical closure counters
record 1,803,229,707 evaluations, 14,864,718 accepted directions, and 7,495,860
retained endpoints (`CLOSURE_COUNTER_FULL_OFF_RAW.log:143`); those are prior
context, not a FirstStone A/B.

Consequently, the 53.930478% figure is a combinatorial ceiling for the unsafe
aggressive candidate, not a wall, node, RSS, or certificate result. PASS fails
both the global-theorem condition and the subsequent >=3% paired-wall
condition. The required path, scoped lemma, sampled invariants, focused tests,
and production isolation showed no KILL counterexample. The final-state
commutation gate is unevaluated, not claimed passed. NULL matches the binding
bar exactly.

## Hostile self-review

Ten attacks are recorded in `FIRSTSTONE_PAIR_NF_COLD_REVIEW_RAW.log`. The
headline survived attempts to promote count shrink, reinterpret zero scoped
misses as global proof, use the pre-resolved NQ2 fixture as unresolved coverage,
delete remote pairs by quietness, or time the unsound filter anyway.

Three attacks found useful failures. Tighter `join_adj1`/`join_adj2` forms omit
both endpoints of the known NQ2 path, so they cannot represent it. Source review
reopened the safe pre-classification orientation opportunity. Chronology review
both refuted the misleading frozen filename and exposed the V1 execution-scope
deviation; a post-seal audit then exposed the eight additional pre-resolved
root exclusions. Finally, inspection rejected the initial census capture and forced
the compact authoritative rerun.

## Residual

The sharp aggressive residual is a theorem or certificate-relative discharge
for every remote-remote FirstStone pair. A sound next candidate can retain
remote pairs that collectively hit all turn-start opponent count-4/count-5
windows, but nonurgent remote blocking/seeding still remains.

Separately, a safe generator-level orientation rule may avoid reverse directed
classification before `seen_pairs`. It needs a direct duplicate-attempt census
and must preserve the only generated direction when candidate membership is
non-monotone, plus newly-legal second cells. Any re-arm must prove exact
unordered-endpoint coverage, all defender branches, and strict verification
before timing.

## Cold-gater checklist

1. Verify `FIRSTSTONE_PAIR_NF_HASHES_RAW.log`, then confirm `tss_verify.rs` has
   no diff and matches the bound hash.
2. Read timestamp-verifiable `FIRSTSTONE_PAIR_NF_PREREG_V1_RAW.log` for binding
   bars; read the chronology before using either later preregistration file.
   Read `FIRSTSTONE_PAIR_NF_SCOPE_AUDIT_RAW.log` for the eight-root sensitivity
   and immutable handoff provenance.
3. Use `FIRSTSTONE_PAIR_NF_CENSUS_RERUN_RAW.log` as the sole authoritative
   census. Check its setup scope, NQ2 lines 8-10, DONE line 94, and exit line 99.
4. Use `FIRSTSTONE_PAIR_NF_ANALYSIS_RAW.log` for copy-paste arithmetic and
   `FIRSTSTONE_PAIR_NF_ARCH_RAW.log` for source/theorem provenance.
5. Confirm the focused and build reruns, then inspect the three gate-attempt
   raws to verify that low-memory checks launched no Cargo.
6. Treat `FIRSTSTONE_PAIR_NF_CENSUS_RAW.log` as disclosed/truncated and never
   as the authoritative per-root stream.
