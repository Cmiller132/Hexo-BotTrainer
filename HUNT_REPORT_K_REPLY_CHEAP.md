# R-CREL-6 Phase 3A — K_reply cheap-trigger re-test

Date: 2026-07-18  
Input branch/HEAD: `hunt/cert-support` / `5f5da82a04d14f645fdbf08ea96937a428182cde`  
Disposition: **NULL — ALREADY-COMPUTED ANALYSIS TRIGGER ALREADY LANDED**  
Deployment state: `TSS_K_REPLY_CONSUME` remains default-off  
Verifier: unchanged

The measurements compiled the named base HEAD plus the inherited Phase-1
cfg(test), default-off zone-emission diff in `tss_solver.rs`. That diff adds 139
lines in two earlier hunks; the K_reply block and `tss_k_reply_shadow.rs` are
byte-identical to base HEAD, and `TSS_LEAF_ZONED_EMIT` was unset. Thus “current
worktree” below is not presented as a clean-HEAD build.

## Headline

The deferred already-computed-`ThreatAnalysis` trigger premise is stale. The exact proposed O(1) trigger,
`eligible && analysis.opp_threat_count > 0`, was already part of the original
G2R8 consumption implementation and therefore was already present during its
negative deep-profile measurements. There is no distinct candidate at current
HEAD to pass or kill.

Fresh current-worktree gates passed and confirmed expected routing. The narrow
300-root leaf probe consumed 398 urgent fallbacks with unchanged 14/14 verdict
counts. Its two fresh wall samples were -0.768% and -12.882%; that spread is not
used to relabel identical code as a new lever. The selected wide/gated route
remained a literal deterministic no-op.

## Frozen scope and bars

The preregistration content was written before the first K_reply measurement
and is archived byte-for-byte at `K_REPLY_CHEAP_PREREG_FROZEN_RAW.log`; that
archive path itself was created later. The timestamp witness and exact
reconstruction are `K_REPLY_CHEAP_PREREG_TRANSCRIPT_RAW.log`, with the cited
source lines preserved verbatim in `K_REPLY_CHEAP_PREREG_LOG_EXCERPT_RAW.log`.
That excerpt is post-run chronology evidence, not a cryptographic precommit.
The live
`K_REPLY_CHEAP_PREREG_RAW.log` is a disclosure-corrected post-audit mirror. The
frozen file enumerated every relevant
narrow quiet fallback and fixed these binding outcomes:

- PASS required the provenance audit to be refuted, a genuinely new trigger,
  all identity gates, aggregate wall improvement of at least 3%, and no cohort
  regression above 3%.
- KILL covered any soundness/identity failure or a distinct candidate's wall
  regression above 3%.
- NULL covered confirmation that the trigger was already in the exact prior
  negative implementation.

The architecture audit confirmed this scoped NULL before timing
(`K_REPLY_CHEAP_ARCH_RAW.log:9-31,42-44`).

## Enumeration architecture

At each `NarrowCompatSearch::prove_choice` quiet-turn fallback:

1. The complete legal domain is generated because the fallback path needs it.
2. Eligibility requires a live claimant `SecondStone` state.
3. Urgency is read in O(1) from the already-recomputed `ThreatAnalysis` as
   `opp_threat_count > 0`.
4. Only an urgent state with consumption or observation enabled calls
   `k_reply_kernel`.
5. A consuming urgent branch replaces the legal domain with exact Q8 cells;
   nonurgent branches retain the full domain.
6. Existing pair-context filtering and deterministic sorting follow.

The selected wide/gated leaf search does not enter this narrow fallback.
Accordingly, no completeness claim is inferred from a sampled subset: this is
a complete source-level enumeration of the trigger's call site.

`git blame` attributes the eligibility check, explicit “no second active-window
walk” contract, O(1) urgency predicate, and guarded kernel call to commit
`c4b496ed5`, the same G2R8 negative implementation. The historical memo states
that its large wall regressions were trajectory effects, not trigger-scan or
shadow overhead (`K_REPLY_CHEAP_ARCH_RAW.log:20,26-31`).

## Fresh measurements

All Cargo invocations used `.target-hunt`, release,
`x86_64-pc-windows-msvc`, and serial tests. Every successful invocation had at
least 10 GiB available, at least 5 GiB free, and zero foreign Cargo processes.

Post-run corpus binding: the manifest now pins the external corpus printed by the
authoritative full leaf run (`K_REPLY_CHEAP_LEAF_RAW.log:7`) to 3,696,030 bytes
/ 6,902 JSONL rows and SHA-256
`54FAE7AEBCEF2A9D19D13C1946FAE36C0565E21BC726C25E2E4E230CFB42A5B7`. The
raw emitted the same path and `eligible_games=6902`, but no corpus byte hash;
this makes future reruns byte-exact and is supporting post-run evidence, not a
cryptographic measurement-time precommit.

| Gate | Result | Authoritative evidence |
|---|---:|---|
| release no-run | passed | `K_REPLY_CHEAP_BUILD_RAW.log:2-5` |
| frozen witness + trigger matrix | 2 passed | `K_REPLY_CHEAP_FOCUSED_RAW.log:2,17-19` |
| double-fork identity, cap 10,000/h=45 | WIN/WIN; certificate equal | `K_REPLY_CHEAP_DOUBLEFORK_RAW.log:7-15` |
| full leaf surface | 806 strict certificates; 0 contradictions | `K_REPLY_CHEAP_LEAF_RAW.log:183-188` |

Double-fork changed 409 to 395 nodes (-3.422983%) and 34.450 to
33.026 ms (-4.133551%), with one urgent consumption and structurally equal certificate values
(`K_REPLY_CHEAP_DOUBLEFORK_RAW.log:7-8`).

The fresh leaf probes were:

| Pass | Verdicts off/on | Total off/on | Delta | Urgent/consumed | Full/retained median |
|---|---:|---:|---:|---:|---:|
| first repeat | 14/14 | 6997.428/6943.712 ms | -0.767654% | 398/398 | 459/2 |
| authoritative full raw | 14/14 | 7268.021/6331.771 ms | -12.881768% | 398/398 | 459/2 |

Sources: `K_REPLY_CHEAP_LEAF_ATTEMPT1_SUMMARY_RAW.log:3-6` and
`K_REPLY_CHEAP_LEAF_RAW.log:182-188`. Only the second file contains complete
verbatim campaign output; the first is explicitly retained as a
non-authoritative summary because its initial console capture was not persisted.

For the selected wide literal comparison at cap 2,000/h=8, E and F both
reported 16/0/284 wins/losses/unknowns, 2,152 nodes, 1,852 expansions, zero TT
hits, 650 stage refreshes, 32,158 peak TT bytes, and identical gate/fragment
counters (`K_REPLY_CHEAP_LEAF_RAW.log:94-101`). Timing noise is not treated as
a trajectory effect when deterministic work is identical.

One initial focused wrapper aborted when PowerShell promoted Cargo's normal
build-progress stderr line into a terminating error; test completion is
unknown. It produced no authoritative test result, is not counted, and is preserved verbatim in
`K_REPLY_CHEAP_FOCUSED_ATTEMPT1_RAW.log`. The separately gated rerun passed.

## Historical comparison and adjudication

Across the 12 of 19 historical forcing rows that completed, the same
implementation held aggregate expansions exactly at 280,002 -> 280,002 while
wall rose 217.526 -> 611.860 s (+181.28%). Seven rows were absent, so this is
not a complete forcing-cohort aggregate. The largest recorded paths were:

- `0hz3hty`: 70.245 -> 291.420 s (+314.86%), only three urgent consumptions.
- `strongloss_a_prefix6`: 0.987 -> 173.567 s (+17,485.31% from the displayed
  endpoints; the historical memo carries +17,487.49%, which is not
  reproducible from those rounded endpoints and likely reflects hidden precision),
  400 consumptions.
- shadow-disabled `0hz3hty`: 70.942 -> 291.084 s (+310.31%).

Those figures are copied from `K_REPLY_CHEAP_ARCH_RAW.log:29-31`, whose source
hash binds the historical memo. A repeat of unchanged eight-minute outliers
would not distinguish the queued candidate, so it was excluded in the frozen
measurement plan.

PASS fails its first condition: the provenance audit was confirmed for the
already-computed analysis trigger. KILL does
not apply because no distinct candidate was introduced and every fresh
soundness/routing gate passed. The binding result is therefore **NULL —
ALREADY-COMPUTED ANALYSIS TRIGGER ALREADY LANDED**.

## Hostile self-review

Six attacks and outcomes are recorded verbatim in
`K_REPLY_CHEAP_COLD_REVIEW_RAW.log`:

- Later-trigger search: failed; exact lines blame to the negative commit.
- Common-branch kernel scan: failed; lazy guarding prevents it.
- Recast direct enumeration as the trigger: rejected as candidate drift.
- Promote the -12.882% sample to PASS: failed against provenance, the first
  -0.768% repeat, the wide no-op, and prior deep negatives.
- Find a soundness failure: failed; focused, certificate, verifier, and leaf
  gates all passed.
- Attribute deep regressions to trigger overhead: failed against the
  shadow-disabled repeat.

## Residual

A separate direct urgent-set constructor could avoid generating and filtering
the full legal vector on urgent states. Its exact candidate set is the union of
claimant count-5 window empties and the intersection of defender count-4/5
window empties, followed by unchanged pair filtering and sort order. Before any
timing it must prove raw-set equality, final-order equality, legality, and B/C
status/node/TT/certificate/verifier identity.

That residual is the separate generation-cost re-arm explicitly allowed by the
handoff doctrine. It only removes generation cost and preserves the same pruned
trajectory responsible for the historical downside, so it does not reopen this
already-computed-trigger queue item without its own preregistered experiment.

## Cold-gater instructions

1. Verify the SHA-256 manifest, then confirm `tss_verify.rs` matches its bound
   hash and has no diff.
2. Read `K_REPLY_CHEAP_PREREG_FROZEN_RAW.log` before interpreting
   measurements; use `K_REPLY_CHEAP_PREREG_RAW.log` for later disclosures.
3. Check `K_REPLY_CHEAP_ARCH_RAW.log:20,26-44`; provenance is decisive.
4. Use `K_REPLY_CHEAP_LEAF_RAW.log` as the authoritative full leaf raw and
   `K_REPLY_CHEAP_DOUBLEFORK_RAW.log` for paired certificate identity.
5. Treat `K_REPLY_CHEAP_LEAF_ATTEMPT1_SUMMARY_RAW.log` only as the disclosed
   timing repeat, and the focused attempt-1 raw only as a disclosed harness
   failure.
6. Use `K_REPLY_CHEAP_PREREG_TRANSCRIPT_RAW.log` and
   `K_REPLY_CHEAP_CHRONOLOGY_RAW.log` for prereg timing, the immutable-object
   audit in `K_REPLY_CHEAP_HANDOFF_PROVENANCE_RAW.log` for handoff provenance, and
   `K_REPLY_CHEAP_COMMANDS_RAW.log` for exact serial commands.
