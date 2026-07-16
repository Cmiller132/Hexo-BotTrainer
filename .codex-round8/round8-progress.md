# Round 8 progress

Build base: `435c67227309f94e0d89adace8acce976ea50042`

## Step 0: binding handoff and workspace audit

- Read `docs/TSS_VCF_WIDTH_BRIEF.md` in full (352 lines), including binding
  ADDENDUM 2 and ADDENDUM 3, plus the explicit Round 8 supersessions.
- Read `.codex-round6/round6-progress.md` (92 lines) and
  `.codex-round7/round7-progress.md` (63 lines) in full.
- Confirmed branch `claude/tss-vcf-width` and HEAD `435c6722`.
- Preserved pre-existing untracked `.codex-round5/`, `.codex-round8/prompt.txt`,
  and `.target-codex/` content.
- Adopted the required order: implement Fix B only, then run prefix-14,
  `lz60mfb`, and the 12-entry matrix before any full `0l4291i_live` rung.
- Operational constraints adopted: no commits; reuse `.target-codex/` on E:;
  `CARGO_BUILD_JOBS=4`; one synchronous solve at a time; log every solve under
  `.codex-round8/`; stop before any predicted >2-hour command.

## Step 1: Fix B implemented

- Added a scheduling-only committed-obligation index to each wide PN entry.
- Every wide Universal now selects its initial unresolved obligation with the
  existing lowest-DN / generator-order rule, then keeps visiting that child
  until a verdict instead of re-interleaving siblings as DN values change.
- PN/DN recomputation, position keys, verifier nodes, and certificate
  materialization are unchanged; narrow/default dispatch remains isolated in
  the legacy path and `WidthOptions::default()` is untouched.
- A true no-progress `Stalled` result releases the commitment and tries each
  remaining unresolved sibling once, reusing the existing linked-child and
  parent-change progress normalization. Depth cutoffs retain commitment across
  staged reopening.
- Added focused tests for commitment persistence, verdict release, cutoff
  persistence, and finite stall failover.
- Ran `rustfmt --edition 2021` on the edited file only.
- Focused release test log: `.codex-round8/fix-b-unit.log`; result: 15 passed,
  0 failed (`wide_pn_` filter), including both new tests and the saturated-TT
  failover regressions.
- `git diff --check` is clean.

## Step 2a: Fix B micro-gate 1 — prefix-14 Universal

- Default wide engine plus Fix B; all `TSS_*` variables cleared before setting
  only the harness inputs.
- Inputs: `TSS_CORPUS_ID=0l4291i_live`, `TSS_BACKWALK_PREFIX=14`, no extras,
  `TSS_BACKWALK_CAP=1000000`.
- Log: `.codex-round8/prefix14-universal-1m.log`.
- Result: **LOSS in 389,569 certified nodes**, 164,841 TT hits, 1,477.95 s.
- Acceptance target met: the complete four-obligation Universal is below the
  requested ~400k ceiling; the previous interleaved engine exceeded 1M.

## Step 2b: Fix B micro-gate 2 — `lz60mfb` ladder

- Default wide engine plus Fix B; only `TSS_CORPUS_ID=lz60mfb` and
  `TSS_CORPUS_MAX_CAP=1000000` were set after clearing all harness variables.
- Log: `.codex-round8/lz60mfb-ladder.log`.
- 10k: UNKNOWN, 10,000 nodes, 2,833 TT hits.
- 100k: UNKNOWN, 100,000 nodes, 8,405 TT hits.
- 1M: **WIN, 194,522 nodes**, 16,910 TT hits, 296.3 s.
- Harness passed (`CORPUS_DONE failures=0`), but the policy misses the explicit
  no-regression acceptance bound: 194,522 exceeds the banked 125,020 nodes.
  The 12-entry micro-gate remains next in the binding order; do not attempt
  full `0l4291i_live` on this regressed policy before revising it.

## Step 2c: Fix B micro-gate 3 — 12-entry matrix

- Default wide engine plus Fix B; the 12 banked WIN IDs were run with
  `TSS_CORPUS_MAX_CAP=100000` after clearing all other `TSS_*` variables.
- Log: `.codex-round8/matrix-12-100k.log`.
- Result: **12/12 WIN**, `CORPUS_DONE failures=0`.

| Entry | Winning rung | Nodes | TT hits |
|---|---:|---:|---:|
| 0hz3hty | 10k | 2,319 | 2,268 |
| acly7kb | 10k | 75 | 4 |
| g2xx6wl | 10k | 4,244 | 1,995 |
| hu01jk4 | 10k | 350 | 0 |
| jh7yo7y | 10k | 1,369 | 132 |
| jnzzmcm | 100k | 13,644 | 5,076 |
| xsnfyll | 10k | 78 | 1 |
| zrugh2x | 100k | 26,400 | 7,078 |
| strongloss_a_prefix6 | 10k | 5,457 | 1,269 |
| strongloss_b_prefix8 | 10k | 629 | 83 |
| hayes_20260712_turn16 | 10k | 9,924 | 2,525 |
| hayes_20260712_placement31 | 10k | 9,924 | 2,525 |

- The formal no-regression gate passes, but `jnzzmcm` moves from its banked
  8,032-node 10k win to 13,644 nodes. Together with the `lz60mfb` miss, this
  rules out shipping commitment indiscriminately at every Universal.

## Step 3a: Fix B scope correction

- Replaced indiscriminate AND commitment with structural high-fanout
  commitment: a Universal qualifies only with at least four distinct linked
  proof-obligation TT entries. Duplicate/transposed edges count once.
- A qualifying Universal remains sequential through its remaining obligations;
  Universals with one to three distinct obligations retain ordinary minimum-DN
  PN re-selection and TT co-staging. This directly preserves the known binary
  `lz60mfb` outer conjunction while targeting the proven four-way prefix-14 /
  full-`0l` interleaving shape, without a corpus ID, coordinate, depth, width,
  or root-order special case.
- Existing commitment remains cutoff-stable and uses finite true-stall
  failover. The PN/DN recurrence and certificate path remain unchanged.
- Added regressions for four-way commitment, binary/TT-converged ordinary
  selection, and four-way finite stall failover.
- Focused release log: `.codex-round8/fix-b-fanout-unit.log`; result: 16 passed,
  0 failed (`wide_pn_` filter).

## Step 3b: high-fanout-only policy rejected

- Restarted the acceptance sequence from prefix-14 after the scope change.
- Inputs: default wide engine, `TSS_CORPUS_ID=0l4291i_live`,
  `TSS_BACKWALK_PREFIX=14`, no extras, `TSS_BACKWALK_CAP=1000000`.
- Log: `.codex-round8/prefix14-fanout-1m.log`.
- Result: LOSS in **561,243 nodes**, 296,977 TT hits, 3,004.68 s.
- The proof is sound and below 1M, but misses the ~400k acceptance target and
  is materially worse than the 389,569-node all-Universal commitment result.
- Conclusion: committing only the four-way parent leaves each selected
  obligation free to re-interleave at descendant Universals. The qualifying
  high-fanout node must establish a sequential commitment domain down its
  selected obligation until verdict; binary Universals outside such a domain
  should retain ordinary PN/TT co-staging (notably `lz60mfb`'s outer split).

## Step 3c: sequential commitment domains implemented

- A four-distinct-obligation Universal now starts a commitment domain. While
  descending its selected child, every descendant Universal keeps one
  obligation committed until verdict, including through intervening Choice
  nodes and staged cutoff reopening.
- Outside a high-fanout domain, one-to-three-obligation Universals still use
  ordinary minimum-DN PN re-selection. Thus the prefix-14 four-way parent
  drives each obligation without descendant re-interleaving, while the known
  binary outer `lz60mfb` conjunction can still co-stage its TT-sharing replies.
- True-stall failover remains finite per distinct TT obligation and propagates
  within the same commitment domain.
- Added an inherited-domain regression for a binary/TT-converged descendant.
- Focused release log: `.codex-round8/fix-b-domain-unit.log`; result: 16 passed,
  0 failed. `git diff --check` is clean.

## Step 3d: commitment-domain micro-gate 1 — prefix-14

- Default wide engine; exact inputs unchanged (`0l4291i_live`, prefix 14, no
  extras, 1M cap).
- Log: `.codex-round8/prefix14-domain-1m.log`.
- Result: **LOSS in 389,569 nodes**, 164,841 TT hits, 1,551.41 s.
- This exactly reproduces the accepted all-Universal visit count and meets the
  ~400k target, confirming that the four-way root's selected obligation is
  sequential throughout its descendant search.

## Step 3e: commitment-domain micro-gate 2 — `lz60mfb`

- Default wide engine; only `TSS_CORPUS_ID=lz60mfb` and
  `TSS_CORPUS_MAX_CAP=1000000` set after clearing all harness variables.
- Log: `.codex-round8/lz60mfb-domain-ladder.log`.
- 10k: UNKNOWN, 10,000 nodes, 2,847 TT hits.
- 100k: UNKNOWN, 100,000 nodes, 10,894 TT hits.
- 1M: **WIN in 122,132 nodes**, 12,165 TT hits, 192.1 s.
- Acceptance met: this is 2,888 nodes below the 125,020 bank and restores the
  binary conjunction's TT-sharing profile (the rejected all-Universal policy
  required 194,522 nodes).

## Step 3f: commitment-domain micro-gate 3 — 12-entry matrix

- Default wide engine; all 12 banked WIN IDs at
  `TSS_CORPUS_MAX_CAP=100000`.
- Log: `.codex-round8/matrix-12-domain-100k.log`.
- Result: **12/12 WIN**, `CORPUS_DONE failures=0`.

| Entry | Winning rung | Nodes | TT hits |
|---|---:|---:|---:|
| 0hz3hty | 10k | 2,319 | 2,268 |
| acly7kb | 10k | 75 | 0 |
| g2xx6wl | 10k | 4,244 | 1,995 |
| hu01jk4 | 10k | 380 | 0 |
| jh7yo7y | 10k | 2,018 | 337 |
| jnzzmcm | 100k | 13,646 | 5,068 |
| xsnfyll | 10k | 81 | 1 |
| zrugh2x | 100k | 39,739 | 11,841 |
| strongloss_a_prefix6 | 100k | 16,245 | 7,699 |
| strongloss_b_prefix8 | 10k | 682 | 151 |
| hayes_20260712_turn16 | 100k | 13,524 | 3,201 |
| hayes_20260712_placement31 | 100k | 13,524 | 3,201 |

- Fix B acceptance is complete: prefix-14 <=~400k, `lz60mfb` <=125,020,
  and the 12-entry matrix remains 12/12. Proceed immediately to the full
  `0l4291i_live` 1M rung.

## Step 4 checkpoint: full `0l4291i_live` 1M rung

- Default commitment-domain engine; exact full-position helper inputs:
  `TSS_CORPUS_ID=0l4291i_live`, `TSS_BACKWALK_PREFIX=0`,
  `TSS_BACKWALK_CAP=1000000`.
- Log: `.codex-round8/0l-full-1m.log`.
- Result: **UNKNOWN at 1,000,000 certified nodes**, 441,328 TT hits,
  2,296.37 s. The helper's test failure is the expected assertion after the
  exact UNKNOWN measurement.
- Observed throughput: ~435.47 certified nodes/s. A full 4M cap projects to
  ~9,186 s (2 h 33 min), exceeding the binding ~2-hour synchronous-shell
  limit. Per the long-run mechanics, STOP here and ask the orchestrator to run
  the required 4M rung externally rather than gambling this session.
- Required orchestrator output log: `.codex-round8/0l-full-4m.log`.
- Do not begin Fix A or consolidation until the 4M verdict is returned: the
  binding order requires this rung after a 1M miss with substantially improved
  micro-benchmarks.

## Step 5a: returned full `0l4291i_live` 4M verdict and saturation audit

- The orchestrator returned `.codex-round8/0l-full-4m.log` for the unchanged
  commitment-domain engine at HEAD `3bc5dd44`.
- Result: **UNKNOWN at 4,000,000 certified nodes**, 441,607 TT hits, 7,000.57 s.
- Relative to the 1M run's 441,328 TT hits, the additional 3,000,000 certified
  nodes produced only **279** additional hits: a 0.0093% marginal hit rate
  versus 44.13% cumulatively through 1M, a roughly 4,745x collapse.
- Code audit confirms that this is the exact expected shape after saturation:
  the wide-PN arena continues retaining entries, while `by_position` stops
  indexing new keys once `current_bytes + key_charge` exceeds the byte cap;
  unindexed transpositions then create separate arena nodes. There is no
  eviction or capacity recovery, and bottom-up stage refresh only recomputes
  retained entries.
- The two supplied full-position logs did not set `TSS_TRACE_PN` and therefore
  contain no retained/indexed-entry, byte-cap, rejection, or stage-refresh
  telemetry. The hit flatline alone is extremely strong evidence but is not a
  conclusive distinction from a transposition-poor region. Per the round-8b
  read-not-guess requirement, a trace-backed 1M reproduction is next.

## Step 5b: trace-only saturation telemetry added

- Added test-only `TSS_TRACE_PN` telemetry for indexed entries, retained arena
  entries, exact TT bytes/cap, rejected index insertions, the first rejection's
  `(certified expansions, retained entries)` checkpoint, and bottom-up refresh
  count. Selection, PN/DN recurrence, certificate materialization, and
  non-test builds are unchanged.
- Ran `rustfmt --edition 2021` on `tss_solver.rs` only.
- Focused release log: `.codex-round8/tt-telemetry-unit.log`; result: **16
  passed, 0 failed** (`wide_pn_` filter), including disabled/full/saturated-TT
  frontier regressions. `git diff --check` is clean.
- Next: rerun full 0l at the 1M cap with only `TSS_TRACE_PN=1` added, so the
  first-rejection and final retained/indexed counts conclusively resolve the
  saturation hypothesis before authorizing the TT-size override.

## Step 5c: 512 MiB saturation conclusively verified

- Full-position trace log: `.codex-round8/0l-full-1m-tt-trace.log`.
- Exact inputs matched the prior 1M run (`0l4291i_live`, prefix 0, no extras,
  cap 1,000,000, 512 MiB TT) with only observational `TSS_TRACE_PN=1` added.
- Result reproduced exactly: **UNKNOWN at 1,000,000 certified nodes**, 441,328
  TT hits. Wall time was 2,424.47 s.
- The exact-key index first rejected a new position at certified expansion
  **603,584**, with 1,831,466 retained arena entries at that point.
- Final telemetry: 1,831,465 indexed entries versus 3,550,627 retained arena
  entries; **1,719,162 rejected index insertions**; 77 bottom-up refresh passes.
- Exact TT usage was **536,870,797 / 536,870,912 bytes** (only 115 bytes free),
  proving the table was saturated. Since rejected keys are retained only as
  separate arena nodes and refresh cannot restore index capacity, the 1M->4M
  hit flatline is causally explained rather than inferred.
- Saturation branch is therefore authorized: add the test-only, default-512MiB
  TT-size override to both ignored corpus helpers, then run full 0l with 2 GiB.

## Step 6: test-only TT resource override implemented and validated

- Added one module-wide ignored-harness resource knob,
  `TSS_BACKWALK_TT_BYTES`, used by both `tss_corpus_backward_walk` and
  `tss_corpus_check`. It defaults to exactly 536,870,912 bytes, so the prior
  harness profile is unchanged when the variable is absent.
- The override exists only in the `#[cfg(test)]` corpus module. Production
  callers, `TssSolver`, `WidthOptions`, and production TT constants/call sites
  are untouched.
- Extended corpus/backwalk output with the requested TT cap and observed peak
  bytes so every banked log self-identifies its resource profile.
- Ran `rustfmt --edition 2021` on `tss_corpus.rs` only and `git diff --check`.
- Focused release log `.codex-round8/tt-override-unit.log`: **16 passed, 0
  failed** (`wide_pn_`), including all full/saturated-TT regressions.
- Default smoke `.codex-round8/tt-override-smoke-default.log`: `xsnfyll` WIN
  in 81 nodes, printed `tt_bytes_cap=536870912`.
- 2 GiB smoke `.codex-round8/tt-override-smoke-2g.log`: the same `xsnfyll`
  WIN in the identical 81 nodes, printed `tt_bytes_cap=2147483648`.
- Next: full `0l4291i_live` exact 4M rung with 2 GiB, no engine A/B variables,
  and observational TT tracing. The prior 4M wall time (7,000.57 s) projects
  just under the binding two-hour shell window.

## Step 7 checkpoint: **full `0l4291i_live` WIN banked**

- Log: `.codex-round8/0l-full-4m-2g.log`.
- Exact inputs: full `0l4291i_live` (`TSS_BACKWALK_PREFIX=0`, no extras),
  certified node cap 4,000,000, `TSS_BACKWALK_TT_BYTES=2147483648`, and only
  observational `TSS_TRACE_PN=1`; all engine A/B variables were absent.
- Result: **WIN in 2,335,295 certified nodes**, 1,492,036 TT hits, 6,969.58 s.
  The helper passed and materialized a certificate.
- Final trace: root `pn=0`, staged depth 78, 6,880,208 indexed entries and the
  identical 6,880,208 retained entries, 2,063,694,498 peak TT bytes, **zero
  rejected inserts**, and no first-rejection event.
- This directly confirms the diagnosis: the 512 MiB run lost transposition
  retention at expansion 603,584 and could not finish, while the unchanged
  certified search proves the position once its working set remains indexed.
- **Fix A is not needed and remains unimplemented.** Per the winning branch,
  proceed immediately to consolidation: keep unconditional wide-mode Fix B
  commitment domains and delete all losing cfg(test)/env experimental search
  scaffolds, including frontier machinery.

## Step 8: winning wide path consolidated

- Deleted every losing test/env search experiment required by the round-8
  one-system directive: legacy-wide/depth-cap dispatch, graph and DAG choice-PN,
  macro DFS, the entire frontier probe/`ProvenFragment` import shell, sequential
  small-choice and force-sequential overrides, the root-tier opt-out,
  placement-PN, and the orphaned pair-rank debug helper.
- Deleted the eight tests and helpers specific to the rejected graph-PN profile.
- Preserved the production narrow `SearchContext`, production `CachedProof` /
  shared positive cache, unconditional root-only width-tier policy, staged
  horizon advance, and all unconditional Fix B commitment-domain machinery and
  regressions.
- The only remaining solver environment hook is observational `TSS_TRACE_PN`.
  The saturation counters and `WIDTH_PN_TT` line remain for reproducible
  resource telemetry; `TSS_BACKWALK_TT_BYTES` remains test-harness-only.
- Ran `rustfmt --edition 2021` on `tss_solver.rs` only; `git diff --check` is
  clean.
- Focused consolidated release log:
  `.codex-round8/consolidation-wide-unit.log`; result: **16 passed, 0 failed**
  (`wide_pn_`), including all Fix B and saturated-TT regressions.
- This is an engine source change, so the required full unit suite and every
  previously closed corpus gate are next before the consolidated 0l re-solve.

## Step 9: consolidated full unit suite

- Cleared every `TSS_*` environment variable and reused the E:-local
  `.target-codex/` target with `CARGO_BUILD_JOBS=4`.
- Log: `.codex-round8/final-unit.log`.
- Result: **95 passed, 0 failed, 3 intentionally ignored**; doc tests green.
- The eight removed graph-profile-only tests account for the count change from
  the pre-consolidation suite. The final rerun is warning-free.
- Next: re-verify the twelve closed corpus WIN entries at <=100k using the
  final 2 GiB resource profile.

## Step 10: consolidated 12-entry matrix

- Final resource profile: `TSS_BACKWALK_TT_BYTES=2147483648`, all engine A/B
  variables absent, and `TSS_CORPUS_MAX_CAP=100000`.
- Log: `.codex-round8/final-matrix-12-2g.log`.
- Result: **12/12 WIN**, `CORPUS_DONE failures=0`.

| Entry | Winning rung | Nodes | TT hits |
|---|---:|---:|---:|
| 0hz3hty | 10k | 2,319 | 2,268 |
| acly7kb | 10k | 75 | 0 |
| g2xx6wl | 10k | 4,244 | 1,995 |
| hu01jk4 | 10k | 380 | 0 |
| jh7yo7y | 10k | 2,018 | 337 |
| jnzzmcm | 100k | 13,646 | 5,068 |
| xsnfyll | 10k | 81 | 1 |
| zrugh2x | 100k | 39,739 | 11,841 |
| strongloss_a_prefix6 | 100k | 16,245 | 7,699 |
| strongloss_b_prefix8 | 10k | 682 | 151 |
| hayes_20260712_turn16 | 100k | 13,524 | 3,201 |
| hayes_20260712_placement31 | 100k | 13,524 | 3,201 |

- Winning-rung nodes and TT hits exactly reproduce the accepted
  commitment-domain bank. Next: consolidated `lz60mfb` ladder.

## Step 11: consolidated `lz60mfb` ladder

- Final 2 GiB resource profile; engine A/B variables absent.
- Log: `.codex-round8/final-lz-2g.log`.
- 10k: UNKNOWN, 10,000 nodes, 2,847 TT hits.
- 100k: UNKNOWN, 100,000 nodes, 10,894 TT hits.
- 1M: **WIN in 122,132 nodes**, 12,165 TT hits, peak TT 98,217,784
  bytes; `CORPUS_DONE failures=0`.
- This exactly reproduces the commitment-domain new-best bank. Next: the
  binding prefix-14 Universal regression, then consolidated full 0l.

## Step 12: consolidated prefix-14 Universal gate

- Exact prefix-14 state, no extras, 1M certified cap, final 2 GiB resource
  profile, and all engine A/B variables absent.
- Log: `.codex-round8/final-prefix14-2g.log`.
- Result: **LOSS in 389,569 nodes**, 164,841 TT hits, peak TT 314,088,521
  bytes, 1,456.84 s.
- This exactly reproduces the accepted Fix B node/hit counts and stays below
  the ~400k ceiling. The improved wall rate leaves enough margin for the
  consolidated full-0l 4M/2GiB re-verification inside the two-hour guard.

## Step 13 checkpoint: consolidated full `0l4291i_live` re-verified

- Log: `.codex-round8/final-0l-4m-2g.log`.
- Exact final inputs: full prefix 0, no extras, certified cap 4,000,000,
  `TSS_BACKWALK_TT_BYTES=2147483648`, observational `TSS_TRACE_PN=1`, and no
  engine behavior variables.
- Result: **WIN in 2,335,295 certified nodes**, 1,492,036 TT hits, 7,290.46 s.
- Trace reproduced exactly: root `pn=0`, staged depth 78, 6,880,208 indexed
  and retained entries, 2,063,694,498 TT bytes, zero rejected inserts, no
  first-rejection event, and 79 bottom-up refresh passes.
- The consolidated engine is node-for-node and hit-for-hit identical to the
  pre-consolidation winning path. Next: all five expect=NO entries under the
  final resource profile; zero may return WIN.

## Step 14: consolidated five-entry `expect=NO` gate

- Log: `.codex-round8/final-no-5-2g.log`.
- Exact final resource profile: `TSS_BACKWALK_TT_BYTES=2147483648`, corpus
  cap 1,000,000, and no engine behavior variables.
- `8is963b`: LOSS at 10k in 1 node, 0 TT hits.
- `94gnnol`: UNKNOWN at 10k in 10,000 nodes / 2,003 hits; UNKNOWN at 100k
  in 100,000 / 38,211; UNKNOWN at 1M in 1,000,000 / 612,111, peak TT
  785,936,412 bytes.
- `dy3dg99`: LOSS at 10k in 1 node, 0 TT hits.
- `l9mxn59`: UNKNOWN at all three rungs in 235 nodes / 18 hits, peak TT
  61,142 bytes.
- `mvp2lvc`: UNKNOWN at 10k in 10,000 / 771; UNKNOWN at 100k and 1M in
  19,207 / 1,847, peak TT 10,743,026 bytes.
- Mechanical log audit found all five requested IDs, zero
  `status=WIN expect=NO` lines, exactly one `CORPUS_DONE failures=0`, and an
  `ok` Rust test result. The outer PowerShell logging cell returned nonzero
  only because Cargo's normal stderr status lines were converted to a native
  error record; the saved test result itself passed.
- Next: narrow default-mode byte-identity gate.

## Step 15: narrow default-mode byte identity

- Raw log: `.codex-round8/final-narrow-default.log`.
- Normalized signature: `.codex-round8/final-narrow-default.sig`.
- All `TSS_*` variables were absent; the production/default narrow path used
  its unchanged 65,536-byte TT cap.
- Rust benchmark test passed. Stable cap-2000 summary: 82 positions, 9,296
  nodes, 53 TT hits, peak TT 21,114 bytes, 13 WIN, 4 LOSS, 65 UNKNOWN.
- Applied the binding timing-only normalizer to CONFIG/BUCKET/SOLVE/SUMMARY
  records: exactly 101 rows.
- Byte identity against `.codex-round5/narrow-b0.sig` is exact: zero line
  differences and SHA-256
  `0098C8BFC6382156979FFE2C022E780EF34D53ABE37477E77A939C052470C4F2`.
- Next: final diff audit and the official all-19 corpus gate. The official
  fixed ladder must be handed to the orchestrator because 0l alone repeats
  a ~40-minute 1M miss followed by the ~121-minute winning rung.

## Step 16 checkpoint: final audit and all-19 orchestrator handoff

- Draft checkpoint/final report: `.codex-round8/round8-final.md`.
- `git diff --check` is clean. Only the intended solver, corpus test helper,
  and round-8 report/note files are changed; no commit was made.
- Mechanical source audit: zero rejected experiment-variable matches remain.
  The only solver environment knob is cfg(test), observational
  `TSS_TRACE_PN`; `TSS_BACKWALK_TT_BYTES` exists only in the cfg(test) corpus
  module and defaults to exactly 512 MiB.
- Independent final diff review found no blocker: Fix B commitment-domain
  scheduling is unconditional in wide mode, saturation counters are
  observational, production/narrow behavior is unchanged, and all losing
  macro-DFS/graph/frontier/sequential/root-opt-out/placement shells are gone.
- The test-only 2 GiB profile is documented as a `SolveCaps` resource choice,
  not a semantic switch: it preserves the fixed corpus node ladder and is
  necessary to reproduce the banked 0l certificate without rejected TT
  inserts.
- All 19 rows have consolidated 2 GiB split-gate banks, recorded in corpus
  order in `round8-final.md`. The required one-process full replay remains
  outstanding; all 19 rows must be replaced/confirmed from that single log.
- The official full gate is predicted well beyond two hours: 0l's measured
  1M miss plus winning rung consume about 161 minutes, and `94gnnol`'s NO
  ladder adds about 59 minutes before the other entries. Per binding long-run
  mechanics, it is handed to the orchestrator rather than launched here.
- Exact clean-environment PowerShell is in `round8-final.md`. It leaves only
  `TSS_BACKWALK_TT_BYTES=2147483648`, uses `.target-codex/`, jobs=4, the
  built-in 10k -> 100k -> 1M -> 20M ladder, and logs to
  `.codex-round8/final-matrix-19-2g.log`.
- Completion condition after return: all 19 IDs, all 14 WINs certified, zero
  NO-as-WIN rows, `CORPUS_DONE failures=0`, and an `ok` Rust test result; then
  replace the pending section of `round8-final.md` with the official matrix.
