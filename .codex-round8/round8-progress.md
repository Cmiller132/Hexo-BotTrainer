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

