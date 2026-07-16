# Group 2, round 2 — exact reference acceleration and first frozen rows

Date: 2026-07-16 (America/New_York)

Base: `b4ec2e73` (round-9b engine plus round-1 honest partial)

## Checkpoints

### C0 — binding context and clean-scope audit

- Read `.codex-group2/round1-progress.md` and `.codex-group2/prompt-r1.txt` in
  full before taking implementation action. The round-1 structural lower bound
  is accepted as binding: stock `tss_reference` cannot credibly close the
  403-cell double-fork frontier without an exact accelerator or construction
  shrink.
- Confirmed `HEAD=b4ec2e7344cf1ed3cb102ec684c4490d34a4fcce`.
- Initial free physical RAM was 13.76 GiB, above the required 8 GiB floor.
- Initial untracked predecessor artifacts (`.codex-group2/*.pid`, the native
  reference executable, `.target-codex*`, and `.codex-round5/`) are treated as
  user-owned and are not oracle verdicts.
- No `AGENTS.md` exists in this worktree. No commit will be attempted.

Checkpoint boundary: implement only a test-gated, independent exact oracle;
normative engine code remains out of scope.

## Differential validation

### C1 — independent accelerator compiled and gated

- Added test-only `tss_reference_fast`, with an independently reconstructed
  full legal set, direct owner-based win scan, exact `(sorted occupancy,
  player, phase including FirstStone coordinate, plies remaining)` keys,
  optional independently implemented 12-image D6 canonicalization, bounded
  exact TT, value short-circuits, and ordering only. It imports no symbol from
  `tss_solver` and uses none of its threat, zone, or wide-universe machinery.
- The TT hard-clamps configuration to 2 GiB. Its final bounded replacement
  policy clears exact entries at the conservative accounted-byte ceiling and
  admits recent transpositions again; replacement changes work only.
- Targeted module tests passed 4/0, including D6 key equivalence, incremental
  frontier/full-rebuild equality through make/unmake, and stock
  recurrence comparison at depths 0, 1, and 2.
- The first differential-gate attempt stopped before any oracle comparison
  because the randomized P1 fixture excluded the sole Opening move and tried
  modulo zero. This was a harness-generation defect, not a verdict mismatch;
  the Opening coordinate was made explicit and the entire gate restarted.
- Differential gate result: **209/209 status agreements**, with 42 WIN, 167
  UNKNOWN, 0 LOSS, 109 Player0 roots, and 100 Player1 roots. Coverage includes
  `compact_urgent_spare` depth 2, `strongloss_a_prefix6` backed off to seven
  stones at depth 2, rejected `spare_tempo_prefix` depth 2, all three again at
  depths 0 and 1, 120 fixed-seed ordinary small positions, and 80 fixed-seed
  randomized four-line tactical positions split across both movers and depths
  1/2. No accelerator/stock disagreement occurred.

Checkpoint boundary: `tss_reference_fast` is now eligible to provide ground
truth. Run one target solve at a time; no commit attempted.

## Oracle and witness results

### C2 — Path A target feasibility result

Every target attempt was serialized, preceded by a free-RAM check, and used
the exact depth 9 (absolute horizon 45) for `double_fork_compact`. No attempt
returned a game status:

| Target/version | TT | Disposition |
|---|---:|---|
| compact, rebuilding legal union | 1.5 GiB | manually stopped at ~5 min; timeout |
| compact, incremental counted frontier | 1.5 GiB | manually stopped at ~10 min; timeout |
| compact, cached ordering + exact depth-one leaf | 1.5 GiB | manually stopped at ~5 min; timeout |
| compact, certificate-derived ordering only | 1.5 GiB | abnormal child exit 1 after 470 s; no verdict |
| compact, bounded replacement TT | 512 MiB | abnormal child exit 1 after 448 s; no verdict |
| dense, incremental oracle | 1.5 GiB | abnormal child exit 1 after 236 s; no verdict |
| compact branch 0/478 at remaining depth 7 | 256 MiB | manually stopped after ~9 min; timeout |

The certificate ordering hint is value-preserving: it tries `(4,0)`, then the
finder certificate's frequent `(4,7)` / `(0,3)` connectors and completion
coordinates first; all legal fallbacks remain. The finder certificate contains
479 Universal nodes / 1,910 explicit edges and chose `(4,7)` 478 times and
`(0,3)` 476 times. The hint did not break the exact universal wall.

The first-Universal range harness found the independently enumerated post-root
frontier is 478 cells (the predecessor's 479 figure included the pre-root
cell). Range 0 alone did not close in nine minutes because the later defender
turn, not the first split, dominates. No partial range was promoted.

### C3 — Path B disposition

Retained constructions were re-triaged at a 1M finder cap:

| Candidate | Default | Wide | Disposition |
|---|---|---|---|
| `double_fork_ordered` | UNKNOWN / 1,392 | UNKNOWN / 2 | no finder proof |
| `shared_target_spare` | WIN / 3 | WIN / 2 | lambda-1; reject |
| `shared_target_block4` | WIN / 3 | WIN / 2 | lambda-1; reject |
| `shared_target_block_endpoints` | UNKNOWN / 500,003 | WIN / 91 | lambda-1 wide win; reject |

The board implementation is sparse/unbounded: legal cells are the union of
radius-eight neighborhoods around every occupied cell. Ordinary prefilled
"walls" therefore expand the exterior legal union while occupying the
interior. A tens-of-cells frontier requires approaching the `i16` coordinate
boundary (not reachable from the mandatory `(0,0)` opening without thousands
of legal bridge placements) or a non-replay state injector, which the corpus
format and round scope do not authorize. No smaller replayable lambda-2 WIN
was found, and no WIN ground truth is claimed.

Checkpoint boundary: Path A remains exact and validated but did not close the
witness; Path B produced no valid replacement. This blocks a truthful
WIN_PENDING row. No commit attempted.

## Frozen corpus and wide baseline

### C4 — honest controls frozen; WIN minimum not met

`rust/corpus/spare_corpus_moves.txt` now freezes the two stock-reference
controls with full replay/provenance:

| ID | Expect | Stock oracle | Reference depth/nodes | Legal | Wide 10k / 100k / 1M |
|---|---|---|---:|---:|---|
| `compact_urgent_spare` | NO | UNKNOWN | 2 / 1,601 | 534 | UNKNOWN/2 at all rungs |
| `strongloss_a_backoff_7` | NO | UNKNOWN | 2 / 129,455 | 324 | UNKNOWN/2 at all rungs |

`tss_spare_corpus_check` is ignored-by-default and wired to that file. It
implements the unchanged semantics: WIN_PENDING fails only on LOSS; NO fails
on WIN; UNKNOWN is printed and accepted. All three 2 GiB TT rungs passed.
The corpus intentionally contains no WIN_PENDING entry because neither exact
oracle produced a WIN. Therefore the requested minimum of one oracle-WIN row
and its capstone baseline is **not met**.

## Round-3 diagnosis memo

The current wide universe dies at the first defender node after root choice
`(4,0)`. The replayed node has `b=2`, `k=1`, two threat windows, no immediate
claimant win, and 478 legal placements. In `WidePnSearch::expand`,
`implicit_dispatch` requires `min_hitting_set == b`; `k=1<b=2` makes it false,
and the node is refuted/exhausted before the engine can reach the later quiet
attacker turn. This is why the wide result is UNKNOWN in two nodes even though
the historical finder selects `(4,7)` and `(0,3)` on nearly every branch.

After those connector placements, the latent families are the horizontal
`r=7` family (direct cells around `(8,7)/(9,7)`), the `QR` family (around
`(8,3)/(9,2)`), and the vertical `q=0` family (notably `(0,1)/(0,2)` and
`(0,7)/(0,8)` across its live six-windows). T3's mandatory zone would start
with those direct cells (`Z_dir`), the protected nonlegal connector/endpoint
seeds and their legal bands (`Z_seed`), every empty of already touched
defender windows (`Z_touch`), and the D-alive virgin-window bands
(`Z_virgin`). By hand this is dozens of locally ranked cells, not the full
478-cell legal set; exact cardinality must be re-derived by the verifier.

Minimal round-3 design:

- default-off `quiet_turn_or_edges` adds complete two-placement attacker OR
  edges that may finish nonforcing, without changing the existing forcing
  pair path;
- default-off `ranked_unforced_defender_zone` supplies the certified T3 union
  only when `k<b`, never a defender-count cap;
- shadow mode computes both additions and records coverage/rank statistics but
  consumes neither;
- verify mode emits candidate evidence, while the independent verifier replays
  the node and re-derives the full T3 union and every represented edge;
- consume only after shadow agreement, preserving finder/verifier separation,
  full defender semantics, and `shadow -> verify -> consume` ownership.

## Regeneration commands

All commands run from the worktree root with PowerShell and serialized builds:

```powershell
rustfmt --edition 2021 packages/hexfield_eq/rust/src/tss_reference_fast.rs
$env:CARGO_TARGET_DIR='.target-codex'
cargo test --release -p hexfield_eq 'tss_reference_fast::tests' -- --ignored --test-threads=1 --nocapture

rustfmt --edition 2021 packages/hexfield_eq/rust/src/tss_spare_corpus.rs
$env:CARGO_TARGET_DIR='.target-codex'
cargo test --release -p hexfield_eq tss_reference_fast_differential -- --ignored --test-threads=1 --nocapture

# Target oracle (no verdict; dispositions above)
$env:TSS_SPARE_MINE_ID='double_fork_compact'
$env:TSS_SPARE_FAST_PLIES='9'
$env:TSS_REFERENCE_FAST_TT_BYTES='536870912'
cargo test --release -p hexfield_eq tss_spare_mine_candidate -- --ignored --test-threads=1 --nocapture

# Exact branch decomposition probe (range 0 timed out)
$env:TSS_SPARE_BRANCH_START='0'
$env:TSS_SPARE_BRANCH_END='1'
cargo test --release -p hexfield_eq tss_reference_fast_compact_branch_batch -- --ignored --test-threads=1 --nocapture

# Frozen-control baseline ladder
$env:TSS_BACKWALK_TT_BYTES='2147483648'
foreach($cap in @('10000','100000','1000000')) {
  $env:TSS_SPARE_CORPUS_CAP=$cap
  cargo test --release -p hexfield_eq tss_spare_corpus_check -- --ignored --test-threads=1 --nocapture
}
```

## Final verification and diff surface

### C5 — final gate and scope audit

- Explicit accelerator unit validation: 4 passed / 0 failed.
- Mandatory differential validation: 209 cases / 0 disagreements.
- Frozen spare-control ladder: all rows green at 10k, 100k, and 1M.
- Full default release suite: **95 passed / 0 failed / 11 ignored**, restoring
  the required 95/0 default count by keeping all new oracle/corpus helpers
  explicitly gated.
- `git diff --check` passed.
- An accidental `rustfmt lib.rs` traversal reformatted unrelated modules. The
  pre-existing `.codex-group2/rustfmt-churn.patch` was reversed mechanically;
  a subsequent diff audit confirms all unrelated formatting churn is gone.
- Normative engine code is unchanged. Tracked source changes are only the
  `#[cfg(test)]` module registration in `lib.rs` and the existing test-only
  `tss_spare_corpus.rs`; new files are the test-only accelerator, spare corpus,
  and this memo. `tss_solver.rs`, verifier semantics, and production/narrow
  bytes are untouched.
- No commit attempted, as required. Pre-existing untracked PID/native/target
  artifacts remain user-owned and were not deleted.

Final disposition: infrastructure and honest NO controls are complete and
green, but the round's central acceptance condition is blocked: there is no
exhaustive oracle WIN and therefore no truthful WIN_PENDING row. Round 3 must
not consume this corpus as a positive acceptance gate until that row exists.
