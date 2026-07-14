# TSS VCF-Width Upgrade — Implementation Brief

## Goal

Add an **opt-in, default-off** wide attacker-move universe to the TSS deep
solver (`packages/hexfield_eq/rust/src/tss_solver.rs`) so that it proves
full-width connect-6 VCF wins — specifically, all 14 WIN positions of the
external forcing corpus at `packages/hexfield_eq/rust/corpus/
forcing_corpus_moves.txt`, with **`0l4291i_live` as the single highest-
priority position**.

## Definition of done (hard gate)

1. `cargo test --release -p hexfield_eq tss_corpus_check -- --ignored
   --nocapture` **passes**: every `expect=WIN` entry reaches
   `ProofStatus::Win` somewhere on the node-cap ladder (10k → 100k → 1M →
   20M), and **no `expect=NO` entry ever returns Win** (Loss/Unknown both
   fine). The acceptance test already exists at `rust/src/tss_corpus.rs` and
   currently fails to compile — it calls the API you must build:
   `TssSolver::set_width_options(WidthOptions::vcf_pair_complete())`.
   - Partial-success fallback (only if full 14/14 proves genuinely
     unreachable after real effort): `0l4291i_live` MUST prove Win. But aim
     for 14/14 — the reference pdspn driver solved every one of these, most
     in <2000 of its nodes, worst case 264s.
2. The **entire existing test suite stays green**:
   `cargo test --release -p hexfield_eq` (the ~58 unit tests, not counting
   ignored harnesses). Narrow-mode (default) behavior must be
   **byte-identical** — `WidthOptions::default()` = all-off = today's
   generator, bit-for-bit.
3. No production semantics change: nothing in `search.rs` / `tree.rs` /
   `tss_async.rs` call sites flips the new option on. Rust-level option
   only; **no Python/TOML plumbing in this task**.

## Where the width is lost today (root cause, verified empirically)

`threat_creating_moves` (tss_solver.rs ~line 1031) only emits empties of
claimant-owned windows with `count >= 3`. In connect-6 the attacking unit is
the PAIR (two placements per turn): pair-builds through count-2 windows and
threat+build tempo moves are structurally invisible. Verified on the corpus:
`hayes_20260712_turn16` exhausts its universe after **5 nodes** at every cap;
14/14 corpus WINs return Unknown; several stall at fixed node counts far
below cap (314, 511, 1787, 1868).

## Specification of the wide universe

When `WidthOptions::vcf_pair_complete` is active, at claimant (OR) plies the
candidate set becomes:

- empties of claimant-owned windows with **count >= 2** (was >= 3), at BOTH
  plies of the turn (FirstStone and SecondStone). Rationale: count-2 + the
  turn's two stones = count-4 (immediate threat); and the classic tempo
  pattern (ply 1 = forcing count-3→4 extension, ply 2 = quiet count-2→3
  build for the NEXT turn) requires the count-2 tier on the second ply too.
- everything the narrow generator already emits (win-now, count>=4
  handling, defender-threat blocks) is unchanged and stays first in
  ordering.
- If, after implementing count>=2 width, specific corpus WINs still
  exhaust-without-proof (watch for stall-below-cap in the test output), add
  an escalation tier behind the same option: empties of claimant count>=1
  windows within distance 3 of any stone (the r3 locality bound from the
  zones work: threat-creating moves are provably within dist 3). Escalate
  only on exhaustion, not by default — branching cost.

The DEFENDER (AND-node) side is already provably complete (hitting universe
+ full-legal fallback) — do not change it.

### Ordering (this decides whether 20M nodes is enough)

The count-2 tier multiplies branching; proof-number search survives via
ordering. Suggested priority within the widened candidate set:
1. moves completing/extending count>=4 (immediate),
2. count-3 extensions that create a new count-4 (forcing),
3. count-2 pair-starts ranked by resulting fork degree (number of distinct
   windows through the cell that would reach count>=3), then by proximity
   to existing own stones.
Also strongly consider enabling the already-implemented pair canonicalization
(`tss_pair_commutation` machinery / P3) semantics inside wide mode to dedupe
(a,b)/(b,a) turn transpositions — this roughly halves the pair space.

### API shape

- `WidthOptions` struct (Default = narrow), constructor
  `WidthOptions::vcf_pair_complete()`.
- `TssSolver::set_width_options(&mut self, opts: WidthOptions)` — follow the
  existing `set_zone_options` pattern INCLUDING dropping the persistent
  positive-fragment cache on option change (profile isolation — same
  rationale as the zone options: cached node-cost provenance must not leak
  across profiles).
- Solver internals thread the option to the OR-node generator. Verifier
  (`tss_verify.rs`) note: WIN certificates witness attacker moves
  explicitly, so a wider searched set should not weaken verification — if
  you find any verifier assumption tied to the narrow generator, fix the
  assumption gap explicitly and say so in the commit message rather than
  silently relaxing a check.

## Corpus reference data (what the original solvers did)

| id | expect | ref driver | ref notes |
|---|---|---|---|
| 0hz3hty | WIN | idtt 0.05s | dfpn 6k nodes |
| **0l4291i_live** | **WIN** | **pdspn 264s, 1058 nodes, 733 leaf solves** | idtt+dfpn both failed at 20M — the monster; PRIORITY |
| 8is963b | NO | all agree | trivially dead |
| 94gnnol | NO | pdspn 21s, 108 nodes | idtt+dfpn failed |
| acly7kb | WIN | idtt 7ms | depth 4 |
| dy3dg99 | NO | all agree | trivially dead |
| g2xx6wl | WIN | idtt 0.15s | depth 6 |
| hu01jk4 | WIN | idtt 18ms | depth 6 |
| jh7yo7y | WIN | idtt 0.11s | depth 6 |
| jnzzmcm | WIN | idtt 0.44s | depth 7 |
| l9mxn59 | NO | dfpn 1.4k nodes | |
| lz60mfb | WIN | idtt 1.2s | depth 13 (deepest) |
| mvp2lvc | NO | dfpn 15k nodes | |
| xsnfyll | WIN | idtt 0.7ms | depth 4 (easiest) |
| zrugh2x | WIN | idtt 1.0s | depth 8 |
| strongloss_a_prefix6 | WIN | idtt 31ms | +2 remote defender pad stones (parity fix; pads are >=8 away and inert — a WIN here transfers a fortiori to the unpadded original) |
| strongloss_b_prefix8 | WIN | idtt 9ms | same padding note |
| hayes_20260712_turn16 | WIN | idtt 0.28s | depth 7; currently dies at 5 nodes |
| hayes_20260712_placement31 | WIN | idtt 0.17s | mid-turn (1 placement left) |

"depth" = attacker turns in the reference forcing line. The corpus format is
documented at the top of `rust/src/tss_corpus.rs`.

## Environment / how to build and test

- Cargo >= 1.95 required (lockfile v4). On this machine that means WSL with
  `export PATH="$HOME/.cargo/bin:$PATH"` (system /usr/bin/cargo is 1.75 and
  FAILS). Run:
  `wsl -e bash -c 'export PATH="$HOME/.cargo/bin:$PATH" && cd /mnt/e/Hexo-BotTrainer-hexgt/.claude/worktrees/tss-vcf-width && CARGO_TARGET_DIR=/tmp/tss-vcf-target cargo test --release -p hexfield_eq tss_corpus_check -- --ignored --nocapture'`
  (plain `cargo` from Windows may also work if a Windows toolchain >= 1.95
  is present — WSL is the proven path).
- ALWAYS set `CARGO_TARGET_DIR=/tmp/tss-vcf-target` — never build into the
  tree.
- The full unit suite: same command with no test filter and without
  `--ignored`.

## Constraints

- Work only in this worktree (`.claude/worktrees/tss-vcf-width`), branch
  `claude/tss-vcf-width`. A LIVE training run executes from a sibling
  worktree — do not touch anything outside this tree.
- Default-off discipline: `WidthOptions::default()` must reproduce today's
  behavior exactly. Production call sites keep defaults.
- Do not modify the acceptance test's assertions or the corpus file to make
  the gate easier (fixing a genuine harness bug is fine — explain it in the
  commit message).
- Commit as you go with clear messages; small commits preferred.

## Out of scope

- Python/TOML exposure of the option, selfplay integration, root-guard
  rungs, zone (AND-side) changes, perf tuning of narrow mode.
