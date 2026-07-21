# R-LOSS-SIDE — loss-frontier theory + deepening (ULTRA, long horizon, owner-directed)

You are in worktree `.claude/worktrees/loss-side` (branch
claude/loss-side, stacked on the gated dual-pass work `6f044e2d`). You
cannot commit; leave changes in the tree and write
`.codex-z/LOSS_SIDE_NOTES.md` (running) — final summary at the end.

## Situation (all measured today, docs/SOLVER_NOTES.md)

The engine now has an unused-budget dual pass (`tss_solver_dual_pass`):
when the wide `Both` WIN attempt returns undecided with budget left, the
leftover goes to the opponent-WIN (loss) attempt. Results at cap 500:
+288 verified losses across the frozen dev sets, wins untouched,
throughput neutral. But the loss frontier is still rationed:
- 58/116 certified atlas loss labels in the puzzle dev split remain
  unprovable at cap 500 (they were proven at the 20k labeling cap with a
  dedicated full-budget loss pass; Lane C data in
  `raws/lanec_labels.jsonl` has per-position win_pass/loss_pass nodes).
- The dual attempt gets NOTHING when the primal is cap-bound — and
  cap-bound grinds are exactly where V1 measured LOSS probes to be
  ~10^3 cheaper than WIN grinds (16-22 microseconds).

## Goal

Make the loss side a first-class, theoretically grounded search
direction, and measurably deepen loss coverage at fixed cap. Two
deliverables:

### 1. Theory: docs/DESIGN_TSS_LOSS_SIDE.md

Written rigorously enough for hostile review:
- The loss-soundness statement, precisely: proving "mover is lost" =
  proving the opponent (as claimant, mover to move) wins. State exactly
  which width restrictions remain sound for that claimant search (width
  restriction = pure strengthening for any claimant-WIN proof) and
  which certificate obligations the strict verifier already checks for
  Loss verdicts (read tss_verify.rs — DO NOT modify it).
- Where the existing zone/deadline machinery applies symmetrically to
  the loss search (opponent completion deadlines create THEIR local
  budgets), and where it is inert (quiet positions), with reference to
  the V1 zone measurements (zone arms: zero pruning at slack budgets).
- Budget-policy analysis: the current leftover policy vs alternatives —
  (a) cheap loss-probe FIRST (bounded, e.g. 32-64 nodes) before the
  primal, (b) a reserved loss floor, (c) leftover-only (current). Use
  the Lane C loss_pass node distribution and the V1 probe-cost data to
  predict each policy's coverage/cost at cap 500 BEFORE implementing.

### 2. Implementation, gated by the harness

Implement the best policy variant(s) as config-gated options (pattern:
`tss_solver_dual_pass`, commit 6f044e2d — solver option + config.py +
divergences + batch API + manifest echo via the shared resolver;
default = current behavior, flag-off bit-identical). Then measure:

    /root/.venvs/twopass-dev/bin/python scripts/tss_harness/runner.py \
        run --label <arm> --tier quick --config-json '{...}'

(harness-dev venv works too if twopass-dev is missing packages; the
runner needs a wheel matching your Rust changes — build with
`/root/.venvs/harness-dev/bin/maturin build --release -o /root/loss-wheels`
from packages/hexfield_eq inside WSL via `bash -lc`, install into a venv
you create, mirror the .pth files from /root/.venvs/twopass-dev; write
.pth files with printf INSIDE one WSL command, never copy them through
Windows shells — CR mangling corrupted them once today.)

Success = more verified losses on the human/puzzle dev samples at cap
500 with wins unchanged and total nodes <= cap per position. Every
verdict still flows through tss_solve_verified.

## Hard constraints

- `tss_verify.rs` zero diff. No weakening anywhere; a policy that would
  skip the primal entirely must still preserve win coverage (measure).
- Flag-off bit-identity, with tests (the dual-pass test patterns in
  tss_solver.rs are your template).
- cargo tests: Windows Git Bash, MSVC target, `--features python`
  (plain `cargo test` silently skips the gated modules;
  `CARGO_TARGET_DIR="E:/cargo-targets/loss-side"`). Known pre-existing
  parallel flake (`cap_resume_discards_...` vs env-mutating warmth
  tests) — rerun serialized if hit, report both.
- Do not modify the dual-pass logic you build on, the harness gates, or
  the frozen sets. New arms may be added to the adapter config
  vocabulary.
- One cargo lane discipline: your builds are yours alone; if the
  machine is compiling elsewhere (it may be), cargo will queue on the
  filesystem — just proceed, your target dir is isolated.

## Definition of done

- DESIGN doc complete with the policy predictions vs measurements table.
- All tests green (existing + new), flag-off identity proven.
- At least one policy variant measured in the harness with its quick-
  tier archive path recorded in the notes.
- Honest verdict: which policy (if any) should be promoted to a
  standard-tier run, and what it predicts for the 58 unreachable atlas
  losses.
