# R-ORDER-PRIOR — net-policy ordering hints for wide-pn (owner-directed)

You are in worktree `.claude/worktrees/order-prior` (branch
claude/order-prior, stacked on the just-gated dual-pass work). Do NOT
commit (the git index is not writable from your sandbox); leave changes in
the tree and write `.codex-z/ORDER_PRIOR_NOTES.md` with what you changed,
test results, and flagged concerns.

## Motivation (measured)

The solver's cost center is unforced/quiet positions where child fanout is
large and the node cap (500 in production) dies before a proof lands. V1
data: proven wins are 45.5% "hot" vs 6% for budget-exhausted grinds —
strong ordering signal exists. We run inside an MCTS engine whose policy
net has ALREADY evaluated these positions; its priors are an ordering
oracle we currently throw away. Ordering is soundness-free: it changes
which child gets budget first, never the proof space, and every decided
verdict still passes the strict verifier.

## Goal

Optional per-solve **ordering hints**: a list of (q, r, weight) move
priors the wide-pn search uses ONLY to order candidate expansion — higher
weight explored first, unhinted moves after hinted ones in their current
order. No pruning, no proof-space change, no verdict consumption of the
hints. Absent hints = bit-identical current behavior.

## Where

- `packages/hexfield_eq/rust/src/tss_solver.rs` — wide-pn candidate
  generation/expansion ordering. Find the real seams yourself; the prize
  sites are (a) attacker candidate ordering (first/second candidates) and
  (b) defender reply ordering at Universal fanouts. If one of the two is
  structurally awkward, land the other and say so in the notes —
  a partial, honest lever beats a forced total one.
- Plumbing mirrors the dual-pass pattern you can read in the tree
  (`tss_solver_dual_pass`, commit 6f044e2d): solver-side setter
  (per-solve hints, cleared after each solve — hints are position-
  specific, they must never leak to another position), batch API
  parameter, manifest echo of a `has_ordering_hints`-style boolean is NOT
  needed (hints are per-position data, not config) — instead echo a
  config-level `ordering: "prior" | "off"` mode flag through the shared
  resolver so the harness can gate it.
- Batch API: `hexfield_eq_deep_solve_batch(..., ordering_hints=None)`
  where hints is an optional list (one entry per state) of lists of
  `(q, r, weight)` tuples. Signature must stay source-compatible
  (default None).

## Constraints (hard)

- `tss_verify.rs` zero diff.
- Hints must be incapable of causing pruning: the candidate SET at every
  node must be exactly what it is today; only the iteration order may
  change. Make this structurally true (sort/stable-partition the existing
  generated list), not policed by discipline.
- No hints => bit-identical to current behavior (not just same verdicts —
  same node counts on a fixture battery).
- Per-solve isolation: hints for position A must never influence position
  B (including via TT/fragment carryover assumptions — if hint-ordered
  solves would poison order-sensitive cached state shared with unhinted
  solves, clear or key the cache accordingly and note the cost).

## Definition of done

Windows Git Bash, MSVC target (the WSL gnu target cannot link pyo3 test
binaries — known):
    cd packages/hexfield_eq/rust
    CARGO_TARGET_DIR="E:/cargo-targets/order-prior" cargo test --features python

NOTE: one pre-existing parallel-run flake exists
(`cap_resume_discards_on_binding_or_cap_mismatch` vs env-mutating warmth
tests). If you hit it, rerun serialized (`-- --test-threads=1`) and
report both results.

1. All existing tests green (`--features python` — the non-default
   feature gates `mod search`/`mod tree`; plain `cargo test` silently
   skips them).
2. New tests:
   - no-hints identity: verdict AND node-count equality with current
     behavior over a fixture battery (win, loss, unknown, quiet).
   - hint-parity: with hints on fixtures where the cap is NOT binding,
     verdicts are unchanged (node counts may differ).
   - constructive win: a fixture where a correct hint (pointing at the
     proving move) strictly reduces nodes-to-proof vs an adversarial
     hint (pointing away). This demonstrates the mechanism is live.
   - no-leak: solving [A-with-hints, B-without] yields for B exactly the
     result of solving B alone.
   - candidate-set invariance: a test asserting the candidate set at the
     root is identical with and without hints (order aside).
3. `git diff --stat packages/hexfield_eq/rust/src/tss_verify.rs` empty.
4. Do not touch `scripts/tss_harness/` (orchestrator wires the harness
   side afterwards) and do not modify the dual-pass logic you build on.

## Out of scope

Real net priors (the orchestrator precomputes them separately and will
A/B via the harness), any pruning/reduction, zone or Group-2 work,
production driver wiring beyond what the flag plumbing pattern requires.
