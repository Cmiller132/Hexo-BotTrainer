# R-TWOPASS-IMPL — unused-budget dual pass for SolveGoal::Both (wide profile)

Owner-approved production change. You are working in this worktree
(branch claude/twopass-leaf, from claude/v1-soak). Do NOT commit (the
worktree git index is not writable from your sandbox); leave changes in
the tree and write .codex-z/TWOPASS_IMPL_NOTES.md summarizing what you
changed, test results, and anything you flagged.

## The measured problem

packages/hexfield_eq/rust/src/tss_solver.rs, solve_goal budget split
(~line 1104-1115): under the wide profile (`vcf_pair_complete`),
`SolveGoal::Both` gives the primal WIN attempt the full budget and the
dual (loss) attempt ZERO. Production consequence, measured 2026-07-20
(docs/SOLVER_NOTES.md §6 P4): when the primal width-exhausts — sometimes
at 2 nodes — Both returns Unknown with 498/500 of the budget UNUSED,
while the loss proof at those positions costs 5-44 nodes. On a 338-
position human sample the harness's adapter-side two-pass protocol
proved 42 losses where Both surfaces 15, at BETTER nodes-per-decision
than the current profile.

## Goal

A config-gated engine-side dual pass: under the wide profile, when the
primal WIN attempt of a `Both` solve returns undecided having consumed
N nodes with N < budget, run the dual (loss/opponent-win) attempt with
the REMAINING budget (budget - N) instead of 0. Total node consumption
must never exceed the original cap. Positions where the primal consumes
the full budget behave exactly as today (dual cap 0).

## Constraints (hard)

- `tss_verify.rs` untouched — zero diff. Verdicts remain consumable only
  through the existing verified path (`tss_solve_verified`); the dual
  attempt's Loss result must flow through the same certificate
  verification as today's loss results.
- Flag OFF (default) = bit-identical behavior to current code. The flag:
  a new solver option (e.g. `dual_pass: bool` alongside the existing
  width/zone options), default false.
- Plumbing: expose the flag end-to-end —
  1. Rust solver option + the wide-profile budget-split change;
  2. optional `dual_pass=False` parameter on `hexfield_eq_deep_solve_batch`
     and `hexfield_eq_solver_manifest` (search.rs pyfunctions; keep
     existing call sites source-compatible via defaults);
  3. a `tss_solver_dual_pass: bool = False` field on the selfplay config
     (packages/hexfield_eq/python/hexfield_eq/config.py) carried through
     `build_divergence_overrides` like `tss_solver_horizon` is, reaching
     the production leaf profile wiring in search.rs;
  4. manifest echo: `dual_pass` must appear in the effective-config dict
     produced by the shared resolver (`effective_solve_config` /
     `sample_runtime_flags` — follow the Lane A pattern in tss_solver.rs
     /search.rs: the manifest must echo what the solver actually uses,
     never a re-derivation).
- The dual attempt must respect the same semantic horizon / caps as the
  primal (same SolveCaps discipline as the existing dual path in the
  non-wide split).

## Definition of done (all must pass; run them yourself)

In WSL from this worktree (cargo via login shell: `bash -lc`):
    cd packages/hexfield_eq/rust
    CARGO_TARGET_DIR=/root/twopass-target bash -lc "cargo test --features python"

1. All existing tests green (NOTE the repo trap: `mod search`/`mod tree`
   are behind the non-default `python` feature — plain `cargo test`
   silently skips them; always use `--features python`).
2. New tests you write:
   - cheap-loss fixture (there is an existing `forced_loss_fixture()` in
     tss_solver.rs tests): with dual_pass ON, `solve_goal(.., Both)`
     returns a verified Loss; with it OFF, Unknown (or whatever current
     behavior is — assert current behavior explicitly).
   - budget guard: total nodes consumed (primal + dual) <= node_cap on a
     position where both attempts run.
   - full-budget primal: a position whose primal consumes the entire cap
     behaves identically ON vs OFF (verdict + node count).
   - win parity: a known-win fixture returns the same verdict and node
     count ON vs OFF.
   - flag-off identity: solve a small fixture battery with the flag off
     and assert results identical to a build without your change
     (equivalently: assert the OFF path never consults the new code —
     e.g. dual cap stays 0 exactly as before).
   - manifest echoes dual_pass truthfully for both values.
3. `git -C ../../.. diff --stat packages/hexfield_eq/rust/src/tss_verify.rs`
   is empty.
4. Keep the change minimal: no ordering changes, no zone changes, no cap
   changes, no driver/eval logic beyond the flag plumbing.

## Notes

- deep_loss counter increments live in tree.rs (~line 872) and will start
  counting naturally once dual verdicts flow — do not add counters.
- Do not touch scripts/tss_harness/ (the orchestrator gates your work
  with it afterwards).
- If you find the budget-split refactor requires touching the non-wide
  `Both` split, leave that split's behavior EXACTLY as-is (its halving is
  pre-existing semantics; out of scope).
