# R-TWOPASS-IMPL notes

## Changes

- Added a default-off `dual_pass` option to `TssSolver` and included it in
  `EffectiveSolveConfig`.
- Kept the existing narrow `SolveGoal::Both` half split unchanged. In the wide
  profile only, an enabled dual pass now assigns an undecided primal's exact
  remaining budget (`node_cap - merged stats.nodes`) to the opponent-WIN
  attempt. A full-cap primal therefore still leaves a zero dual cap, and the
  aggregate node count cannot exceed the original cap.
- Plumbed `tss_solver_dual_pass` through `SelfplayConfig`,
  `build_divergence_overrides`, Rust divergence resolution, root/inline/async
  production solves, and async requests.
- Added source-compatible `dual_pass=False` parameters to
  `hexfield_eq_deep_solve_batch` and `hexfield_eq_solver_manifest`.
- Added the `dual_pass` manifest field from the same shared effective-config
  resolver used by real solves.
- Left `tss_verify.rs`, `scripts/tss_harness/`, solver ordering, zones, and all
  pre-existing cap semantics unchanged. Dual Loss certificates continue
  through `tss_solve_verified` and its existing verifier/mint path.

## Tests

- Full required Rust suite (Python feature enabled): **PASS** - 190 passed,
  0 failed, 37 ignored; doc tests also passed.
  - Command run from `packages/hexfield_eq/rust`:
    `cargo test --features python`
  - `CARGO_TARGET_DIR` used:
    `E:/Hexo-BotTrainer-hexgt/.claude/worktrees/twopass-leaf/.cargo-target/twopass`
  - A repeat parallel run transiently failed the pre-existing
    `cap_resume_discards_on_binding_or_cap_mismatch` test when another test's
    process-global `TSS_SHARED_FRAGMENTS` mutation overlapped it. That test
    passed immediately in isolation, and the complete serialized suite
    (`-- --test-threads=1`) passed 190/190.
- New coverage includes:
  - existing root `forced_loss_fixture` current behavior pinned and verified;
  - non-lambda-one cheap Loss: flag off Unknown, flag on verified Loss;
  - combined-node budget guard when both attempts run;
  - full-budget primal ON/OFF verdict and node-count identity;
  - known-WIN ON/OFF verdict and node-count identity;
  - flag-off wide `Both` identity with the legacy primal-only `Win` path over a
    small fixture battery;
  - manifest echo for both boolean values through the shared resolver and the
    Python-facing dict;
  - strict Rust override-key acceptance and Python config-map propagation.
- `git diff --check`: **PASS**.
- `git -C ../../.. diff --stat packages/hexfield_eq/rust/src/tss_verify.rs`:
  **empty**.
- `git -C ../../.. diff --stat scripts/tss_harness`: **empty**.

## Flagged environment limitations

- This host has no installed WSL distribution, so the requested WSL run and
  `/root/twopass-target` were unavailable. Git Bash maps `/root` under its
  protected installation directory. The same login-shell Cargo suite was run
  with the workspace-local target directory listed above.
- The host Python lacks `pytest` and `numpy`, so the added standalone Python
  config test could not be run separately. Its Rust-facing plumbing is covered
  by the passing `--features python` suite, and the Python source change is a
  direct dataclass-to-dict mapping.
