# main_4 launch plan

Clean fresh-start run, WEIGHTS-ONLY warm-started from main_3 `epoch_000090.pt`
(owner ruling 2026-07-20). Fresh optimizer state, fresh replay window, epoch
counter restarts at 0, new run dir. This document is the exact ordered launch
procedure plus first-hour health checks and rollback.

Branch: `claude/main4-integration` (the merged solver + value-signal tree — the
dual-pass / G2 / ordering-hints solver code and the harness live ONLY here).
Config: `configs/hexfield_eq_main_4.toml`.
Arch: main_3 A5 additive ray-tap (`scripts/prefit_env/hexfield_eq_raytap_a5_lut2.env`).

---

## 0. Preconditions

- `claude/main4-integration` checked out in a worktree whose WSL path you will
  use as `ROOT` (default in the launch script resolves to the tree the script
  lives in). Current integration worktree:
  `/mnt/e/Hexo-BotTrainer-hexgt/.claude/worktrees/consolidate-main`.
- GPU free (main_3 is STOPPED). Confirm no `hexfield_eq` trainer is running.
- Host free RAM healthy; no competing heavy build.
- main_3 `epoch_000090.pt` present:
  `/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main_3/checkpoints/epoch_000090.pt`
  (confirmed 2026-07-20; the run reached ep111 so ep90 is well inside history).

## 1. Fill the park window (PARK_PLACEHOLDER)

`configs/hexfield_eq_main_4.toml` sets
`tss_solver_park_timeout_ms = 150  # PARK_PLACEHOLDER`. Replace `150` with the
value the running park sweep settles on. If the sweep is unresolved at launch,
leave 150 (the main_3-proven default). This is the ONLY placeholder in the
config.

## 2. Build the trainer from WSL into the run venv (trainer-builds-from-WSL law)

The trainer's native `_rust` extension MUST be built from WSL (not Windows) into
the GPU venv, or you hit the **stale-.pth trap**: an old `_rust` silently
shadows the new solver code and the run trains against the wrong engine.

```bash
# from WSL, ROOT = the merged worktree
ROOT=/mnt/e/Hexo-BotTrainer-hexgt/.claude/worktrees/consolidate-main
cd "$ROOT/packages/hexfield_eq/rust"
source /root/.venvs/hexgt-build/bin/activate
# release build so selfplay/train run at production speed
maturin develop --release --features python
# sanity: the freshly-built module must import and expose the solver seam
python -c "import _rust; print('deep_solve_batch' in dir(_rust))"
```

Verify no stale `_rust*.pth` / old `_rust*.so` lingers in the venv
site-packages after the develop (the trap): `pip show` / `find` the venv for a
second `_rust` and remove it if present.

## 3. Repackage the ep90 weights into the soak-init (weights-only warm start)

`scripts/_main4_repackage.sh` runs `eq_ladder_runner.py --repackage`, which calls
`build_soak_init(ckpt, out, weights="raw")`: it takes the `"model"` state dict
only (drops optimizer + train_state), wraps `{"meta", "model"}`, and
cross-checks the arch meta against the sourced `HEXFIELD_EQ_*` env. Fresh
optimizer/window/epoch=0 come from the trainer treating this as
`initialize_from` (NOT `resume_from`).

```bash
# from WSL; sources the SAME a5_lut2 arch env so the meta-vs-env check passes
ROOT=/mnt/e/Hexo-BotTrainer-hexgt/.claude/worktrees/consolidate-main \
  bash "$ROOT/scripts/_main4_repackage.sh"
# → /mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main4_prefit/main3ep90/soak_init.pt
```

Watch the repackage output for `soak-init arch meta vs env MISMATCH` — if it
warns, the warm start would silently drop mismatched shapes. Fix the env before
proceeding. `configs/hexfield_eq_main_4.toml`'s `initialize_from` already points
at this exact path.

## 4. Launch

```bash
# from WSL, via a bare `wsl -e` from the Windows side or directly in WSL
ROOT=/mnt/e/Hexo-BotTrainer-hexgt/.claude/worktrees/consolidate-main \
  bash "$ROOT/scripts/_hexfield_eq_launch_main4.sh"
```

The launch script sources the a5_lut2 arch env, sets `CONFIG` = main_4 toml and
`RUNDIR` = `/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main_4`, verifies the
soak_init exists, and starts `scripts/_hexfield_eq_supervise_main1.sh`
(nohup+setsid). The supervisor's FIRST LAUNCH (no checkpoints yet) uses
`initialize_from` per the config (weights only, epoch 0). Every relaunch after a
checkpoint exists injects `resume_from` into `_resume_config.toml` (model +
optimizer + epoch — normal resume). No manual resume wiring needed.

## 5. First-hour health checks

Tail `RUNDIR/supervisor.log` and the current `RUNDIR/train.<stamp>.out.log`, and
the epoch JSON under `RUNDIR/` (the `tss` block). Confirm:

- **`deep_verify_failed == 0`** (cumulative). Any nonzero verify failure = a
  soundness regression → HALT immediately (see rollback).
- **`deep_win_backups` and `deep_loss_backups` BOTH nonzero** by the first
  full epoch — proves mode-3 consumption is actually landing WIN and LOSS
  proofs, not silently degrading to one side.
- **`park_bailed`** small relative to gated leaves (near 0 is healthy). A high
  bail share means the park window is too tight or worker capacity is short —
  revisit `tss_solver_park_timeout_ms` / `tss_solver_async_threads_max`.
- **pos/s (selfplay throughput)** in the expected band (compare to the main_3
  ep~90 epoch seconds; a large regression means the solver is over-spending).
- **LR** in the epoch JSON `lr` field starts at `2e-4` at epoch 0 and steps
  down along the cosine (≈`1.1e-4` by ep75, `2e-5` by ep150, held after). The
  curve is unit-pinned in `tests/test_hexfield_eq_lr_schedule.py`.
- First eval (ep5 / multi-stage every 5) produces a pooled number and a strix
  score — sanity that warm-started weights load and play.

### Crash-loop signature (the wedged-dxg trap)

If the supervisor keeps relaunching AND the GPU probe reports OK while the
train loop dies in a fast loop (**probe-ok + fast relaunch loop = wedged dxg**),
the WSL CUDA layer is wedged. Remedy: from Windows, `wsl --shutdown`, then wait,
then relaunch (step 4). Also check `C:\...\Temp\wsl-crashes` and C: free space
first — a CUDA abort can dump the ~20GB trainer image and exhaust C: (host-disk
exhaustion incident); `.wslconfig maxCrashDumpCount=0` should be in place.

## 6. Rollback / kill switches (main_3 conventions)

- **Halt the run:** `touch RUNDIR/supervisor_halted.flag` — the supervisor
  stops relaunching after the current child. To also kill the in-flight driver:
  `kill $(cat RUNDIR/driver.pid)`.
- **Disable the solver without stopping training:** flip
  `tss_enabled = false` (or drop `tss_solver_mode` to 0 for SHADOW) in the
  config and let the supervisor relaunch — self-play falls back to the pure
  net. The default-OFF merged extensions (`tss_solver_group2`,
  `tss_solver_loss_reserve_nodes`, ordering-hints) are already off and are not
  in the play path.
- **Full stop:** halt flag + kill driver + `wsl --shutdown` if the VM is wedged.
- **Config was wrong:** edit `configs/hexfield_eq_main_4.toml`, remove
  `RUNDIR/_resume_config.toml` if the bad value was already injected, then
  relaunch. To restart truly fresh (discard the run), move `RUNDIR` aside and
  re-run from step 4 (the soak_init from step 3 is reusable).

---

## Config summary (the deltas that define main_4)

| Knob | Value | Note |
|------|-------|------|
| `initialize_from` | main4_prefit/main3ep90/soak_init.pt | weights-only, epoch 0 |
| `learning_rate` / `lr_final` | 2e-4 / 2e-5 | cosine |
| `lr_decay_epochs` / `lr_warmup_steps` | 150 / 0 | hold floor after 150 |
| `lr_schedule` | cosine | `scheduled_lr`, clamped-hold |
| `tss_solver_mode` | 3 | WIN + hard-LOSS consumption |
| `tss_solver_node_cap` | 500 | harness-decided |
| `tss_solver_dual_pass` | true | unused-budget dual pass |
| `tss_solver_horizon` | 0 | unbounded |
| `tss_solver_horizon_ladder` | false | ladder off |
| `tss_zone` | false | |
| `tss_solver_park` / `_park_timeout_ms` | true / PARK_PLACEHOLDER(150) | fill from park sweep |
| `tss_solver_async` / threads / inline16 | true / 8 / 4 | |
| `tss_solver_root_guard` / `tss_interior_guard` | true / true | |
| `tss_solver_group2` / `_loss_reserve_nodes` | false / 0 | default-off merged extensions |
| profile | wide | always-on production leaf profile |
