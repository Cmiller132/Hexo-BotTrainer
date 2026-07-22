# main_10 launch plan (delta over MAIN4_LAUNCH_PLAN.md)

Continuation run of the main_4 line: WEIGHTS-ONLY warm start from main_4
`epoch_000025.pt` (owner-prepped 2026-07-21; main_4 was stopped clean at ~ep25
for the solver drive) under the **~2x-faster bit-identical solver** at
**cap 750 / J2near off**. The full procedure, health checks, crash-loop
signatures, and rollback in `MAIN4_LAUNCH_PLAN.md` apply verbatim except for
the deltas below. DO NOT LAUNCH without the owner's go.

Branch: `claude/main4-integration` (consolidate-main worktree; production tip
includes candidate-gen r1+r2 `2a1bdf97` and attacker-gen r3 `cc75b304`).
Config: `configs/hexfield_eq_main_10.toml`.
Arch: unchanged a5_lut2 (`scripts/prefit_env/hexfield_eq_raytap_a5_lut2.env`).

## Deltas vs the main_4 procedure

| Step | main_4 | main_10 |
|------|--------|---------|
| Warm-start source | main_3 ep90 | main_4 **ep25** (`epoch_000025.pt`) |
| Repackage one-shot | `scripts/_main4_repackage.sh` | `scripts/_main10_repackage.sh` → `/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main10_prefit/main4ep25/soak_init.pt` |
| Launch script | `_hexfield_eq_launch_main4.sh` | `scripts/_hexfield_eq_launch_main10.sh` |
| Run dir | `runs/hexfield_eq_main_4` | `runs/hexfield_eq_main_10` |
| Solver cap | 500 (later 750 in-place) | **750** from epoch 0 |
| Engine | pre-fold | **~2x solve wall, bit-identical** (same certs/nodes) |
| Park placeholder | 150 → sweep | **5000 strict-mode** (already final, no placeholder) |

## Pre-launch state (prepped 2026-07-21 by the orchestrator)

- **hexo_utils venv fix APPLIED**: the hexgt-build venv's `hexo_utils`
  editable install dangled at the deleted `Hexo-BotTrainer-hexgtfeat` path
  (would have failed the launch at import). Reinstalled editable from the
  INFRA_TREE worktree (`resume-run-crash-fdef2b`) per the supervisor split
  ruling. `import hexo_utils` verified OK.
- **hexfield_eq editable** in hexgt-build points at THIS worktree
  (consolidate-main). The launch-plan step-2 `maturin develop --release
  --features python` MUST still be run from WSL before launch so the venv
  carries the release-built merged engine (the stale-.pth trap) — see the
  staged-status note below.
- Repackage: run `scripts/_main10_repackage.sh` once from WSL (staged-status
  note below records whether this has already been done).
- main_4's run dir is left untouched (halt flag intact). main_10 uses a fresh
  run dir; the supervisor injects `resume_from` only after main_10's own first
  checkpoint exists.

## First-epoch validation gates (unchanged + one addition)

All MAIN4_LAUNCH_PLAN §5 gates bind: `deep_verify_failed == 0`,
`deep_win_backups` AND `deep_loss_backups` nonzero by first full epoch,
`park_bailed` ≈ 0, LR starts at 2e-4, first ep5 eval produces pooled + strix
scores. Addition for main_10:

- **Pace**: with the ~2x solver, selfplay pos/s should be AT OR ABOVE main_4's
  ep1-25 band (the all-leaves -10% bench was pre-fold). A pace BELOW main_4's
  band is unexplained and worth a halt-and-look. Queue metrics
  (`workers_spawned`, park wait) should ease relative to the ep20 retune
  measurements.

## Launch (when the owner says go)

```bash
# 1. (once) build the release engine into the venv — from WSL
ROOT=/mnt/e/Hexo-BotTrainer-hexgt/.claude/worktrees/consolidate-main
cd "$ROOT/packages/hexfield_eq/rust" && source /root/.venvs/hexgt-build/bin/activate \
  && maturin develop --release --features python

# 2. (once) repackage ep25 → soak_init
ROOT=$ROOT bash "$ROOT/scripts/_main10_repackage.sh"

# 3. launch
ROOT=$ROOT bash "$ROOT/scripts/_hexfield_eq_launch_main10.sh"
```
