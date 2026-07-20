# TSS Run-Book — flags, profiles, rung mechanics

Deployment mechanics for the TSS solver stack in the trainer. The
plan-of-record for WHAT deploys and in what order is
`PLAN_TSS_MCTS_INTEGRATION.md` (v3.1); this doc is the HOW: flags,
memory profiles, build/test commands, health metrics.

Everything below is BUILT, tested, and default-off in this tree.
Attaching to a run = rebuild the extension → relaunch at an epoch
boundary (the launch script doubles as the relauncher) → flip ONE flag
per rung. Revert of any rung = flip the flag back + relaunch
(checkpoints unaffected; target-semantics rungs additionally note
`target_regime` below).

## Deployment state (main_3, STOPPED at ep111 on 2026-07-20)

The main_3 run deployed the full built ladder before its deliberate
stop: `tss_solver_mode=3` + `tss_interior_guard` +
`tss_policy_target_sharpen` + `tss_solver_root_guard` +
`tss_solver_async` (8 threads, max 12) + `tss_solver_park` (150 ms
bail), `tss_solver_node_cap=500`, `tss_zone=false` (measured inert at
the flat +12 horizon — see the horizon section of the integration
plan). NOT built: the Lever-2 train-read swap and the horizon ladder.
**Test checkpoint for the integration program = ep90
(`epoch_000090.pt`, best-eval, owner-designated).**

## Canonical solver profiles and offline ladder

There are three memory contexts, all expressed as caller-owned byte caps:

- The ordinary offline test profile is **512 MiB**. This is the bare default
  in the forcing and spare corpus harnesses.
- The official deep-solve acceptance profile is **2 GiB**, selected with
  `TSS_BACKWALK_TT_BYTES=2147483648`. The all-19 forcing gate is one process
  and uses the fixed node ladder **10k → 100k → 1M → 20M** for WIN
  rows; NO rows stop after 1M because their acceptance condition is non-WIN.
- Trainer leaf, root-guard, and async-worker solves use the **256 KiB
  per-solve cap** in `RustSearch::TSS_SOLVER_TT_BYTES`. The default narrow
  profile splits that cap equally between the solve-local TT and the
  persistent positive-fragment cache. Pair-complete offline profiles use the
  caller's whole cap locally.

The official all-19 gate command from the solver worktree root is:

```powershell
$env:CARGO_TARGET_DIR='.target-codex'
$env:TSS_BACKWALK_TT_BYTES='2147483648'
cargo test --release -p hexfield_eq tss_corpus_check -- --ignored --test-threads=1 --nocapture
```

The spare corpus is not a second positive ladder. `NO` rows are soundness
controls and must remain non-WIN. A positive row may be labelled
`WIN_PENDING` only after an exhaustive oracle establishes WIN or after a
strict-verifier-accepted certificate is recorded with explicit provenance.

## The flags

| Flag (SelfplayConfig / TOML `[model.config.selfplay]`) | What it does |
|---|---|
| `tss_interior_guard` (bool) | Lever 0: interior nodes at the fully-forced boundary (`k == B`) narrow to the hitting-cell universe. |
| `tss_policy_target_sharpen` (bool) | Lever 1: recorded visit/π′ targets get the guard math; shards carry `target_regime=1`. |
| `tss_solver_mode` (0/1/2/3) | Stage 4 ladder: off / shadow / +verified hard LOSS / +verified hard WIN at gated leaves. |
| `tss_solver_node_cap` (int, 2000) | Deterministic per-solve node cap. Size from shadow histograms. |
| `tss_solver_sample_16` (int, 16) | Leaf solve subsample (of 16). Lower it if shadow shows CPU drag. |
| `tss_solver_root_guard` (bool) | Verified root solves; a certified WIN's move is always played (`action_selection="tss_deep_root_win"`); row proofs deep-upgrade. |
| `tss_solver_async` (bool) | Async rung: gated leaves ENQUEUE to a background worker pool (identical solver→verifier→mint path); results drain into the memo and are consumed by the selection descent-stop on later visits. Verified `Done` entries persist across moves (binding re-checked at every consumption). Flag-ON self-play is NOT bit-reproducible (arrival timing); flag-off unchanged. Wired for continuous self-play AND the lockstep eval/arena loop. |
| `tss_solver_async_threads` (int, 8) | Base pool worker count (validated 1–32). This is also the fixed worker count for non-park async. |
| `tss_solver_async_threads_max` (int, 0) | Park-mode dynamic-worker ceiling: `0` auto-sizes `available_parallelism - 6` between the base count and 24 (if base is already above 24, base is the ceiling); an explicit value must be between base and 64. Park workers scale up one at a time when queue depth exceeds 2× the current worker count and never shrink. Ignored when parking is off, preserving the legacy fixed-size pool. |
| `tss_solver_park` (bool) | Wait-at-leaf rung; requires `tss_solver_async=true`. An accepted gated leaf waits in the scheduler pen for its async result instead of entering the GPU queue. A consumable proof backs up immediately; Unknown/non-consumable results and timed-out leaves rejoin normal eval. Selection never blocks, and flag-off is bit-identical. Park-on play is not bit-reproducible because worker scheduling/cache warmth can vary, but every non-bailed accepted leaf gets practical first-touch consumption. |
| `tss_solver_park_timeout_ms` (int, 100) | Per-leaf liveness bail, validated 1–5000 ms. A bail releases the leaf to normal eval while any late result may still populate the memo. |
| `tss_solver_async_inline_16` (int, 0) | Legacy hybrid inline tier under non-park async: gated leaves with `(hash & 0xF)` below this solve inline and the rest enqueue. **Ignored when `tss_solver_park=true`**: every accepted gated leaf parks, so the park rung supersedes this tier. |
| `tss_zone` (bool) | Zone-theorem AND generation (P0–P3 solver rewrite). With the flag on, every solve runs the **horizon ladder**: a tight `+8` deadline on half the node budget first (defender budget 4 at the first Universal ⇒ zones prune the initial fanout — at the flat `+12` the budget is exactly 6 and the generator must take the full legal set, which is why ep32 showed `zone_nodes=0`), then the unchanged `+12` solve only if the tight attempt is Unknown. Zone-off is bit-identical to pre-zone. |

Numeric controls are range-validated at the Rust seam: `tss_solver_mode` 0–3, `tss_solver_sample_16`/`tss_solver_async_inline_16` 0–16, `tss_solver_async_threads` 1–32, `tss_solver_async_threads_max` 0 or base–64, `tss_solver_park_timeout_ms` 1–5000, and `tss_solver_node_cap` ≥1. `tss_solver_park=true` with async disabled is rejected. An out-of-band TOML value fails launch loudly instead of silently changing behavior. Changing the worker settings on a live config resizes/rebuilds the pool at the next run boundary. Flipping zone/commutation options mid-session drops the solver's persistent fragment cache (profile isolation).

For park-mode production sizing, leave `tss_solver_async_threads` at the known-safe base and start with `tss_solver_async_threads_max=0`. Auto preserves six logical CPUs for the rest of the pipeline, never goes below base, and caps additional scaling at 24 (an explicitly higher base remains unchanged). Set an explicit ceiling only after observing sustained queue pressure or park bails and checking host CPU headroom. Non-park async ignores this setting and stays fixed at the base count. Measured worker ceiling on the 32-core host: ~25k nodes/s per worker at 12 workers; 24 workers collapse to ~7.9k/worker — the practical ceiling is 12–16.

## Rung order (one per relaunch; watch ≥1–2 epochs + the next eval before the next rung)

0. **Merge only, all flags off.** This alone turns on the Stage-0 shadow
   telemetry (λ¹ class column, proof scalar, counters). Play and targets are
   bit-identical (golden-digest-proven). Start collecting the gate metrics.
1. **`tss_solver_mode=1` (shadow).** Adds deep solves + verification at gated
   leaves, consuming nothing (bit-identical, twin-run-proven). Collects
   `deep_*` histograms — the cost/UNKNOWN data that sizes `node_cap`.
2. **`tss_interior_guard=true`.** First behavior change (search-only).
3. **`tss_policy_target_sharpen=true`** — ONLY if the shadow gate says it
   matters: `tss.win_retained_mass_mean` clearly below ~1.0 (mass actually
   moves). If ≈1.0, skip: π′ is already proof-sharp.
4. **`tss_solver_mode=2`** (verified hard LOSS + eval elision). Before/after,
   deliberately probe avoided lines offline (false LOSS is the silent
   failure): re-run the solver harness + spot-solve avoided positions.
5. **`tss_solver_mode=3`** (verified hard WIN).
6. **`tss_solver_async=true`** (background fire-and-forget route). Keep
   `tss_solver_async_inline_16=0`; watch worker/queue health before parking.
7. **`tss_solver_park=true`** (wait-at-leaf first-touch consumption; the
   production async consumption rung — supersedes the hybrid inline tier).
8. **`tss_solver_root_guard=true`** (serve too — the eval arena inherits it).
9. **Integration-era rungs** (engine port R0, horizon ladder, Lever-2
   train-read swap, class head): sequencing and gates live in
   `PLAN_TSS_MCTS_INTEGRATION.md` §9 — one lever per boundary still binds.

## Watch on every rung (epoch JSON `hexfield.selfplay.epoch_*.json`, `tss` block)

- **`deep_verify_failed` — MUST BE 0.** Nonzero = a solver claim failed its
  certificate check (values degraded safely, but the solver has a bug): set
  `tss_solver_mode=0`, keep shadow data, investigate before re-enabling.
  This counter is timing-safe: every scheduler exit quiesces the async pool
  and folds late-banked failures into the same epoch's total (scheduler keys
  `tss_async_verify_failed_tail` / `tss_async_worker_panics_tail` carry the
  tail; the driver adds the verify tail into `tss.deep_verify_failed` before
  the epoch JSON is written).
- `injection_fire_rate` — the closed-loop metric; should fall as the net
  internalizes tactics.
- `win_retained_mass_mean` / `sharpened_rows` — Lever-1 gate + effect.
- `forced_defense_fraction`, `prune_eligible/dropped` — Lever-0 scale.
- `deep_calls/win/loss/unknown/nodes` by epoch — solver cost + UNKNOWN rate
  at production caps in the threat-dense endgame. `horizon_cut` — Unknowns
  refused by the ply deadline (the depth-frontier gate metric).
- `proof_rows` / `proof_disagreements` — the Lever-2 gate.
- **Park rung: `park_bailed / park_parked` should be ≈0.** This is the primary
  health signal: `park_hard` is first-touch proof consumption,
  `park_released` is a completed Unknown/non-consumable solve, and
  `park_wait_ms_sum/max` report resolution latency including bails. Sustained
  bails mean the pool cannot keep up: raise `tss_solver_async_threads_max`
  if CPU headroom exists, or lower `tss_solver_node_cap`. Never remove the
  timeout; it is the scheduler's liveness backstop.
- Non-park async rung: the queue is a LIFO with oldest-eviction (cap 16384;
  workers serve the NEWEST request so freshness is workers × solve-time, not
  backlog latency). `async_dropped` counts evicted oldest entries (never a
  correctness issue — dropped leaves take the plain net eval); `async_stale`
  informational. With parking enabled the queue instead serves FIFO and never
  evicts accepted work. At the 16,384-entry memory bound it rejects a fresh
  request (that leaf takes normal eval) rather than orphaning an existing
  parked leaf; the per-leaf bail is the latency overload valve. Watch
  `async_workers_spawned` for pressure-triggered scaling and
  `deep_hard_backups`/`park_hard` for impact.
- Standard health: pinned entropy metric (L0+L1 compound sharpening), pos/s,
  eval cadence h2h (pool/Strix/SealBot), earlyoom/VM memory.

## Build + test (WSL/Linux ONLY — Windows cargo artifacts ICE the Linux rustc)

The three frozen command lines (run from the tree being built; substitute
`<TREE>` with its absolute /mnt/e path):

```bash
# 1. non-python cargo suite
wsl -e bash -c 'source ~/.cargo/env; cd <TREE> && CARGO_TARGET_DIR=packages/hexfield_eq/rust/target-wsl cargo test -p hexfield_eq'
# 2. python-feature cargo suite
wsl -e bash -c 'source ~/.cargo/env; cd <TREE> && CARGO_TARGET_DIR=packages/hexfield_eq/rust/target-wsl-py PYO3_PYTHON=/root/.venvs/hexfield-dev/bin/python RUSTFLAGS="-L /usr/lib/x86_64-linux-gnu -C link-arg=-lpython3.12" cargo test -p hexfield_eq --features python'
# 3. rebuild extension + pytest
wsl -e bash -c 'source ~/.cargo/env; source /root/.venvs/hexfield-dev/bin/activate; cd <TREE> && CARGO_TARGET_DIR=packages/hexfield_eq/rust/target-wsl-py maturin develop -m packages/hexfield_eq/Cargo.toml && PYTHONPATH=packages/hexfield_eq/python:packages/hexo_runner/python:packages/hexo_models/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python /root/.venvs/hexgt-build/bin/python -m pytest tests/test_hexfield_eq_tss_shadow.py tests/test_hexfield_eq_window_streaming.py -q'
```

All three must be green pre-merge. Deployment builds: `maturin develop
--release -m packages/hexfield_eq/Cargo.toml` in the `hexfield-dev` venv
FROM THE RUN TREE being deployed (editable install drops the `.so` into
that tree). Never build from a different tree into the live one
mid-epoch; relaunch picks it up.

## Mechanics reminders

- The Stage-0 golden digest (`tests/data/hexfield_eq_tss_stage0_golden.json`)
  pins flag-off bit-identity; regenerate ONLY for an intentional
  behavior-change baseline (two-build procedure in the test docstring).
- Solver memory: trainer solves are capped at 256 KiB per solve, and the
  per-move memo has ≤8192 entries (cleared every move inline; async retains
  verified `Done` entries for the life of the game's search object, still
  ≤8192) — no unbounded growth (host earlyoom discipline). Async adds:
  request queue ≤16384 state clones + one persistent solver TT per worker
  thread (byte-capped per solve).
