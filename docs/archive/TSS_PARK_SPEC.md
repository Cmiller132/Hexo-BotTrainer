# TSS Leaf Parking — Build Spec (wait-at-leaf synchronous consumption)

Owner-ordered 2026-07-14: make the search WAIT on the deep solver so that
100% of gated leaves consume their solve result at FIRST TOUCH ("consistent"),
with minimal throughput cost. Design agreed with the owner; this spec freezes
it for implementation. Base: branch `claude/tss-v2-build` @ 24ef8961 (all
suites green: cargo 58 no-python + 126 python-feature, pytest 29).

## Goal

Behind a new default-off flag `tss_solver_park`, a TSS-gated leaf is PARKED
instead of being sent to the GPU eval queue: its solve request goes to the
async pool (as today), the leaf is held in a scheduler-owned pen, and when
the solve result lands the leaf either

- backs up the verified hard value directly (GPU eval elided — the existing
  hard-leaf semantics), or
- is released into the normal eval queue (Unknown / non-consumable tier), or
- is released by a bounded timeout ("bail": today's fire-and-forget behavior
  for exactly that leaf; the late result still lands in the memo as usual).

The select loop must NEVER block on a solve. Parking is per-leaf; all other
slots/leaves keep flowing to the GPU. Little's-law sizing: production runs
~430 solves/s at ~2–5 ms mean solve latency ⇒ ~2 leaves parked on average,
~9 at p95 — GPU flush sizes are essentially unchanged.

## Why (context you should know)

- The pool exists and works (`tss_async.rs`): gated leaves enqueue, workers
  run `tree::tss_solve_verified` (solver → independent verifier → sealed
  HardValue mint), results drain into the per-move memo, and a descent-stop
  consumes them on LATER visits. The weakness: consumption timing is
  arrival-dependent. Empirically, first-touch (inline) consumption has ~7×
  the per-solve search influence of late-landing async results.
- The current hybrid inline tier (`tss_solver_async_inline_16`) solves 4/16
  of gated leaves inline ON the select thread. Parking supersedes it: when
  `tss_solver_park` is on, the inline tier is IGNORED (everything parks; do
  not solve on the select thread). Document this in the flag comment.

## Hard constraints (non-negotiable)

1. **Soundness firewall unchanged.** Nothing new mints or consumes a
   `HardValue` outside the existing paths. Pen resolution consumes via the
   existing memo + `tss_consume_gate` tiering (LOSS at mode≥2, WIN at
   mode≥3). Verify failures stay fatal-counted exactly as today.
2. **Flag-off bit-identity.** With `tss_solver_park=false` every code path,
   counter, and byte of behavior is identical to 24ef8961. The Stage-0
   golden digest test must pass unmodified.
3. **No deadlocks, ever.** The scheduler loops must provably make progress
   with a non-empty pen and an empty eval queue (the bail deadline is the
   liveness backstop). `continuous_has_work` (and the lockstep loop
   condition) must count parked leaves as pending work; the "scheduler
   stalled" hard error must not fire while leaves are parked.
4. **Virtual-loss discipline.** A parked leaf holds exactly the same
   `mark_pending` / virtual-loss state as a leaf sitting in the eval queue,
   and releases it exactly once (hard backup, eval backup, or bail-then-eval
   backup). No double backups, no leaked pending marks. Slot `in_flight`
   accounting (continuous path) must include parked leaves.
5. **Never run cargo on Windows in this worktree** (Windows artifacts ICE
   the Linux rustc). Build/test ONLY with the WSL command lines below.
6. Do NOT git-commit; leave the working tree for review.

## Seam anchors (verified against 24ef8961)

- Gated-leaf hook: `RustSearch::tss_deep_leaf(&mut self, state, hash) ->
  Option<HardValue>` at `packages/hexfield_eq/rust/src/tree.rs:1148`. Its
  async branch enqueues (`TssAsyncHandle::try_enqueue`) and returns `None`.
  For parking you need a tri-state instead — suggested:

  ```rust
  pub enum TssLeafRoute { Hard(HardValue), Parked, Miss }
  ```

  Keep the existing `Option<HardValue>` signature working for flag-off (or
  refactor all callers to the enum with `Miss` ≡ today's `None`; your call,
  but flag-off behavior must be bit-identical). `Parked` is returned when
  park is on AND the request was accepted by the queue; a shutdown/closed
  pool returns `Miss` (leaf takes the plain GPU eval — never park without a
  request in flight).
- Callers (both build the `RustLeaf` on miss):
  - lockstep `select_leaf_batch` arm at `search.rs:~2121-2147`
  - continuous `select_continuous_leaves` arm at `search.rs:~2215-2241`
- Existing consumption/backup pattern to reuse for pen resolution: the
  hard-leaf arms right above those sites (`backup_virtual(&path, leaf_player,
  hard.value(), virtual_loss, None)`).
- Pool: `packages/hexfield_eq/rust/src/tss_async.rs` — `RequestQueue`
  (LIFO + oldest-eviction, cap 16384), `TssAsyncPool` (workers, generations,
  alarms, `quiesce_for_telemetry`), worker loop with `finish_one` in-flight
  tracking.
- Drains: `wire_tss_async` / `drain_tss_async` (continuous, loop-top at
  `search.rs:~1245`), `wire_tss_async_searches` / `drain_tss_async_searches`
  (lockstep, `run_searches_to_targets`). Memo writes:
  `apply_tss_async_response{,_stale}` in tree.rs.
- Scheduler loops to wire: `run_continuous` lockstep loop (PRODUCTION),
  `run_continuous_pipeline_depth2` (env-gated, off in production — wire it
  anyway, same semantics), `run_searches_to_targets` (eval/arena lockstep).
- Flags plumbing pattern: `Divergences` in tree.rs (fields + defaults),
  `KNOWN_DIVERGENCE_KEYS` + `resolve_divergences` in search.rs (WITH range
  validation — follow the existing tss_* validation style),
  `SelfplayConfig` + `build_divergence_overrides` in
  `packages/hexfield_eq/python/hexfield_eq/config.py`.
- Telemetry route: `TssCounters` (tree.rs) → payload `tss_counters` →
  `diagnostics.tss` dict (search.rs `to_pydict`) → accumulation in
  `ContinuousDriver.__call__` + `stats()` + `_merge_epoch_diag` int_keys in
  `packages/hexfield_eq/python/hexfield_eq/selfplay.py`.

## Design decisions (follow these)

### Pen

A scheduler-owned holding pen per loop (continuous: one pen for all slots,
entries tagged with slot index; lockstep: tagged with search index). Entry =
the `RustLeaf` (+ slot/search tag) + enqueue `Instant` + the generation it
was parked under. Resolution pass runs right after the existing pool drain
each iteration:

1. Look up the leaf's `state_hash` in the owning search's memo.
   - `Done` with binding == the leaf's position and a tier-consumable hard
     value → hard backup via the existing pattern; remove from pen. Bump
     `park_hard`. (Reuse `tss_async_descent_hard`-style full-binding
     re-check — never consume on hash alone.)
   - `Done` decided-but-not-consumable at this tier, or `Done(Unknown)` →
     release to the eval queue; bump `park_released`.
2. `now - parked_at > timeout` → release to eval queue; bump `park_bailed`.
3. Move advance / slot rebind / search drop: any pen entries belonging to a
   slot whose search is replaced/advanced/cleared MUST be handled — their
   node ids are invalidated by `advance_root`. Continuous move completion
   already gates on `in_flight == 0`; counting parked leaves in `in_flight`
   (constraint 4) makes this safe by construction: a move cannot complete
   while its slot still has parked leaves, so resolve-or-bail must precede.
   Verify this invariant and add a debug assertion.
4. On loop exit (scheduler end), any remaining pen entries are released to
   the eval path or backed up before return — the pen must be empty when the
   scheduler returns (the existing tail quiesce runs after).

Timing note: with park on, prefer resolving the pen BEFORE the flush
decision so a resolved leaf's release can join the current flush.

### Queue order under park

Parked requests must not starve: when the pool is constructed with park
enabled, `RequestQueue` serves OLDEST-first (FIFO) and NEVER evicts
(the bail timeout replaces eviction as the overload valve). Park disabled →
today's LIFO + oldest-eviction, unchanged. Decide at pool construction
(a `park: bool` constructor arg from the divergences that created the pool).

### Dynamic workers

- `tss_solver_async_threads` stays the BASE worker count (default 8).
- New `tss_solver_async_threads_max` (u32, default 0 = auto). Auto resolves
  to `clamp(available_parallelism - 6, base, 24)`. Validation: 0, or
  `base..=64`.
- Scale-up trigger: on `push`, if queue depth > 2 × current workers and
  workers < max, spawn one worker (spawn outside the queue lock; an atomic
  or mutex-guarded worker registry on the pool — careful: `workers:
  Vec<JoinHandle>` is behind `&self` on push paths, so use interior
  mutability, e.g. `Mutex<Vec<JoinHandle>>`; Drop joins them all as today).
- Never shrink (idle workers cost ~nothing: 50ms condvar timeouts).
- Count spawns in a pool stat exposed through the drain into
  `TssCounters.async_workers_spawned` (or equivalent) so it reaches epoch
  telemetry.

### Flags (all via the Divergences route, validated)

| flag | type | default | validation |
|---|---|---|---|
| `tss_solver_park` | bool | false | — |
| `tss_solver_park_timeout_ms` | u32 | 100 | 1..=5000 |
| `tss_solver_async_threads_max` | u32 | 0 (auto) | 0 or base..=64 |

`tss_solver_park` requires `tss_solver_async` (reject the combination
park=true, async=false at resolve time with a clear error). When park is on,
`tss_solver_async_inline_16` is ignored (comment at the read site + runbook).

### Telemetry (new TssCounters fields, all through to the epoch JSON)

`park_parked` (leaves parked), `park_hard` (resolved to hard backup),
`park_released` (resolved unknown/non-consumable → eval), `park_bailed`
(timeout → eval), `park_wait_ms_sum` + `park_wait_ms_max` (resolution
latency incl. bails), `async_workers_spawned`. Add to the Python
accumulation + `stats()` block + `_merge_epoch_diag` int_keys (max-merge for
`park_wait_ms_max`).

## Definition of done (run all of these; all must pass)

```bash
# non-python suite
wsl -e bash -c 'source ~/.cargo/env; cd /mnt/e/Hexo-BotTrainer-hexgt/.claude/worktrees/orbit-channels-equivariance-ee23df && CARGO_TARGET_DIR=packages/hexfield_eq/rust/target-wsl cargo test -p hexfield_eq'
# python-feature suite (tree/search/async)
wsl -e bash -c 'source ~/.cargo/env; cd /mnt/e/Hexo-BotTrainer-hexgt/.claude/worktrees/orbit-channels-equivariance-ee23df && CARGO_TARGET_DIR=packages/hexfield_eq/rust/target-wsl-py PYO3_PYTHON=/root/.venvs/hexfield-dev/bin/python RUSTFLAGS="-L /usr/lib/x86_64-linux-gnu -C link-arg=-lpython3.12" cargo test -p hexfield_eq --features python'
# rebuild extension + pytest
wsl -e bash -c 'source ~/.cargo/env; source /root/.venvs/hexfield-dev/bin/activate; cd /mnt/e/Hexo-BotTrainer-hexgt/.claude/worktrees/orbit-channels-equivariance-ee23df && CARGO_TARGET_DIR=packages/hexfield_eq/rust/target-wsl-py maturin develop -m packages/hexfield_eq/Cargo.toml && PYTHONPATH=packages/hexfield_eq/python:packages/hexo_runner/python:packages/hexo_models/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python /root/.venvs/hexgt-build/bin/python -m pytest tests/test_hexfield_eq_tss_shadow.py tests/test_hexfield_eq_window_streaming.py -q'
```

New tests required (in addition to every existing test passing unmodified —
the golden digest test especially):

1. Cargo (python feature): pen unit/integration —
   - park-on `run_continuous`-level test in the existing shadow-test style
     is Python's job; in Rust, test the pen mechanics directly where
     practical: a parked leaf resolving to a consumable hard backs up
     exactly once and releases its pending mark; a bail releases to eval;
     the pool in park mode serves FIFO and never evicts; dynamic spawn
     fires when depth > 2×workers and respects max.
   - a no-deadlock test: park on, pool with a solver that always returns
     Unknown (or positions with no proof) and a tiny timeout — the loop
     terminates with `park_bailed > 0` and correct visit counts.
2. pytest (`tests/test_hexfield_eq_tss_shadow.py`, follow
   `test_async_pool_routes_solves_end_to_end_and_stays_sound`):
   `test_park_first_touch_consumption_end_to_end` — park on (async on,
   threads 4, timeout 200ms) over the threat fixture games: assert
   `park_parked > 0`, `park_hard > 0`, `deep_verify_failed == 0`,
   `park_bailed / park_parked < 0.10`, digest diverges from the flag-off
   golden (consumption changes play), and flag-off digest still matches
   golden.

Also update `docs/TSS_RUNBOOK.md`: new flag rows, the park rung entry
(supersedes inline tier), watch items (`park_bailed` ≈ 0 is the health
signal; sustained bails ⇒ raise threads_max or lower node cap), and the
threads_max sizing note.

## Out of scope

- Solver node-cap changes (cap 500 is a separate, config-only lever).
- Zone/commutation machinery (measured null; flag stays off).
- Bit-reproducibility under park (worker cache warmth is scheduling-
  dependent; practical first-touch consistency is the goal — document).
- Any change to serve/inference paths.
