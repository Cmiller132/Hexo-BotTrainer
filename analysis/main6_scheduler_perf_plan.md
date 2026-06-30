# main_6 continuous-scheduler performance plan (locked design)

Author: lead engineer. Scope: four changes to the continuous self-play scheduler
in the **worktree** `E:/Hexo-BotTrainer-gumbel` only. Never touch the main tree or
the installed `hexgt-build` packages.

## Problem statement (measured)

Per-pass loop in `packages/hexfield/rust/src/search.rs::run_continuous` (~L1021):
`select_continuous_pass` (parallel rayon) → flush to GPU evaluator (`submit/finish_eval_cached`)
→ `backup_continuous_items` (SERIAL, L1520) → `complete_continuous_slots` (SERIAL, L1605).

Symptoms: main thread ~68% busy while rayon workers idle, GPU ~34% util, mean GPU
batch ~213 vs `flush_target`=1024. Root cause: `backup` and `complete` run serially
on the main thread; the GPU drains and starves while they execute, and select can't
refill until backup frees the virtual-loss paths.

The four changes, in dependency order:
1. `active_games` 96 → 192 (config-only) — bigger batches/more parallel select; **stresses** the serial backup/complete that #2/#3 fix.
2. Parallelize `backup_continuous_items` per-slot. **MUST be byte-identical.**
3. Parallelize payload-**building** in `complete_continuous_slots` per-slot; `on_move` stays serial in slot order. **MUST be byte-identical.**
4. Double-buffer the GPU eval (one eval in flight while host selects next batch). **NOT byte-identical → flag-gated, default OFF.**

Ship order: **#1 + #2 + #3 together** (all parity-safe), validate, then **#4** behind a flag.

---

## Change 1 — `active_games` 96 → 192 (config only)

### Files / functions
- `configs/hexfield_main_6.toml` L155: `active_games = 96` → `active_games = 192`.
- No code change. `active_games` flows `selfplay.py` L426 `slots = min(sp.active_games, remaining)` → `ContinuousDriver(active_limit=slots)` → number of `ContinuousSlot`s passed into `session.run_continuous`.
- Leave `active_root_limit = 192`, `virtual_batch_size = 4`, `flush_target = 1024` unchanged for the first bench; `active_root_limit` already ≥ new `active_games`, and the eval chunk cap (`EVAL_CHUNK_STATES`=1024 in payload.rs) still single-chunks a 192×4=768 worst-case flush.

### Expected effect
- 2× slots → ~2× selectable leaves per pass → the queue reaches `flush_target` (1024) on real progress instead of `no_progress` early flushes → mean GPU batch rises from ~213 toward 1024, raising GPU util.
- More independent slots → more parallel work for the rayon select pass (and for the new parallel backup/complete).
- **Stress note:** doubling slots roughly doubles the per-pass work in the *serial* backup and complete phases. Without #2/#3 this would just move the main-thread bottleneck, not remove it — which is exactly why #1 ships **with** #2/#3.

### Data-race argument
None. Pure config scalar; each slot still owns its own `&mut` tree.

### GIL handling
N/A.

### Determinism argument
`active_games` only changes how many independent games run; per-slot RNG is seeded by
`(base_seed, game_key, ply)` (see `mix_seed` call sites), not by slot count or order.
Same games + same seed → same per-game stream regardless of how many slots co-run.

### Parity-safety argument
The parity test (`tests/test_hexfield_continuous_parity.py`) pins its own small config
(`active_root_limit=16`, `flush_target=24`) and a fixed game set; production `active_games`
does not enter it. Safe.

### Test that catches a regression
- Bench wall-clock pos/s + `flush_size_histogram`/mean batch from `ContinuousSchedulerStats`
  (logged to the per-epoch `hexfield.selfplay.epoch_*.json`). Mean batch should rise; if
  pos/s *drops*, the serial phases are the new bottleneck (expected if #2/#3 absent).
- VRAM watch: 192×4 in-flight states + larger trees must fit. Abort criterion below.

### Go / no-go
GO together with #2/#3. NO-GO alone if it regresses pos/s or OOMs (then fall back to 128).

---

## Change 2 — Parallelize `backup_continuous_items` per-slot (byte-identical)

### Files / functions
- `packages/hexfield/rust/src/search.rs::backup_continuous_items` (L1520–1602). Rewrite the
  body; keep the signature.
- Touches `RustSearch::add_node_from_eval`, `mark_pending`, `backup_virtual` (tree.rs) — no
  changes to those, only how they are dispatched.

### Design
Today: one serial `for (item, evaluation) in items.zip(evaluations)`. Two item kinds:
`Leaf{root_index,…}` and `RootInit{slot_index,…}`. Both index a slot.

New structure (deterministic, two stages):
1. **Bucket by slot, preserving in-flush order.** Walk `items.into_iter().zip(evaluations)`
   once, pushing each `(local_seq, item, Arc<eval>)` into `per_slot[slot_idx]` where
   `slot_idx` = `leaf.root_index` for Leaf / `slot_index` for RootInit. `local_seq` is the
   item's original index in the flush. Bucketing is a serial O(n) pass (cheap, no tree work).
2. **Process slots in parallel; serial within a slot.** Take a mutable, disjoint handle to
   each slot's tree and run, for that slot, the **exact same operations in the exact same
   in-flush order** as today. Use `slots.par_iter_mut().enumerate()` so rayon hands each
   closure exactly one `&mut ContinuousSlot` (disjoint borrows — the same pattern
   `select_continuous_pass` already uses at L1478). Each closure drains its bucket in
   `local_seq` order.

```rust
py.detach(|| {
    // stage 1: bucket (serial, cheap)
    let mut per_slot: Vec<Vec<(usize, ContinuousEvalItem, Arc<RustEvaluation>)>> =
        vec![Vec::new(); slots.len()];
    for (seq, (item, eval)) in items.into_iter().zip(evaluations.iter()).enumerate() {
        let s = match &item { Leaf(l) => l.root_index, RootInit{slot_index,..} => *slot_index };
        per_slot[s].push((seq, item, Arc::clone(eval)));
    }
    // stage 2: parallel across slots, serial within (already in seq order)
    slots.par_iter_mut().enumerate().try_for_each(|(slot_index, slot)| -> PyResult<()> {
        for (_seq, item, eval) in per_slot[slot_index].drain(..) {
            apply_backup_item(slot, slot_index, item, &eval, /* move_policy, widening, base_seed, virtual_loss, divergences */ )?;
        }
        Ok(())
    })
})?
```
`apply_backup_item` is the existing per-item body lifted verbatim (the Leaf arm and the
RootInit arm), with `slots[idx]` replaced by the closure's `slot`.

### Data-race argument (who owns what `&mut`)
- A Leaf/RootInit item targets **exactly one** slot (`root_index`/`slot_index`). Bucketing
  is total and disjoint: bucket `k` contains only items for slot `k`.
- `par_iter_mut` yields **disjoint** `&mut ContinuousSlot`s (rayon's safety guarantee). Each
  closure mutates only its own slot's `RustSearch` (`add_node_from_eval` appends to that
  search's `nodes`/`node_table`; `backup_virtual` mutates only nodes on that search's path).
  No two threads touch the same tree → no race. The shared `evaluations` are `Arc<…>`
  (immutable, read-only `Arc::clone`).
- Within a slot, items run **serially** in the closure, so the intra-slot hazards (node-id
  allocation order, overlapping backup paths from two leaves of the same slot) are
  identical to today's serial loop — no intra-slot parallelism is introduced.

### GIL handling
The entire backup body is **pure Rust** (no Python calls). Wrap the bucket+parallel section
in `py.detach(|| { … })` so the GIL is released for the whole phase (rayon workers must not
hold the GIL). The current serial loop runs while holding the GIL for no reason; releasing it
also lets the in-flight eval/select overlap better.

### Determinism argument
- Bucketing is order-preserving (`local_seq`), so within every slot items execute in the
  same in-flush order as the serial version.
- Across slots the only shared sink is the per-slot trees, which are **independent** — there
  is no cross-slot reduction whose order could vary. Two runs with the same seed produce the
  same buckets and the same per-slot sequences → bit-identical trees.
- `slot.in_flight` is mutated only by its own closure.

### Parity-safety argument
The serial and parallel forms perform the **same operations on the same data in the same
per-slot order**; only independent slots are reordered relative to each other, and they share
no mutable state. Result is byte-identical → `test_continuous_parity_full_machinery` and the
golden/parity suite stay green. This is the **same correctness argument already accepted for
`select_continuous_pass`** (L1474–1476 comment), applied to backup.

### Test that catches a regression
- `tests/test_hexfield_continuous_parity.py::test_continuous_parity_full_machinery` —
  byte-compares the full on_move stream (action ids, visit counts, visit-policy bytes,
  root_value) against dense_cnn. ANY tree-state divergence from a backup-order bug surfaces here.
- Plus a **focused multi-leaf-per-slot determinism test** (new, `tests/test_backup_parallel_determinism.py`):
  run `run_continuous` twice with the same seed and a config that guarantees ≥2 leaves per
  slot in a flush (`virtual_batch_size` ≥ 2, `flush_target` large, few slots); assert the two
  recorded streams are identical AND identical to a baseline captured with a `HEXFIELD_SERIAL_BACKUP=1`
  escape hatch (keep the old loop behind that env for one release as the parity oracle).
- Run cargo suite: expect **41 passed**.

### Go / no-go
GO only if parity test + 41 cargo tests are green AND the serial-vs-parallel stream
comparison is bit-identical. Otherwise NO-GO (keep serial).

---

## Change 3 — Parallelize payload-building in `complete_continuous_slots` (byte-identical)

### Files / functions
- `packages/hexfield/rust/src/search.rs::complete_continuous_slots` (L1605–1850). Split into
  two phases. `build_search_result_payloads` (L1999) is reused unchanged.

### Design
Today the body does, per slot, serially: (a) completion check + early-stop bookkeeping,
(b) build payload via `build_search_result_payloads` (**pure Rust** but currently runs under
the GIL), (c) `on_move.call1(...)` (Python, needs GIL), (d) apply the on_move response
(advance/replace/end) which mutates the slot tree.

Restructure into **build-parallel / dispatch-serial**:

**Phase A (parallel, GIL released):** for each slot, decide completion and — if complete —
build its payload + collect everything `on_move` needs into a plain Rust struct
`PreparedMove { slot_index, game_key, move_class, early, payload_fields…, action_id, init_sampled, stats_deltas }`.
This includes the visit-policy bytes, `pcr_full`, `policy_init`, `lcb_override`, the
Init-class prior sample, and `action_id`. Produce `Vec<Option<PreparedMove>>` indexed by slot
via `slots.par_iter().enumerate()` (note `par_iter`, **read-only** — payload building reads
the tree, does not mutate it). Wrap in `py.allow_threads`/`py.detach`.

Problem: `build_search_result_payloads` currently takes `py: Python` and returns a
`Py<PyAny>` (it builds a `PyDict`). To run it without the GIL we must **not** build Python
objects inside the parallel region. Two options — pick **Option A**:
- **Option A (chosen):** add a pure-Rust core `build_search_result_payload_native(search, baseline, temperature, seed, c_puct, forced_k) -> PayloadNative` that returns a Rust struct of plain
  bytes/scalars (the same values the dict gets). Run THAT in the parallel region. Then in the
  serial Phase B, convert `PayloadNative` → `PyDict` (cheap, GIL held) right before `on_move`.
  Keep the existing `build_search_result_payloads` as a thin wrapper around the native core so
  other callers (e.g. `search`/eval_arena) are untouched and the multi-search batch path stays
  identical.
- (Option B — building PyDicts in the parallel region under per-thread `Python::with_gil` —
  rejected: re-acquiring the GIL per slot serializes anyway and adds contention.)

**Phase B (serial, GIL held):** iterate `slot_index in 0..slots.len()` (same order as today).
For each `Some(PreparedMove)`: apply early-stop side effects to the slot (`early_stopped`,
`target_visits = completed_visits`, stats), convert native payload → `PyDict`, call
`on_move.call1((game_key, payload_dict))`, then apply the response (advance/replace/end)
exactly as today — including `advance_root`, re-init/noise, pushing `RootInit` to `queue`,
phase transitions, `in_flight = 0`. All slot mutation that depends on Python output stays here.

Care: early-stop bookkeeping currently mutates the slot **before** building the payload (it
sets `target_visits = completed_visits`, which changes `remaining_visits`). `build_search_result_payloads`
reads visit *counts* (`completed_visits` / edge visits), not `remaining_visits`, so the payload
is unaffected by the early-stop write — **verify by capturing `early`/`remaining` in Phase A and
asserting the payload built in A equals one built after the mutation** (covered by the parity
test). Keep early-stop **stat** mutation in Phase B to avoid mutating in the read-only `par_iter`.

### Data-race argument (who owns what `&mut`)
- **Phase A is read-only over slots** (`par_iter`, `&ContinuousSlot`): it reads each slot's
  `search` to build a payload and reads the completion predicate. No tree mutation, no shared
  mutable sink — each closure writes only its own `Option<PreparedMove>` slot in the output
  vec (disjoint by index via `enumerate`/`collect`). Safe.
- **Phase B is serial** on the main thread: it owns `&mut slots` and mutates one slot at a
  time. Same single-owner discipline as today. No race.

### GIL handling
- Phase A: `py.detach(|| slots.par_iter()...)` — pure Rust native payload core, GIL released,
  rayon-safe.
- Phase B: GIL held for the `PyDict` construction and every `on_move.call1`. `on_move` is a
  Python callback (single-threaded, GIL-bound) and **must** stay serial — no parallelism here.

### Determinism argument
- Phase A produces a per-slot result keyed by slot index; building is a pure function of
  `(search snapshot, baseline, temperature, seed, c_puct, forced_k)` — no cross-slot state.
- Phase B dispatches `on_move` in **slot-index order** (`0..len()`), identical to today, so the
  `on_move` record sequence is unchanged. `queue.push(RootInit)` ordering is preserved (same
  slot order). Same seed → same payloads → same dispatch order.

### Parity-safety argument
The on_move stream (move classes, action ids, visit counts, visit-policy bytes, root_value)
is byte-identical because (1) payloads are the same pure function as before, just computed off
the GIL, and (2) `on_move` is called in the same slot order with the same dicts. The
`Recorder` in the parity test compares this exact 8-tuple sequence → stays green.

### Test that catches a regression
- `test_continuous_parity_full_machinery` (the on_move stream byte-compare) — primary gate.
- New `tests/test_complete_parallel_determinism.py`: same-seed double run with several slots
  completing in the same flush; assert identical streams, and identical to a
  `HEXFIELD_SERIAL_COMPLETE=1` oracle (old path kept one release).
- Native-vs-dict equivalence unit test: assert `build_search_result_payload_native` →
  PyDict equals the old `build_search_result_payloads` dict field-for-field on a fixed search.
- 41 cargo tests green.

### Go / no-go
GO only if both parity-stream comparisons are bit-identical and 41 cargo tests pass. NO-GO
otherwise.

---

## Change 4 — Double-buffer the GPU eval (FLAG-GATED, default OFF, NOT byte-identical)

### Why it is not byte-identical
Double-buffering keeps eval N in flight while the host **selects N+1 and backs up N-1**. Select
N+1 therefore reads a tree state in which flush N's results have **not yet** been backed up
(extra virtual losses still pending, real values not yet integrated). That changes *which*
leaves get selected relative to the strict lockstep order → different (still search-faithful,
but different) streams. Hence it CANNOT pass the byte-identical parity test and MUST be opt-in.

### Flag / config
- New scheduler-mode knob, default OFF. There is no existing `scheduler_mode` plumbing, so add:
  - Env gate read in `run_continuous`: `let pipeline_depth2 = std::env::var("HEXFIELD_PIPELINE_DEPTH2").is_ok();`
    (mirrors the existing `HEXFIELD_ASYNC_EVAL` / `HEXFIELD_NO_PREFETCH` env pattern at L1019–1020 — lowest-risk, no Python signature churn).
  - Optional config surfacing later: `SelfplayConfig.scheduler_pipeline_depth: int = 1` in
    `config.py`, passed through `selfplay.py` as an env set before the call. Default 1 (OFF).
- Requires `HEXFIELD_ASYNC_EVAL=1` (depth-2 is a deepening of the async submit/finish pipeline);
  if depth2 is set without async, fall back to the lockstep path and log a warning.

### Files / functions
- `packages/hexfield/rust/src/search.rs::run_continuous` (L1021–1153): add an alternate loop
  body selected by `pipeline_depth2`. Keep the **existing lockstep loop untouched** as the
  default (byte-identical production path).

### Exact loop restructuring
Maintain **one** in-flight eval handle across iterations (`Option<(PendingEval, Vec<ContinuousEvalItem>)>`):

```
inflight: Option<(PendingEval, Vec<Item>)> = None;     // (handle, the items it will resolve)
loop until done:
  1. select N  -> new_leaves; queue.extend
  2. if flush_decision(queue) == Flush:
        items_N = take(queue)
        pending_N = submit_eval_cached(items_N)          // enqueue GPU forward, NO sync
        if let Some((pending_P, items_P)) = inflight.take():
            evals_P = finish_eval_cached(pending_P)       // drain the PREVIOUS eval (now done)
            backup_continuous_items(slots, items_P, evals_P)   // (parallel, change #2)
        inflight = Some((pending_N, items_N))             // N is now the in-flight one
  3. complete_continuous_slots(slots, ...)                // change #3
  4. (loop tail) when no more selectable work AND queue empty AND inflight.is_some():
        drain the final inflight: finish -> backup -> complete  (flush the pipeline)
```

So at steady state, between submit N and finish N the host does: select N+1, backup N-1,
complete N-1. The GPU is busy with N while the host works on N-1 — exactly one eval in flight,
one staged. (Depth is **2**: one on GPU, one being submitted. Not deeper — deeper buffering
compounds staleness and complicates the stall/Gumbel safety net.)

### Virtual-loss / search-faithfulness
- Virtual loss is the mechanism that keeps this faithful: when select N picks a leaf it calls
  `apply_virtual_visit` (tree.rs:1491), pessimistically penalizing that path so select N+1 does
  **not** re-pick the same leaf while N is in flight. `backup_virtual` (after finish) restores
  the virtual loss and folds in the real value. This is the SAME virtual-loss discipline the
  current single-buffer async path already relies on (submit → prefetch-select → finish); we
  are only extending the window by one flush. The per-slot `in_flight` counter and
  `leaf_batch_per_root` cap bound how many leaves a slot can have pending, preventing runaway
  selection on stale state.

### Avoiding lost / double backups
- **Exactly-once backup:** every flush's `items` are backed up exactly once — when its handle
  is drained by `finish_eval_cached`. The `inflight` slot holds *its own* `items` vector
  alongside the handle, so finish always pairs the right items with the right evals. A flush is
  never submitted twice (it's `take`n from the queue) and never finished twice (the `Option` is
  `take`n on drain).
- **No skipped backup at shutdown:** the loop tail explicitly drains a non-empty `inflight`
  before exit (the final flush). The existing stall/`Stop` and Gumbel `force_stuck_gumbel`
  rescue (L1155) runs **after** the pipeline is drained, so completion never fires on a slot
  whose last eval hasn't landed.
- **in_flight accounting:** select increments `slot.in_flight`; backup decrements it. Because
  backup N-1 happens before complete in the same iteration, completion still sees the correct
  `in_flight==0` predicate for slots whose evals have all landed; a slot with an eval still in
  the in-flight buffer has `in_flight>0` and is correctly **not** completed yet.
- **Cache safety:** `submit_eval_cached` dedups/cache-checks on the main thread before the next
  submit; `finish` of N-1 completes (inserts into cache) before `submit` of N+1 in the next
  iteration, so cache writes never interleave across two flushes (submit-N then finish-(N-1)
  are ordered within an iteration). Evals are `Arc<RustEvaluation>` (immutable), safe to hold
  across the staleness window.

### Data-race argument
- GPU forward for N runs on the device; the host mutates **only** slot trees (select N+1 via
  `par_iter_mut`, backup N-1 via change-#2 parallel). The in-flight handle holds on-device
  tensors / `Arc` results — not aliased with any slot tree. No host/device data race.
- Backup N-1 and select N+1 do **not** run concurrently (they are sequential statements in the
  loop body); only the GPU eval overlaps the host. So no two CPU phases touch the same tree at
  once.

### GIL handling
- `submit_eval_cached` / `finish_eval_cached` are Python-evaluator calls (GIL held). Select and
  backup run under `py.detach` (GIL released) so the GPU kernels and rayon overlap. Same GIL
  choreography as the current async path, just with the finish of N-1 moved to the top of the
  next iteration.

### Determinism argument
- Determinism is preserved **for a fixed pipeline depth**: with `HEXFIELD_PIPELINE_DEPTH2=1`
  and a fixed seed, the interleave (submit N, finish N-1, backup N-1, select N+1) is fully
  determined by the loop structure — no wall-clock-dependent branching. Two runs with the same
  seed + same flag produce identical streams. (They differ from depth-1, which is the whole
  point and why it's flagged.)
- The default path (flag OFF) is **untouched** and remains byte-identical to dense.

### Parity-safety argument
- Default OFF → production and all golden/parity tests run the existing lockstep loop →
  byte-identical, green.
- The depth-2 path gets its **own** parity-style test that pins the depth-2 stream to a
  committed golden (self-consistency / determinism), NOT to dense. It must NOT be compared to
  the byte-identical corpus.

### Test that catches a regression
- `test_continuous_parity_full_machinery` with the flag UNSET must stay green (proves default
  path untouched).
- New `tests/test_pipeline_depth2_determinism.py`: run twice with `HEXFIELD_PIPELINE_DEPTH2=1`
  + same seed; assert identical streams (determinism) and assert no lost moves (every started
  game reaches a terminal/`max_plies`, move counts match a depth-1 run's *count* even though
  the stream differs).
- Stall/Gumbel test: a tiny-budget Gumbel endgame run under depth2 must not deadlock (exercises
  the drain-then-rescue tail).
- A no-double-backup assertion: instrument `in_flight` to never go negative and total
  backups == total submitted leaves (debug assert under a test feature).

### Go / no-go
GO to **merge** if: default-OFF parity green, 41 cargo tests green, depth-2 determinism test
green, no deadlock. GO to **enable in production** only after a bench shows GPU util ↑ and
pos/s ↑ on the depth-2 run with strength unchanged (eval-arena Elo within noise vs depth-1).
NO-GO to enable if strength regresses or stalls appear; the flag stays OFF and shipping is
unaffected.

---

## Ordered summary

1. **#1 active_games 96→192** (config, `configs/hexfield_main_6.toml` L155). Bigger GPU
   batches + more parallel select; **stresses** serial backup/complete → ship with #2/#3.
2. **#2 parallel backup** (`backup_continuous_items`, search.rs L1520): bucket items by slot
   in-order, `par_iter_mut` across slots, serial within slot, `py.detach`. **Byte-identical.**
3. **#3 parallel payload build** (`complete_continuous_slots`, search.rs L1605): read-only
   `par_iter` builds native payloads off-GIL (new `build_search_result_payload_native`); then
   serial GIL-held `on_move` dispatch in slot-index order. **Byte-identical.**
4. **#4 double-buffer eval** (`run_continuous` loop, search.rs L1021): one eval in flight,
   one staged; finish N-1 at top of next iter; drain tail before exit. Virtual loss keeps it
   faithful; in-flight buffer holds its own items for exactly-once backup. **NOT byte-identical
   → `HEXFIELD_PIPELINE_DEPTH2` env, default OFF.**

Gates: #2 and #3 MUST be byte-identical (production non-gumbel parity); verified by
`test_continuous_parity_full_machinery` + serial-oracle escape hatches
(`HEXFIELD_SERIAL_BACKUP` / `HEXFIELD_SERIAL_COMPLETE`). #4 MUST be flag-gated, default OFF,
with the lockstep path left untouched.

---

## Established build / test / bench commands (worktree, isolated venvs)

**Build (release, dev venv — editable install puts the `.so` in the worktree):**
```
wsl.exe -e bash -lc "source /root/.venvs/hexfield-dev/bin/activate; export PATH=/root/.cargo/bin:\$PATH; cd /mnt/e/Hexo-BotTrainer-gumbel && maturin develop --release -m packages/hexfield/Cargo.toml 2>&1 | tail -8"
```

**Cargo tests (must stay green — expect 41 passed):**
```
wsl.exe -e bash -lc "source /root/.venvs/hexfield-dev/bin/activate; export PATH=/root/.cargo/bin:\$PATH; cd /mnt/e/Hexo-BotTrainer-gumbel; export PYO3_PYTHON=\$(which python); export RUSTFLAGS='-L /usr/lib/x86_64-linux-gnu -C link-arg=-lpython3.12'; cargo test -p hexfield --features python 2>&1 | grep -E 'test result:|error'"
```

**Parity / determinism pytest (worktree shim):**
```
wsl.exe -e bash -lc "source /root/.venvs/hexgt-build/bin/activate; export PYTHONPATH=/mnt/e/Hexo-BotTrainer-gumbel/packages/hexfield/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/dense_cnn_restnet/python; cd /mnt/e/Hexo-BotTrainer-gumbel; python -m pytest tests/test_hexfield_continuous_parity.py -q 2>&1 | tail -20"
```

**E2E / bench (torch venv, GPU free, run with run_in_background=true — a plain `&` is killed by WSL SessionLeader teardown):**
```
wsl.exe -e bash -lc "source /root/.venvs/hexgt-build/bin/activate; export PYTHONPATH=/mnt/e/Hexo-BotTrainer-gumbel/packages/hexfield/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/dense_cnn_restnet/python; export HEXFIELD_SUPPORT_RADIUS=4 HEXFIELD_CHANNELS=128 HEXFIELD_TRAIN_FLEX=1 HEXFIELD_SERVE_FLEX=1 HEXFIELD_ASYNC_EVAL=1 HEXFIELD_DEFER_DECODE=1 CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8; cd /mnt/e/Hexo-BotTrainer-gumbel; python -u -m hexo_train.cli.train_model <CONFIG> 2>&1 | tee <RUNDIR>/bench.log"
```
For the #4 bench, add `HEXFIELD_PIPELINE_DEPTH2=1` to the env. GPU is a single shared
resource — run benchmarks one at a time. Do NOT launch the systemd supervisor. Do NOT git push.
