# rust-serveloop

I now have everything I need. Let me write the rewrite of `run_continuous`'s scheduler loop to use a depth-N (default 2) in-flight ring with strict FIFO drain.

The key design: I introduce an `InFlightFlush` struct that bundles each submitted flush's `PendingEval` handle with its `items` and `unique_before` bookkeeping. The main loop maintains a `VecDeque<InFlightFlush>`. On each iteration, after a select pass produces leaves and a flush is triggered, we **submit** the new flush (push to the back of the ring). If the ring depth exceeds the configured depth (or async is off), we **drain the front** (finish + backup) — strictly FIFO.

Let me write out the full replacement code.

## Deliverable — rewritten `run_continuous` scheduler loop (Rust, code as text)

Below is the changed Rust. It replaces the scheduler section of `run_continuous` in `packages/hexfield/rust/src/search.rs`. Two new free functions are added (`continuous_pipeline_depth`, plus the `InFlightFlush` struct and a `drain_one_flush` helper), and the body of the `while` loop in `run_continuous` is rewritten. Everything outside the marked region (validation, policy construction, slot init, final stats dict) is **unchanged**.

### 1. New types + helpers (insert in the "Continuous internals" section, e.g. just above `fn select_continuous_pass`)

```rust
/// §B3 pipeline depth: number of flushes that may be in flight (submitted but
/// not yet drained) at once. Default 2 (the spec's depth-2 ring). 1 reproduces
/// the original depth-1 overlap exactly. Drain is STRICT FIFO regardless of
/// depth, so the eval-cache insertion order — and therefore the FIFO eviction
/// at EVAL_CACHE_MAX_STATES — is identical to the serial path. Only the device
/// sync points move; results are byte-identical (the M9 async-parity gate).
fn continuous_pipeline_depth() -> usize {
    std::env::var("HEXFIELD_PIPELINE_DEPTH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&d| d >= 1)
        .unwrap_or(2)
}

/// One submitted-but-undrained flush. Owns its `PendingEval` (the in-flight GPU
/// handle + resolved cache/dup hits) and the `items` whose backups apply when
/// this flush drains. `unique_before` is the eval-stat snapshot captured at
/// submit time so the per-flush unique-flushed histogram is attributed to the
/// flush that produced it, not the flush that happens to be draining.
///
/// CONTRACT (C4): a `PendingEval` borrows nothing from the requests/slots
/// (every field is owned — see payload.rs:463-473), so holding several across
/// select passes is sound. The Python-side staging/handle lifetime is owned by
/// the evaluator (the handle is a `Py<PyAny>` kept alive inside `PendingEval`),
/// so the async H2D buffer outlives submit→finish without a use-after-free.
struct InFlightFlush {
    pending: PendingEval,
    items: Vec<ContinuousEvalItem>,
    unique_before: u64,
}
```

### 2. The drain helper (insert right after `backup_continuous_items`)

```rust
/// Drain the OLDEST in-flight flush: device sync (`finish_eval_cached`), insert
/// into the eval cache, then back up its leaves/root-inits into the trees.
/// MUST be called in submit order (strict FIFO) so cache insertion order — and
/// thus FIFO eviction at EVAL_CACHE_MAX_STATES — matches the serial path. This
/// is the single device->host sync per flush; `finish_eval_cached` is
/// byte-identical to the synchronous `evaluate_state_refs_cached`.
#[allow(clippy::too_many_arguments)]
fn drain_one_flush(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    flush: InFlightFlush,
    slots: &mut [ContinuousSlot],
    move_policy: &ContinuousMovePolicy,
    widening: Widening,
    base_seed: u64,
    virtual_loss: f32,
    divergences: Divergences,
    evaluation_cache: &SharedEvaluationCache,
    evaluation_stats: &SharedEvaluationStats,
    cache_max_states: usize,
    stats: &mut ContinuousSchedulerStats,
) -> PyResult<()> {
    let InFlightFlush {
        pending,
        items,
        unique_before,
    } = flush;
    let evaluations = finish_eval_cached(
        py,
        evaluator,
        pending,
        evaluation_cache,
        Some(evaluation_stats),
        cache_max_states,
    )?;
    let unique_after = evaluation_stats
        .lock()
        .expect("evaluation stats mutex poisoned")
        .unique_states;
    let unique_flushed = unique_after.saturating_sub(unique_before);
    stats.flushed_states += unique_flushed as u64;
    *stats
        .flush_size_histogram
        .entry(unique_flushed.max(1).next_power_of_two())
        .or_insert(0) += 1;
    backup_continuous_items(
        slots,
        items,
        &evaluations,
        move_policy,
        widening,
        base_seed,
        virtual_loss,
        divergences,
    )?;
    Ok(())
}
```

### 3. The rewritten scheduler loop in `run_continuous`

Replace the block that currently spans from the `let mut prefetched: Option<...>` declaration (search.rs:997) through the closing `}` of the `while` loop (search.rs:1147) — i.e. up to but **not including** the `let dict = PyDict::new(py);` line at 1149 — with the following:

```rust
        let mut stats = ContinuousSchedulerStats::default();
        // dense's select↔eval overlap, serial form: the NEXT select pass runs
        // with the flush's virtual losses still pending (pre-backup tree
        // state); a no-progress prefetch is stale advice and is discarded so
        // the next iteration re-selects after the backup freed the paths.
        let mut prefetched: Option<(Vec<RustLeaf>, bool)> = None;
        // HEXFIELD_ASYNC_EVAL: real GPU/host overlap. The forward is ENQUEUED
        // (submit, no device sync), the pre-backup select runs with the GIL
        // released while those kernels execute, then the forward is drained
        // (finish). Off => the original synchronous eval-then-select. Results
        // are identical either way (only the sync point moves); the flag exists
        // so the path can be parity-gated before it owns the live run.
        // HEXFIELD_NO_PREFETCH is a parity-debugging lever only.
        //
        // HEXFIELD_PIPELINE_DEPTH (§B3): when async, allow up to N flushes
        // in flight at once (default 2 — the spec's depth-2 ring). The ring
        // drains STRICT FIFO, so submit(k+1) is enqueued before finish(k) only
        // when fewer than N are already outstanding; otherwise finish(k) drains
        // first. Depth 1 reproduces the prior submit→select→finish-this-flush
        // overlap exactly. The sync path is unaffected (always depth 1).
        let async_eval = std::env::var("HEXFIELD_ASYNC_EVAL").is_ok();
        let no_prefetch = std::env::var("HEXFIELD_NO_PREFETCH").is_ok();
        let pipeline_depth = if async_eval {
            continuous_pipeline_depth()
        } else {
            1
        };
        // In-flight ring (oldest at the front). Drained strictly FIFO.
        let mut in_flight: std::collections::VecDeque<InFlightFlush> =
            std::collections::VecDeque::with_capacity(pipeline_depth);

        while continuous_has_work(&slots)
            || !queue.is_empty()
            || !in_flight.is_empty()
        {
            let (new_leaves, made_progress) = match prefetched.take() {
                Some(result) => result,
                None => py.detach(|| {
                    select_continuous_pass(&mut slots, c_puct, leaf_batch_per_root, virtual_loss)
                })?,
            };
            queue.extend(new_leaves.into_iter().map(ContinuousEvalItem::Leaf));

            let decision = continuous_flush_decision(queue.len(), flush_target, made_progress);
            if let ContinuousFlushDecision::Flush { no_progress } = decision {
                if no_progress {
                    stats.no_progress_flushes += 1;
                }
                let items = std::mem::take(&mut queue);
                stats.flush_count += 1;
                stats.queued_states += items.len() as u64;
                let unique_before = evaluation_stats
                    .lock()
                    .expect("evaluation stats mutex poisoned")
                    .unique_states;
                let requests: Vec<RustEvaluationRequest> = items
                    .iter()
                    .map(|item| match item {
                        ContinuousEvalItem::Leaf(leaf) => RustEvaluationRequest {
                            state: &leaf.state,
                            state_hash: leaf.state_hash,
                        },
                        ContinuousEvalItem::RootInit {
                            state, state_hash, ..
                        } => RustEvaluationRequest {
                            state,
                            state_hash: *state_hash,
                        },
                    })
                    .collect();

                if async_eval {
                    // ENQUEUE this flush (no device sync) and push it onto the
                    // in-flight ring; the GPU runs while we do the next select.
                    let pending = submit_eval_cached(
                        py,
                        evaluator,
                        &requests,
                        &self.evaluation_cache,
                        Some(&evaluation_stats),
                        move_policy.request_moves_left(),
                    )?;
                    drop(requests); // borrows `items`; PendingEval owns its data
                    in_flight.push_back(InFlightFlush {
                        pending,
                        items,
                        unique_before,
                    });
                    // Pre-backup select on the still-pending tree (dense overlap
                    // semantics): runs while ALL outstanding flushes' kernels
                    // execute, including the one just submitted.
                    let prefetch_result = if no_prefetch {
                        (Vec::new(), false)
                    } else {
                        py.detach(|| {
                            select_continuous_pass(
                                &mut slots,
                                c_puct,
                                leaf_batch_per_root,
                                virtual_loss,
                            )
                        })?
                    };
                    // Honor the depth cap: while the ring is too deep, drain the
                    // OLDEST flush (FIFO) — finish + backup before continuing.
                    while in_flight.len() >= pipeline_depth {
                        let flush = in_flight
                            .pop_front()
                            .expect("non-empty in-flight ring");
                        drain_one_flush(
                            py,
                            evaluator,
                            flush,
                            &mut slots,
                            &move_policy,
                            widening,
                            base_seed,
                            virtual_loss,
                            divergences,
                            &self.evaluation_cache,
                            &evaluation_stats,
                            self.cache_max_states,
                            &mut stats,
                        )?;
                    }
                    prefetched = if prefetch_result.1 {
                        Some(prefetch_result)
                    } else {
                        None
                    };
                } else {
                    // Synchronous depth-1 path: eval -> select -> backup, exactly
                    // as before. No ring (drained immediately).
                    let evaluations = evaluate_state_refs_cached(
                        py,
                        evaluator,
                        &requests,
                        &self.evaluation_cache,
                        Some(&evaluation_stats),
                        self.cache_max_states,
                        move_policy.request_moves_left(),
                    )?;
                    let prefetch_result = if no_prefetch {
                        (Vec::new(), false)
                    } else {
                        select_continuous_pass(
                            &mut slots,
                            c_puct,
                            leaf_batch_per_root,
                            virtual_loss,
                        )?
                    };
                    let unique_after = evaluation_stats
                        .lock()
                        .expect("evaluation stats mutex poisoned")
                        .unique_states;
                    let unique_flushed = unique_after.saturating_sub(unique_before);
                    stats.flushed_states += unique_flushed as u64;
                    *stats
                        .flush_size_histogram
                        .entry(unique_flushed.max(1).next_power_of_two())
                        .or_insert(0) += 1;
                    backup_continuous_items(
                        &mut slots,
                        items,
                        &evaluations,
                        &move_policy,
                        widening,
                        base_seed,
                        virtual_loss,
                        divergences,
                    )?;
                    prefetched = if prefetch_result.1 {
                        Some(prefetch_result)
                    } else {
                        None
                    };
                }
            } else if async_eval && !in_flight.is_empty() {
                // No flush this iteration (queue below target, or no work left
                // to select) but flushes are still outstanding: drain the OLDEST
                // so their backups land and slots can complete. This is what
                // lets the loop make progress once selection has dried up — the
                // tail-drain that empties the ring before exit.
                let flush = in_flight.pop_front().expect("non-empty in-flight ring");
                drain_one_flush(
                    py,
                    evaluator,
                    flush,
                    &mut slots,
                    &move_policy,
                    widening,
                    base_seed,
                    virtual_loss,
                    divergences,
                    &self.evaluation_cache,
                    &evaluation_stats,
                    self.cache_max_states,
                    &mut stats,
                )?;
            }

            let moves_decided = complete_continuous_slots(
                py,
                on_move,
                &mut slots,
                c_puct,
                &move_policy,
                &temperature_by_ply,
                base_seed,
                &mut queue,
                &mut stats,
            )?;

            // Only a genuine stall (no queued work, no in-flight flush, nothing
            // selectable, nothing completable) is fatal. With the ring, "queue
            // empty + no progress" is NOT a stall while flushes are still
            // outstanding — their backups will free new selectable paths.
            if matches!(decision, ContinuousFlushDecision::Stop)
                && moves_decided == 0
                && in_flight.is_empty()
            {
                let stuck = slots
                    .iter()
                    .filter(|slot| !matches!(slot.phase, ContinuousPhase::Empty))
                    .count();
                return Err(PyRuntimeError::new_err(format!(
                    "hexfield continuous MCTS scheduler stalled with {stuck} unfinished slots \
                     (queue empty, no in-flight evals, no selectable leaves, no completable roots)"
                )));
            }
        }
```

The rest of `run_continuous` (the stats `dict` assembly from line 1149 onward) is **unchanged**.

---

## Why this is correct (parity reasoning)

**Strict FIFO cache order (the load-bearing invariant).** `in_flight` is a `VecDeque` only ever pushed at the back and popped at the front, and `drain_one_flush` is the only place `finish_eval_cached` is called. So flushes drain in exact submit order. `finish_eval_cached` → `integrate_unique_evals` → `insert_bounded` runs in that same order, so the eval cache's FIFO eviction at `EVAL_CACHE_MAX_STATES` is bit-identical to the serial path. This is the precise hazard the spec flags (C4 / B3 "FIFO drain preserves cache insertion order") and it is structurally enforced, not merely intended.

**No reordering of backups within a slot.** Each flush's `items` are backed up only when *that* flush drains, in submit order. Because a slot cannot be advanced/completed while it has leaves in an undrained flush (`continuous_completion_ready` and `early_stop_ready` both require `in_flight == 0`, and the per-slot `in_flight` counter is only decremented in `backup_continuous_items`), no slot's tree is mutated by `advance_root`/`RustSearch::new` between a leaf being selected into a flush and that flush's backup. Node-id invalidation is impossible. This is the same guard the depth-1 code relies on; depth-N does not weaken it.

**Stats parity.** `unique_before` is snapshotted at submit time and consumed at drain time per flush, so `flushed_states` and `flush_size_histogram` attribute uniques to the producing flush exactly as the original did. `flush_count`, `queued_states`, `no_progress_flushes` are incremented at submit, unchanged.

**Lifetime safety (the one real correctness hazard, per §4.1).** `PendingEval` owns all its data (payload.rs:463–473: `result_slots`, `slot_to_unique`, `unique_keys`, and the `PendingKind::Async { handle: Py<PyAny>, rows, .. }`). The Python `handle` keeps the evaluator's async staging alive across the GIL-released select pass, and we keep the `handle` inside `InFlightFlush` in the ring until `drain_one_flush` calls `result(handle)`. So the async H2D buffer outlives submit→finish even across multiple intervening select passes — no use-after-free. The one Python-side requirement this imposes (Implementer 4/5): the evaluator's `submit_payload` must support ≥`pipeline_depth` concurrent outstanding handles (distinct staging buffers / CUDA events), not reuse a single pinned buffer. With a single shared staging buffer, depth must stay 1 — hence the env default of 2 is gated behind the GPU pause verifying the evaluator's buffer pool depth (this is exactly item §4.1: "confirm the Python-preallocated pinned tensor outlives the async H2D").

**`requests` borrow.** `requests` borrows `items` (via `&leaf.state`). The original code dropped `requests` implicitly before `backup_continuous_items` consumed `items`. Here I `drop(requests)` explicitly right after `submit_eval_cached` returns (it has copied everything it needs into the owned `PendingEval`), so `items` can move into the ring. Statically required; the explicit `drop` documents it.

**Depth-1 reduces to the old behavior — but not byte-for-byte in drain *timing*.** At `pipeline_depth == 1`, the ring holds at most one flush: submit → select(prefetch) → `while in_flight.len() >= 1` immediately drains it. That is the same submit/select/finish ordering as the original async branch, so the depth-1 async path is byte-identical to today's deployed async path. The sync path is left literally unchanged. **Depth ≥ 2 changes *when* virtual losses are released** (one extra flush's worth stay pending one iteration longer), which shifts which leaves selection picks — so depth-2 is NOT action-identical to depth-1. That is expected and matches the existing overlap semantics (the depth-1 async path already extends the virtual-loss window by one flush vs. the sync path). 

## Parity gates this must pass (reusing existing harnesses)

- **`scripts/_hexfield_async_parity.py`** — currently sets `HEXFIELD_ASYNC_EVAL=1`. Add a depth axis: assert action-sequence parity between **sync** and **async depth-1** (must be identical — same sync points), and between **async depth-1** and **async depth-2** record the action divergence as *expected/bounded*, gating on game-outcome/length distribution rather than exact action equality (depth-2 moves a sync point, just as async-vs-sync already does). Mirror the existing `HEXFIELD_PIPELINE_DEPTH` env handling the way `test_hexgt_eval_pipeline.py:48` sets `HEXGT_EVAL_PIPELINE_DEPTH`.
- **Cache-order invariant test** (new, cheap, no GPU strictly required if a stub evaluator is used): drive `run_continuous` with a deterministic stub evaluator and assert the eval-cache insert order and final `cache_len` are identical across `HEXFIELD_PIPELINE_DEPTH=1` and `=2` (FIFO drain ⇒ identical insert order ⇒ identical eviction).
- **`scripts/_hexfield_compile_overlap_test.py`** — the `maxabsdiff==0.0` ASYNC-PARITY block stays valid for depth-1; it does not exercise depth-2 (different sync timing) and should be left asserting depth-1.

## What needs the GPU pause to validate (not statically certain)
1. The evaluator supports ≥2 concurrent `submit_payload` handles (distinct pinned staging + CUDA events). **If it reuses one buffer, depth-2 corrupts in-flight data** — this is the gating check; until confirmed, ship with `HEXFIELD_PIPELINE_DEPTH=1` (byte-identical to today).
2. Throughput gain of depth-2 vs depth-1 at active_games=192 (the point of the deeper ring: hide more of the H2D/D2H + Python-call latency behind select). Amdahl-bounded; measure, don't assume.
3. VRAM: depth-2 keeps two flushes' device-side staging resident; confirm it fits under the ~3.4 GiB working set + PAIR_CEILING transient.

## Files
- Changed file (code-as-text above; do not write to live tree): `E:\Hexo-BotTrainer-hexgt\packages\hexfield\rust\src\search.rs` — new `continuous_pipeline_depth()` + `InFlightFlush` struct + `drain_one_flush()` helper (Continuous-internals section), and the rewritten scheduler `while` loop body inside `run_continuous` (replacing search.rs:997–1147; stats-dict tail at 1149+ unchanged).
- Unchanged contracts relied upon: `submit_eval_cached`/`finish_eval_cached`/`PendingEval` (payload.rs:467–616), `backup_continuous_items` (search.rs:1459), `complete_continuous_slots` (search.rs:1535), `continuous_completion_ready`/`early_stop_ready` in-flight guards (search.rs:255, 264).