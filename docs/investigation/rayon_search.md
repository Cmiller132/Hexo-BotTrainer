# OPTION 3 — Parallelise the Rust select pass with rayon

Design agent notes. Baseline = HEAD (commit 8fa37a0; build_attn_bias rewrite + active_games=64).
GPU free, run stopped. Do NOT modify packages/ or configs/ in the design phase.

## Target

`select_continuous_pass` (packages/hexfield/rust/src/search.rs:1428) is the only
large pure-CPU Rust chunk that is structurally a per-slot independent loop:

```
for (slot_index, slot) in slots.iter_mut().enumerate() {
    if !Active { continue }
    cap = leaf_batch_per_root - slot.in_flight; if cap==0 { continue }
    search = slot.search.as_mut(); if !search.needs_visits() { continue }
    (slot_leaves, progressed, added_in_flight) =
        select_continuous_leaves(search, slot_index, c_puct, cap, virtual_loss)?;
    slot.in_flight += added_in_flight;
    made_progress |= progressed;
    leaves.extend(slot_leaves);
}
```

`select_continuous_leaves` (search.rs:1375) operates ENTIRELY on its own
`&mut RustSearch` (the slot's tree): `select_pending_leaf` (tree.rs:570) does a
deterministic PUCT descent (no RNG), `apply_virtual_visit`/`backup_virtual`
mutate only `search.nodes`, `threats::analyze`/`tactical_cells` are pure (no
global/`static`/`thread_local`/`Rc` — verified by grep over hexfield/src and
hexo_models/rust/src/threats_shared.rs). There is NO cross-slot shared mutable
state in the select pass: the eval cache and eval stats are NOT touched here
(they are touched only in submit/finish/backup, which stay serial). So the slots
are embarrassingly parallel for the duration of the select pass.

## Parity constraints (HARD — math must stay fp16/bit-identical)

1. **Per-slot determinism is already guaranteed**: no RNG in the descent, no
   float-reduction reordering inside a slot (the serial inner loop is untouched).
2. **Leaf ORDER must be byte-identical to the serial pass.** The returned
   `leaves` Vec feeds the flush queue, and the eval dedup
   (`evaluate_state_refs_cached`/`submit_eval_cached`, payload.rs:313/479) builds
   `unique_states` in arrival order via `unique_index_by_key.insert(...)`. The
   featurize step (payload.rs:47 `featurize_and_sort`) sorts rows by support DESC
   then **restores caller order** — so a different leaf order would change which
   duplicate is canonical and the cache insertion order, i.e. it would change the
   eval batching and (in fp16) potentially the values. THEREFORE: leaves must be
   emitted in `(slot_index ASC, then per-slot select order)` — exactly the serial
   layout. `par_iter().map(...).collect::<Vec<_>>()` preserves input index order,
   so concatenating the per-slot leaf Vecs in slot-index order reproduces it
   exactly.
3. `made_progress` is an OR-reduction (order-independent — bit-exact).
4. `slot.in_flight += added_in_flight` is a per-slot write — must be applied to
   the same slot the rayon task owned (returned alongside the leaves).

## Prototype design (minimal, mechanical)

Rewrite ONLY the body of `select_continuous_pass`. rayon 1.10 is ALREADY a
workspace dependency (Cargo.toml:20) but hexfield's package Cargo.toml does NOT
yet list it — add `rayon.workspace = true` to packages/hexfield/Cargo.toml
`[dependencies]`.

The borrow problem: `slots: &mut [ContinuousSlot]`. We need `&mut` to each slot's
`search` in parallel. Use `slots.par_iter_mut().enumerate()` (rayon gives disjoint
`&mut ContinuousSlot` to each task — sound). Each task does the per-slot guard +
`select_continuous_leaves` + writes `slot.in_flight`, and returns
`(slot_index, slot_leaves, progressed)`. Collect into an index-ordered Vec, then
flatten in slot order to preserve leaf ordering. `select_continuous_leaves`
already takes `&mut RustSearch` and returns `PyResult` — but PyResult/PyErr is
NOT `Send` (it may hold a `Py<...>`). The select path here cannot actually fail
(`select_pending_leaf` only returns `Err` via `move_error` from
`apply_placement`, which in practice never fires on legal trees) — but to keep
the signature honest, map the error inside the task to a `String` (Send) and
re-raise as `PyValueError` after the parallel region.

### Draft patch (search.rs) — REPLACE the body of `select_continuous_pass`

```rust
fn select_continuous_pass(
    slots: &mut [ContinuousSlot],
    c_puct: f32,
    leaf_batch_per_root: u32,
    virtual_loss: f32,
) -> PyResult<(Vec<RustLeaf>, bool)> {
    use rayon::prelude::*;

    // Per-slot, fully independent. par_iter_mut hands each task a disjoint
    // &mut ContinuousSlot. Returns (slot_index, leaves, progressed) so the
    // collected Vec can be flattened in slot-index order — byte-identical leaf
    // ordering to the serial pass (the eval dedup is order-sensitive). Errors
    // are stringified (PyErr is !Send) and re-raised after the parallel region.
    let per_slot: Vec<Result<(usize, Vec<RustLeaf>, bool), String>> = slots
        .par_iter_mut()
        .enumerate()
        .map(|(slot_index, slot)| {
            if !matches!(slot.phase, ContinuousPhase::Active) {
                return Ok((slot_index, Vec::new(), false));
            }
            let cap = leaf_batch_per_root.saturating_sub(slot.in_flight);
            if cap == 0 {
                return Ok((slot_index, Vec::new(), false));
            }
            let Some(search) = slot.search.as_mut() else {
                return Ok((slot_index, Vec::new(), false));
            };
            if !search.needs_visits() {
                return Ok((slot_index, Vec::new(), false));
            }
            let (slot_leaves, progressed, added_in_flight) =
                select_continuous_leaves(search, slot_index, c_puct, cap, virtual_loss)
                    .map_err(|e| e.to_string())?;
            slot.in_flight = slot.in_flight.saturating_add(added_in_flight);
            Ok((slot_index, slot_leaves, progressed))
        })
        .collect();

    let mut indexed: Vec<(usize, Vec<RustLeaf>, bool)> = Vec::with_capacity(per_slot.len());
    for r in per_slot {
        indexed.push(r.map_err(PyValueError::new_err)?);
    }
    // par collect already preserves input index order, but sort defensively so
    // leaf ordering is provably (slot_index ASC, per-slot order) == serial.
    indexed.sort_by_key(|(i, _, _)| *i);

    let mut leaves = Vec::new();
    let mut made_progress = false;
    for (_, slot_leaves, progressed) in indexed {
        made_progress |= progressed;
        leaves.extend(slot_leaves);
    }
    Ok((leaves, made_progress))
}
```

Notes:
- `rayon::prelude` brings `par_iter_mut`. `ContinuousSlot`/`RustSearch`/`RustLeaf`
  must be `Send`. `RustSearch` holds `Vec<RustNode>`, `HashMap`, primitives, and
  `Arc<RustEvaluation>` inside nodes — `Arc<T: Send+Sync>` is Send, so the tree is
  Send PROVIDED `RustEvaluation` is `Send+Sync` (it is shared via `Arc` across the
  cache already, so it must be). `RustHexoState` is plain data → Send. If the
  compiler rejects Send, that surfaces a real shared-state hazard and the
  prototype is abandoned (honest gate).
- `RustLeaf` holds `RustHexoState` + ids → Send.
- The `.collect::<Vec<Result<...>>>()` allocates 64 small results — negligible.
- Determinism: rayon's work-stealing changes which THREAD runs a slot but NOT the
  per-slot computation nor the final ordering (we re-sort by slot_index). The
  result is bit-identical to serial.

### Do NOT parallelise backup (this round)

`backup_continuous_items` (search.rs:1460) zips `items` with `evaluations`
positionally and dispatches by `leaf.root_index`. Parallelising it requires
grouping items by slot AND the `RootInit` arm calls `RustSearch::new` +
`threats::tactical_cells` (heavier, but rare — one per new root). Backup is also
cheaper than select (no tree descent, just a path walk of length = depth).
Keep backup serial for the prototype; it is a strictly larger parity surface and
a smaller share of the time. If select parallelisation shows a real win, backup
can be a follow-up.

## Honest expected-gain assessment

The Rust tree-walk is a SMALL slice of per-decision time; the GPU dominates.
Established profiling facts:
- EARLY/MID flushes are GPU-LAUNCH-bound: host `submit` ~17 ms enqueue while the
  GPU does only ~11 ms; GPU ~57% idle. The select pass runs (GIL released) DURING
  the async submit/finish window, overlapping the GPU.
- LATE/DEEP flushes are GPU-COMPUTE-bound: GPU ~97%. Here the Rust select is
  ALREADY fully hidden behind GPU compute — parallelising it gains **zero** pos/s.

So a measurable win is only possible in the EARLY/MID regime, and ONLY to the
extent the serial select time is actually ON the critical path (i.e. select +
host submit on the single thread together exceed the GPU compute, leaving the GPU
starved — consistent with the "GPU 57% idle" finding). Parallel select shrinks
the host-side select term; if select is, say, a few ms of the ~17 ms host
critical path at 64 games, an 8–16x core speedup on that term could recover a few
ms per flush → low-single-digit % pos/s in early/mid, blended down by the
late/deep regime where it does nothing. **Plausible blended gain: ~0–4% pos/s.**

Risks that could make it NET-NEGATIVE:
- rayon thread-pool spin-up / work-stealing overhead per pass (called twice per
  flush in the async path) can EXCEED the tiny per-slot work when most slots are
  inactive or `cap==0` (common when in_flight is saturated). With virtual_batch=4
  and 64 slots the per-pass work is ~256 leaf descents max — rayon's overhead
  (~1–5 µs dispatch + steal) may dominate at this granularity.
- The select pass holds the GIL released; adding 32 threads competing for memory
  bandwidth with the GPU host-copy/featurize can hurt the very submit it overlaps.
- Determinism: SOUND by construction (re-sorted by slot index), but any accidental
  reliance on leaf order being interleaved across slots (it is not — it is slot
  order) would break parity.

## Recommendation

worth_prototyping = FALSE (marginal). The change is mechanically clean and
parity-safe, BUT the expected blended pos/s gain is ~0–4% and could be negative
due to rayon dispatch overhead at this fine granularity (256 leaf descents/pass,
many slots inactive), while the dominant late/deep regime gets nothing because
the GPU already hides the select. Higher-leverage options (cutting the ~17 ms
host submit enqueue / fusing the hundreds of tiny launch kernels in the
GPU-launch-bound early/mid regime) attack the actual bottleneck. If prototyped
anyway, gate it behind an env flag (e.g. HEXFIELD_RAYON_SELECT) so it can be
parity- and bench-compared without owning the live path, and MEASURE before
trusting — do not assume a win.

## Exact validate recipe (if the validate phase proceeds)

Baseline number to beat: run the baseline .so FIRST and record `pos/s` at
active_games=64 (this is the number to beat; ~steady-state mid/late mix).

1. Record baseline pos/s (current .so, no code change):
   ```
   wsl.exe -- bash -lc 'cd /mnt/e/Hexo-BotTrainer-hexgt && PYTHONPATH=packages/hexfield/python /root/.venvs/hexgt-build/bin/python scripts/_hexfield_lategame_bench.py /mnt/e/Hexo-BotTrainer/runs/hexfield_main_1/checkpoints/epoch_000031.pt 90 "64" /tmp/rayon_baseline.json'
   ```
   Note `BEST pos/s` from the output.

2. Apply the patch: add `rayon.workspace = true` to packages/hexfield/Cargo.toml
   [dependencies], replace the `select_continuous_pass` body as above (ideally
   behind an `if std::env::var("HEXFIELD_RAYON_SELECT").is_ok()` branch that
   falls back to the existing serial loop, so parity/bench can A/B the same .so).

3. Rebuild the .so:
   ```
   wsl.exe -- bash -lc 'cd /mnt/e/Hexo-BotTrainer-hexgt && bash scripts/_rebuild_hexfield.sh'
   ```

4. PARITY — action sequences must match a fixed-seed baseline (run with the
   serial path AND the rayon path; they must be IDENTICAL):
   ```
   wsl.exe -- bash -lc 'cd /mnt/e/Hexo-BotTrainer-hexgt && PYTHONPATH=packages/hexfield/python /root/.venvs/hexgt-build/bin/python scripts/_hexfield_async_parity.py /mnt/e/Hexo-BotTrainer/runs/hexfield_main_1/checkpoints/epoch_000031.pt'
   ```
   Expect `RESULT: PASS`. Then run the continuous-parity unit test:
   ```
   wsl.exe -- bash -lc 'cd /mnt/e/Hexo-BotTrainer-hexgt && PYTHONPATH=packages/hexfield/python:packages/dense_cnn_restnet/python /root/.venvs/hexgt-build/bin/python -m pytest tests/test_hexfield_continuous_parity.py -q'
   ```
   If gated behind HEXFIELD_RAYON_SELECT, also run the parity harness with the
   flag SET to prove the rayon path matches:
   ```
   wsl.exe -- bash -lc 'cd /mnt/e/Hexo-BotTrainer-hexgt && HEXFIELD_RAYON_SELECT=1 PYTHONPATH=packages/hexfield/python /root/.venvs/hexgt-build/bin/python scripts/_hexfield_async_parity.py /mnt/e/Hexo-BotTrainer/runs/hexfield_main_1/checkpoints/epoch_000031.pt'
   ```

5. BENCH — rayon pos/s vs baseline:
   ```
   wsl.exe -- bash -lc 'cd /mnt/e/Hexo-BotTrainer-hexgt && HEXFIELD_RAYON_SELECT=1 PYTHONPATH=packages/hexfield/python /root/.venvs/hexgt-build/bin/python scripts/_hexfield_lategame_bench.py /mnt/e/Hexo-BotTrainer/runs/hexfield_main_1/checkpoints/epoch_000031.pt 90 "64" /tmp/rayon_on.json'
   ```
   Compare `BEST pos/s` against step 1. A win must clear noise (>~3%) AND parity
   must be PASS. If pos/s is flat or down, REJECT (the prototype confirmed the
   tree-walk is hidden / rayon overhead dominates).
