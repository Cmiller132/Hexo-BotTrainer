# main_6 GPU-saturation rewrite — chosen architecture + incremental plan

Author: principal engineer. Scope: the continuous self-play scheduler in the
**worktree** `E:/Hexo-BotTrainer-gumbel` ONLY. Never touch the main tree
(`E:/Hexo-BotTrainer-hexgt`) or the installed `hexgt-build` packages.

---

## 0. Decision (read this first)

**Chosen approach: "Staged depth-N pipeline" — the incremental extension of the
EXISTING async (`submit/finish`) + `run_continuous_pipeline_depth2` machinery,
NOT a ground-up worker-pool/actor rewrite.**

Why this and not the full continuous-batching worker-pool rewrite:

1. **The profile forbids the expensive parts of the rewrite from paying off.**
   The forward is FLOP-bound and *flat per-state* (~0.10–0.14 ms/state across
   batch 145–265) and `mean_batch` is already healthy at 245. So the worker-pool's
   headline feature — coalescing many tiny per-game requests into one big batch —
   buys **nothing** here: batches are already big and bigger batches are not
   cheaper. The ONLY win available to anyone is filling the ~0.6 idle fraction,
   and that idle is caused by `select`(~31 ms) + `complete`(~31 ms) running as
   **serial phases between forwards**. You do not need 24 free-running GIL-juggling
   worker threads to overlap two host phases with a forward — the existing depth-2
   pipeline already overlaps `select(N+1)` + `backup(P)` with the forward of `N`.
   The one phase it does **not** yet hide is `complete` (~28% of the loop). That
   is the single highest-value, lowest-risk change, and it is ~30 lines.

2. **The codebase already paid for the hard infrastructure.** `submit_payload`/
   `result` (inference.py) split enqueue from the D2H sync with deferred decode
   (`HEXFIELD_DEFER_DECODE`); `submit_eval_cached`/`finish_eval_cached` (payload.rs)
   split the cache-checked eval into an enqueue + a drain with an owned
   `PendingEval`; `HEXFIELD_RUST_PACK` does grouping/padding/f16 assembly GIL-free
   in parallel Rust; `select_continuous_pass`/`backup_continuous_items` already run
   under `py.detach` + rayon `par_iter_mut` over disjoint slots;
   `run_continuous_pipeline_depth2` already keeps one eval in flight, double-buffers
   submit before drain, and is flag-gated + documented as virtual-loss-faithful but
   NOT byte-identical. The full rewrite re-implements all of this against the
   `unsendable` `#[pyclass]` / `Py<PyAny>`-is-not-`Send` constraint (HexfieldEvaluator
   and the session handles are not `Send`), which is multi-week, parity-critical, and
   — given the bounded ceiling — high risk for the same ~1.7x a few targeted changes
   reach.

3. **The prior NULL result is the cautionary tale that picks this approach.** A
   previous parallel-backup + double-buffer rewrite measured NULL because it
   parallelized the CHEAP phase (`backup` = 1.0 ms) and did naive double-buffering
   without overlapping the EXPENSIVE serial `select`/`complete`. This plan does the
   opposite: it explicitly moves `complete` (and the GIL-held `on_move`) into the
   forward-overlap window, and benchmarks the GPU-idle fraction after *each*
   increment so we abort the moment an increment shows no util gain.

**Honest ceiling: util ~40–50% → ~75–85%, i.e. ~1.6–1.9x pos/s (~11–12 → ~18–22).
NOT >2x.** Bounded by Amdahl on the host MCTS that cannot be hidden and by the
~38 ms/flush forward becoming the true serial floor once the pipe is full.

**Everything below is opt-in / flag-gated. The production parity path
(`HEXFIELD_ASYNC_EVAL` off, or on but byte-identical; lockstep `run_continuous`)
and `tests/test_hexfield_continuous_parity.py` MUST stay green and unchanged.**

---

## 1. Files / functions touched

| File | Function | Change |
|---|---|---|
| `packages/hexfield/rust/src/search.rs` | `run_continuous_pipeline_depth2` (L1309) | Increment 1: pull `complete_continuous_slots` into the in-flight-forward overlap window (submit N → complete P-evals-landed → drain P). |
| `packages/hexfield/rust/src/search.rs` | new `run_continuous_pipeline_depthn` (sibling of depth2) | Increment 3: generalize the 1-in-flight ring to a K-in-flight ring (K=2..3) behind a depth knob. |
| `packages/hexfield/python/hexfield/inference.py` | `submit_payload` / `result` / new stream plumbing | Increment 3: hold K outstanding handles, each forward on its own `torch.cuda.Stream`, so H2D(k+1)∥compute(k)∥D2H(k-1). |
| `packages/hexfield/rust/src/search.rs` | `run_continuous` env parsing (L1021–1040) | Add `HEXFIELD_PIPELINE_DEPTH` (int, default 1 = lockstep-async; 2 = current depth2; ≥3 = ring). Keep `HEXFIELD_PIPELINE_DEPTH2` as an alias for depth=2. |
| `packages/hexfield/python/hexfield/selfplay.py` | epoch driver (L459–472) | Increment 0: add a `cuda.Event`-based GPU-idle-fraction + per-phase timing instrument, logged to `hexfield.selfplay.epoch_*.json`. Bench-only; gated by `HEXFIELD_PERF_TRACE=1`. |
| `configs/_gumbel_e2e.toml` (bench), `configs/hexfield_main_6.toml` (prod, later) | `active_games`, `virtual_batch_size`, `flush_target` | Increment 2: raise `active_games` to feed the deeper pipeline; config-only. |

No changes to `select_continuous_pass`, `backup_continuous_items`,
`submit_eval_cached`, `finish_eval_cached`, the cache, or `featurize_and_sort` —
they are reused verbatim. No changes to the lockstep `run_continuous` body or the
parity test.

---

## 2. GIL strategy (unchanged from the existing design — this is why it works)

The profile already proves this is **not** GIL/dispatch-bound (async submit overlap
works: `submit_host` 22–25 ms < `gpu_forward` 36–46 ms; `select` runs under
`py.detach` + rayon). The plan preserves that property and never adds GIL pressure:

- **One thread touches Python.** Only the scheduler's main thread ever calls into
  the evaluator (`submit_payload` / `result`) and `on_move`. There is no second
  Python-calling thread, so there is no GIL contention to create.
- **All MCTS work stays GIL-free.** `select_continuous_pass`, `backup_continuous_items`,
  virtual loss, tree mutation already run inside `py.detach { rayon par_iter_mut }`.
  The pack/H2D is the existing GIL-free Rust path (`HEXFIELD_RUST_PACK=1`).
- **The GIL is held only for the irreducible spots:** the `submit_payload` dispatch
  (torch releases the GIL during the CUDA forward and the H2D/D2H copies — measured
  `submit_host` < forward, so dispatch hides under the previous batch's compute),
  the one `result()` D2H sync per drained flush, and the `on_move` callback. The
  whole point of increment 1 is to put the `select(N+1)` (GIL-released) and the
  GPU forward of `N` concurrently with the `complete`/`on_move`(GIL-held) of an
  EARLIER flush, so the GIL-held work overlaps GPU compute instead of stalling it.
- **Dispatch stays single-threaded** for `torch.compile`/`mark_dynamic` safety — we
  never call `fpv` from two threads. The K-in-flight ring (increment 3) issues K
  forwards from the **same** thread onto K **different CUDA streams**; the streams
  give device-side overlap, not host-side concurrency, so compile/mark_dynamic are
  untouched.

Required env (already in the run command): `HEXFIELD_RUST_PACK=1`,
`HEXFIELD_DEFER_DECODE=1`, `HEXFIELD_ASYNC_EVAL=1`. (Note: the run command in the
task sets `HEXFIELD_TRAIN_FLEX`/`HEXFIELD_SERVE_FLEX`/`HEXFIELD_ASYNC_EVAL`; verify
`HEXFIELD_RUST_PACK=1` and `HEXFIELD_DEFER_DECODE=1` are also exported for the bench
— without them `submit_payload` is not a pure non-blocking enqueue and the overlap
collapses.)

---

## 3. Faithfulness / determinism contract

**Anchor that is PRESERVED, every increment:** virtual loss. A leaf gets
`apply_virtual_visit` at selection (incrementing both `completed_visits` and the
selected edge's visits) and `backup_virtual` restores it at backup. A leaf with an
in-flight eval therefore carries a pending penalty so the next (overlapped) select
does not re-pick it. Each game's tree is touched by exactly one logical owner (its
slot's `&mut`, disjoint under `par_iter_mut`), so there are no torn nodes and no
cross-game tree sharing — identical to today.

**What is RELAXED (and must be flag-gated, NEVER on the parity path):** strict
determinism / byte-identical search. Deepening the in-flight window past depth-1
widens the leaf-selection staleness window: which leaves land in which batch, and
the interleaving of `complete` against in-flight forwards, becomes timing-dependent.

**Why it stays UNBIASED (not a directional bias, a nuisance variable):**
1. The eval cache is keyed by `state_hash` and dedup/order is restored by request
   index (`integrate_unique_evals`), so identical states always get identical
   evaluations regardless of batching timing.
2. Per-game RNG is seeded by `mix_seed(base_seed, game_key, ply, stream)` — independent
   of scheduling/execution order (confirmed at the `run_continuous` seed call sites:
   ROOT_NOISE, GUMBEL, move-select, PCR, policy-init all key on `(game_key, ply)`).
   So Dirichlet/Gumbel/PCR/temperature draws are timing-invariant.
3. Virtual loss makes the wider window self-correcting: it changes WHICH leaves are
   explored in what order within the same visit budget, exactly as KataGo's
   asynchronous batched search does — it cannot systematically favor any move/value
   because batch composition is independent of position content.
   The only genuine perturbation is bounded in-flight staleness, capped per game by
   `leaf_batch_per_root` (a slot has at most `virtual_batch_size` leaves in flight),
   so staleness ≤ the existing depth-2 flush window × pipeline depth.

**Flag firewall (matches what the code already documents):**
- `HEXFIELD_ASYNC_EVAL` (depth-1): byte-identical — only the sync point moves
  (`submit/finish` share the cache+order machinery; identical to
  `evaluate_state_refs_cached`). **Safe for the parity/differential-harness path.**
- `HEXFIELD_PIPELINE_DEPTH ≥ 2` (incl. the existing `HEXFIELD_PIPELINE_DEPTH2`,
  the new complete-overlap, and the K-ring): NOT byte-identical. **NEVER on the
  parity/harness path.** Default OFF. Production self-play stays on the lockstep or
  depth-1 path until a depth-N path is proven equal-strength.
- A determinism/parity mode (single-flight, fixed order, depth=1) is retained for
  the differential harness and the per-game seed-stream golden vectors (which pass
  because property 2 above is untouched).

**Promotion gate to own the live run:** a non-byte-identical depth-N path may
become the production self-play driver ONLY after an **eval-arena Elo A/B** shows it
is equal strength (not byte-equality; aggregate-equivalence: same mean game length,
value-target distribution, policy-target KL within MC noise over a few hundred
games). This matters because the live main_5/main_6 bottleneck is policy-target
variance — a silent search-quality regression would be costly.

---

## 4. Incremental build order (each increment independently benchmarkable; abort if no GPU-util gain)

Ship strictly in this order. After each increment, run the same bench (§5) and apply
the go/no-go (§6). **Do not proceed to the next increment if the current one fails
its go criterion.**

### Increment 0 — Instrumentation (no behavior change; prerequisite)
Add a `cuda.Event`-based per-flush instrument behind `HEXFIELD_PERF_TRACE=1`:
GPU-busy fraction (sum of `start.elapsed_time(end)` over forwards ÷ wall), per-phase
wall (`select`, `submit`, `finish/result`, `backup`, `complete`), `mean_batch` and
`flush_size_histogram` (already in `ContinuousSchedulerStats`), and python-process
thread occupancy (sample `top`/`/proc` or `psutil` cpu_percent). Emit to
`hexfield.selfplay.epoch_*.json`. This is the measurement substrate for every
go/no-go below; it must impose ~0 overhead when the flag is off.
- **Go:** trace reproduces the profile baseline (idle ~0.6, mean_batch ~245,
  gpu_forward/state ~0.10–0.14 ms) on the representative deep-game bench. If it does
  NOT reproduce the baseline, STOP and reconcile the instrument with the profile
  before trusting any later delta.

### Increment 1 — Overlap `complete` with the in-flight forward (the headline change)
In `run_continuous_pipeline_depth2`, today `complete_continuous_slots` runs after
the drain, GPU idle. Reorder the steady state so per iteration:
`select(N) → submit(N, enqueue, no sync) → complete(slots whose evals already
landed, GIL-held on_move) → drain(P) (finish + backup) → stash N as inflight`.
The completes of already-resolved slots (and their `on_move`) now run while the GPU
computes `N`. (`complete` only finalizes slots with `in_flight == 0`, so a slot with
an eval still buffered in `inflight` is correctly NOT completed — the existing
invariant holds.) ~30 lines, all inside the already-flag-gated depth2 path.
- **Hypothesis:** moves the ~31 ms `complete` phase off the GPU-idle critical path
  → idle 0.6 → ~0.45–0.5, pos/s +15–30%.
- **Go:** GPU-idle fraction drops ≥ 0.08 absolute AND pos/s rises ≥ 10% vs depth2
  baseline, on the deep-game bench. **Kill if** idle unchanged (complete did not
  actually move into the window — re-check ordering with the cuda.Event trace, do
  NOT trust wall-clock) or pos/s flat/down.

### Increment 2 — Raise `active_games` to feed the deeper pipeline (config-only)
Once `complete`/`backup`/`select` overlap the forward (inc. 1), more independent
slots translate to filled idle instead of a longer serial phase (the prior 96→192
NULL was because select didn't overlap the forward). Raise `active_games` in the
bench config (e.g. 96 → 144 → 192), keep `virtual_batch_size`/`flush_target`; watch
VRAM and that the flush still single-chunks under `EVAL_CHUNK_STATES`.
- **Go:** at the chosen `active_games`, GPU-idle drops further (target ≤ 0.4) and
  pos/s rises, with `mean_batch` staying healthy and VRAM in budget. **Kill the
  step (revert to the prior value) if** pos/s drops (serial host work re-dominates)
  or VRAM blows the budget. This increment is cheap and reversible — sweep it.

### Increment 3 — K-in-flight CUDA-stream ring (depth-N), only if inc.1+2 leave residual idle
If after inc.1+2 the GPU-idle fraction is still ≥ ~0.25 and the trace shows a
visible gap *between* forwards (the stream draining before the next is enqueued),
generalize depth2's single `inflight` to a ring of K=2–3 outstanding
`(PendingEval, items, snapshot)` tuples, and extend `submit_payload`/`result` to
hold K handles each on its own `torch.cuda.Stream` so the next forward is already
queued on the device when the current finishes (zero inter-forward gap). Bound K by
VRAM (K× in-flight activations) and by the staleness cap. This is the highest-effort,
highest-risk increment — **only build it if the trace proves residual inter-forward
idle that inc.1+2 did not remove.**
- **Go:** inter-forward gap in the trace closes, GPU-idle → ~0.15–0.25, pos/s rises
  further toward the ~1.6–1.9x ceiling. **Kill if** the ring adds no util over
  depth2-with-complete-overlap (the forward is already the serial floor — we've hit
  the ceiling, ship inc.1+2 and stop) or if VRAM/staleness forces K back to 1.

### Increment 4 (gate, not code) — strength validation + promotion
For whichever depth-N path won, run an eval-arena Elo A/B vs the lockstep/depth-1
path over a few hundred games + the aggregate-equivalence checks (§3). Only on a
PASS may the flag default flip for the live run; otherwise the perf path stays
opt-in and the lockstep path keeps owning production.
- **Go:** Elo within noise (no significant regression) AND aggregate distributions
  match. **Kill (do not promote, keep opt-in) if** any significant Elo/length/KL
  regression appears — the speedup is not free and must not silently degrade search.

---

## 5. Measurement plan (same bench every increment)

- **Harness:** the RUN/BENCH command from the task header (torch venv + worktree
  shim, GPU free, `run_in_background=true`), with `HEXFIELD_PERF_TRACE=1` added and
  `HEXFIELD_RUST_PACK=1 HEXFIELD_DEFER_DECODE=1 HEXFIELD_ASYNC_EVAL=1` confirmed.
- **Primary metric:** GPU-busy fraction via cuda.Event (NOT `nvidia-smi` sampling,
  which the profile already showed is noisy at 100 ms granularity) — this is the
  abort signal, since the entire thesis is "convert idle to busy."
- **Secondary:** pos/s (wall-clock, the user-facing number), `mean_batch` +
  `flush_size_histogram` (must NOT shrink — if it does, the deepening broke
  coalescing), per-phase wall (`select`/`submit`/`finish`/`backup`/`complete`),
  python-process CPU% / cores busy (should rise from ~600% as host work overlaps).
- **Workload:** a representative **deep-game** steady-state window (the profile's
  worst idle, 0.55–0.83) AND a mid-game window — report both; the deep-game window
  is where the win must show. Fixed seed + fixed game set so runs are comparable.
- **Build/test discipline every increment:** `cargo test -p hexfield --features
  python` must stay **41 passed**; `maturin develop --release` clean; the parity
  test untouched and green.

---

## 6. Per-increment go/no-go + kill criteria (summary table)

| Inc | Go (proceed) | Kill / abort |
|---|---|---|
| 0 Instrument | Trace reproduces profile baseline (idle ~0.6, batch ~245, ms/state 0.10–0.14) | Trace disagrees with profile → fix instrument before trusting deltas |
| 1 complete-overlap | idle drops ≥0.08 abs AND pos/s +≥10% vs depth2 | idle unchanged (complete not in window) or pos/s flat/down → revert, the gain isn't there |
| 2 active_games sweep | idle ≤~0.4, pos/s up, batch healthy, VRAM ok | pos/s down (serial host re-dominates) or VRAM over budget → revert to prior value |
| 3 K-stream ring | inter-forward gap closes, idle ~0.15–0.25, pos/s up | no util over inc.1+2 (forward is the floor — ship inc.1+2, stop) or VRAM/staleness forces K=1 |
| 4 Elo A/B (gate) | Elo within noise, distributions match | any significant Elo/length/KL regression → keep opt-in, do not promote |

**Overall success criterion:** on the representative deep-game bench, sustained
GPU-busy fraction rises from ~0.40–0.50 to **≥0.75** and self-play throughput rises
from ~11–12 pos/s to **≥18 pos/s (~1.6x)**, with `cargo` at 41 passed, the parity
test untouched and green, and an eval-arena Elo A/B showing the enabled depth-N path
is equal strength — all behind flags, default OFF, with the lockstep production
scheduler unchanged and unregressed.

---

## 7. Explicit non-goals / guardrails

- NOT a Go rewrite, NOT a new actor/thread-pool framework, NOT a worker-pool over
  `slots`. The profile shows coalescing is already solved (`mean_batch` 245); the
  worker pool's headline feature would buy nothing here at large concurrency cost.
- Do NOT chase >2x — the forward is FLOP-bound and flat per-state; the ceiling is
  the ~38 ms/flush serial floor once util saturates.
- Do NOT regress the existing scheduler: the lockstep `run_continuous` body and the
  `HEXFIELD_ASYNC_EVAL` byte-identical path are frozen; all new behavior is additive
  and flag-gated, default OFF.
- Abort early and cheaply: every increment is independently benchmarked against the
  cuda.Event idle fraction; if an increment shows no idle reduction, stop — do not
  build the next, more expensive one on faith.
