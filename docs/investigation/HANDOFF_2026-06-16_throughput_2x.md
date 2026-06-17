# Handoff — hexfield self-play throughput (~2× effort)
**Date:** 2026-06-16 · **Branch:** `claude/wizardly-johnson-x90akh` · **Goal:** ~2× self-play pos/s, functionally equivalent.

---

## 0. TL;DR

- **Measured 1.58×** so far from two shipped, parity-verified changes: `torch.compile` of the serve
  forward + a **Rust parallelism restore** (hexfield was a *de-parallelized* port of dense_cnn).
- A **background writer thread** (dense_cnn port) is implemented and was **validated live** (shards
  tracked game completions exactly), adding a live-only win the bench can't show.
- **Multi-process is dead** on this box (no CUDA MPS under WSL2 → contexts time-slice: 0.39×).
  **CUDA graphs are dead** (1.03–1.09× over the already-compiled forward).
- **Open blocker:** enabling `torch.compile` for **all** Npad shapes caused an Inductor
  **compile hang** on the first *deep* late-game shape (single self-play thread froze in
  `async_compile._wait_futures`, GPU idle). Quick-stabilized by capping at `Npad<=1024`. **The owner
  wants compile-all working** — §4 has the concrete fix (pre-warm + synchronous compile). This is the
  top next step and the most likely remaining path to 2×.
- Remaining exact levers to close 1.58×→2×: **get compile-all working** (deep-shape forward win),
  **`thread::scope` overlap** (dense_cnn port, Rust), **decode-fuse** (NEW, not a port).

---

## 1. Current run / machine state (VERIFY FIRST)

- The live run is `hexfield_main_1` under WSL, managed by **systemd** (`systemctl status hexfield-supervisor`).
- At handoff the supervisor is **active + enabled** with a freshly **auto-restarted** trainer (low PID,
  e.g. 288). The **Inductor disk cache was cleared** (`/tmp/torchinductor_root` empty → it will recompile
  all shapes), and the keepalive `sleep` process is gone — symptoms of a **WSL distro restart**
  (see the supervisor-persistence memory; the *only* reliable keep-alive is an attached `wsl.exe` client
  or a Windows logon Scheduled Task running `wsl -d <distro> -e bash -lc 'exec sleep infinity'`).
- **ACTION:** confirm the auto-restarted trainer is *progressing, not hung* — `live.json` `elapsed_seconds`
  must advance and GPU must cycle (not stuck at 0%/10W). It should be safe now because the source default
  is capped at `Npad<=1024`. If it hung, kill it (`pgrep -f cli.train_model` then `kill -9 <pid>` — do
  **not** `pkill -f "train_model"`, the pattern matches your own shell and kills it) and restart.
- Latest checkpoint: `epoch_000034.pt`. Epoch 35 has been wiped/regenerated several times this session.
- The hexfield `.so` in the tree is the **parallel-Rust** build (mtime 2026-06-16 13:18:45), parity-verified.

---

## 2. What changed (all UNCOMMITTED — the run loads from source + tree `.so`)

| File | Change | Status |
|---|---|---|
| `packages/hexfield/python/hexfield/inference.py` | Enable compile for large Npad (`HEXFIELD_COMPILE_MAX_NPAD` default **now 1024** after the hang; was 512→1e6); `cache_size_limit` 64→256; fixed the stale "matmul-bound" comment (forward is **bandwidth-bound**, compile helps large N too). | shipped |
| `packages/hexfield/Cargo.toml` | `rayon.workspace = true` | shipped |
| `packages/hexfield/rust/src/search.rs` | `select_continuous_pass` → `par_iter_mut` (parallel MCTS select); `use rayon::prelude::*`. **The layer the hexfield port dropped vs dense_cnn.** | shipped, parity-pass |
| `packages/hexfield/rust/src/payload.rs` | `featurize_and_sort` → `par_iter` + SIMD f16 cast; `parse_chunk_reply` → `par_iter` w/ prefix-offset (parallel priors/value decode). | shipped, parity-pass |
| `packages/hexfield/python/hexfield/selfplay.py` | **Background writer thread** (`_write_queue`/`_writer_loop`/`_start_writer`/`_stop_writer`; `_finish` enqueues; start/stop around `run_continuous`). dense_cnn_restnet port. | shipped, validated live |
| `configs/hexfield_main_1.toml` | `active_games` 64→96 (from earlier in session) | shipped |
| `scripts/_hexfield_compile_largeN_bench.py` | new: real-state eager-vs-compiled forward bench | new |

**Rebuild Rust:** `bash scripts/_rebuild_hexfield.sh` (dev venv `/root/.venvs/hexfield-dev`, mirrors `.so`
into the tree). The script exits non-zero (rc=2) cosmetically due to its tail `.so`-mirror `ls` under
maturin's *editable* install — **the build itself succeeds**; verify by the fresh `.so` mtime + parity.
**Parity gates:** `pytest tests/test_hexfield_search_parity.py tests/test_hexfield_continuous_parity.py`
(run in the `hexgt-build` venv via `PYTHONPATH=packages/hexfield/python`). Both pass with the parallel build.

**Recommend committing** these once the run is confirmed healthy (clean restore point before further work).

---

## 3. Measurements (the evidence)

Harness: `scripts/_hexfield_lategame_bench.py <ckpt> 150 "96" <out> refill` — ag96, 150s, refill.
**Caveats:** early-game biased (understates late-game wins), and its synthetic `on_move` **writes no
shards** (so it cannot show the writer-thread win). Ratios are trustworthy; absolute pos/s is inflated
vs live full-epoch.

| build | pos/s | note |
|---|---|---|
| original: no-compile + serial Rust | **5.91** | session-start baseline (git-stash measured) |
| compile + serial Rust | 6.89 | compile alone ≈ 1.17× |
| no-compile + parallel Rust | 6.51 | parallel Rust alone ≈ 1.10× |
| **compile + parallel Rust (current)** | **9.36** | **1.58× over original** |

**Live (new build, before the hang):** early pos/s 11→9→7.6→7.0 (well above old build), GPU 67–78%,
**writer thread validated** (npz == games_finished exactly: 2/2, 9/9, 15/15, 16/16).

**Dead ends (don't re-chase):**
- **Multi-process N=2:** 0.39× aggregate; GPU util *fell* to 22%, VRAM near-OOM. WSL2 has **no CUDA MPS**,
  so two contexts time-slice the GPU. Confirms (and explains) the stale 1.03× note.
- **CUDA graphs** (`scripts/_hexfield_cudagraph_proto.py`): graph-replay only **1.03–1.09×** over the
  *already-compiled* forward (compile already removed the launch overhead, ~2.9–3.1× over eager). Not worth it.

**Profile (py-spy on the live trainer):** ONE Python thread pegged ~100% — **~80% host-side forward
orchestration** (the per-group decode/softmax at `inference.py:293–320` + kernel launches; `numpy` pack ≈ 0),
**~18% Rust search**. GPU ~50–60% with idle duty-cycle gaps. NOT CPU-aggregate-bound (3% of 32 cores),
NOT VRAM-bound (2.6/12.3 GiB), NOT power-bound (~55%). The wall is the single GIL-bound host thread.

---

## 4. ★ FIX COMPILE-ALL (top next step — owner wants this) ★

**Compile-all is NOT impossible.** It worked for 16 games (small/mid shapes compiled fine); it **hung**
on the first *deep* late-game shape (S≈1500+, a giant `(B,heads,S,S)` attention kernel). py-spy showed the
main thread stuck in `torch/_inductor/async_compile.py` `_wait_futures` / `get_result`, and the kernel
cache **stopped growing** — i.e. the **async compile-worker subprocess pool deadlocked** (a worker likely
OOM'd/died on the huge kernel and never returned its future). This is a *pool* failure, not a "can't compile."

**Two fixes, combine them:**

1. **Synchronous compile — `TORCHINDUCTOR_COMPILE_THREADS=1`.** Removes the worker pool entirely; compile
   runs inline in the main process (full memory, no `_wait_futures`, no dead-worker deadlock). Deep shapes
   then *stall* (finite) instead of *hang*. Set it in the supervisor env (`scripts/_hexfield_supervise_main1.sh`).

2. **Pre-warm the disk cache offline** so the live run is **all cache-hits** → compile-all win on *every*
   shape with **zero runtime stalls**. Write `scripts/_hexfield_prewarm_compile.py`: for each 64-quantized
   Npad bucket from small→max (~3072), build a synthetic batch (see `_hexfield_compile_largeN_bench.py`'s
   `make_payload`/`group_near`), run the compiled `forward_policy_value` once **with a per-shape watchdog
   timeout**, print `Npad -> seconds`. This (a) populates `/tmp/torchinductor_root` and (b) **proves which
   shapes are compilable**. Then restart the run with `HEXFIELD_COMPILE_MAX_NPAD=1000000`.

**The one thing to confirm:** is the deep-shape compile *slow-but-finite* (then pre-warm solves it fully) or
*genuinely infinite* (an Inductor limitation on that exact shape)? The pre-warm watchdog answers it. If one
specific deep shape truly never compiles, cap just **above** the largest compilable shape (likely ≫1024) and
leave only that rare tail eager. Either way you get compile-all (or as-close-as-physically-possible), not 1024.

**Also worth ruling out:** the writer thread (added this session) is a background Python thread; torch's
compile-worker pool is created at first compile. A fork-with-threads interaction is *unlikely* (the pool
served 16 games fine before hanging), but `TORCHINDUCTOR_COMPILE_THREADS=1` eliminates that risk too. If you
want to be certain, A/B the pre-warm with the writer thread present vs a vanilla forward.

---

## 5. Remaining path to 2× (after compile-all)

Levers, honest estimates, and whether they're **ports** (proven, low-risk) or **new**:

1. **Get compile-all working (§4)** — recovers the deep-shape forward win the cap gives up. *Config/infra.*
2. **`thread::scope` overlap (Phase 2)** — **PORT** of dense_cnn `mcts.rs:1002-1025`: run the (now parallel)
   select on a `std::thread::scope` worker that overlaps the *whole* GIL-held forward, not just the GPU-exec
   gap that hexfield's current `py.detach` covers. ~1.05–1.1×. Rust rebuild + parity gate. hexfield dropped
   this from its template too. Implementation sketch: replace the `async_eval`/`py.detach` flush block in
   `search.rs` (~1010, ~1058) with the dense_cnn `thread::scope(|scope| { let h = scope.spawn(select…);
   let evals = evaluate…; (evals, h.join()) })` shape.
3. **Decode-fuse — NEW (not in any lineage).** Fold the legal-prefix softmax + value/ml decode
   (`inference.py:293–320`, ~8–10 eager launches/group) into the compiled forward (`forward_serve_decoded`,
   keep only `priors[legal]` gather eager). ~1.2–1.35× per the design workflow — the biggest single
   in-process lever, but higher validation burden (numerically-close-gated parity). **Note:** restnet does
   the softmax in Python too (`inference.py:342`), so this is genuinely new, not a port.
4. **Writer thread (DONE)** — live-only win not visible in the bench.

Compounding is **not** the product (host cuts shrink the GPU gaps a 2nd thread/process would fill); measure
the stack together. Realistic ceiling 1.6–1.9×; 2× is the optimistic edge. The cleanest "is it 2×?" test is a
**live full-epoch** (the bench can't represent late-game or shard writing).

---

## 6. Key architecture context (why this works)

- **hexfield is a de-parallelized port of dense_cnn.** `search.rs` header says so. dense_cnn / restnet /
  hexgt / hexgnn ALL parallelize self-play the same way; hexfield dropped the layers. The wins above are
  mostly *restoring its own template*: `par_iter` select (done), `thread::scope` overlap (todo),
  rayon featurize/decode (done), background writer thread (done). References:
  `packages/hexo_models/dense_cnn/rust/src/mcts.rs` (`1002-1025` scope overlap, `1458-1493` par select,
  `mcts_eval.rs:248/274/372` parallel encode/cast/decode), `dense_cnn_restnet/.../selfplay.py:1229-1356`
  (writer thread).
- **No lineage drives the forward from Rust** (none link libtorch/tch). The forward is always a Python
  GIL-held call. So "rewrite the forward in Rust" is NOT the path; the lineages keep the forward in Python
  and shove everything *else* (select/featurize/decode/write) onto GIL-free Rust/background threads.
- **Bottleneck:** single GIL thread; the GPU forward is a minority of wall-clock (compile made it ~3 ms);
  the host decode + Rust search are the rest. Filling GPU idle needs more in-process concurrency, which the
  GIL blocks — hence multi-process (dead, no MPS) and free-threaded-Python (3.13t nogil) being the only true
  "many threads" routes; nogil is a major rewrite recorded but not attempted.

## 7. Run management cheat-sheet

- **Build Rust:** `bash scripts/_rebuild_hexfield.sh` (dev venv). **Tests:** parity files in §2.
- **Start/stop run:** `systemctl {start,stop,status} hexfield-supervisor` (systemd is the single launch
  authority — do NOT also launch via `wsl.exe` background tasks; see the supervisor-persistence memory).
- **Wipe an epoch fresh:** `rm -f <run>/samples/epoch_0000NN/game_*.{npz,json} <run>/selfplay/epoch_0000NN*.hxr`
  then restart (resumes from latest `epoch_*.pt`). Run dir: `/mnt/e/Hexo-BotTrainer/runs/hexfield_main_1`.
- **Watch pos/s:** `<run>/diagnostics/hexfield.selfplay.live.json` (`positions_per_second`, `elapsed_seconds`).
- **Keep WSL alive unattended:** an attached `wsl.exe` client (`wsl -e bash -c 'exec sleep infinity'` in
  background) or a Windows Scheduled Task; the distro tears down ~15s after the last client detaches.
- **Bench:** `_hexfield_lategame_bench.py` (ag/pos/s/VRAM), `_hexfield_serve_profile.py` (host vs GPU split),
  `_hexfield_compile_largeN_bench.py` (eager vs compiled per Npad).
- **Profile the live trainer:** `py-spy dump --pid $(pgrep -f cli.train_model | head -1)`.

## 8. Immediate next-session checklist

1. Verify/stabilize the auto-restarted run (§1). Commit the uncommitted changes (§2) as a checkpoint.
2. Write + run `_hexfield_prewarm_compile.py` (§4) with a per-shape watchdog → confirm which shapes compile.
3. Restart with `TORCHINDUCTOR_COMPILE_THREADS=1` + warmed cache + `HEXFIELD_COMPILE_MAX_NPAD=1000000`;
   confirm no hang on a deep-game epoch.
4. Measure a **live full-epoch** vs the pre-session baseline → the real 2× verdict.
5. If short: add the `thread::scope` overlap (port), then the decode-fuse (new), re-measuring each.
