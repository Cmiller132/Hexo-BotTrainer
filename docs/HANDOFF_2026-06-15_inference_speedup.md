# Handoff — hexfield inference / self-play throughput work (2026-06-15)

Goal of the session: figure out why `hexfield_main_1` self-play was only ~3 pos/s and
make inference as fast as possible **without changing search behaviour or model quality**.

This doc covers: the live run state, every change committed this session, what is
validated vs uncertain, the staged (not-yet-validated) inference rewrite, and the open
decisions.

---

## 0. TL;DR

- **Root cause of "~3 pos/s":** the serve `torch.compile` silently fell back to eager
  (Inductor `CantSplit`), so the documented ~2.4× fusion was never active; and the GPU
  was under-fed (low concurrency).
- **Shipped & validated** (branch `claude/wizardly-johnson-x90akh`, pushed):
  working compile, GPU/host overlap, resumable epochs, and a pure-efficiency config tune
  (`active_games` 64→192). All numeric-path changes are **bit-identical / fp16-parity**
  gated.
- **Honest perf status:** large GPU-throughput gains were measured in *early-game* sweeps
  (evals/s 416→699). The **full-epoch** benefit is **not yet confirmed** — late-game
  large-support positions are compute-bound and the live cumulative pos/s settles low
  (~2.3 when all 192 games are deep in late game). The honest metric to compare is
  **epoch wall-clock**, not instantaneous pos/s.
- **Staged, NOT validated** (branch `claude/inference-rewrite`, worktree
  `E:/Hexo-BotTrainer-hexgt-rewrite`): a 29-agent designed inference rewrite (`hexflash`
  Triton kernel hybrid). It does not build as-assembled (6 contradictions, resolved on
  paper) and its own perf review doubts it beats the baseline. Needs a GPU pause to
  build + parity + benchmark as a go/no-go.

---

## 1. The live run

- **Run:** `hexfield_main_1` → dir `/mnt/e/Hexo-BotTrainer/runs/hexfield_main_1`
- **Supervisor:** `scripts/_hexfield_supervise_main1.sh` (bash loop; auto-resumes from the
  latest checkpoint; circuit breaker: fast-crash<300s ×3 OR >8/hr → writes
  `supervisor_halted.flag` and stops). Holds `supervisor.lock`.
- **Venv:** `/root/.venvs/hexgt-build` (has torch 2.12+cu130, CUDA). NOTE:
  `/root/.venvs/hexfield-dev` is the **build** venv (Rust toolchain + maturin) and has **no
  torch** — use it to BUILD the `.so`, not to run.
- **Process model:** the trainer (`hexo_train.cli.train_model`) runs **all 200 epochs in one
  process**; the supervisor only relaunches on crash. So `torch.compile` warmup is a
  one-time cost (also disk-cached at `/tmp/torchinductor_root`).
- **hexfield loads from source** via `PYTHONPATH` (set in the supervisor) —
  `packages/hexfield/python` — and the in-tree `_rust*.so`. There is **one** `.so`; the
  live process mmaps it, so rebuilding on disk only takes effect on the next process start.
- **Hardware:** RTX 4070 Ti, 12 GB, SM 8.9, WSL2 Ubuntu-24.04.

### Operating gotchas
- **Never stop the run mid-epoch without the resumable-epoch path.** Each finished game
  writes its own `samples/epoch_NNNNNN/game_*.npz` immediately; training reads a rolling
  mtime window over all `epoch_*/game_*.npz`. The resumable-epoch patch (this session) makes
  a restart KEEP finished shards and generate only the remainder (non-colliding keys), so a
  restart no longer recomputes finished games.
- **Clean stop:** `kill <supervisor_pid>` first (so it won't relaunch), then `kill
  <trainer_pid>`. The supervisor trap removes `supervisor.lock`.
- **Clean restart:** `cd /mnt/e/Hexo-BotTrainer-hexgt && nohup bash
  scripts/_hexfield_supervise_main1.sh >> <run>/_supervisor_console.log 2>&1 &` (clear
  `supervisor_halted.flag` first if the breaker tripped).
- **`pkill -f <pattern>` foot-gun:** the Bash tool's own command line contains the pattern,
  so `pkill -f 'epoch_000031.pt'` (etc.) kills the calling shell. Kill by explicit PID.

---

## 2. Changes committed this session

Branch **`claude/wizardly-johnson-x90akh`** (the live branch), pushed to origin.

### Commit `ec253eb` — compile fix + GPU/host overlap + resumable epochs
- **`inference.py` — torch.compile fix (the #1 win).** The old
  `torch.compile(dynamic=True)` always fell back to eager: Inductor `CantSplit: 96*s+768
  not divisible by s+8` (it can't prove `CHANNELS*(Npad+8)` divisible by the symbolic
  seq-len). Replaced with: `automatic_dynamic_shapes=False`, `cache_size_limit≥64`, and per
  forward-group `mark_dynamic(batch dim 0)` + `mark_static(cell dim 1)`, **gated to small
  `Npad` (`HEXFIELD_COMPILE_MAX_NPAD`, default 512)**; large `Npad` runs eager (already
  SDPA flash-class). Result: 0 CantSplit, 0 recompiles, bit-exact fp16 parity, ~2.4× forward
  on small-S. Opt out with `HEXFIELD_NO_COMPILE=1`.
  - *Gotcha learned:* a per-bucket compile **dict does not work** — dynamo caches by
    function code, so the 2nd shape still triggers automatic-dynamic → CantSplit. The
    `automatic_dynamic_shapes=False` + explicit marks is what fixes it. And `mark_static`
    on `Npad` alone over-specializes (60+ shapes → recompile-limit → eager); hence the
    small-`Npad` gate.
- **`inference.py` + `payload.rs` — GPU/host overlap (`HEXFIELD_ASYNC_EVAL`).** Split the
  serve forward into `submit_payload` (enqueue, **no** device sync) + `result` (the single
  D2H drain). Rust adds `submit_eval_cached` / `finish_eval_cached`; the `search.rs` flush
  loop, under the flag, does **submit → (GIL-released) prefetch select → finish** so the
  existing pre-backup select overlaps the in-flight forward. Bit-identical on identical
  inputs (the deferred `.cpu()` syncs before any read). On by default via the supervisor.
- **`selfplay.py` — resumable epochs.** `generate_selfplay_epoch` counts existing
  `game_*.npz`, generates only `games_per_epoch − already_done` with keys past the max
  existing key, guards `remaining==0`, and writes a separate `*_resumeNNN.hxr` so the
  interrupted run's replays aren't truncated.
- **Snapshot note:** this commit also bundled pre-existing in-tree (uncommitted) eval-v2 /
  config / test changes that the live run was already using (`config.py`, `eval_arena.py`,
  `evaluation.py`, `multistage_eval.py`, the eval-v2 spec, some tests) — not authored this
  session, committed because they were the live working state.
- New harnesses: `scripts/_hexfield_compile_overlap_test.py` (compile + async parity,
  maxabsdiff gate), `scripts/_hexfield_async_parity.py` (end-to-end self-play action parity
  + throughput).

### Commit `43f9c0d` — pure-efficiency config tune + host-side patches
- **`configs/hexfield_main_1.toml`:** `active_games` 64→192, `active_root_limit` 192,
  `flush_target` 1024 (in-flight 192×4=768; constraint `flush_target > active_games×vbs`,
  and `active_root_limit ≥ active_games` per `search.rs:836`). cache_max_states unchanged
  (sweep showed it inert). **No search-behaviour knobs touched.**
- **`inference.py` P1:** `result()` emits `flat_priors` directly (it is already the exact
  concatenated `sum(legal_counts)` layout) — drops a per-row Python loop + `np.concatenate`.
- **`inference.py` P2:** pinned + `non_blocking` H2D in `_forward_group` — async DMA
  overlap. Both bit-identical (parity 0.0).
- New harness: `scripts/_hexfield_batch_sweep.py` (config/cache throughput sweep).

### `.so`
Rebuilt via `scripts/_rebuild_hexfield.sh` (hexfield-dev venv → maturin → copies
`_rust*.so` into the source tree). The `.so` is **not tracked** (build artifact); rebuild
after any Rust change.

---

## 3. Validation performed

- **Compile parity:** compiled vs eager bit-exact (maxabsdiff 0.0; one 1e-5 fp16 rounding)
  across single / uniform-64 / prod-mean-144 / skewed-200.
- **Async parity:** `result(submit(p)) == evaluate_payload(p)` exactly (0.0).
- **Live:** 0 CantSplit, 0 recompile-limit, 0 OOM/tracebacks at 128 and 192 games; VRAM
  healthy (≈3.4–4.8 GB; flat across active_games — per-group `PAIR_CEILING` caps the
  transient).
- **Note on parity gating:** self-play is inherently fp16-nondeterministic (off-vs-off runs
  diverge), so "identical games" is NOT a valid gate. Correctness rests on exact
  *same-input* parity, which passed.

### Sweep results (`_hexfield_batch_sweep.py`, 60s/config, early-game)
| active_games | flush_target | cache | evals/s | pos/s | VRAM |
|---|---|---|---|---|---|
| 64 | 1024 | 262k | 416 | 4.47 | 3.0 GB |
| 128 | 1024 | 262k | 598 | 6.87 | 3.4 GB |
| 192 | 1024 | 262k | 699 | 9.83 | 3.4 GB |
| 256 | 2048 | 262k | 711 | 12.0 | 3.4 GB |

- `active_games` is the dominant lever; `flush_target` and `cache_max_states` made <2%
  (both effectively inert here).
- evals/s **plateaus at 192** (GPU saturated); 256 needs `games_per_epoch≥256` to matter
  (it's `min(active_games, games_per_epoch=192)`), so 192 is the deployed choice.

---

## 4. Honest throughput status (read this before claiming a speedup)

- The sweep numbers are **early-game-biased** (60s window = small supports = fast). The
  **full-epoch** picture is different: late-game positions have huge supports (Npad up to
  ~3300), are O(S²) **compute-bound**, and do **not** benefit from concurrency (the GPU is
  already saturated there). Live cumulative pos/s was seen settling to ~2.3 when all 192
  games were deep in late game.
- Therefore the **true** benefit of `active_games=192` is **unconfirmed**. The clean
  measurement is **epoch wall-clock** (time to generate 192 games) vs the baseline epoch 30
  (~4322 s at active_games=64). That comparison has not been made (needs full epochs to
  complete). It is possible 192 is a wash or modest win at the full-epoch level.
- The shipped compile + async fixes are real and parity-safe regardless; they remove a
  genuine eager-fallback bug. The *config* lever is the one with unconfirmed full-epoch
  payoff.
- **Realistic pure-efficiency ceiling** (per the design workflow): ~5–6 pos/s steady-state
  (~1.5–1.8× over 3.3), then the GPU is matmul-bound on large-S and no scheduling/packing
  change helps.

---

## 5. The inference rewrite (STAGED — NOT validated)

Branch **`claude/inference-rewrite`**, worktree `E:/Hexo-BotTrainer-hexgt-rewrite`,
commit `4a0afdc`. Produced by a 29-agent workflow (off-GPU, off the live tree).

- **Architecture (hybrid, all dark behind env flags):**
  - **A. `hexflash`** — hand-written shape-generic Triton FA2 kernel that reconstructs the
    learned rel-pos bias **in-kernel** from `model._exact_lut` (S is a runtime arg → one
    binary, Npad 64..3300+). Primary for the large-S tail.
  - **A-fb. FlexAttention** `score_mod` fallback.
  - **B.** Rust-owned pinned serve plumbing + depth-2 submit/finish pipeline (default
    depth=1 — depth≥2 needs a pinned pool that is NOT built).
  - **C.** Keep the deployed gated compile (small-S) + universal fallback.
- **Where it lives:** `docs/inference_rewrite/` in the worktree —
  `00_CHOSEN_SPEC.md`, `01_INTEGRATION_RUNBOOK.md` (the Part-0 reconciliation + the
  GPU-pause validation plan), `02_REVIEWS.md`, `components/*.md` (10 code drafts), plus a
  drafted `packages/hexfield/python/hexfield/hexflash.py`.
- **Why it's not validated / not merged:**
  1. **Does not build as-assembled** — 7 adversarial reviews found 6 contradictions
     (3-way ABI field schism, two incompatible `inference.py` rewrites, coords dtype, flex
     pad-mask/scale parity gap, Triton `HEAD_DIM` module-global, depth-2 pinned-buffer
     unsoundness; plus non-compiling Rust tests). Resolved **on paper** in the runbook
     (R0.1–R0.8), not applied.
  2. **Payoff is doubted by its own perf reviewer:** the "70% bias machinery" win is a
     *small-S launch-bound* figure; large-S is matmul-bound (conv+heads ≈ half the forward,
     untouched). The only solid win is VRAM-freeing → bigger batches, unquantified.
     `hexflash` may *regress* the band it targets.
  3. A deploy footgun was caught and neutralized: the auto-generated parallel-supervisor
     hardcoded the **live** run dir (data-mixing risk). The runbook supersedes it with
     dark-flag rollback (no `$ROOT` swap).

---

## 6. What we are considering (open decisions)

1. **Validate the rewrite (go/no-go), or shelve it.** Recommended path: a deliberate GPU
   pause to (a) apply runbook R0, (b) build `hexflash`, (c) parity-test, (d) benchmark vs
   baseline on **real mid/late-game shapes (Npad 900–3300)** — a hard go/no-go. Do this
   BEFORE finishing the ~2,500-line assembly, because the payoff is unproven.
2. **Confirm the `active_games=192` win at the full-epoch level** (epoch wall-clock vs the
   ~4322 s baseline). If it's a wash/regression in late-game-heavy epochs, reconsider the
   value (possibly a lower active_games, or accept it for the early/mid-game gain).
3. **Bigger speedups need tradeoffs we've ruled out for now:** reducing `search_visits` /
   PCR (~1.5–2× but changes strength → arena A/B + owner sign-off), or a fixed-max-support
   representation (KataGo-style → unlocks TensorRT/CUDA-graphs/big-batch, but needs a
   retrain). Neither was pursued (the brief was pure efficiency).

### Ruled out (don't re-chase)
- `torch.compile(dynamic=True)` CantSplit "fix" — intrinsically unfactorable.
- Full-range static compile — recompile-limit blowup → eager (this is *why* the small-Npad
  gate exists).
- Flush-*floor* — hang risk (every flush is already select-exhausted).
- FlexAttention as primary — head_dim=24 (+33% FLOPs), batch-dependent BlockMask + dynamic
  bugs.
- TensorRT/ONNX — not live-safe, parity drift, ONNX seq-len specialization landmine.
- LRU cache / larger cache — sweep showed cache inert here.

---

## 7. Key flags, tools, branches

- **Env flags:** `HEXFIELD_ASYNC_EVAL` (overlap, default 1 in supervisor),
  `HEXFIELD_NO_COMPILE=1` (disable compile), `HEXFIELD_COMPILE_MAX_NPAD` (compile gate,
  default 512), `HEXFIELD_NO_PREFETCH` (parity-debug). Rewrite-only (staged):
  `HEXFIELD_ATTN_IMPL`, `HEXFIELD_PAYLOAD_ABI`, `HEXFIELD_PIPELINE_DEPTH`.
- **Build:** `scripts/_rebuild_hexfield.sh` (hexfield-dev venv; copies `.so` into the tree).
- **Tests/bench:** `_hexfield_compile_overlap_test.py`, `_hexfield_async_parity.py`,
  `_hexfield_batch_sweep.py`, `_hexfield_profile_fwd.py` (SDPA backend detector).
- **Branches:** `claude/wizardly-johnson-x90akh` (live, shipped) ·
  `claude/inference-rewrite` (staged rewrite, worktree `E:/Hexo-BotTrainer-hexgt-rewrite`).
- **Memory:** `hexfield-throughput-fix.md` (in the agent memory dir) has the condensed
  version of all of the above.
