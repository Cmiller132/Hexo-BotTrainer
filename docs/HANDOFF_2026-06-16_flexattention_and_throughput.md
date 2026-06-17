# Handoff — hexfield serve throughput: compile-all + FlexAttention (2026-06-16)

**Branch:** `claude/wizardly-johnson-x90akh` · **Run:** `hexfield_main_1` (WSL/systemd) · **GPU:** RTX 4070 Ti (12 GiB) · **torch:** 2.12.0+cu130 (already the latest — no update available/needed).

> ⚠️ This handoff CORRECTS earlier docs. The prior `docs/investigation/HANDOFF_2026-06-16_throughput_2x.md` premise — "the single GIL host thread is the wall" — was **early-game-biased profiling and is wrong** for the regime the run actually spends its time in. Trust the measurements here, not older overconfident conclusions.

---

## 0. TL;DR — what shipped this session

1. **Compile-all fix (shipped, live):** the serve `torch.compile` now works for **every** shape via ONE dynamic compile. Real root cause of the old failure was Inductor `CantSplit: 96*s+768 not divisible by s+8` on **concrete batch=1** (size-1 groups), NOT the async-pool deadlock the prior handoff guessed. Fix: `compile(dynamic=True)` + duplicate size-1 groups to batch 2. ~1.6× on the forward.
2. **Rigorous disproof of the host levers:** on a **representative** deep-game bench, the handoff's host-overlap levers — line-312 sync fix, async overlap, `thread::scope`, padding-waste, `active_games` — are **all NULL**. The large-game forward was **GPU-FLOP-bound**, dominated by the relative-position **bias build** (65% of the forward; SDPA matmul only 10%).
3. **FlexAttention (shipped, ENABLED via `HEXFIELD_SERVE_FLEX=1`):** computes the rel-pos bias **inside** the attention kernel (score_mod), eliminating the `(B,heads,S,S)` materialization. **Forward halved**, **~20× less serve VRAM**, **1.18–1.26× end-to-end pos/s**, parity within shipped fp16 tolerance.
4. **Bottleneck shifted:** after FlexAttention, GPU util **dropped** (74→65%) — the run is **no longer GPU-bound**. The wall is now **host-side per-decision work** (`window_scan` + `HexfieldSampleData` construction on the search thread). The host levers that were null before may now matter.

---

## 1. Current run state (VERIFY FIRST)

- Live run `hexfield_main_1` under WSL `Ubuntu-24.04`, managed by **systemd** (`systemctl status hexfield-supervisor`), dashboard on `:8080` (`hexfield-dashboard`).
- **FlexAttention is ENABLED**: `scripts/_hexfield_supervise_main1.sh` exports `HEXFIELD_SERVE_FLEX=1`. Confirm in the trainer env: `tr '\0' '\n' < /proc/$(pgrep -f cli.train_model|head -1)/environ | grep HEXFIELD_SERVE_FLEX`.
- **WSL persistence:** the distro tears down ~15s after the last client detaches. A background `wsl … sleep infinity` keepalive must be running, else systemd (and the run) die. Services are systemd-managed; do NOT relaunch via wsl.exe background tasks.
- Health check: newest `train.*.out.log` must show `CantSplit=0`/`InductorError=0`; `diagnostics/hexfield.selfplay.live.json` `positions_per_second>0` and `elapsed_seconds` advancing; GPU cycling (not stuck 0%/10W). One-time ~10s flex compile on the first serve forward of an epoch is expected (breaker tolerates it).

---

## 2. The compile-all fix (`inference.py`, `model.py`)

**Symptom:** the prior session capped compile at `Npad<=1024` because compiling all shapes "hung". **Real cause (measured, torch 2.12):** dynamo keeps a size-1 dim specialized (concrete), so for the serve path's **size-1 groups** (one big late-game row) Npad is the only free symbol and Inductor trips on the attention head-merge transpose-copy: `CantSplit: 96*s+768 not divisible by s+8` (`96=CHANNELS`, `768=96*NUM_TOKENS`, `s+8 = seq-len`). With batch≥2 the batch stays a free symbol and the SAME graph compiles for every Npad.

**Fix (`inference.py`):** ONE `torch.compile(forward_policy_value, dynamic=True)`; mark BOTH batch (dim0) and Npad (dim1) dynamic; **duplicate size-1 groups to batch 2** (`.repeat(2,…)`, slice `out[:1]` — pad-/batch-inert so exact). Removed the `HEXFIELD_COMPILE_MAX_NPAD` cutoff.

**Verified:** 1 compile (~27s first, disk-cached), serves Npad 256..2560 with 0 CantSplit; real-state parity max|dvalue| 6e-4–1.5e-3; ~1.6× on a 96-state late-game flush.

---

## 3. Throughput investigation — methodology + the GPU-bound finding

### 3.1 Representative bench (the key methodology fix)
Naive benches start games from an empty board → spend the window in the cheap early-game ramp → **overstate** pos/s 2×. The live run's slow time is **96 concurrently-LARGE games** (support ~1700–1900). `scripts/_hexfield_repr_bench.py`:
- Seeds each of 96 games from a **real deep .hxr position** (replay a random prefix of a real game; `SIZE_FLOOR` enforces large support).
- Replicates the live **per-decision host work** (`window_scan` + `HexfieldSampleData` over the growing record list — O(game-length)).
- Fixed seed → comparable across code versions. **Must** run with `HEXFIELD_ASYNC_EVAL=1` (the live eval path) or it's unrepresentative.

**Calibration:** `SIZE_FLOOR=1200` (support mean ~1880) → **3.1 pos/s ≈ live 3.3** (no mysterious 2× — it was the support distribution). `SIZE_FLOOR=0` (mix) → ~6.9 pos/s.

### 3.2 The disproof (all NULL on the representative regime)
| lever | result |
|---|---|
| #1 kill the per-group blocking `legal_counts.to(cuda)` (profiled "60%") | 3.14 → 3.16 (null; sync & async). The 60% was the host **spin-waiting on the GPU**, not removable work. Reverted. |
| async overlap vs sync eval | identical — overlap buys nothing |
| `active_games` 96→224 | util 62→72%, flush 226→458, pos/s +4% (padding waste cancels it) |
| padding-waste ↓ (WASTE_FRACTION 0.18→0.02) | 3.12 → 3.07 (null) |

**Conclusion:** GPU util 62% (mix) / 73–75% (large); the forward is GPU-FLOP-bound at ~308 large-evals/s. Host levers can't move it.

### 3.3 Forward breakdown (`scripts/_hexfield_forward_profile.py`, Npad=1920, B=8)
| component | eager ms | share |
|---|---|---|
| full forward | 21.45 | 100% |
| **rel-pos bias build** (`build_attn_bias`) | 14.0 | **65.2%** |
| SDPA matmul | 2.2 | 10.3% |
| conv trunk + projections/MLP + heads | 5.3 | 24.8% |

Compiled profiler: the three `(B,heads,S,S)` bias-construction triton kernels (`constant_pad` + `gather`) ≈ **68%**, fmha 17%, GEMMs 5%. **The forward is dominated by materializing the bias, not the attention matmul.**

---

## 4. FlexAttention — the real win

**Idea:** compute the bias **inside** the attention kernel via a `score_mod` (the same additive bias: token/cell region select → cell-cell `lut[clamp(dq)*W+clamp(dr)]` → `bias_table[row,h]`, pad-key mask folded as a `-3e4` fill), so the `(B,heads,S,S)` tensor is never built.

**Implementation (all in `model.py`, serve-only, flag-gated `HEXFIELD_SERVE_FLEX`, default behavior unchanged when unset):**
- `trunk()` builds a `_FlexBias` carrier (raw tensors) **only** when `_serve_flex and not torch.is_grad_enabled()`. Training/grad path (`build_attn_bias` + `_BiasGather`) is **untouched**.
- `RelPosAttention.forward` builds the `score_mod` **closure locally** (same frame) and calls flex.
- **Two integration fixes that are mandatory (else it crashes):**
  1. `_flex_call` is wrapped in `@torch.compiler.disable(recursive=False)` so the flex op **graph-breaks** out of the outer `compile(dynamic=True)` serve graph and compiles in its OWN inner graph — tracing the flex HOP into the outer graph trips `lift_tracked_freevar_to_input should not be called on root SubgraphTracer`.
  2. Build the `score_mod` in `RelPosAttention.forward`, NOT in `trunk()` (threading a closure two frames down re-triggers the same freevar crash). `_FlexBias` carries tensors; the closure closes over them locally.
- `score_mod`-ONLY for pad masking (no `create_block_mask`) → no per-shape `BlockMask` object → no extra dynamic-shape recompile surface.

### 4.1 Benchmark results (RTX 4070 Ti, epoch-34 ckpt, `HEXFIELD_ASYNC_EVAL=1`)
**Parity (gate):** flex-OFF bit-exact (all deltas 0.0 @ tol 1e-6); flex-ON max|dvalue| **1.24e-3**, max|dprior| 6.15e-4, max|dml| 0.0 — within the shipped 3e-3 fp16 tolerance (bias is bit-exact on every live key; only divergence is a softmax-invisible fp16 floor on pad keys).

**Dynamic-shape robustness:** exactly **1 compile (~10.3s)**, 21 Npad shapes {256..2560} at B∈{2,8} + size-1→batch-2 reuse in 4–109ms. **0 CantSplit, 0 InductorError, 0 recompiles, 0 eager-fallback.**

**Forward:** the three bias-construction kernels are **GONE** (fused into one flex kernel). Eager forward **21.14 → 10.59 ms**. Peak serve VRAM **1.51 → 0.07 GiB (~20×)**.

**End-to-end throughput:**
| regime | pos/s OFF → ON | speedup | GPU util | peak VRAM |
|---|---|---|---|---|
| large (SIZE_FLOOR=1200) | 2.77 → **3.48** | **1.26×** | 74% → 65% | 1.51 → 0.07 GiB |
| mix (SIZE_FLOOR=0) | 4.70 → **5.56** | **1.18×** | 70% → 58% | 1.48 → 0.13 GiB |

**Why 1.2× end-to-end and not 13.7× (the attention-kernel microbench):** Amdahl. The forward roughly halved, but per-decision host work (`window_scan` + sample construction) is untouched. Evidence the forward genuinely got cheaper: **GPU util dropped** and VRAM collapsed ~20×.

---

## 5. Next steps (the bottleneck moved)

After FlexAttention the run is **host-bound**, not GPU-bound. The levers that were null when GPU-bound now have headroom:
1. **Reduce/offload per-decision host work** — the O(game-length) `window_scan` + `HexfieldSampleData` construction runs on the search thread every full PCR decision. Move it off the critical thread (background) or make it incremental.
2. **`thread::scope` overlap / re-test active_games** — now that the GPU isn't the limiter, host-overlap may finally pay.
3. **Decode-fuse** — smaller, but the host softmax/decode launches are now a larger relative share.
Re-measure each on `_hexfield_repr_bench.py` (SIZE_FLOOR=1200, `HEXFIELD_ASYNC_EVAL=1`, flex ON).

---

## 6. Tooling / scripts (new this session)
- `scripts/_hexfield_repr_bench.py` — **representative** throughput (deep-seed + per-decision overhead + GPU-util sampler). The instrument to use.
- `scripts/_hexfield_serve_ref.py` — deterministic serve-output parity (`save`/`check <tol>`).
- `scripts/_hexfield_forward_profile.py` — forward ablation + kernel breakdown.
- `scripts/_hexfield_flex_probe.py` — standalone FlexAttention parity/speed probe.
- `scripts/_hexfield_compile_diag.py`, `scripts/_hexfield_flex_compile_diag.py` — dynamic-compile / CantSplit diagnostics.

## 7. Run management cheat-sheet
- Start/stop: `systemctl {start,stop,restart,status} hexfield-supervisor` (+ `hexfield-dashboard`). Wipe a partial epoch before restart: `rm -f <run>/samples/epoch_0000NN/game_*.{npz,json} <run>/selfplay/epoch_0000NN*.hxr`.
- Toggle flex: `HEXFIELD_SERVE_FLEX` in `scripts/_hexfield_supervise_main1.sh` (1=on, unset/0=materialized bias). Toggle compile: `HEXFIELD_NO_COMPILE=1`.
- Bench (GPU must be free → stop supervisor): `HEXFIELD_ASYNC_EVAL=1 python scripts/_hexfield_repr_bench.py <ckpt> 75 96 /tmp/b.json real full 1200`.
- Keepalive: a background `wsl … 'exec sleep infinity'` must stay attached or the distro tears down.
