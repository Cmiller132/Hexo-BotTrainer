The ground-truth confirms the reviewers' key claims: `set_attention_impl` is a global mutable flip across all 3 blocks (model.py:256-258), the live `_forward_group` uses positional args and per-call `pin_memory()` (inference.py:206-249), and `build_attn_bias` is already fp16-optimized on the inference path (model.py:314-315). I have what I need to write the runbook.

---

# HEXFIELD INFERENCE REWRITE — INTEGRATION + VALIDATION RUNBOOK

**Target worktree:** `E:\Hexo-BotTrainer-hexgt-rewrite` · branch `claude/inference-rewrite`
**Live tree (READ-ONLY, do not write):** `E:\Hexo-BotTrainer-hexgt`
**Default posture:** every new path is dark. `HEXFIELD_ATTN_IMPL=sdpa`, `HEXFIELD_PAYLOAD_ABI=1`, `HEXFIELD_PIPELINE_DEPTH=1` reproduce the deployed forward byte-for-byte.

This runbook supersedes the build/deploy component's parallel-supervisor mechanism, which the deploy-risk review correctly flagged as data-unsafe. **Rollback is the env flag, not a `$ROOT` swap.**

---

## PART 0 — RECONCILIATION PASS (do this BEFORE writing any file)

The 7 reviews surface **6 hard contradictions that make "assemble the accepted components" produce a non-building tree.** These are static (no GPU) and MUST be resolved first. The decisions below are frozen for the rest of the runbook.

### R0.1 — FREEZE the v2 wire schema (resolves the 3-way field schism)
Three components invented three key vocabularies (`gather_index` / `nbr_local`+`nbr` / `v2_*`). **Frozen single schema** — Rust emits, Python reads, exactly:

```
abi:               2            (int)
shape:             [total_nodes] / per-group meta as v1
node_feats:        f16  bytes, node-major, NUM_FEATURES wide
node_coords:       i32  bytes, node-major, 2 wide  (axial q,r)
nbr:               i32  bytes, node-major, 6 wide  (NBR_SENTINEL = 65535 preserved)
cu_seqlens:        i32  bytes, B+1                 (== v1 node_row_offsets values)
legal_counts:      i32  bytes, B
request_moves_left: bool
```

- **DROP** Rust's 7-wide `gather_index`/`gather_width`/`gather_sentinel`. No consumer wants it — `HexfieldNet.trunk` builds `self_idx = arange` + `cat([self_idx, nbr])` itself (model.py:346-347, verified). Trunk stays unchanged.
- **Sentinel is 65535 (`NBR_SENTINEL_U16`, constants.py:35), NOT -1.** The i32 `nbr` buffer carries 65535; Python's `np.where(row_nbr == NBR_SENTINEL, pad_to, row_nbr)` (inference.py:226) matches as written.
- `_v2_fields_present` (fallback-compat) must be rewritten to probe these exact keys; kill the `v2_*` namespace entirely.

### R0.2 — ONE inference.py (resolves the two-incompatible-rewrites blocker)
evaluator-packing ships a full-file rewrite with `_forward_group(self, src, ...)`; fallback-compat ships diffs against the live positional `_forward_group(self, feats, qr, nbr, offsets, ...)`. They cannot coexist.
- **Base = evaluator-packing's full rewrite** (it has the `_parse_v1`/`_parse_v2` dispatch and per-group `src` bundle).
- **Rebase fallback-compat's logic onto it**: keep fallback-compat's `resolve_group_impl` + `_run_forward` (it has degrade/strict-mode/`finally: set_attention_impl(base)` restore that the bare `_ensure_impl` lacks).
- **DELETE evaluator-packing's `_select_impl`/`_ensure_impl`.** One routing mechanism only.

### R0.3 — coords dtype, ONE owner (resolves the triple-cast hazard)
- **`trunk` owns the int32 cast** (`kcoords = coords.to(torch.int32)` for the kernel path; model.py path under `use_kernel`).
- **evaluator-packing always sends int64 coords.** Remove the int32 cast from both evaluator-packing `_forward_group` and fallback-compat `_run_forward`.
- `build_attn_bias` keeps int64 indexing (`_exact_lut[(clamped_q*17+clamped_r)]`, model.py:278) → SDPA-path parity preserved; the degrade-from-hexflash-to-sdpa path now hands int64 to `build_attn_bias` unconditionally.

### R0.4 — flex must match `materialized`, not `mask_mod` semantics (resolves the worst parity gap)
The live `materialized` oracle adds the literal finite `PAD_KEY_MASK_VALUE = -3.0e4` (model.py:329); flex's `mask_mod` would write `-inf`. Bit-parity against the Tier-2 oracle requires:
- **Fold `PAD_KEY_MASK_VALUE` additively into `score_mod`** (`score + bias_val + (live ? 0 : -3.0e4)`), do **not** use `mask_mod` for pad keys. This makes flex match `materialized` exactly. (You may keep a `mask_mod` returning all-True or omit it.)
- flex explicitly passes `scale=1/sqrt(24)` (overrides the 1/sqrt(32) padded-dim default). Confirmed required.

### R0.5 — `HEAD_DIM` as `tl.constexpr`, not a module global (resolves the Triton JIT build-risk)
The hexflash kernel references `HEAD_DIM_CONST` as a module global defined *after* the `@triton.jit` function. Triton global-capture ordering is fragile and untested.
- **Pass `HEAD_DIM: tl.constexpr` as an explicit kernel argument.** This is what the 24→32 zero-lane mask exactness depends on; it must be compile-time. Removes the only execution-time unknown in head-dim padding.
- Add explicit assertion **before** any hexflash/flex call: `assert bool(seq_mask[:, :NUM_TOKENS].all())` — the token-key-always-live invariant is the *only* thing preventing a 0/0 (or -inf NaN) softmax row. Make it load-bearing-explicit, not implicit.

### R0.6 — depth defaults to 1; pinned-pool prerequisite is NOT built
The pinned-staging-pool that depth≥2 requires does not exist in any delivered component (evaluator-packing still does per-call `t.pin_memory()`, inference.py:243-244 — verified). With depth≥2 the host source for flush k is dropped/reused before the async H2D drains → **silent corruption from freed pinned memory.**
- **`HEXFIELD_PIPELINE_DEPTH` default = 1** in `search.rs` AND in every Python resolver. Reconcile the serveloop default (it ships 2) down to 1.
- Depth-2 is **experimental, off by default**, and gated behind an explicit pinned-pool implementation that is NOT in scope for first pause. Ship the FIFO `VecDeque<InFlightFlush>` plumbing (it's sound) but cap inflight at 1.

### R0.7 — Rust unit tests do not compile (resolves the void static de-risk)
The delivered `#[cfg(test)]` fixtures call `RustHexoState::new()`, `state.is_legal_placement()`, `state.apply_placement()` — none exist. Real API: free fns `rules::is_legal_placement(&HexoState, HexCoord) -> Result<(), MoveError>` and `apply_placement(&mut state, Placement{coord}) -> Result<...>`. **Rewrite the three fixtures against the free-fn API and the real `RustHexoState` constructor before they can serve as the gather-layout gate.** Confirm `DIRECTIONS` import path against `support.rs`.

### R0.8 — fix the test/script path errors
- Tests live at **repo-root `tests/test_hexfield_model.py`**, NOT `packages/hexfield/python/tests/`. Point all validate scripts there.
- Fix `_make_payload`'s dead `if False else` ternary (parity-tests author flagged but shipped it).
- Align benchmark attr introspection (`_attn_impl`/`_supports_abi2`) to the real evaluator attr names or it reports `<absent>` and misleads the operator.

> **Gate G0 (no GPU):** none of R0.1–R0.8 require a GPU. They must ALL be closed before assembly. If any is open, STOP — the tree will not build.

---

## PART 1 — FILE-BY-FILE ASSEMBLED CHANGE LIST

Write into the rewrite worktree. (paths relative to `E:\Hexo-BotTrainer-hexgt-rewrite`)

| # | File | Action | Incorporated fixes |
|---|------|--------|--------------------|
| 1 | `packages/hexfield/python/hexfield/hexflash.py` | **NEW**. `hexflash_attention(...)` (Triton FA2, frozen §A1 sig) + `flex_attention_relpos(...)`. | R0.4 (additive pad fill in score_mod), R0.5 (`HEAD_DIM: tl.constexpr` arg), R0.5 token-key assert. Import all constants from `constants.py` — never duplicate `BIAS_*`/`PAD_KEY_MASK_VALUE`/`HEAD_DIM`/`NUM_TOKENS`. |
| 2 | `packages/hexfield/python/hexfield/model.py` | **EDIT**. `RelPosAttention.forward` + `AttnBlock.forward` + `trunk`: add `hexflash`/`flex` branches that skip `build_attn_bias` and thread `coords,seq_mask,bias_table` when impl∈{hexflash,flex} AND `not torch.is_grad_enabled()`. `set_attention_impl` accepts the 2 new strings. | R0.3 (trunk owns int32 cast). q/k/v proj+scale+out_proj bit-identical to SDPA. `*m` re-zero (model.py:193) untouched. `build_attn_bias` untouched (training + oracle). |
| 3 | `packages/hexfield/python/hexfield/inference.py` | **REPLACE** with merged file (R0.2). `submit_payload` dispatches abi via `_parse_v1`/`_parse_v2`; per-group impl via `resolve_group_impl`+`_run_forward`; large-S (`pad_to > HEXFIELD_LARGE_NPAD`=512) → kernel, else compile/SDPA. | R0.1 (`nbr` i32, sentinel 65535), R0.2 (single routing), R0.3 (always int64 coords), R0.6 (depth default 1). Single-D2H decode discipline UNCHANGED (inference.py:166-204). |
| 4 | `packages/hexfield/python/hexfield/constants.py` | **EDIT** (assert-only). Assert kernel constants imported here, not duplicated. No new constants. | — |
| 5 | `packages/hexfield/rust/src/payload.rs` | **EDIT**. Add `build_chunk_payload_v2` emitting the frozen R0.1 schema. v1 path + reply ABI (`parse_chunk_reply` payload.rs:154, `finalize_priors` payload.rs:621) + cache + dedup UNCHANGED. | R0.1 (drop 7-wide gather_index; emit 6-wide `nbr` i32 sentinel 65535; `cu_seqlens` i32 with `i32::try_from` overflow guard). |
| 6 | `packages/hexfield/rust/src/search.rs` | **EDIT**. `run_continuous`: FIFO `VecDeque<InFlightFlush>`, strict in-order `drain_one_flush`. **Inflight cap = `HEXFIELD_PIPELINE_DEPTH` default 1.** `py.detach()` GIL points (search.rs:1010,1058) preserved. | R0.6 (default depth 1; depth≥2 disabled until pinned pool exists). |
| 7 | `packages/hexfield/rust/Cargo.toml` | **EDIT** (if any FFI dep). Likely none after dropping zero-copy DLPack. | — |
| 8 | `tests/test_hexfield_model.py` | **EDIT**. Add Tier-1 bias-index oracle + Tier-2 fp16 oracle for `hexflash` and `flex` vs `materialized` AND `sdpa`. | R0.4, R0.5; add key-COLUMN check (not query-row-only) to `test_attention_kernel_core_matches_materialized`. |
| 9 | `packages/hexfield/rust/src/payload.rs` (`#[cfg(test)]`) | **EDIT**. Rewrite 3 fixtures against real engine API. | R0.7. |
| 10 | `scripts/_hexfield_compile_overlap_test.py` | **EDIT**. Add hexflash/flex evaluator builds; extend `cases` with large-S (1024/2048/3300); add v2-vs-v1 `torch.equal` scatter assert. | reuses `TOL=3e-3` (line 118), `maxabsdiff==0.0` async gate (line 130). |
| 11 | `scripts/_hexfield_async_parity.py` | **EDIT**. Add a run with `HEXFIELD_ATTN_IMPL=hexflash` (and one `=flex`). Depth stays 1. | — |
| 12 | `scripts/_hexfield_batch_sweep.py` / `_hexfield_profile_fwd.py` | **EDIT** (bench). Add per-impl ms/flush + peak-VRAM + evals/s@192 columns; the **go/no-go throughput gate**. | perf-review fix: no cutover without a throughput gate. |

---

## PART 2 — APPLY ORDER

Dependency-ordered. Each step builds on frozen contracts from PART 0.

1. **constants.py** (#4) — single source of truth; everything imports from it.
2. **hexflash.py** (#1) — pure function, no model/inference imports. Build it standalone.
3. **model.py** (#2) — depends on #1 for the kernel call.
4. **payload.rs v2 + Cargo** (#5,#7) — Rust wire, independent of Python attn.
5. **payload.rs tests** (#9) — depends on #5.
6. **inference.py merged** (#3) — depends on #2 (impl routing) and #5 (v2 parse).
7. **search.rs** (#6) — depends on #3's submit/finish API (unchanged).
8. **tests + harnesses** (#8,#10,#11,#12) — depend on everything above.

> **Gate G1 (no GPU):** `cargo build` + `cargo test -p <crate> payload` (Rust fixtures compile and PASS — R0.7), `python -c "import hexfield.hexflash"` (import order OK — R0.5), `pytest tests/test_hexfield_model.py -k pair_index` (**Tier-1 bias-index oracle — the single most important pre-GPU gate**, statically certain: integer math reusing `model._exact_lut`). If Tier-1 fails, the whole hexflash bet is dead on arrival — fix before touching a GPU.

---

## PART 3 — GPU-PAUSE BUILD + PARITY + BENCH SEQUENCE (go/no-go)

Run ONLY during the operator's deliberate pause, after the live training process is stopped. **Use an ISOLATED run dir** seeded with a COPY of the live checkpoint (see PART 0 deploy-risk + PART 4). No live-tree writes.

Execute in order; STOP at the first failed bar.

| Step | Command (rewrite worktree, hexfield-dev venv) | GO bar | NO-GO → action |
|------|-----------------------------------------------|--------|----------------|
| **3.0 Build** | `maturin develop --release` into hexfield-dev; mirror `.so` to source | `_rust.cpython-312-*.so` present | wrong py-minor → fix glob (R0.8); else fix build |
| **3.1 Tier-2 fp16 oracle** | `pytest tests/test_hexfield_model.py -k "fp16_cuda"` | `hexflash` vs `materialized` AND vs `sdpa`: fp16 diff ≤ **2e-3** (expect ~1.2e-4), fp32 ≤ 1e-4 | hexflash fails → switch primary to `flex`, rerun. flex fails → **Layer A blocked**, ship B+C only |
| **3.2 Layer-B byte parity** | `_hexfield_compile_overlap_test.py` with abi2 vs abi1 | v2 scatter `torch.equal` v1; ASYNC `maxabsdiff == 0.0` | any nonzero → ABI bug, keep abi=1 |
| **3.3 Compile-parity (incl large-S)** | `_hexfield_compile_overlap_test.py`, cases += 1024/2048/3300 | values/priors/moves_left vs eager ≤ **TOL=3e-3** at ALL sizes | large-S fail → hexflash off for that band |
| **3.4 Action parity** | `_hexfield_async_parity.py` with `HEXFIELD_ATTN_IMPL=hexflash`, depth=1 | end-to-end self-play **action sequence identical** to baseline | any divergence → Layer A off |
| **3.5 Pinned lifetime** | `_hexfield_async_parity.py` depth=1, stress flush sizes | no corruption; no use-after-free (depth=1 is the safe path) | — |
| **3.6 Throughput gate (THE perf go/no-go)** | `_hexfield_batch_sweep.py` per-impl | large-S **ms/flush(hexflash) < ms/flush(baseline eager SDPA)** on the SAME band, AND evals/s@192 ≥ baseline | **hexflash slower → stay `sdpa` for that band.** This is mandatory; correct-but-slower must NOT deploy |
| **3.7 VRAM headroom** | bench peak VRAM @192 | hexflash path frees the `(B,4,S,S)` bias transient; measure the new ceiling | record actual; raise `PAIR_CEILING` only by the measured margin |

**flex-only extra (if 3.1 routed to flex):** confirm head_dim 24→32 hits a real kernel (not slow tail) on torch 2.12; all 3 A-blocks compile + share BlockMask; dynamic seqlen recompiles bounded; fp16+pad precision bug (torch #163588) does not fire with the additive-score_mod approach.

> All of 3.1–3.7 are GPU-only and **not statically certain.** Only Tier-1 (G1) is certain.

---

## PART 4 — FLAG-GATED DEPLOY + INSTANT ROLLBACK

**The deploy vehicle is the env flag picked up by the NORMAL live supervisor on its next epoch relaunch — NOT a parallel training supervisor on the shared run dir.** (Deploy-risk review: the parallel supervisor runs full `train_model`, advancing weights/self-play/eval_pool into the live dir; a numeric defect would bake into `epoch_000032.pt` irreversibly. Delete that mechanism.)

### Pre-cutover (mandatory, deploy-risk fixes)
1. **Backup, verified.** Copy `epoch_000031.pt` + `eval_pool.json` + `manifest.json` to an immutable backup path with a checksum. Rollback restores *that exact* checkpoint, not "latest" (which may be rewrite-produced).
2. **Isolated validation already done** in PART 3 against a copied run dir — no rewrite data in `hexfield_main_1`.

### Cutover (data-safe)
3. Merge accepted rewrite CODE into the live tree during the pause (code only — forward semantics, no weight change).
4. Relaunch the normal live supervisor with defaults still dark: `HEXFIELD_ATTN_IMPL=sdpa`, `HEXFIELD_PAYLOAD_ABI=1`, `HEXFIELD_PIPELINE_DEPTH=1` → byte-identical to deployed. Confirm one clean epoch.
5. Flip ONE flag at a time, large-S only:
   - `HEXFIELD_PAYLOAD_ABI=2` (Layer B; byte-gated, lowest risk) — confirm an epoch.
   - then `HEXFIELD_ATTN_IMPL=hexflash` (Layer A; only if 3.6 throughput gate passed).

### Instant rollback (the primary mechanism)
- **`HEXFIELD_ATTN_IMPL=sdpa` + `HEXFIELD_PAYLOAD_ABI=1` + `HEXFIELD_PIPELINE_DEPTH=1`, relaunch.** Forward becomes byte-identical to baseline (`set_attention_impl('sdpa')` is just an attribute flip; abi=1 is the live numpy path). Reversible and data-safe because it never produces divergent weights.
- If a process must be killed: SIGTERM → 20s → SIGKILL, **then poll `nvidia-smi` compute-apps == empty BEFORE removing `supervisor.lock` and relaunching** (deploy-risk fix: lock removal before GPU-free confirmation can double-launch trainers and OOM the saturated 12 GB card).

---

## PART 5 — HONEST EXPECTED-SPEEDUP VERDICT + RESIDUAL RISKS

### The defensible win is **VRAM → larger flushes**, NOT the attention kernel.
The performance review's recomputation is sound and I concur:
- **Layer A (hexflash) likely REGRESSES the large-S band it targets.** The +33% head-pad tax (24→32) lands on the most expensive part of the large-S forward; a hand Triton FA2 realistically runs 0.5–0.7× of the deliberately-steered fp16 mem-efficient SDPA (model.py:317–331). Net end-to-end vs baseline ≈ **0.64–0.91×** across Npad 900–3300. Even best-case (k=1.0 util) ≈ 0.83–0.91× from the pad tax alone. The "1.2–1.5× large-S" claim is **not supported by the compute structure.** Removing the `build_attn_bias` transient saves only ~1/29th of the quadratic cost (bias S² coefficient ~20 vs attention-matmul ~576). **hexflash is experimental-behind-flag; it deploys ONLY if step 3.6 proves it faster.**
- **Real lever #1 — VRAM:** eliminating the `(B,4,S,S)` fp16 bias transient (the thing `PAIR_CEILING` exists to bound, inference.py:26) frees memory → higher `PAIR_CEILING` → larger flushes → better saturation at active_games=192. Quantify at step 3.7; that margin is the acceptance target, not a 1.2× forward.
- **Real lever #2 — Layer B (Rust pinned v2 ABI):** Amdahl-orthogonal. Removes the numpy per-row pack (inference.py:220–232) at the host/launch-bound regime that dominates at 192 games. Cleanest throughput story, byte-exact. **Land B first.** Note the headline-narrowing fact (completeness review, verified at model.py:314–315): `build_attn_bias` is *already* fp16-optimized, so the "~70% bias cost" baseline being beaten is the *current optimized* path — the win is smaller than naive.
- **flex** is a correctness fallback only, not a perf peer (recompiles BlockMask per (B,S); worse than SDPA at S<128).

**Bottom line:** expect a **throughput gain from Layer B + raised PAIR_CEILING (modest, single-digit to low-double-digit % at 192 games)**, with Layer A neutral-to-positive ONLY on the large-S tail and ONLY if the 3.6 gate passes. Do not promise a forward-kernel multiplier.

### Residual risks / what could still kill it
1. **hexflash throughput** (3.6) — most likely outcome is it fails the gate; then large-S stays SDPA and Layer A delivers nothing. Mitigated: dark by default, gate-protected.
2. **Native Dh=24 kernel not written** — the pad tax is structural until someone writes `BLOCK_DMODEL=24` tiles. Without it, hexflash is "a regression dressed as a speedup" (perf review). The flag protects, but the win is contingent on this future work.
3. **Triton compile on installed version** (3.0/3.1) — `tl.constexpr HEAD_DIM` (R0.5) removes the known NameError risk, but first-call JIT on this Triton is unverified until the pause.
4. **flex fp16+pad precision** (torch #163588) — only if flex is the chosen path.
5. **Depth-2 pipeline is NOT shipping** — the pinned-pool prerequisite is unimplemented; depth stays 1. The expected overlap win from depth-2 is deferred, not delivered.

### Unresolved review issues flagged for the operator
- **OPEN (deferred, not fixed):** depth-2 pinned-staging pool — no component implements it; depth-2 cannot ship. Closing requires an evaluator-owned depth-N pinned pool referenced by the `PendingEval` handle. **Until then, the depth-2 throughput win is vaporware.**
- **OPEN (needs GPU):** the entire Layer-A speedup is unverifiable until 3.6. The honest expectation is *regression* on the kernel; only VRAM/Layer-B are confidently positive.
- **OPEN (needs GPU):** native-Dh=24 kernel to clear the pad tax — not in scope for this assembly; hexflash ships only if it somehow clears 3.6 *with* the tax (unlikely).
- **CLOSED by R0 (static, must verify in assembly):** v2 field schism (R0.1), dual inference.py (R0.2), coords dtype (R0.3), flex/materialized parity (R0.4), Triton global ordering + token-key assert (R0.5), depth default (R0.6), non-compiling Rust tests (R0.7), wrong test paths + dead ternary + bench attrs (R0.8). **If any R0 item is left open during assembly, the tree will not build — these are the gating blockers, not the GPU steps.**

**Net verdict:** Ship Layer C (unchanged) + Layer B (byte-gated, real win) + raised PAIR_CEILING (real win). Treat Layer A hexflash as experimental-behind-flag, deploy-blocked by the 3.6 throughput gate; expect it to fail that gate until a native-Dh=24 kernel exists. No path endangers the live run as long as the env-flag rollback (PART 4) replaces the parallel-supervisor mechanism and the checkpoint backup is taken first.