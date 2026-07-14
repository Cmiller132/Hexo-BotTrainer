I have full ground truth on every interface, constant, and parity gate. The designs' claims check out against the source. Now I'll commit to the architecture and write the implementation spec.

The decision: the three judges split 2-for-hexflash, 1-for-FlexAttention, and all three converge on the *same hybrid* (small-S keep compile, large-S new kernel, orthogonal Rust serve plumbing). I commit to that hybrid with hexflash (Triton) as the primary large-S attention kernel and FlexAttention as the gated fallback, because hexflash is the only `works_all_shapes=yes` design and its bias-index oracle is statically certifiable today (the one thing I can de-risk before the GPU pause), while FlexAttention's edges (head_dim dispatch, dynamic BlockMask) mirror the exact symbolic-shape failure already documented in this codebase.

Here is the implementation spec.

---

# HEXFIELD INFERENCE REWRITE — IMPLEMENTATION SPEC

## 0. Decision

**Adopted architecture: three orthogonal, independently-gated layers.** Each lands behind a switch, parity-gated by the existing harnesses, with zero risk to the live path until the deliberate GPU pause.

| Layer | Component | Regime it wins | Status |
|---|---|---|---|
| **A. Attention kernel** | `hexflash` — hand-written shape-generic fused Triton FA2 that reconstructs the rel-pos-bias row in-kernel from the model's own `_exact_lut`. S is a runtime arg → ONE binary, Npad 64..3300+. | Large-S (the eager tail today) | PRIMARY |
| **A-fb. Attention fallback** | `flex` — `torch.nn.attention.flex_attention` with `score_mod` gather + `mask_mod` pad keys. | Large-S, if hexflash autotune/head_dim=24 underperforms at the pause | FALLBACK |
| **B. Serve plumbing** | Rust-owned pinned host pack handed to Python by buffer-protocol over a Python-preallocated pinned tensor + depth-2 submit/finish pipeline. | Small-S (launch/host-bound) | COMPANION |
| **C. Keep** | Deployed gated `torch.compile` SDPA-over-materialized-bias. | Small-S default + the universal fallback the harness already exercises. | UNCHANGED |

**Rationale for committing to this over a single-paradigm replacement:**
1. The deployed baseline is *already good at small-S* (gated compile ~2.4x, 0 CantSplit). The Amdahl ceiling (ESTABLISHED FACTS: conv+MLP+heads are 40-55% of large-S, untouched by any attention kernel) caps a pure attention rewrite at ~1.2-1.5x end-to-end, concentrated entirely on the large-S band the baseline leaves eager. There is no justification for ripping out a working small-S path to chase a large-S win.
2. hexflash is the only design rated `works_all_shapes=yes` and the only one whose core parity assertion (bias-index equality) is **statically checkable now** by reusing `model._exact_lut` (`model.py:231`) and the identical index expressions (`model.py:274-296`). That removes the single largest unverifiable risk before any GPU is touched.
3. FlexAttention's hard edges — head_dim=24 outside its fast dispatch set, batch-dependent dynamic BlockMask — are the *same* symbolic-shape class of failure already documented in this tree (`inference.py:82-88`, `CantSplit: 96*s+768 not divisible by s+8`). It is therefore the fallback, not the first bet, but it shares ~80% of hexflash's plumbing so building both behind one switch is cheap insurance.
4. Layer B is Amdahl-orthogonal (it removes host serialization, not FLOPs), so it stacks on top of A and is the *only* component that helps the small-S throughput regime that dominates at active_games=192. It has the cleanest parity story of all (byte-exact, reply ABI unchanged).

**Hard invariants preserved (the basis for "no retrain"):**
- The exactness trio (`model.py:19-22`): every parameter op re-applies the node mask; pad KEY columns are additively masked; pad QUERY rows are re-zeroed by `AttnBlock`'s `*m` (`model.py:193`). Padding a row to any Npad is therefore output-bit-identical. **Every new kernel MUST reproduce this.**
- `PAD_KEY_MASK_VALUE = -3.0e4` (`model.py:55`), finite in fp16. **Not** `-inf`.
- `scale = 1/sqrt(24)` applied to `q@kᵀ` **before** the bias add (`model.py:151,170`). **Not** `1/sqrt(32)`.
- The bias row index is the exact selection at `model.py:274-296`: clamp ±8 → `_exact_lut[(dq+8)*17+(dr+8)]` for `d≤8`; `BIAS_ON_AXIS_BASE/BIAS_OFF_AXIS_BASE + (d-9)` for `9≤d≤16` (on-axis test `dq==0|dr==0|dq+dr==0`); `BIAS_FAR_ROW` beyond; token-class rows (234/235/236) for any pair touching slots `<NUM_TOKENS=8`.
- Search behaviour (visits/PCR/vbs/dirichlet/widening/fpu/temperature) is **not touched** by anything here. This is forward-compute + serve only.

---

## 1. Component breakdown — file-by-file responsibilities

Ten implementers can each own one of these without colliding. Interfaces between them are the **frozen contracts** in §2.

### A1. `packages/hexfield/python/hexfield/hexflash.py` — NEW (Implementer 1)
The Triton kernel + an autograd-free Python wrapper. **No model/inference imports** beyond constants — pure function.

Public surface (FROZEN — A2 and the test depend on it):
```python
def hexflash_attention(
    q, k, v,            # (B, H, S, Dh=24) fp16, S = NUM_TOKENS + Npad
    coords,             # (B, Npad, 2) int32, axial (q,r); pad coords arbitrary
    bias_table,         # (BIAS_ROWS=237, H) — model.bias_table (fp16 view ok)
    seq_mask,           # (B, S) bool, True = live (tokens always True)
    exact_lut,          # (289,) int32 — model._exact_lut
    scale: float,       # 1/sqrt(24)
    num_tokens: int,    # 8
) -> torch.Tensor:      # (B, H, S, Dh) fp16
```
Kernel structure: standard FA2 (tile Q rows `BLOCK_M`, stream K/V `BLOCK_N`, online softmax, fp32 accumulator, fp16 loads). Inside the inner loop, per (q_idx, kv_idx):
- if `q_idx < num_tokens` or `kv_idx < num_tokens`: pick token-class row (236 token-token / 235 token-cell / 234 cell-token per `model.py:293-295`).
- else compute `dq,dr = coords[kv]-coords[q]` (int32), `d = max(|dq|,|dr|,|dq+dr|)`, then the exact branch selection above.
- Load `bias_table[row, h]` (load the whole head column into SRAM once per program — 237 fp16 = trivial — so per-pair is an SRAM index, not HBM). Add to `qkᵀ*scale`. Add `PAD_KEY_MASK_VALUE` where `seq_mask[kv]` is false. **Token keys never masked** (seq_mask True for slots `<8`).
- `head_dim=24` → `BLOCK_DMODEL=32` with a load mask on the last 8 lanes (zeros). Exact: zero q/k lanes add 0 to every score; v's extra lanes never written.
- Launch grid `(B, H, ceil(S/BLOCK_M))`. S, Npad, num_tokens passed as args; one autotuned binary covers all sizes. Autotune configs keyed by an S-bucket for num_stages/BLOCK (still ONE binary, config selection only).

Constraints: take `coords` as int32 (see B/A2 contract), guarantee no `dq+dr` overflow (board offsets `|q|,|r|<~60`, safe in int32). Pad-query rows may compute garbage — that is correct, they are re-zeroed downstream by `AttnBlock`'s `*m`.

### A1-fb. `hexflash.py` (same file) — `flex` path (Implementer 2)
A second public function, same signature shape, behind the same module:
```python
def flex_attention_relpos(q, k, v, coords, bias_table, seq_mask, exact_lut, scale, num_tokens) -> Tensor
```
Uses `torch.nn.attention.flex_attention`: `score_mod` replays the row selection (captures `coords`, `exact_lut`, `bias_table`); `mask_mod` from `seq_mask` compiled into an `lru_cache`d `BlockMask` keyed by `(B, S)`. head_dim 24→32 zero-pad. `mark_dynamic` the seq dim. This is the fallback; it must pass the **same** oracle (§3) as hexflash.

### A2. `packages/hexfield/python/hexfield/model.py` (Implementer 3)
Thread the raw attention inputs through so the inference path **never calls `build_attn_bias`**. Changes:
- `RelPosAttention.forward`: add `impl in {"hexflash","flex"}` branches. New overload that accepts `coords, seq_mask` (not a prebuilt `(B,H,S,S)` bias) and routes to `hexflash_attention` / `flex_attention_relpos`. Keep `"sdpa"`/`"materialized"` exactly as-is (training + oracle + fallback).
- `AttnBlock.forward`: signature change to thread `coords` and `seq_mask` through to the attn when impl is hexflash/flex; pass the prebuilt `attn_bias` otherwise. **Re-zero with `*m` is unchanged** (`model.py:193`) — the pad-query inertness invariant must survive.
- `HexfieldNet.trunk`: when `set_attention_impl` is hexflash/flex AND `not torch.is_grad_enabled()`, **skip `build_attn_bias` entirely** and pass `coords, seq_mask, bias_table` down. Otherwise unchanged. `build_attn_bias` itself is untouched (still the training path + the oracle ground truth).
- `set_attention_impl` already exists (`model.py:256`) — just accepts the two new strings.

**Hard rule for Implementer 3:** the `q/k/v` projection + scale + out_proj must be bit-identical to the SDPA path; only the score+softmax+@v core changes kernel.

### B1. `packages/hexfield/python/hexfield/inference.py` (Implementer 4)
- `HexfieldEvaluator.__init__`: on CUDA, read `HEXFIELD_ATTN_IMPL` (`sdpa`|`hexflash`|`flex`, default `sdpa`). When hexflash/flex, set it ONLY for groups with `pad_to > HEXFIELD_LARGE_NPAD` (default = `HEXFIELD_COMPILE_MAX_NPAD` = 512), via a per-group impl switch in `_forward_group`. Below the cutover, keep gated-compile SDPA (Layer C, unchanged). This is the regime routing — small-S compile, large-S hexflash.
- `_forward_group`: pass int32 `coords` (kernels want int32, not int64) when routing to hexflash/flex; keep int64 for the SDPA path. **No change to decode/softmax/D2H discipline** — single-D2H stays exactly as `submit_payload`/`result` have it (`inference.py:166-204`).
- Consume the new pinned buffers from B2 (see §2 contract) instead of the per-row numpy `_forward_group` loop (`inference.py:216-232`) when the v2 ABI is present; fall back to the v1 loop otherwise.

### B2. `packages/hexfield/rust/src/payload.rs` (Implementer 5)
- v2 ABI: Rust writes the **final on-device-ready flat buffers** itself into a **Python-preallocated pinned torch tensor** (passed down as a writable buffer pointer — the SAFE variant, NOT raw-Rust DLPack): flat node-major f16 feats, flat int32 coords, flat int32 gather-index (tap0=self + 6 nbr already remapped sentinel→pad-row per the model's convention), `cu_seqlens` (i32, B+1 — these ARE `node_row_offsets`), `legal_counts`. The f32→f16 cast already lives in Rust (`payload.rs:118`); keep it there.
- The gather-index remap MUST match `model.trunk`'s `self_idx=arange` + `nbr→Npad` convention (`model.py:346-347`) per group. **Pin with a Rust unit test against a known support** — an off-by-one silently corrupts conv neighbours.
- Reply ABI (`values_bytes`/`priors_bytes`/`moves_left_bytes`) **UNCHANGED** → `parse_chunk_reply` (`payload.rs:154`) and `finalize_priors` (`payload.rs:621`) reused verbatim. The transposition cache and dedup (`cache.rs`, `evaluate_state_refs_cached` at `payload.rs:313`) **UNCHANGED**.

### B3. `packages/hexfield/rust/src/search.rs` (Implementer 6)
- `run_continuous`: ring of in-flight `PendingEval`s (depth N, default 2). `submit(flush k+1)` enqueued before `finish(flush k)` drains. **Strict in-order drain**: `finish(k)` must complete (and insert into cache) before `finish(k+1)` — FIFO cache eviction at 262,144 is order-sensitive. Out-of-order completion is forbidden.
- Pinned-buffer lifetime: the Python-preallocated pinned staging tensor for flush k must outlive the async H2D. Keep it referenced inside the `PendingEval` handle until `result()` syncs.

### B4. `packages/hexfield/rust/Cargo.toml` (Implementer 5, trivial) — any pinned-buffer FFI deps.

### Test/harness owners
- **`tests/test_hexfield_model.py`** (Implementer 7) — the parity oracles (§3).
- **`scripts/_hexfield_compile_overlap_test.py`** (Implementer 8) — add hexflash/flex evaluator builds + large-S cases.
- **`scripts/_hexfield_async_parity.py`** (Implementer 9) — verify depth-2 pipeline preserves action parity.
- **`packages/hexfield/python/hexfield/constants.py`** (Implementer 10) — no new constants needed; owns the assertion that the kernel constants (`BIAS_*`, `PAD_KEY_MASK_VALUE`, `HEAD_DIM`, `NUM_TOKENS`) are imported from here, never duplicated in `hexflash.py`.

---

## 2. Frozen interfaces (the contracts between implementers)

**C1. Attention-core contract (A1/A1-fb ↔ A2).** Exactly the `hexflash_attention` signature in §A1. Inputs in the `(B,H,S,Dh)` layout the existing `forward` already produces at `model.py:161-163`. Output `(B,H,S,Dh)`, consumed by the existing `out.transpose(1,2).reshape(b,s,c)` + `out_proj`. The wrapper owns head_dim padding internally; A2 passes raw Dh=24 tensors.

**C2. Model serve contract (A2 ↔ B1).** `forward_policy_value(feats, nbr, mask, coords, *, request_moves_left)` signature **unchanged**. The only behavioural change is internal routing under `set_attention_impl`. B1 must pass `coords` as **int32** when impl∈{hexflash,flex}.

**C3. v2 wire contract (B2 ↔ B1).** Python pre-allocates pinned tensors of capacity ≥ flush size; passes their raw pointers + capacities to Rust; Rust fills flat node-major f16 feats / int32 coords / int32 gather-idx + `cu_seqlens` i32(B+1) + `legal_counts` i32(B). Payload carries `"abi": 2` to select this path; `"abi": 1` keeps the v1 numpy path. The ragged→dense scatter to `(g, pad_to, *)` is one vectorized GPU op per group keyed off `cu_seqlens` (NOT a Python loop). For the hexflash/flex large-S branch the dense scatter may be skipped if the kernel consumes ragged directly — but **v1 dense remains the default**; ragged-direct is a later optimization gated separately.

**C4. Pipeline contract (B3 ↔ B1).** `submit_payload`/`result` Python API **unchanged**. B3 only holds ≤N handles concurrently and drains FIFO. No reordering of cache inserts.

---

## 3. Parity strategy

Reuse the existing gates; invent no new thresholds. Three tiers, ordered by what is verifiable WITHOUT a GPU.

**Tier 1 — STATICALLY CHECKABLE NOW (no GPU; do this first, it de-risks the whole bet):**
- **Bias-index oracle.** Extend `test_pair_index_matches_geometry` (`test_hexfield_model.py:218`). Run hexflash with `bias_table = arange(BIAS_ROWS)` broadcast across heads, `scale=0`, q@k masked off so the output equals the gathered row value; assert `torch.equal` vs `build_attn_bias`'s integer pair index for the SAME coords/mask. Because the kernel consumes the model's OWN `_exact_lut` (`model.py:231`) and the identical clamp/d/on-axis/token-class expressions, the bias VALUE per pair is bit-identical **by construction**. This is the single most important gate and it needs no GPU to reason about (the index math is integer; only the kernel *execution* needs the pause).

**Tier 2 — fp16 OUTPUT ORACLE (needs the GPU pause):**
- Extend `test_sdpa_equals_materialized_fp16_cuda` (`test_hexfield_model.py:295`) to assert `impl="hexflash"` vs `"materialized"` (the canonical math oracle) AND vs `"sdpa"`, within the SAME budget already in the file: `diff <= 2e-3` fp16 (observed ~1.2e-4), `<= 1e-4` fp32. Same test for `impl="flex"`. Padded row (`mask[2,-5:]=False`) already exercises the pad-key mask — keep it.

**Tier 3 — END-TO-END (needs the GPU pause):**
- `scripts/_hexfield_compile_overlap_test.py`: add hexflash/flex evaluator builds; COMPILE-PARITY block reuses `TOL=3e-3` (`line 118`) on values/priors/moves_left vs eager; **extend `cases` with large-S sizes** (e.g. `1024`, `2048`, `3300`) so the new band is actually covered. ASYNC-PARITY block (`line 130`) `maxabsdiff==0.0` must still hold — the single-D2H discipline is unchanged so it does.
- For Layer B: add an explicit `torch.equal` assert that the ragged→dense scatter produces the same `(g,pad_to,*)` tensor the v1 numpy loop produced. ASYNC byte gate (`maxabsdiff==0.0`) is the v2-vs-v1 parity gate (math is bit-identical: same fp16 feats, same gather idx, same coords).
- `scripts/_hexfield_async_parity.py`: action-sequence parity must hold with depth-2 pipeline (overlap only moves sync points; FIFO drain preserves cache insertion order).

---

## 4. What MUST be validated in the GPU pause (NOT statically certain)

In sequencing order:

1. **Layer B first (lowest risk, byte-exact).** v2 ABI byte parity (`maxabsdiff==0.0`) + depth-2 action parity + pinned-buffer lifetime (no use-after-free: confirm the Python-preallocated pinned tensor outlives the async H2D — this is the one real correctness hazard in B).
2. **hexflash Tier-1 bias-index oracle executes** and returns `torch.equal` (the math is certain; the kernel *compiling and indexing correctly* on the installed Triton is not).
3. **hexflash Tier-2 fp16 oracle** ≤ 2e-3. If it fails the gate, the approach is blocked → switch to `flex` fallback and re-run the same oracle.
4. **hexflash autotune warmup** populates (configs across Npad 64..3300) and the **end-to-end speedup** vs the deployed baseline on the large-S band (the ~1.2-1.4x claim — Amdahl-capped, unverifiable until measured). Confirm small-S is at worst neutral if hexflash is ever routed there.
5. **VRAM** drops (no `(B,4,S,S)` bias transient on the hexflash path) so PAIR_CEILING can rise — measure, don't assume.
6. **flex-specific (only if used as fallback):** head_dim 24→32 hits a real kernel not a slow tail on torch 2.12; three flex calls (3 A-blocks) all compile and share the BlockMask; dynamic seqlen recompiles bounded-ly; fp16+padding precision bug (#163588) does not fire with `mask_mod`.

**Honesty:** every speedup number is static reasoning + library facts. Only the Tier-1 bias-index equality is statically certain (it is integer math reusing the model's own buffers). Everything runtime — kernel compilation, fp16 output parity, throughput, VRAM — requires the deliberate pause. Nothing in this spec runs a GPU workload, writes to the live tree, or alters search behaviour or learned-weight semantics.

**Deliverable assembly:** the operator assembles accepted components into `E:\Hexo-BotTrainer-hexgt-rewrite` (branch `claude/inference-rewrite`). Ship order: Layer C stays default; land Layer B (byte-gated), then Layer A behind `HEXFIELD_ATTN_IMPL` defaulting to `sdpa` so the live path is untouched until the pause flips it for large-S only.