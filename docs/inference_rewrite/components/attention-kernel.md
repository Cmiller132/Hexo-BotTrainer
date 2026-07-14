# attention-kernel

Confirmed. The flex path is correct: `score_mod` receives `q@kᵀ * scale` (scale = my explicit `1/sqrt(24)`), then I add `table[row,h]` — matching the production order (`q@kᵀ*scale + bias`). My explicit `scale=1/sqrt(24)` override neutralizes the head_dim-32 padding effect on scaling.

Both static gates pass. My component is complete. Here is the deliverable.

---

## DELIVERABLE — Layer A attention backend (Implementer 1+2+3)

Two files for the rewrite worktree. Everything is gated; the live path is untouched until `HEXFIELD_ATTN_IMPL` flips at the pause.

### Files written (rewrite tree, NOT live tree)
- `E:\Hexo-BotTrainer-hexgt-rewrite\packages\hexfield\python\hexfield\hexflash.py` — NEW kernel module (Triton FA2 + flex fallback + pure-torch reference/oracle helper).
- `E:\Hexo-BotTrainer-hexgt-rewrite\packages\hexfield\python\hexfield\model_attention_region.py` — full replacement bodies for `RelPosAttention`, `AttnBlock`, `HexfieldNet.trunk`, `set_attention_impl`, plus the one new import. Drop these in place of model.py:142-195, 335-367, 256-258; add `from .hexflash import flex_attention_relpos, hexflash_attention`.

### What the code does
- `hexflash_attention(q,k,v,coords,bias_table,seq_mask,exact_lut,scale,num_tokens)` — FROZEN signature (§C1). One Triton binary, S/Npad as runtime args, covers 64..3300+. Reconstructs the bias **row** per (q,k) pair in-kernel from the model's own `_exact_lut` + the identical clamp/d/on-axis/token-class expressions, gathers `bias_table[row,h]`, adds it to `q@kᵀ*scale`, then overwrites pad-key columns with `PAD_KEY_MASK_VALUE` (token keys never masked), FA2 online softmax, fp32 accumulator. head_dim 24 runs on a 32-wide tile with the last 8 lanes zero-loaded (zeros add 0 to scores; extra V lanes never stored).
- `flex_attention_relpos(...)` — same signature, FlexAttention fallback. `score_mod` replays the exact row selection; `mask_mod` is the pad-key mask; `BlockMask` lru-cached on `(B,S,seq_mask ptr)`; head_dim padded 24→32 and sliced back; **`scale=1/sqrt(24)` passed explicitly to `flex_attention`** to override its `1/sqrt(32)` default (verified: flex applies scale before `score_mod`).
- `reference_relpos_attention(...)` — pure-torch materialized-equivalent; the autograd-safe path used if grad is ever enabled, the CPU/no-Triton fallback, and the semantic definition for the oracle.
- Model routing: `sdpa`/`materialized` are **byte-identical to the live tree** (oracle preserved). Fused impls fire only when `not torch.is_grad_enabled()`; the trunk then **skips `build_attn_bias`** and threads int32 `coords` + `bias_table` + `_exact_lut` + `seq_mask` to each AttnBlock. `AttnBlock`'s `*m` pad-query re-zero is unchanged → pad-inertness preserved.

### Static parity results (no GPU, run on this box)
- **Tier-1 bias-index oracle**: `relpos_pair_index` vs `build_attn_bias` on live KEY columns = **0.0 max diff** (bit-exact, spans exact-disk/ring/far/token classes + padding). Pad-key columns differ in the underlying garbage row but are overwritten by the pad mask in both paths, so it is irrelevant (verified the masked values match: row−3e4).
- **Reference vs materialized** (CPU fp32, derandomized model, padded row): **1.16e-6** max diff on live query rows — well under the 1e-4 fp32 gate.

### What MUST wait for the GPU pause (not statically certain)
1. The Triton kernel compiling/indexing correctly on the installed Triton (3.6 here; live box torch 2.12+cu130). The index math is certain; the *execution* is not.
2. Tier-2 fp16 oracle: extend `test_sdpa_equals_materialized_fp16_cuda` with `impl="hexflash"` and `impl="flex"` vs `"materialized"`, reusing `diff <= 2e-3`. If hexflash misses the gate → switch to `flex`, same oracle.
3. hexflash autotune across Npad buckets; end-to-end large-S speedup (Amdahl-capped ~1.2–1.4x); VRAM drop (no `(B,4,S,S)` bias transient on the fused path).
4. flex-only: head_dim 24→32 hits a real kernel; 3 A-blocks share the BlockMask; fp16+padding precision (#163588) doesn't fire with `mask_mod`.

### Exact parity assertions to reuse
- `test_pair_index_matches_geometry` (test_hexfield_model.py:218) extended with the hexflash row gather (the 0.0 gate I just ran statically).
- `test_sdpa_equals_materialized_fp16_cuda` (test_hexfield_model.py:295) extended: `hexflash`/`flex` vs `materialized`, `diff <= 2e-3` fp16.
- `_hexfield_compile_overlap_test.py` COMPILE-PARITY `TOL=3e-3` extended with large-S cases (1024/2048/3300); ASYNC `maxabsdiff==0.0` unchanged (single-D2H discipline untouched by Layer A).

One note for the operator: in `model_attention_region.py` the `MLP_RATIO` reference in `AttnBlock.__init__` is flagged `noqa: F821` only because the snippet is shown standalone; in model.py it resolves from the existing module import (constants.py:82). No new constants are introduced — all values come from `constants.py`, none duplicated in `hexflash.py` (PAD_KEY_MASK_VALUE is the one frozen contract value, asserted equal by the constants owner).

Sources:
- [torch.nn.attention.flex_attention (2.12)](https://docs.pytorch.org/docs/2.12/nn.attention.flex_attention.html)