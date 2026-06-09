# dense_cnn_restnet — Architecture & Code Review

**Date:** 2026-06-09
**Scope:** `packages/dense_cnn_restnet/` (Python-only ResTNet fork of `hexo_models.dense_cnn`)
**Type:** review only — no model code was modified and no live run was touched.
**Reviewer baseline:** the released ResTNet code (github.com/rlglab/restnet, IJCAI 2025
arXiv:2410.05347), the parent `hexo_models.dense_cnn`, and the run config
`configs/dense_cnn_restnet_main1.toml`.

> **Note on the referenced background doc.** The task pointed at
> `docs/analysis/DENSE_CNN_RESTNET_FEASIBILITY.md`. **That file does not exist** in the
> repo (tracked or untracked); the only feasibility doc present is the unrelated
> `TSS_TO_DENSE_CNN_FEASIBILITY.md`. This review was therefore done against the code,
> the config's inline design notes, and the released-paper citations in
> `architecture.py`. Capturing the feasibility rationale as a committed doc is itself a
> recommendation (see §1).

---

## 1. What the package is

`dense_cnn_restnet` is a **pure-Python additive fork** of `hexo_models.dense_cnn`. It
swaps **only the PyTorch trunk** — replacing the homogeneous gated-residual stack with an
interleaved Residual (`R`) + Transformer (`T`) trunk per the ResTNet paper — while keeping
the dense_cnn input contract, the four-head output surface
(`policy` / `value` / `opp_policy` / `stvalue_<h>`), the binned-value loss, the NPZ replay
schema, MCTS, and the native accelerator.

Of the ~22 module files, **16 are byte-identical copies** of the parent
(`trainer.py`, `losses.py`, `selfplay.py`, `evaluation.py`, `replay.py`, `samples.py`,
`mcts.py`, `input.py`, `geometry.py`, `constants.py`, `d6.py`, `rust_bridge.py`,
`compact_io.py`, `performance.py`, `debug_artifacts.py`, `trt_backend.py`). The meaningful
divergence is concentrated in **`architecture.py`** (309 → 587 lines); the rest are
small surface edits (`plugin.py` name/build, `config.py` six new arch keys, `inference.py`
class-name swap, `__init__.py` exports, and a `"model"` metadata label in
`checkpoints.py` / `player.py`).

**Native reuse is correct and verified:** `rust_bridge.py` is unchanged and imports the
shared `hexo_models._rust.dense_cnn`; `test_coexists_with_dense_cnn` asserts the fork and
the parent resolve to the *same* native submodule object. No Rust rebuild is required, and
the fork cannot perturb the parent's native contract because `constants.py` (the byte
layout) is identical.

### Trunk design (`architecture.py`)

- `parse_blocks_type("R_R_R_T_R_R_T_R")` → tuple of block kinds; `R` = `ResidualBlock`,
  `T` = `TransformerBlock`.
- `ResidualBlock`: plain post-activation AlphaZero block
  (`Conv3×3 → BN → ReLU → Conv3×3 → BN → +identity → ReLU`), hex-masked 3×3 conv by default.
- `TransformerBlock`: pre-norm `x += MSA(LN(x)); x += MLP(LN(x))`, GELU MLP at ratio 2.
- `RelPosMHSA`: full O(N²) attention over all 41×41 = 1681 tokens, with a learned
  relative-position bias table `((2H-1)(2W-1), heads)`. Two numerically-identical
  implementations: `materialized` (the test oracle) and `sdpa` (default, additive-mask).
- R↔T boundary is a parameter-free `(B,C,H,W) ↔ (B,N,C)` reshape; `trunk()` always returns
  a `(B,C,41,41)` map for the heads.

---

## 2. Strengths (this is a careful, well-tested fork)

These are worth stating plainly, because they are the things most likely to be eroded by
future edits:

1. **Minimal, contract-preserving surface.** Only the trunk changed; every I/O contract
   the rest of the pipeline depends on is byte-for-byte the parent's. This is exactly the
   right way to add a model lineage without forking the whole stack.

2. **Two attention implementations with an equivalence test.** Keeping a `materialized`
   oracle alongside the `sdpa` production path, and asserting `allclose` at the MHSA,
   block, and whole-network levels (`test_msa_sdpa_equals_materialized`,
   `test_transformer_block_sdpa_equals_materialized`,
   `test_network_sdpa_equals_materialized`), is excellent — it pins the most error-prone
   numerical surface.

3. **The relative-position scheme is genuinely hex-faithful.** Because axial→(row,col) is
   an affine map (`row = r - cr + half`, `col = q - cq + half`), a relative `(drow, dcol)`
   offset uniquely identifies the axial offset between two cells. So the square-grid
   relative-bias table — copied straight from the square-board released code — encodes the
   *hex* relative geometry without modification. `test_relative_index_matches_bruteforce_3x3`
   verifies the indexing against a brute-force reference.

4. **Strong, targeted test suite.** Beyond the SDPA equivalence: brute-force rel-index,
   token/conv round-trip, R-block accepting both forms, full-board head shapes,
   `forward_policy_value` subset, trunk-returns-conv-even-when-last-block-is-T,
   input-shape rejection, fresh-init finiteness, loss+backward (incl. asserting the
   rel-bias table receives gradient), inference-fold equivalence, plain-conv variant, and
   config parsing/rejection. This is high-quality coverage for the parts that differ.

5. **Fail-fast validation throughout** — `parse_blocks_type`, `RelPosMHSA` (channels %
   heads, square grid), `set_impl`, input rank/shape — matching the project's
   reject-don't-repair convention.

6. **The warm-start bootstrap is conscientious.** `bootstrap_dense_cnn_restnet_hf.py`
   replays the human corpus through the real engine, gates on legality + a winner
   cross-check, aborts if >10% of games fail to replay (coordinate-mismatch guard), reuses
   the production sample/shuffle/trainer machinery (no bespoke poisoning-prone path), and
   verifies a strict reload into a fresh `RestnetNetwork` before saving.

---

## 3. Findings & improvement suggestions

Ordered by category. Severity tags: **[correctness]**, **[efficiency]**, **[hygiene]**,
**[process]**, **[experimental]**. None are blockers; the lineage is sound.

### 3.1 Weight-init divergence from the parent — **[correctness, low risk]**

`RestnetNetwork.__init__` ends with `self.apply(_init_weights_trunc_normal)`, which
`trunc_normal_(std=0.02)`-inits **every** `nn.Linear` in the module tree. That correctly
covers the ViT-style trunk (q/k/v/out projections, MLP), but it **also re-inits the
`ValueBinnedHead` Linear layers** (`Linear(1681→64)` and `Linear(64→65)`) — including the
short-term-value heads. The parent `dense_cnn` has **no custom init at all** (confirmed:
its `architecture.py` defines no `apply`/init function), so it relies on PyTorch defaults
(Kaiming-uniform) for those exact layers.

Consequences:
- A `Linear(1681→64)` initialized at `std=0.02` produces near-zero pre-activations, so the
  value head starts close to degenerate (≈uniform bins). That is *probably benign* —
  arguably even mildly helpful from scratch — and is fully overwritten under the
  warm-start path (`initialize_from` loads prefit weights). But it is an **unintended**
  divergence from the parent's head initialization, not a deliberate one.
- `PolicyHead` is conv-only, so it is unaffected; only the value heads diverge.

**Suggestion:** decide intentionally. If head parity with dense_cnn matters, scope the
init to the trunk only (e.g. apply `_init_weights_trunc_normal` over `self.stem_*` +
`self.blocks` and leave the heads at PyTorch defaults). If the current behavior is
intended, add a one-line comment in `__init__` saying the heads are deliberately
trunc-normal'd too, so a future reader doesn't "fix" it.

### 3.2 Relative-bias table initialized to zeros — **[correctness, confirm-vs-paper]**

`relative_bias_table` is a `Parameter` created with `torch.zeros(...)`. Because
`_init_weights_trunc_normal` only touches `Linear`/`LayerNorm`/`BatchNorm2d`, the bias
table is left at **zeros**, and `test_fresh_init_has_finite_params_and_forward` explicitly
asserts that. Swin-Transformer convention (the lineage this indexing scheme comes from)
`trunc_normal_(std=0.02)`-inits the relative-bias table.

Zeros means "no positional preference at init, learn it from scratch," which is fine and
will converge — but it is slower to acquire a positional prior than a small random init,
and it is a divergence worth a deliberate decision.

**Suggestion:** confirm what the released ResTNet code does for its bias table. If it
trunc-normals it, match that (and update the `test_fresh_init` zero-assertion). If zeros is
intentional, note it in the docstring (the current docstring says "released-code init"
without a citation line).

### 3.3 `blocks_type` default is a custom 8-block variant, not the paper canonical — **[process]**

The default `"R_R_R_T_R_R_T_R"` is an 8-block, 2-transformer trunk; the paper's canonical
R3 form is the 10-block `"R_R_R_T_R_R_T_R_R_T"` (3 transformers). The config comment
already flags this for owner confirmation — good. The thing to guard is that the **prefit
checkpoint and the RL config must agree on `blocks_type`**: a mismatch makes the strict
load fail (`RestnetNetwork` builds its block list from the string, so the state-dict keys
differ). Today they agree because the bootstrap reads the arch from the same config, but
this is an easy footgun if someone edits one side. Consider stamping `blocks_type` (and
`channels`) into the checkpoint metadata and checking it on load with a clear error
message. (The bootstrap already records arch in metadata; the *RL* checkpoint loader does
not appear to assert it.)

### 3.4 Per-block duplication of the relative-index buffer — **[efficiency, optional]**

Each `RelPosMHSA` registers its own `relative_index` buffer of shape `(N·N,) = 2.82M`
int64 ≈ **22.6 MB**, and the contents are identical for every transformer block (same
board geometry). With 2 T-blocks that is ~45 MB of redundant buffers; a 3-T canonical
trunk would be ~68 MB. Additionally, `_relative_bias()` re-gathers and builds the
`(1, heads, N, N)` additive bias on **every forward of every block**.

**Suggestion (optional, perf is a stated non-goal):** compute the index once at the
`RestnetNetwork` level (or a module-level cache keyed by board size) and share it across
blocks. This is purely a memory/allocation win and does not change numerics.

### 3.5 Verify the SDPA backend actually avoids the N×N score matrix — **[efficiency, verify]**

The `_forward_sdpa` docstring claims it "never materializes the `(B, heads, N, N)` score
matrix." That is true **only if PyTorch dispatches to the memory-efficient attention
kernel.** With a *dense float additive* `attn_mask` of shape `(1, heads, 1681, 1681)`,
PyTorch's SDPA **cannot use the FlashAttention backend** (it rejects arbitrary additive
bias). On CUDA it should fall to the memory-efficient (xFormers-style) kernel — which does
support an additive bias without materializing scores — but if that kernel is unavailable
for the given dtype/shape, it falls back to the **math kernel, which *does* materialize the
full N×N scores.** If the math fallback is what's running, the `batch_size=64` OOM ceiling
documented in the config is being set by the fallback, not by an irreducible cost.

Two things follow:
- The additive **bias tensor itself** (`(1, heads, N, N)`, ~22 MB fp16 per T-block) is
  materialized regardless of backend — so "no N×N materialization" is never literally true;
  it's the *score* matrix that the efficient kernel avoids.
- **Suggestion:** add a one-off runtime check (e.g. under
  `torch.backends.cuda.sdp_kernel(...)` or `torch.nn.attention.sdpa_kernel`) confirming the
  efficient kernel is selected for this mask shape on the target GPU, and log it during
  calibration. If only the math kernel is available, batch 64 is pessimistic and could be
  raised; if efficient is confirmed, document that and move on.

### 3.6 No CUDA/AMP test in the suite — **[hygiene]**

Every test is CPU-only and explicitly `torch.no_grad()` / fp32. But the things that *differ*
between this fork and dense_cnn — SDPA backend selection, AMP (fp16) attention numerics,
and the OOM ceiling — only manifest on CUDA. There is a `scripts/_restnet_gpu_sanity.py`
script, but it is not part of the suite.

**Suggestion:** promote a tiny CUDA test guarded by
`@pytest.mark.skipif(not torch.cuda.is_available())` that runs a forward+backward under
`autocast` and re-checks SDPA↔materialized `allclose` at fp16 tolerance. This catches AMP
regressions and backend changes that the CPU tests structurally cannot.

### 3.7 Dead branch in `parse_blocks_type` — **[hygiene, trivial]**

```python
kinds = tuple(token for token in blocks_type.split("_"))
if any(kind not in ("R", "T") for kind in kinds):
    ...raise...
if not kinds:          # <-- unreachable
    raise ValueError("blocks_type must contain at least one block")
```

`str.split("_")` on a non-empty string never yields an empty tuple (empty/whitespace input
is caught by the earlier `not blocks_type.strip()` guard; embedded empty tokens like
`"R__T"` are caught by the `not in ("R","T")` check, since `""` is not a valid kind). The
`if not kinds` branch is therefore unreachable. Harmless — can be deleted for clarity.

### 3.8 `residual_blocks` config key is silently ignored — **[hygiene]**

`config.py` still parses `residual_blocks` (inherited from dense_cnn) but the comment notes
trunk depth is governed entirely by `blocks_type`, so the field is ignored by
`build_model`. A user copying a dense_cnn config and tweaking `residual_blocks` to "make
the model deeper" would see no effect and no warning.

**Suggestion:** emit a parse-time warning (or reject) when `residual_blocks` is present in a
restnet config, pointing the user at `blocks_type`. The project already prefers fail-fast
config handling, so rejecting is in keeping with convention.

### 3.9 Dropout is 0.0 everywhere — **[experimental, data-dependent]**

The transformer blocks add capacity (attention + MLP params) on top of the conv trunk. From
scratch on a finite replay window, that raises overfitting risk relative to the conv-only
parent. `dropout` is wired through (`RelPosMHSA.out_drop`, config `dropout`) but defaults to
0.

**Suggestion:** keep 0 for the faithful baseline, but treat a small attention/MLP dropout
(0.05–0.10) as the first lever if early RL shows policy/value train-vs-eval divergence.
This is data-dependent — note it, don't pre-emptively set it.

### 3.10 Global attention spends capacity on never-relevant cells — **[experimental]**

Attention is full and unmasked over all 1681 tokens, but the dense view is a *window on an
infinite sparse board* — the crop borders are frequently all-empty and will never be played
in the near term. Those cells are pure positional noise that attention still distributes
mass over, which can hurt sample efficiency.

**Suggestion (experimental, stays faithful to block structure):** the input already carries
a `PLANE_LEGAL` / `PLANE_EMPTY` signal. An additive attention-key bias derived from "is this
cell plausibly relevant" (e.g. a large negative bias on cells that are illegal *and* far
from any stone) would let the model ignore dead borders without changing the R/T topology.
This is a divergence from the released square-board code (where all cells are always valid),
so gate it behind a config flag and A/B it rather than baking it in. Worth a small ablation
once the faithful baseline has a SealBot number to beat.

---

## 4. Summary

`dense_cnn_restnet` is a clean, contract-preserving, well-tested fork that adds a faithful
ResTNet trunk while reusing the entire dense_cnn pipeline and native accelerator with zero
Rust risk. The relative-position scheme is correctly hex-faithful, and the dual-impl
attention with an equivalence test is exactly the right safety net for the riskiest code.

The findings are almost all low-risk polish. The two worth an explicit decision before/early
in the RL run are:

- **§3.1 / §3.2 — init divergences** (value-head Linears and the zero-init bias table differ
  from the parent / Swin convention). Decide and document; both are benign under warm-start
  but should be intentional.
- **§3.5 — confirm the SDPA backend** actually avoids the N×N score matrix on the target
  GPU, since the documented batch-64 ceiling hinges on it.

The rest (§3.3 checkpoint arch-stamp, §3.4 shared index buffer, §3.6 a CUDA test, §3.7 dead
branch, §3.8 ignored config key, §3.9–§3.10 dropout / attention masking) are optional
improvements that can land opportunistically.

And independently of the model: **commit the feasibility doc** the work was supposed to be
based on (§intro) and add a short package README, so the design baseline and the explicit
performance-non-goal are recorded next to the code rather than only in commit history and
config comments.
