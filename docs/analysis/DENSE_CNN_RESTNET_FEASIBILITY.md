# dense_cnn_restnet — Faithful ResTNet Implementation: Feasibility & Design

**Status:** analysis only. No code in `dense_cnn`, `hexgt`, `hexgnn`, or any live run is modified by
this document.
**Date:** 2026-06-09 (rev 2 — re-oriented to *paper-faithful* after owner refinement)
**Owner intent:** a true *fork* of the `dense_cnn` lineage (the way `hexgnn` was forked from
`hexgt`) called `dense_cnn_restnet`, implementing the ResTNet architecture **as faithfully to the
paper as possible, with performance treated as an explicit non-goal.** No throughput-motivated
substitutions: full global O(N²) self-attention over all board-cell tokens (the 41×41 crop = 1681
tokens), the paper's exact Transformer block, the paper's interleaving patterns (RRTRRT, R3(RRT)),
relative position encoding, norm placement, MLP ratio, and heads. Throughput cost is documented as
an FYI only.

**Primary sources used (the paper's *released code*, not just the PDF):**
- Paper: Ju, Wu, Shih, Wu, *Bridging Local and Global Knowledge via Transformer in Board Games*,
  IJCAI 2025. [arXiv:2410.05347](https://arxiv.org/abs/2410.05347) ·
  [project page](https://rlg.iis.sinica.edu.tw/papers/restnet/).
- **Official code: [`github.com/rlglab/restnet`](https://github.com/rlglab/restnet)** — built on
  [MiniZero](https://github.com/rlglab/minizero). The exact architecture lives in
  `restnet/learner/network/{block_unit.py, alphazero_network.py, embed_unit.py, policy_unit.py,
  value_unit.py}`. **This report's architecture spec is transcribed from that code** (verbatim
  excerpts in Appendix A), because the PDF deliberately omits the block internals (it only says
  "standard Transformer, Vaswani 2017, relative position encoding Shaw 2018, four heads").

---

## 0. TL;DR / Verdict

**Fork mechanics: Easy. Faithful architecture: Easy–Moderate — we have their exact code to mirror.
The only real constraint is GPU memory at training time, and it has a *non-compromising* fix.**

- **The change is 100% Python (a PyTorch `nn.Module` swap).** dense_cnn's Rust crate is a pure
  featurizer + MCTS that calls back to Python for the forward pass and is blind to the trunk. The
  fork can reuse `hexo_models._rust.dense_cnn` read-only — **no native rebuild, no risk to the live
  `dense_cnn_rl_main1` run.**
- **We can mirror ResTNet almost line-for-line.** Their trunk is a `blocks_type` string
  (`"R_R_T_R_R_T"`) split into `ResidualBlock`/`TransformerBlock`; the R↔T boundary is pure einops
  reshape with no projection (token dim = conv channels); the T block is a standard pre-norm
  Transformer with a learned relative-position-bias table and a GELU MLP at ratio 2, 4 heads. All of
  this ports directly. (Appendix A has the exact code; Appendix B is the port mapping.)
- **Full attention over 1681 tokens is the design, not a problem to engineer around.** Per owner
  intent we keep it. The honest cost FYI: the attention logit tensor is `(B, 4, 1681, 1681)` ≈
  **22.6 MB per board per T-block** (fp16). At self-play leaf-batch sizes (~99–256, no autograd,
  blocks run one at a time) peak is a few GB — **runnable**. At the paper's training batch (1024)
  with autograd across 3 T-blocks it would need **>100 GB** — so **training must use a small batch
  and/or `F.scaled_dot_product_attention`**, which computes the *identical* full attention without
  materializing the N² matrix (it is NOT windowing/downsampling — same math, fused kernel). That is
  the one practical accommodation and it is architecturally faithful.
- **Throughput FYI (non-goal):** at the paper-faithful width (256 channels) with full 1681-token
  attention, expect self-play forward to be **roughly 5–20× slower** than the live 64-channel
  conv-only run, i.e. **single-digit search pos/s**, and the TensorRT FP16 path likely unavailable
  for the attention trunk. The owner accepts this.
- **Worth doing for Hexo?** Yes — the paper's **19×19 Hex** result (50.4% → **58.0%** vs a 10R
  baseline, R3(RRT)) is the directly relevant datapoint, and the fork is additive/non-disruptive.
- **Rough effort:** **~1 week** to a faithful, launchable `dense_cnn_restnet` (RRTRRT and R3(RRT),
  full attention, paper hyperparameters), plus the open-ended cost of actually training it.

**Recommended first build:** fork → standalone additive package reusing `_rust.dense_cnn` → port
`block_unit.py` (ResidualBlock + TransformerBlock + MSA_rel + MLP) and the `blocks_type`-driven
trunk verbatim → **256 hidden channels, 4 heads, MLP ratio 2, embed_kernel_size 3** (the paper's
19×19 settings) → headline config **R3(RRT)** = `"R_R_R_T_R_R_T_R_R_T"` (mirrors the winning 19×19
Hex net), with **RRTRRT** = `"R_R_T_R_R_T"` as the smaller bring-up. Keep dense_cnn's existing
featurizer, heads, loss, MCTS, and replay schema (the Hexo training contract); insert a token→conv
rearrange before the heads. Implement attention both ways: **MSA_rel verbatim as the correctness
oracle**, and an SDPA-backed equivalent for memory-bound training. Details in §6.

---

## 1. ResTNet, transcribed from the released code

The PDF is sparse on internals; the code is authoritative. Everything below is from
`github.com/rlglab/restnet` (`restnet/learner/network/`). Verbatim excerpts in Appendix A.

### 1.1 Trunk construction & interleave (`alphazero_network.py`)
```python
self.num_head = 4          # hardcoded
self.mlp_ratio = 2         # hardcoded
self.embed = EmbedNet(num_input_channels, num_hidden_channels, embed_kernel_size)
self.blocks = nn.ModuleList(
    [self.get_backbone(bt) for bt in blocks_type.split("_")]   # "R_R_T_R_R_T" -> [R,R,T,R,R,T]
)
# get_backbone("R") -> ResidualBlock(C, H)
# get_backbone("T") -> TransformerBlock(C, C*mlp_ratio, num_head=4, H, W)
# forward: x = embed(state); for b in blocks: x = b(x); policy/value heads
```
- **The trunk is literally a string of `R`/`T` split on `_`.** `RRTRRT` → `"R_R_T_R_R_T"` (6 blocks,
  2 T). `R3(RRT)` → `"R_R_R_T_R_R_T_R_R_T"` (10 blocks, 3 T; one leading R then the `RRT` triplet
  ×3). `10R` baseline → ten `R`s. This is exactly the config-driven interleave we need.
- **Heads** are selectable: `policy_type ∈ {P (conv), TP (transformer)}`, `value_type ∈ {V (conv),
  TV (transformer)}`. The paper's runs use **`P_TV`** (conv policy, transformer value — see the
  folder name `..._P_TV_n64-...` in the README).
- **Weight init:** `self.apply(_init_weights_trunc_normal)` → `trunc_normal_(weight, std=0.02)` on
  every `nn.Linear`; LayerNorm/BatchNorm get bias 0, weight 1. (ViT-style; matters for stability.)

### 1.2 Residual block — plain AlphaZero, post-activation (`block_unit.py`)
```python
conv1 = Conv2d(C, C, 3, padding=1, bias=False); bn1 = BatchNorm2d(C)
conv2 = Conv2d(C, C, 3, padding=1, bias=False); bn2 = BatchNorm2d(C)
# forward (accepts tokens or conv map):
#   if x.dim()==3: x = Rearrange("b (h w) c -> b c h w")(x)
#   input = x; x = relu(bn1(conv1(x))); x = bn2(conv2(x)); return relu(input + x)
```
**No SE, no gating, no pre-activation.** Note dense_cnn's existing block is a *gated* variant
(`GatedResBlock`) and uses hex-masked convs (`HexConv2d`); see §3.1 for the faithfulness decision.

### 1.3 Transformer block — pre-norm, GELU MLP (`block_unit.py`)
```python
ln1 = LayerNorm(C); ln2 = LayerNorm(C)
MSA = MSA_rel(C, n_head=4, drop=0.0, H, W)      # relative-position multi-head self-attn
MLP = Sequential(Linear(C, 2C), GELU(), Linear(2C, C))   # ratio 2, GELU, no dropout
# forward (accepts conv map or tokens):
#   if x.dim()==4: x = Rearrange("b c h w -> b (h w) c")(x)
#   x = x + MSA(ln1(x))        # pre-norm self-attention, residual
#   x = x + MLP(ln2(x))        # pre-norm FFN, residual
#   return x                   # OUTPUTS TOKENS (b, H*W, C)
```
Key faithful details often gotten wrong:
- **Pre-norm** (`LN → sublayer → add`), LayerNorm (not BatchNorm) inside the T block.
- **GELU** in the MLP (the PDF doesn't say this; the code does). **Ratio 2.** No dropout by default.
- A T block **consumes** a conv map *or* tokens and **emits tokens**; the next R block reshapes back.
  So the trunk freely alternates `(B,C,H,W) ↔ (B,H·W,C)` with **zero learned parameters at the
  boundary** — token dim == channel count, no projection.

### 1.4 Attention with learned relative-position bias (`MSA_rel`, `block_unit.py`)
```python
projq/projk/projv = Linear(C, C) -> Rearrange("b n (h d) -> b h n d", h=4)   # d = C/4
scaling = (C / 4) ** -0.5
dots = einsum("b h q d, b h k d -> b h q k", q, k) * scaling   # (B, 4, N, N)  FULL O(N^2)
# learned relative bias table, shape ((2H-1)(2W-1), heads), gathered by precomputed relative_index:
relative_bias = relative_bias_table.gather(0, relative_index.repeat(1, heads))  # (N*N, heads)
dots = dots + Rearrange("(h w) c -> 1 c h w", h=N, w=N)(relative_bias)           # broadcast over B
attn = softmax(dots, dim=-1)
x = einsum("b h d i, b h i v -> b h d v", attn, values)
x = Linear(C, C)(rearrange(x)); x = Dropout(0.0)(x)
```
- **Standard scaled dot-product MHSA, 4 heads, full board-to-board attention.** No attention dropout
  (softmax is applied directly); output projection has a dropout that defaults to 0.
- **Relative position encoding = a learned bias table** indexed by 2D (Δrow, Δcol) offsets (Shaw
  2018-style, implemented Swin-style). Table size `(2H−1)(2W−1) × heads`; for 41×41 that's
  `81×81×4 = 26,244` params — trivial. `relative_index` is a precomputed `(N², 1)` buffer (≈22 MB at
  N=1681) added to the logits each forward. The construction assumes a **square grid (H==W)** — true
  for the 41×41 crop. (Verbatim in Appendix A.)

### 1.5 Stem / embedding (`embed_unit.py`)
```python
EmbedNet = Conv2d(in, C, kernel_size=embed_kernel_size, padding=k//2, bias=False) -> BN -> ReLU
# embed_kernel_size: 1 (pure per-cell projection — "positional embedding") or 3 (local mixing).
```

### 1.6 Heads, as the paper ships them (`policy_unit.py`, `value_unit.py`)
- **`P` (conv policy):** `Conv1x1 → BN → ReLU → flatten → Linear → action_size`.
- **`TP` (transformer policy):** `Linear(C,C) → Tanh → Linear(C,1)` per token → flatten (+ zero pads
  for extra non-board actions).
- **`V` (conv value):** `Conv1x1 → BN → ReLU → flatten → Linear → ReLU → Linear → Tanh` (scalar).
- **`TV` (transformer value, the paper default):** `AdaptiveAvgPool2d(1) → Linear(C,2C) → SiLU →
  Linear(2C,1) → Tanh` (scalar).
- All heads accept tokens *or* conv maps and `Rearrange` as needed.

These heads produce a **scalar tanh value** and a flat policy — they are *not* compatible with
dense_cnn's binned-value / opp-policy / short-term-value training contract. The faithfulness
decision for heads is in §3.4.

### 1.7 Hyperparameters (Table 4 + configs)
| | 9×9 Go | 19×19 Go | **19×19 Hex** |
|---|---|---|---|
| Blocks | 6 (RRTRRT) | 10 (R3RRT) | **10 (R3RRT)** |
| Hidden channels | 256 | 256 | **256** |
| Heads | 4 | 4 | **4** |
| MLP ratio | 2 | 2 | **2** |
| embed_kernel_size | 3 | 3 | **3** |
| Heads used | P_TV | P_TV | **P_TV** |
| Batch size | 1024 | 1024 | **1024** |
| Training steps | 100k | 150k | **100k** |
| LR | 0.02→0.005 | 0.1 | **0.02** |
| MCTS sims (train) | 64 | — | **32** (Gumbel 16) |

The training loop is **Gumbel AlphaZero in MiniZero** — *not* portable verbatim to the Hexo
pipeline; treat these as the architecture/optimization reference, and drive training through the
existing dense_cnn pipeline (§3.4, §5).

### 1.8 Results (the case for trying)
9×9 Go 54.6→**60.8%**; 19×19 Go 53.6→**60.9%**; **19×19 Hex 50.4→58.0%** (all vs the residual-only
baseline). The Hex row — a hex connection game, +7.6 points from interleaving T-blocks into a 10R
trunk — is the most transferable evidence for Hexo.

---

## 2. Fork mechanics (unchanged by the re-orientation)

### 2.1 What dense_cnn looks like (what gets forked)
Python package `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/`: `architecture.py`
(`Model1Network` — trunk = `nn.Sequential` of `GatedResBlock`; `HexConv2d`, `PolicyHead`,
`ValueBinnedHead`), `config.py` (`parse_model1_config` + `Model1ArchitectureConfig`), `constants.py`
(`BOARD_SIZE=41`, `BOARD_AREA=1681`, `INPUT_CHANNELS=13`), `plugin.py` (`DenseCNNPlugin`),
`inference.py`, `trainer.py`, `selfplay.py`, `input.py`/`d6.py`/`geometry.py`,
`samples.py`/`replay.py`/`compact_io.py`, `evaluation.py`, `performance.py`, `losses.py`, etc.
Rust crate `dense_cnn/rust/src/` (`encoding.rs`, `mcts*.rs`, `mcts_eval.rs`, `sample_gen.rs`) —
**architecture-blind** (`lib.rs`: *"deliberately contains no model logic"*; `mcts_eval.rs`: the
Python/Torch evaluator boundary). **=> the ResTNet change is purely the PyTorch trunk.**

### 2.2 The hexgnn → hexgt template (the recipe)
hexgnn forked hexgt as a **standalone top-level additive package** (`packages/hexgnn/`, namespace
`hexgnn`): copy the dir tree; rename every model-named identifier (`HexgtNetwork→HexgnnNetwork`,
`HexgtPlugin→HexgnnPlugin`, the config dataclasses, `parse_hexgt_config→parse_hexgnn_config`,
`hexgt_loss→hexgnn_loss`, the plugin `name`, the `model_family` string); add a `pyproject.toml`
with entry point `hexgnn = "hexgnn.plugin:get_plugin"`; add a config + launch script writing to an
isolated `runs/` dir. (hexgnn *also* compiled its own renamed Rust submodule into
`hexo_models._rust.hexgnn` via a `#[path]` include in `packages/hexo_models/rust/src/lib.rs` — we do
**not** need that; see §2.4.)

### 2.3 dense_cnn → dense_cnn_restnet renames
Standalone package `packages/dense_cnn_restnet/python/dense_cnn_restnet/…`, entry point
`dense_cnn_restnet = "dense_cnn_restnet.plugin:get_plugin"`. Renames: `Model1Network`→
`RestnetNetwork`, `DenseCNNPlugin`→`DenseCNNRestnetPlugin`, `parse_model1_config`→
`parse_restnet_config`, `Model1ArchitectureConfig`→`RestnetArchitectureConfig`, `DenseCNNTrainer`→
`RestnetTrainer`, `DenseCNNInference`→`RestnetInference`, plugin `name`, `model_family`,
`__init__.__all__`.

### 2.4 No Rust work — reuse `_rust.dense_cnn` (recommended)
The featurizer (13-plane 41×41 encode), MCTS, sample facts, and the `(N,13,41,41)` byte contract are
**identical** for a ResTNet trunk. So `dense_cnn_restnet` needs **no Rust crate**: its
`rust_bridge.py` points at the existing `hexo_models._rust.dense_cnn` submodule. Benefits: **no
`lib.rs` edit, no native rebuild, no `.so` churn** → the live run's loaded native module is never
touched. (Unlike hexgnn, which added a submodule and thus required a rebuild — unnecessary here.)
The fork becomes a **pure-Python additive package**.

### 2.5 Python-only vs native
| Work item | Python-only? | Native? |
|---|---|---|
| TransformerBlock / MSA_rel / MLP | ✅ | ❌ |
| `blocks_type` interleave trunk | ✅ | ❌ |
| Relative-position bias table | ✅ | ❌ |
| R↔T reshape + head adapter | ✅ | ❌ |
| Config keys + plugin wiring | ✅ | ❌ |
| Featurizer / MCTS / replay | reuse `_rust.dense_cnn` & dense_cnn Python | ❌ |
| **Everything** | **✅ Python-only** | **❌ none** |

---

## 3. The architecture change — faithful port

### 3.1 Where the trunk lives, and the block-type decision
dense_cnn's `Model1Network` builds `self.blocks = nn.Sequential(*[GatedResBlock(...) for _ in
range(blocks)])` over a `(N,C,41,41)` map. The fork replaces this homogeneous build with the
ResTNet `blocks_type`-driven build (§1.1).

**Residual block — faithfulness vs hex-correctness (a real choice):**
- The paper's R block is a **plain** AlphaZero block with **plain `Conv2d`** (§1.2). dense_cnn uses
  a **gated** block with **hex-masked `HexConv2d`** (corners zeroed for hex adjacency).
- For maximum fidelity to ResTNet, port the **plain post-activation ResidualBlock verbatim**. To
  *also* respect Hexo's hex geometry, the only principled tweak is to keep the 3×3 conv **hex-masked**
  (`HexConv2d`) — i.e. the paper's plain residual block, with dense_cnn's hex conv as the local
  operator. This changes nothing about ResTNet's contribution (the R block is still "two convs +
  BN + ReLU + residual"); it just uses the correct local kernel for a hex board.
- **Recommendation:** port the plain ResidualBlock (drop the gating, to match the paper), and make
  the conv choice a config flag `residual_conv ∈ {hex, plain}` defaulting to `hex` (hex-correct) with
  `plain` available for a literal paper match. Do **not** silently keep `GatedResBlock` — gating is a
  dense_cnn deviation the paper doesn't have.

### 3.2 The Transformer block — port `block_unit.py` directly
Port `TransformerBlock`, `MSA_rel`, and `MLP` essentially verbatim (Appendix A). Notes for the port:
- Keep **pre-norm LayerNorm, GELU MLP at ratio 2, 4 heads, scaling `1/√(C/4)`, learned
  relative-bias table** — all exactly as shipped.
- The block emits **tokens**; ensure the downstream module (next R block, or the head) reshapes. The
  R block already handles `dim==3`. The dense_cnn heads do **not**, so add the rearrange in §3.4.
- Use `einops` (already a small, common dep) or hand-rolled `flatten(2).transpose(1,2)` /
  `reshape` to avoid adding `einops` if undesirable — the math is identical.
- **Square-grid assumption:** `MSA_rel`'s relative-index math assumes `H==W`. The 41×41 crop
  satisfies this. Add an assert.

### 3.3 Interleave + positional encoding via config
Add to `RestnetArchitectureConfig` (fail-fast parsing per project convention):
- `blocks_type: str = "R_R_T_R_R_T"` — validated: tokens in `{R,T}` split on `_`; length is the
  block count. Provide a tiny `R3(RRT)` expander helper (`"R" + "_R_R_T"*3` → `"R_R_R_T_R_R_T_R_R_T"`)
  for convenience, but store the explicit string.
- `hidden_channels: int = 256` (paper width; the fork is independent of the live 64-ch run).
- `attention_heads: int = 4`, `mlp_ratio: int = 2`, `embed_kernel_size: int = 3` (paper values; expose
  for completeness).
- `residual_conv: str = "hex"` (§3.1).
Positional encoding is **the learned relative-position bias inside `MSA_rel`** — nothing to add at
the trunk level (no absolute/sinusoidal embedding; the paper's encoding is the relative bias). This
is fully faithful and we keep it as-is.

### 3.4 Heads & training contract — the one principled deviation
ResTNet's shipped heads (`P`, `TV`) emit a **scalar tanh value** + flat policy. dense_cnn's pipeline
is built around **binned value (65 bins)**, an **opponent-policy** auxiliary head, optional
**short-term-value** heads, the `model1_loss`, the NPZ replay schema, and the Rust MCTS evaluator's
value/prior decode. Swapping in `P`/`TV` would cascade into the loss, replay schema, sample facts,
and the MCTS value decode — a large, pipeline-wide change orthogonal to the ResTNet contribution.

**Recommendation (keep dense_cnn's heads):** the ResTNet *contribution is the interleaved trunk*,
not the heads. Keep dense_cnn's `PolicyHead`, `ValueBinnedHead`, opp/STV heads, `model1_loss`,
replay schema, and MCTS decode unchanged, and insert a **token→conv rearrange** after the trunk so
the heads receive the `(N,C,41,41)` map they expect (the last trunk block is a `T`, which emits
tokens):
```python
x = self.embed(state)                 # (N, C, 41, 41)
for b in self.blocks: x = b(x)        # may end as tokens (N, 1681, C) if last block is T
if x.dim() == 3:                      # faithful boundary, same trick the paper's heads use
    x = x.transpose(1, 2).reshape(N, C, 41, 41)
policy = self.policy_head(x); value = self.value_head(x); ...
```
This keeps 100% of the Hexo training/inference contract and is the standard ResTNet boundary
operation (their own heads do exactly this `Rearrange`). Document it as the deliberate, minimal
deviation: **trunk = paper-faithful; heads = dense_cnn (pipeline contract).** If the owner later
wants the paper's exact `P`/`TV` heads too, that's a separate, larger pipeline change (note it as an
option, not a default).

### 3.5 Stem
Replace dense_cnn's `HexConv2d→ReLU` stem with the paper's `EmbedNet` semantics: `Conv(kernel=
embed_kernel_size)→BN→ReLU` (add the BN the paper has; keep `HexConv2d` if `residual_conv="hex"` for
consistency). `embed_kernel_size=3` matches all paper configs.

---

## 4. Full attention at 1681 tokens — cost documented (FYI, non-goal)

Per owner intent we **keep full global O(N²) self-attention** over all 1681 board-cell tokens. No
windowing, no downsampling, no sparsification. This section is an honest FYI on what that costs, and
the one *non-architectural* accommodation needed to make training fit in memory.

### 4.1 Scale vs the paper
Paper max board: 19×19 = **361 tokens**. dense_cnn crop: 41×41 = **1681 tokens** = **4.66×**, and
attention is O(N²) → **≈21.7× more attention-matrix work per T-block** than 19×19. At the paper's 256
channels this is a large but bounded constant.

### 4.2 Memory (the only thing that can *block* a run)
The attention logits `dots` and the softmax `attn` are each `(B, heads=4, N=1681, N=1681)`:
- Per board, per T-block, fp16: `4 · 1681² · 2 B` ≈ **22.6 MB** (×2 for `dots`+`attn` ≈ 45 MB).
- **Self-play / inference** (no autograd; blocks run sequentially; real leaf batch mean ≈99, p95 228,
  chunk cap 1024): peak ≈ one block's `dots`+`attn` + the relative-bias broadcast. At B=128 that's
  **~6 GB** — **fits the single GPU.** At the chunk cap 1024 it would be ~46 GB → keep the MCTS eval
  chunk modest (the existing `MCTS_EVAL_CHUNK_STATES`/calibration already buckets to ~128–256).
- **Training** with autograd retaining intermediates across (e.g.) 3 T-blocks at batch 1024:
  `1024 · 45 MB · 3` ≈ **>130 GB** → **OOM**. So training the faithful net needs a **smaller batch**
  and/or **gradient checkpointing**, and/or `F.scaled_dot_product_attention`.

### 4.3 SDPA is faithful, not a compromise
`torch.nn.functional.scaled_dot_product_attention` (flash / memory-efficient backends) computes the
**identical full attention** — every query attends to every one of the 1681 keys — but tiles the
computation so the N² matrix is never resident in HBM (memory drops to O(N)). It is **not**
windowed/local/downsampled; the result is bit-comparable up to floating-point reordering. The
paper's **relative-position bias** is an additive term on the logits, and SDPA accepts an additive
`attn_mask`, so the relative bias is passed through and the math stays exact.

**Recommendation:** implement attention **twice**: (1) port `MSA_rel` verbatim (materialized `dots`)
as the **numerical correctness oracle** for tests and small-batch analysis; (2) an SDPA-backed
`MSA_rel_sdpa` that takes the same relative-bias tensor as `attn_mask` for memory-bound training and
larger self-play batches. Unit-test that (2) matches (1) within fp tolerance. This respects "no
architectural compromise" while making the faithful net trainable on one GPU.

### 4.4 Throughput FYI (non-goal, for planning only)
Grounded in measured dense_cnn numbers (live run 64ch×10, 512 visits, 256 active; conv-only forward
≈70% of the search wall; ~38 search pos/s torch FP16 / ~90 with TRT at 96ch×6):
- The faithful config is **256 channels** (16× the FLOPs/conv vs 64ch) **plus** 2–3 full
  1681-token T-blocks. Expect self-play forward **~5–20× slower** than the live conv-only run →
  **low single-digit search pos/s**. Self-play epochs will be long; plan run duration accordingly.
- **TensorRT FP16** (dense_cnn's 2.3× lever) is unlikely to cover the relative-bias attention
  cleanly; assume the ResTNet trunk runs on the **torch** backend. (Not a blocker — perf is a
  non-goal.)
- VRAM, not speed, is the only thing that can *prevent* a run; §4.2/§4.3 resolve it.

---

## 5. Effort breakdown (S / M / L)

| Work item | Size | Notes |
|---|---|---|
| Fork scaffolding (copy dir, rename ids, `pyproject` entry point, reuse `_rust.dense_cnn`) | **S–M** | hexgnn is the template; no native work. |
| Port `ResidualBlock` (plain, optional hex conv) | **S** | Verbatim from `block_unit.py`; drop gating. |
| Port `TransformerBlock` + `MLP` | **S** | Verbatim; pre-norm, GELU, ratio 2. |
| Port `MSA_rel` (materialized) + relative-bias table | **S–M** | Verbatim; assert H==W; precompute `relative_index` buffer. |
| SDPA-backed attention equivalent + parity test | **M** | Same math, for memory-bound training (§4.3). |
| `blocks_type` interleave build + `R3(RRT)` expander + config keys/validation | **S** | String split on `_`; fail-fast parse. |
| Stem (`EmbedNet` semantics) + trunc_normal init | **S** | Add BN to stem; `self.apply` init. |
| Token→conv head adapter; keep dense_cnn heads/loss/replay | **S** | One rearrange; preserves training contract. |
| Plugin/config/launch wiring (`parse_restnet_config`, `build_model`, `configs/…`, `_rl_train_…`) | **S–M** | Copy dense_cnn pipeline; isolated `runs/` dir. |
| Tests | **M** | Block shape/dtype, MSA materialized-vs-SDPA parity, relative-bias index correctness, interleave parser, full-forward smoke, D6/sample-pipeline correctness (trunk is no longer D6-equivariant — verify training-time D6 augmentation still valid). |
| Memory/throughput validation on a CPU smoke + short warm-GPU probe | **M** | Confirm no OOM at the chosen self-play leaf-batch; record pos/s as FYI. |
| **Training to a comparable checkpoint** | **L (open-ended)** | Dominant cost; from-scratch interleaved net, slow self-play. |

### Hard parts & risks (re-prioritized for the faithful build)
1. **Training-time VRAM at 1681 tokens** — the only thing that can stop a run. Resolved by
   small batch + SDPA + (if needed) gradient checkpointing (§4.2–4.3). Verify early.
2. **Relative-bias index correctness** — easy to get subtly wrong (the paper's code uses
   `indexing="xy"` and reuses `H-1` for both axes, valid only for square boards). Port verbatim,
   assert H==W, and unit-test against a tiny hand-computed grid.
3. **Training stability of a from-scratch interleaved net** — the paper uses large LRs (0.02–0.1)
   tuned per game and ViT-style trunc_normal init; our LR/data differ. Keep the trunc_normal init,
   start conservative on LR, watch early loss.
4. **D6 / symmetry** — the conv-only dense_cnn trunk pairs with training-time D6 augmentation; a
   global-attention trunk is not D6-equivariant by construction, but D6 *augmentation* remains valid
   (it augments samples, doesn't assume model equivariance). Add a correctness test to be sure no
   symmetry assumption is baked where attention breaks it.
5. **Throughput** — large but accepted (non-goal). Documented in §4.4.

---

## 6. Verdict & recommended faithful first config

**Difficulty:** the fork + faithful architecture is **Easy–Moderate, ~1 week** of engineering —
materially easier than rev 1 implied, because the paper's *released code* lets us mirror every block
verbatim instead of guessing. **No Rust work.** The dominant cost is training the model, not building
it.

**Central constraint (not "risk"):** GPU memory for full 1681-token attention at training time —
fully resolved by small batch + SDPA (identical math) + optional gradient checkpointing. Throughput
is slow but explicitly a non-goal.

**Worth doing for Hexo:** yes — 19×19 Hex 50.4→58.0% (R3(RRT)) is the directly relevant evidence, and
the fork is additive and non-disruptive to the live run.

**Recommended first build (paper-faithful):**
- **Fork** as a standalone additive package `packages/dense_cnn_restnet/`, **reusing
  `hexo_models._rust.dense_cnn`** read-only — zero native rebuild, zero risk to `dense_cnn_rl_main1`.
- **Port `block_unit.py` verbatim:** plain `ResidualBlock` (hex-conv variant for Hexo), pre-norm
  `TransformerBlock` (GELU MLP ratio 2), `MSA_rel` with the learned relative-position bias table,
  4 heads, `1/√(C/4)` scaling, `trunc_normal_(std=0.02)` init.
- **Width = 256 hidden channels, embed_kernel_size = 3** — the paper's 19×19 settings.
- **Headline config: `blocks_type = "R_R_R_T_R_R_T_R_R_T"` (R3(RRT), 10 blocks, 3 T)** — mirrors the
  exact net that won 19×19 Hex. **Bring-up config: `blocks_type = "R_R_T_R_R_T"` (RRTRRT, 6 blocks,
  2 T)** — smaller, faster to first-light, the paper's 6-block sweet spot. Also wire `10R`
  (`"R_R_R_R_R_R_R_R_R_R"`) as the in-lineage baseline to reproduce the paper's +7.6-point comparison
  on Hexo.
- **Full global attention over all 1681 tokens** — no windowing/downsampling. Materialized `MSA_rel`
  as the correctness oracle; SDPA-backed equivalent (relative bias as additive mask) for
  memory-bound training and self-play.
- **Keep dense_cnn's heads/loss/replay/MCTS** unchanged; insert the token→conv rearrange before the
  heads. (One principled deviation: heads stay dense_cnn for the Hexo training contract; the trunk is
  paper-faithful.)
- **Validate VRAM and forward on a CPU smoke + short warm-GPU probe** into an isolated
  `runs/dense_cnn_restnet_*` dir before any real launch; do not contend with the live run for GPU.

**One-line recommendation:** the faithful build is cheap to stand up because we have ResTNet's exact
code — port `block_unit.py` into a dense_cnn fork, run R3(RRT)@256 with full 1681-token attention
(materialized oracle + SDPA for memory), keep dense_cnn's heads, and accept slow self-play; the only
thing to verify early is training-time VRAM, which SDPA + small batch resolves without compromising
the architecture.

---

## Appendix A — Reference implementation (verbatim, `github.com/rlglab/restnet`)

`restnet/learner/network/block_unit.py`:
```python
class ResidualBlock(nn.Module):
    def __init__(self, num_channels, input_channel_height):
        super().__init__()
        self.token_to_conv = Rearrange("b (h w) c -> b c h w", h=input_channel_height)
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)
    def forward(self, x):
        if x.dim() == 3: x = self.token_to_conv(x)
        input = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(input + x)

class TransformerBlock(nn.Module):
    def __init__(self, emb_size, MLP_hsize, n_head, input_channel_height, input_channel_width,
                 drop=0.0, MLP_drop=0.0):
        super().__init__()
        self.conv_to_token = Rearrange("b c h w -> b (h w) c")
        self.MSA = MSA_rel(emb_size, n_head, drop, input_channel_height, input_channel_width)
        self.MLP = MLP(emb_size, MLP_hsize, MLP_drop)          # MLP_hsize = emb_size * mlp_ratio
        self.ln1 = nn.LayerNorm(emb_size); self.ln2 = nn.LayerNorm(emb_size)
    def forward(self, x):
        if x.dim() == 4: x = self.conv_to_token(x)
        x = x + self.MSA(self.ln1(x))     # pre-norm self-attention
        x = x + self.MLP(self.ln2(x))     # pre-norm FFN
        return x                          # tokens (b, h*w, c)

class MSA_rel(nn.Module):
    def __init__(self, emb_size, num_heads, dropout, H, W):
        super().__init__()
        self.emb_size, self.num_heads = emb_size, num_heads
        self.token_len = H * W
        self.projq = nn.Sequential(nn.Linear(emb_size, emb_size), Rearrange("b n (h d) -> b h n d", h=num_heads))
        self.projk = nn.Sequential(nn.Linear(emb_size, emb_size), Rearrange("b n (h d) -> b h n d", h=num_heads))
        self.projv = nn.Sequential(nn.Linear(emb_size, emb_size), Rearrange("b n (h d) -> b h n d", h=num_heads))
        self.rearrange_out = Rearrange("b h n d -> b n (h d)")
        self.rearrange_rel = Rearrange("(h w) c -> 1 c h w", h=self.token_len, w=self.token_len)
        self.relative_bias_table = nn.Parameter(torch.zeros((2*H-1)*(2*W-1), num_heads))
        coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="xy")
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[0] += H - 1
        relative_coords[1] += H - 1
        relative_coords[0] *= 2*H - 1
        relative_coords = rearrange(relative_coords, "c h w -> h w c")
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)
        self.scaling = (emb_size / num_heads) ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.out = nn.Sequential(nn.Linear(emb_size, emb_size), nn.Dropout(dropout))
    def forward(self, x):
        q, k, v = self.projq(x), self.projk(x), self.projv(x)
        dots = torch.einsum("b h q d, b h k d -> b h q k", q, k) * self.scaling
        rel = self.relative_bias_table.gather(0, self.relative_index.repeat(1, self.num_heads))
        dots = dots + self.rearrange_rel(rel)
        attn = self.attend(dots)
        x = torch.einsum("b h d i, b h i v -> b h d v", attn, v)
        return self.out(self.rearrange_out(x))

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))
    def forward(self, x): return self.net(x)
```
`alphazero_network.py` (trunk): `num_head=4`, `mlp_ratio=2` hardcoded; `blocks = ModuleList(
get_backbone(bt) for bt in blocks_type.split("_"))`; `get_backbone("R")→ResidualBlock`,
`("T")→TransformerBlock`; `self.apply(_init_weights_trunc_normal)` → `trunc_normal_(std=0.02)` on
Linear. `embed_unit.py`: `EmbedNet = Conv(kernel=embed_kernel_size, bias=False)→BN→ReLU`.

## Appendix B — Port mapping (restnet → dense_cnn_restnet)
| restnet (MiniZero) | dense_cnn_restnet (Hexo) |
|---|---|
| `EmbedNet`, in=feature planes | stem over **13** input planes, 41×41 |
| `num_hidden_channels` (256) | `hidden_channels` config (default 256) |
| `blocks_type` string split `_` | `blocks_type` config (same semantics) |
| `ResidualBlock` (plain Conv2d) | plain block; conv = `HexConv2d` if `residual_conv="hex"` |
| `TransformerBlock`/`MSA_rel`/`MLP` | ported verbatim (materialized + SDPA variants) |
| relative-bias table, H==W | 41==41 ✓ (assert) |
| heads `P`/`TV` (scalar tanh) | **replaced by** dense_cnn `PolicyHead`+`ValueBinnedHead`+opp/STV (token→conv adapter) |
| Gumbel-AlphaZero/MiniZero training | dense_cnn `TrainingPipeline` + replay + Rust MCTS (unchanged) |
| LibTorch/C++ inference | Rust MCTS calls Python torch evaluator (unchanged) |

## Appendix C — In-repo grounding (read-only)
`packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/{architecture.py,config.py,constants.py,
plugin.py,inference.py,performance.py}`; `…/dense_cnn/rust/src/{lib.rs,mcts_eval.rs,encoding.rs}`
(arch-agnostic); `packages/hexgnn/` + `packages/hexo_models/rust/src/lib.rs` (fork template);
`configs/dense_cnn_rl_main1.toml` (live: 64ch×10, 512 visits, 256 active);
`analysis/throughput_understanding.md` (≈38→90 search pos/s; forward ≈70% of wall; leaf batch mean
≈99/p95 228); `CLAUDE.md` (≥128 searched pos/s calibration target — for the conv baseline, not this
faithful trunk).
