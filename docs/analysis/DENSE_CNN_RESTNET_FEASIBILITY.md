# dense_cnn_restnet — Feasibility & Design Report

**Status:** analysis only. No code in `dense_cnn`, `hexgt`, `hexgnn`, or any live run is
modified by this document.
**Date:** 2026-06-09
**Author intent:** a true *fork* of the `dense_cnn` lineage — copy the package into a new
`dense_cnn_restnet` lineage (the way `hexgnn` was forked from `hexgt`), modify only the copy,
and have **both** lineages coexist and be independently launchable. Implement the ResTNet
architecture from *"Bridging Local and Global Knowledge via Transformer in Board Games"*
(Ju, Wu, Shih, Wu — IJCAI 2025, Academia Sinica;
[arXiv:2410.05347](https://arxiv.org/abs/2410.05347),
[project page](https://rlg.iis.sinica.edu.tw/papers/restnet/)) "almost exactly as defined in
the paper."

---

## 0. TL;DR / Verdict

**Forking the package is Easy. Implementing the architecture is Easy. Making it run at an
acceptable self-play throughput at the current 41×41 / 1681-token crop is the entire risk, and
it is a real one.**

- **Fork scaffolding: S–M.** The `hexgnn` fork is a clean, documented template. dense_cnn is a
  bigger directory (more files, a Rust crate), so it's a larger copy than hexgnn, but
  mechanically identical. **The Rust crate does not need any architecture work** — it is a pure
  featurizer + MCTS that calls back into Python for the forward pass, and is completely blind to
  the trunk. You can even *skip copying the Rust crate* and have `dense_cnn_restnet` reuse
  `hexo_models._rust.dense_cnn` read-only (recommended — see §2.4), which is simpler and removes
  all native-rebuild risk to the live run.
- **The Transformer block itself: S.** The codebase *already contains* a working, tested
  PyTorch attention-over-tokens block (`hexgt`'s `GraphTransformerLayer`). The dense_cnn trunk is
  a clean `nn.Sequential` of residual blocks (`architecture.py`), so interleaving R and T per a
  config-driven pattern string is a small, localized change.
- **The central risk is attention cost at 1681 tokens.** The paper used **81 tokens (9×9 Go)**
  and **361 tokens (19×19 Go / 19×19 Hex)**. dense_cnn's fixed 41×41 crop is **1681 tokens** —
  **4.66× the largest board the paper tested, and attention is O(N²), so ~21.7× more
  attention-matrix work per T-block than 19×19.** A naïve `nn.MultiheadAttention` would
  materialize a **1681×1681 score matrix ≈ 2.9 GB at batch 128 per T-block** and tank both VRAM
  and throughput. This is **the** feasibility question; it is solvable (flash/SDPA kernels +
  modest mitigations) but it is the whole job.
- **Worth trying?** Yes, with eyes open. The paper's **19×19 Hex** result (50.4% → **58.0%** vs a
  10R baseline) is the single most relevant datapoint we have — Hexo is hex-based and connection-
  driven, exactly the regime where global/long-range reasoning helps. But "almost exactly as the
  paper" at our token count is not free; budget the throughput mitigation as the main task.
- **Rough effort: ~1–2 focused weeks** for a from-scratch, launchable, throughput-sane
  `dense_cnn_restnet` with an RRTRRT (or R3(RRT)) trunk and a token-grid strategy that keeps
  attention tractable — *plus* the open-ended cost of actually training it to a comparable
  checkpoint, which dwarfs the engineering.

**Recommended first cut:** fork → keep 64 channels → **RRTRRT** (6 blocks, 2 T-blocks) →
attention via `F.scaled_dot_product_attention` (flash/mem-efficient, never materialize N²) →
**2× spatial downsample of the token grid for the T-blocks** (41×41 → 21×21 = 441 tokens,
~14.5× cheaper than 1681 while preserving the paper's "attention over board positions" spirit).
Validate forward pos/s on a CPU/dev smoke before committing GPU. Details in §5.

---

## 1. The paper, pinned down

Extracted from the arXiv HTML and project page (the project page omits the block internals; the
arXiv version has them). Numbers are quoted where the paper gives them.

### 1.1 What ResTNet is
An AlphaZero-style trunk that **interleaves Residual (R) and Transformer (T) blocks** to combine
**local** (conv) and **global** (attention) knowledge. R blocks are standard pre-activation conv
residual blocks; T blocks are standard Transformer encoder blocks operating over the board cells
as tokens.

### 1.2 Block interleaving notation
- `R` = one residual block, `T` = one Transformer block. The trunk is a left-to-right string.
- **`RRTRRT`** = R, R, T, R, R, T = "2R1T2R1T" — **the best 6-block config**, 60.8% vs KataGo in
  9×9 Go (up from 54.6% for 6R).
- **`R3(RRT)`** expands to **`RRRTRRTRRT`** — one leading R, then the `RRT` triplet ×3 = 10 blocks.
  This is the best 10-block config and is what won **19×19 Go** and **19×19 Hex**.
- General empirical rule (not a theorem): a **repeating `RRT` pattern** balances global extraction
  (the T) against local pattern preservation (the surrounding Rs). They only explored permutations
  with **two T-blocks among six** (and the RRT-triplet extension) for cost reasons.

### 1.3 The Transformer block (the part that matters for us)
- **Standard Transformer encoder block.** Multi-head self-attention + position-wise FFN, each with
  a residual connection.
- **Heads:** **4** attention heads per T-block (all experiments).
- **FFN / MLP ratio:** **2** (hidden = 2× embedding) (Table 4, all experiments).
- **Embedding dim = channel count C.** No projection at the R↔T boundary — the token embedding
  size equals the trunk's hidden channels (e.g. 256 in their experiments). This is important: it
  means a T-block drops into the trunk **without changing the channel width**.
- **Positional encoding:** **relative position encoding, Shaw et al. (2018)** — *not* learned
  absolute, *not* sinusoidal-added-to-input. Tokens are ordered by board position (row-major or
  column-major); the relative-position bias carries 2D spatial relationships.
- **Norm:** the paper says "standard Transformer" and does not explicitly state pre- vs post-norm
  or LN-vs-other. (We will use pre-norm LayerNorm — it's what `hexgt` already uses and is the
  stable modern default; see §3.3.)
- **Activation:** not specified (ReLU is the safe match to the rest of the trunk and to hexgt).

### 1.4 Conv ↔ token conversion
- **Conv → tokens:** flatten the `(C, H, W)` feature map to `H·W` tokens of dim `C`, ordered by
  board position (row-major). No channel projection.
- **Tokens → conv:** reshape back to `(C, H, W)`, each token returned to its original cell.
- One-to-one spatial mapping; the only learned parameters in the boundary are inside the attention
  and FFN. **The token count = the spatial cell count of the feature map** (this is the crux for
  us — see §4).

### 1.5 Reported cost (their scale, 256ch, their token counts)
From Table 1 (inference time / params at their settings):
- `6R`: 3.067 ms, 7.146 M params
- `6T`: 8.130 ms, 3.358 M params (T-blocks have *fewer* params than R-blocks but are *slower*)
- `RRTRRT`: 4.566 ms, 5.967 M params

So at **their** token counts (81 / 361) a T-block is ~2.6× the latency of an R-block but only adds
~1.5 ms each. **The relative cost of a T-block explodes with token count** because attention is
O(N²) — that ratio does **not** carry over to our 1681 tokens (see §4).

### 1.6 Training details (their runs)
- Batch size **1024**; LR schedule e.g. 0.02→0.01→0.005 (9×9 Go), 0.1 (19×19 Go), 0.02 (19×19 Hex);
  100k–150k steps. Optimizer unspecified. **No special T-block init or warmup is mentioned** — they
  train the interleaved net from scratch like a plain ResNet.
- **No windowing/downsampling/efficiency tricks** — they use full-board attention. They could afford
  to because their largest board is 361 tokens.

### 1.7 Results (the case for trying)
| Game | Baseline (NR) | ResTNet | Config |
|---|---|---|---|
| 9×9 Go | 54.6% | **60.8%** | RRTRRT (6) |
| 19×19 Go | 53.6% | **60.9%** | R3(RRT) (10) |
| **19×19 Hex** | **50.4%** | **58.0%** | R3(RRT) (10) |

The **Hex** row is the directly transferable evidence: a hex connection game, a +7.6-point swing
from interleaving T-blocks into a 10R trunk.

---

## 2. Fork mechanics

### 2.1 How dense_cnn is structured (what gets forked)

**Python package** — `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/`:

| File | Purpose | Touched by the fork? |
|---|---|---|
| `architecture.py` | `Model1Network` (trunk = `nn.Sequential` of `GatedResBlock`), `HexConv2d`, `PolicyHead`, `ValueBinnedHead` | **YES — the core change** |
| `config.py` | `parse_model1_config()` + `Model1ArchitectureConfig` frozen dataclass | **YES — add block-pattern keys** |
| `constants.py` | `BOARD_SIZE=41`, `BOARD_AREA=1681`, `INPUT_CHANNELS=13`, `DEFAULT_CHANNELS`, `DEFAULT_BLOCKS`, plane indices | rename only |
| `plugin.py` | `DenseCNNPlugin` (`build_model`, training overrides, selfplay/eval/calibrate) | **YES — rename + build_model wiring** |
| `inference.py` | `DenseCNNInference` (batch build, AMP, TRT) | rename; arch-agnostic |
| `input.py`, `geometry.py`, `d6.py` | dense-plane build + D6 augmentation | copy as-is |
| `samples.py`, `compact_io.py`, `replay.py` | sample facts / NPZ replay schema | copy as-is |
| `trainer.py` | `DenseCNNTrainer` (loss, batching) | rename; arch-agnostic |
| `losses.py` | `model1_loss`, value binning | copy as-is |
| `selfplay.py`, `player.py`, `mcts.py` | self-play loop + Rust MCTS wrapper | rename; arch-agnostic |
| `evaluation.py`, `performance.py` | SealBot eval + calibration | rename; arch-agnostic |
| `checkpoints.py`, `debug_artifacts.py`, `rust_bridge.py`, `trt_backend.py` | misc | copy/rename |

**Rust crate** — `packages/hexo_models/dense_cnn/rust/src/`: `encoding.rs` (state→13-plane
tensor), `mcts.rs`/`mcts_tree.rs` (PUCT search), `mcts_eval.rs` (the Python-callback boundary),
`sample_gen.rs`, `constants.rs`, `state.rs`, `lib.rs`. **None of these know anything about the
network architecture.** Evidence: `lib.rs` — *"It deliberately contains no model logic"*;
`mcts_eval.rs` — *"MCTS owns the tree … while PyTorch remains the neural evaluator … Call the
Python/Torch evaluator through the strict byte contract."* The Rust side encodes states, runs the
tree, batches leaves, and hands `(N,13,41,41)` bytes to Python; what happens inside the forward
pass is opaque to it.

**=> The ResTNet change is 100% a PyTorch `nn.Module` change. No Rust logic changes.**

### 2.2 The hexgnn → hexgt fork template (the canonical recipe)
`hexgnn` was forked from `hexgt` as a **standalone top-level package** (`packages/hexgnn/`, Python
namespace `hexgnn`, not nested under `hexo_models`). The mechanical steps it followed:

1. **Copy the directory tree** (`hexgt/` → `hexgnn/`), every `.py` (and, in hexgnn's case, every
   `.rs`) verbatim.
2. **Rename every model-named identifier:** plugin class (`HexgtPlugin`→`HexgnnPlugin`), network
   (`HexgtNetwork`→`HexgnnNetwork`), trainer, checkpoint IO, the 7 config dataclasses,
   `parse_hexgt_config`→`parse_hexgnn_config`, `hexgt_loss`→`hexgnn_loss`, the Rust session
   (`HexgtMctsSession`→`HexgnnMctsSession`), Rust pyfns (`hexgt_candidate_ids`→`hexgnn_…`), the
   `rust_bridge` getter (`_hexgt_module`→`_hexgnn_module`), the `name = "…"` plugin attribute, and
   the `"model_family"` metadata string.
3. **New `pyproject.toml`** with `[project.entry-points."hexo_train.models"]`:
   `hexgnn = "hexgnn.plugin:get_plugin"`.
4. **Wire the Rust submodule** in `packages/hexo_models/rust/src/lib.rs` via a `#[path]` include and
   a `register_pybridge` + `add_submodule` call, exposing `hexo_models._rust.hexgnn`.
5. **`rust_bridge.py`** imports `from hexo_models import _rust` and reads its **own** submodule
   (`getattr(_rust, "hexgnn")`).
6. **New config** `configs/hexgnn_model.toml` with `[model] name="hexgnn" module="hexgnn.plugin"`,
   and a launch script `scripts/_rl_train_hexgnn.py` / `_rl_launch_hexgnn.sh` that bootstraps
   `sys.path` to `packages/hexgnn/python` and writes to an isolated `runs/hexgnn_rl*` dir.

The key lib.rs wiring it added (the template to copy):
```rust
#[cfg(feature = "python")]
#[path = "../../../hexgnn/rust/src/lib.rs"]
mod hexgnn;
// …in _rust(): create PyModule "hexgnn", hexgnn::register_pybridge(&m),
// sys.modules["hexo_models._rust.hexgnn"] = m, module.add_submodule(&m)
```

### 2.3 What gets copied / renamed for dense_cnn → dense_cnn_restnet
Same recipe, dense_cnn names. Two viable layouts:

**Layout A — standalone package (mirror hexgnn exactly):** `packages/dense_cnn_restnet/python/
dense_cnn_restnet/…`, top-level namespace `dense_cnn_restnet`, own `pyproject.toml`, entry point
`dense_cnn_restnet = "dense_cnn_restnet.plugin:get_plugin"`. Renames:
`Model1Network`→`RestnetNetwork`, `DenseCNNPlugin`→`DenseCNNRestnetPlugin`,
`parse_model1_config`→`parse_restnet_config`, `Model1ArchitectureConfig`→`RestnetArchitectureConfig`,
`DenseCNNTrainer`→`RestnetTrainer`, `DenseCNNInference`→`RestnetInference`, the plugin `name`
attribute, `model_family` string, and `__init__.py` `__all__` exports.

**Layout B — nested under hexo_models (mirror dense_cnn/hexgt):**
`packages/hexo_models/dense_cnn_restnet/…`, entry point
`dense_cnn_restnet = "hexo_models.dense_cnn_restnet.plugin:get_plugin"`. This keeps it inside the
monolithic package like the original dense_cnn.

Either works. **Layout A (standalone, additive) is recommended** precisely because it is what
hexgnn did to stay *non-disruptive to a live run*: installing/uninstalling an additive top-level
package never rebuilds the native module the live `dense_cnn_rl_main1` run has loaded.

### 2.4 Does the Rust crate need to be involved? (No — and you can skip it)
Because the trunk swap is Python-only and the Rust featurizer/MCTS are identical between dense_cnn
and a ResTNet trunk, **`dense_cnn_restnet` does not need its own Rust crate at all.** It can import
`hexo_models._rust.dense_cnn` read-only — its `rust_bridge.py` simply keeps pointing at the
existing `dense_cnn` submodule. This is strictly better for the live-run constraint:

- **No `lib.rs` edit, no native rebuild, no `.so` churn.** The live run's loaded
  `hexo_models._rust` is never touched. (Compare hexgnn, which *did* add a submodule and therefore
  *did* require a native rebuild — unnecessary risk here since the featurizer is unchanged.)
- The 13-plane encoding, MCTS, sample facts, and the `(N,13,41,41)` byte contract are all identical
  — there is nothing for a forked Rust crate to do differently.

**Recommendation:** Layout A + reuse `_rust.dense_cnn`. The fork becomes a **pure-Python additive
package**: copy the Python dir, rename identifiers, point `rust_bridge` at the existing dense_cnn
submodule, add a `pyproject.toml` entry point, add a config + launch script. Zero native work, zero
risk to the live run. (If you later want the ResTNet lineage to diverge in featurization, you can
add a renamed Rust submodule then — but that's not needed for the architecture change.)

### 2.5 Python-only vs native, summarized
| Work item | Python-only? | Native (Rust) needed? |
|---|---|---|
| Transformer block module | ✅ | ❌ |
| R/T interleave pattern | ✅ | ❌ |
| Positional encoding | ✅ | ❌ |
| R↔T token reshape adapters | ✅ | ❌ |
| Config keys + plugin wiring | ✅ | ❌ |
| Featurizer / encoding | unchanged | reuse `_rust.dense_cnn` |
| MCTS search | unchanged | reuse `_rust.dense_cnn` |
| **Everything** | **✅ Python-only** | **❌ none (if reusing dense_cnn's submodule)** |

---

## 3. The architecture change

### 3.1 Where the trunk lives today
`architecture.py` → `Model1Network.__init__` builds:
```python
self.conv_in = HexConv2d(in_channels, channels, kernel_size=3, padding=1)
self.activation = nn.ReLU(inplace=True)
self.blocks = nn.Sequential(*[GatedResBlock(channels, dropout) for _ in range(blocks_count)])
self.policy_head = PolicyHead(channels)
self.value_head  = ValueBinnedHead(channels)
# + opp_policy_head, short_term_value_heads
```
The trunk is a **flat `nn.Sequential` of `GatedResBlock`** over a `(N, C, 41, 41)` feature map. This
is the ideal shape to fork: interleaving is "replace the homogeneous `for _ in range(blocks)` build
with a pattern-driven build that emits an `R` or a `T` module per character."

Note `HexConv2d` masks the 3×3 kernel corners for hex adjacency — that is a property of the **R**
blocks and stays. T blocks operate on flattened tokens and don't use the hex mask (attention is
all-to-all over cells anyway).

### 3.2 (a) Adding the Transformer block module — reuse what's already here
The codebase already has a correct, tested attention block: **`hexgt`'s `GraphTransformerLayer`**
(`packages/hexo_models/hexgt/python/hexo_models/hexgt/architecture.py`):
```python
self.ctx_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
self.norm_ctx1 = nn.LayerNorm(dim); self.norm_ctx2 = nn.LayerNorm(dim)
self.ffn_ctx = nn.Sequential(nn.Linear(dim, ffn_dim), nn.ReLU(True), nn.Linear(ffn_dim, dim))
# pre-norm residual:  x = norm1(x + attn(x));  x = norm2(x + ffn(x))
```
That is *exactly* the paper's T-block (4 heads, FFN ratio 2, pre-norm, ReLU) — minus the
candidate/context cross-attention split (we don't need it; dense attention is self-attention over
all cells). A dense T-block for dense_cnn is essentially the **self-attention half** of
`GraphTransformerLayer`, wrapped with a `(C,H,W)↔(N_tok,C)` reshape:

```python
class TransformerBlock(nn.Module):       # ~40 lines, mostly copied from hexgt
    def __init__(self, channels, heads=4, mlp_ratio=2, dropout=0.0):
        self.norm1 = nn.LayerNorm(channels)
        self.attn  = nn.MultiheadAttention(channels, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn   = nn.Sequential(nn.Linear(channels, mlp_ratio*channels), nn.ReLU(True),
                                   nn.Linear(mlp_ratio*channels, channels))
        # + relative-position bias (Shaw 2018) — see §3.4
    def forward(self, x):                # x: (N, C, H, W)
        n, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2) # (N, H*W, C)  row-major tokens
        t = t + self.attn(self.norm1(t), ...)[0]   # pre-norm self-attn (use SDPA, §4)
        t = t + self.ffn(self.norm2(t))
        return t.transpose(1, 2).reshape(n, c, h, w)
```
**Effort: S.** The only non-trivial parts are (i) using `F.scaled_dot_product_attention` instead of
`nn.MultiheadAttention` so the N² score matrix is never materialized (§4), and (ii) the relative
positional bias (§3.4).

### 3.3 (b) Config-driven R/T interleave
Replace the homogeneous trunk build with a pattern string:
```python
# config: block_pattern = "RRTRRT"  (or expand R3(RRT) -> "RRRTRRTRRT")
blocks = []
for ch in pattern:
    blocks.append(GatedResBlock(channels, dropout) if ch == "R"
                  else TransformerBlock(channels, heads, mlp_ratio, dropout, token_grid=...))
self.blocks = nn.Sequential(*blocks)
```
Add to `RestnetArchitectureConfig`:
- `block_pattern: str = "RRTRRT"` (validated: only `R`/`T`, length = total blocks; reject unknown
  chars per the project's fail-fast config convention).
- `attention_heads: int = 4`
- `mlp_ratio: int = 2`
- `token_grid: str = "full" | "downsample2" | "downsample4"` (the throughput lever — §4).
- (optionally keep `residual_blocks` as a derived/ignored field, or drop it; the pattern subsumes
  it.)

The paper's R3(RRT) "macro" can be a tiny expander: `R3(RRT)` → `"R" + "RRT"*3`. **Effort: S.**

### 3.4 Positional encoding
The paper uses **Shaw et al. (2018) relative position encoding**. Faithful implementation: a learned
relative-position bias added to the attention logits, indexed by the (Δrow, Δcol) offset between
token i and token j on the 41×41 grid. For a `H×W` grid that's a `(2H−1)×(2W−1)` table of per-head
biases gathered into an `(N_tok, N_tok)` bias matrix.

**Caveat:** a full relative-bias matrix is itself `N_tok × N_tok` and **breaks flash-attention's
no-materialization property** (SDPA only fuses additive masks/bias if passed as `attn_mask`, and a
1681×1681 bias is 11 MB/head fp16 — materializable but it forces the math backend, losing the flash
speedup). Options, in order of recommended:
1. **2D learned absolute positional embedding** added to tokens at the R→T boundary (a
   `(H*W, C)` parameter, or factorized row+col embeddings). Simple, flash-friendly, and for a
   *fixed* 41×41 crop, absolute ≈ relative in expressiveness. This is the pragmatic first cut and a
   defensible deviation. **Effort: S.**
2. **Relative bias on a downsampled token grid** (§4): at 21×21 = 441 tokens the bias matrix is
   441² ≈ 194k entries/head — cheap to materialize, and you keep the paper's relative encoding.
   **Effort: M.**
3. **Full relative bias at 1681** faithfully — possible but couples to the slow attention path; only
   if you've already solved §4 by downsampling. **Effort: M, and it costs throughput.**

For "almost exactly as the paper," option 2 (relative bias on a downsampled grid) is the best
trade; option 1 is the fastest path to a running net.

### 3.5 R↔T boundary channel/token handling
Because **embedding dim = channels** (paper §1.3), there is **no projection** at the boundary — the
adapter is pure `flatten`/`reshape` (shown in §3.2). The only subtlety is the **token grid**: if a
T-block downsamples (§4), it must pool `(C,41,41)→(C,h,w)`, attend, then upsample back to
`(C,41,41)` before the next R-block. That's a `nn.AvgPool2d`/`F.interpolate` pair (or a strided conv
down + transposed conv up if you want it learned). **Effort: S–M** depending on pooled vs learned.

---

## 4. The central feasibility question: attention cost at 1681 tokens

### 4.1 Why this is the whole ballgame
The paper's largest board is **19×19 = 361 tokens**. dense_cnn's crop is **41×41 = 1681 tokens**.
Self-attention compute and the score-matrix memory both scale as **O(N_tok²)**:

- Token ratio: 1681 / 361 = **4.66×**
- Attention-matrix ratio: 4.66² = **≈21.7×** more score-matrix work per T-block than 19×19, and
  **≈430×** more than the 9×9 (81-token) setting where they got the cleanest wins.

### 4.2 Memory (the hard wall, with naïve attention)
A materialized score tensor is `(N_batch, heads, N_tok, N_tok)`. At the live run's forward batch
(mean ≈ 99, padded to 128; p95 228; chunk cap 1024):
- batch 128, 4 heads, fp16: `128 × 4 × 1681 × 1681 × 2 B` = **2.89 GB** for **one** T-block's scores.
- batch 256: **5.8 GB**. With 2 T-blocks (RRTRRT) and activations, a naïve implementation **OOMs**
  on a single consumer GPU.

**Mitigation that removes this wall entirely:** use
`torch.nn.functional.scaled_dot_product_attention` (flash / memory-efficient backend). It never
materializes the N² matrix → attention memory drops to **O(N_tok)** activations. With SDPA, batch
128 at 1681 tokens is **memory-tractable** (a few hundred MB of activations). *This is mandatory;
do not use plain `nn.MultiheadAttention` with `need_weights=True` here.*

### 4.3 Compute (the throughput tax, even with flash)
Flash attention removes the memory wall but **not** the O(N²) FLOPs. Per-board MAC estimate at
C=64, 1681 tokens, 4 heads, FFN ratio 2:
- QKV + out projections: ~4 · N·C² = 4·1681·64² ≈ **27.5 M**
- Attention scores QKᵀ + AV: 2 · N²·C = 2·1681²·64 ≈ **362 M**  ← dominates
- FFN (ratio 2): 2 · N·C·2C = 2·1681·64·128 ≈ **27.5 M**
- **T-block ≈ ~417 M MACs/board.**

Compare a `GatedResBlock` (two masked 3×3 convs, C→C, over 41×41):
- 2 · H·W·C²·9 = 2·1681·64²·9 ≈ **124 M MACs/board.**

So at our token count **one T-block ≈ 3.4× the FLOPs of one R-block** (vs ~2.6× *latency* at the
paper's 361 tokens — the ratio worsens with N because the conv term is linear in cells while
attention is quadratic). For **RRTRRT** (4 R + 2 T) vs **6R**:
- 6R ≈ 6·124 = 744 M; RRTRRT ≈ 4·124 + 2·417 = **1330 M** → **~1.8× trunk FLOPs.**
- For a **10-block R3(RRT)** (7R + 3T) vs **10R**: 1240 M → **1.24·1240 ≈ 2.12 G**, **~1.7×**.

### 4.4 Translating to pos/s (grounded in measured numbers)
Measured dense_cnn facts (from `analysis/throughput_understanding.md`, `performance.py`,
`configs/dense_cnn_rl_main1.toml`):
- Live config: **64ch × 10 blocks, 512 visits, 256 active games.**
- Self-play throughput (96ch×6 baseline, torch FP16): **~38 search pos/s** (~35 full); with
  TensorRT FP16 + bucketing **~90 search pos/s**.
- **Evaluator callback = ~78% of search wall; ~90% of that is forward compute** → forward compute is
  **~70% of the search wall** (Amdahl). Real forward batch: mean ≈99, p50 70, p95 228, max 245;
  chunk cap `MCTS_EVAL_CHUNK_STATES = 1024`.

If the trunk forward gets **~1.8× more expensive** (RRTRRT) and forward is ~70% of the wall, the
end-to-end search wall grows by roughly `0.30 + 0.70·1.8 = 1.56×` → **throughput ≈ 38 / 1.56 ≈ 24
search pos/s** (torch FP16), best case. That is a **~35% throughput hit** for RRTRRT at C=64 *with
flash attention and full 1681 tokens*. R3(RRT) (10 blocks, 3 T) lands similarly (~1.7× trunk →
~1.5× wall → ~25 pos/s). These are optimistic: they assume SDPA is well-utilized at C=64/seq=1681
(a 64-dim, 4-head attention is narrow and may underutilize tensor cores, making the real hit worse).

**Caveat on TensorRT:** dense_cnn's big throughput lever is the TRT FP16 backend (2.3×). Attention/
SDPA export to TRT is far less turnkey than convs; the ResTNet trunk may **lose the TRT path** and
be stuck on the torch backend, which compounds the slowdown relative to the live conv-only run. Flag
this as a real secondary risk.

### 4.5 Mitigations (ranked) — pick one for the first cut
1. **Downsample the token grid for T-blocks (recommended).** Pool `(C,41,41)→(C,21,21)` (441 tokens)
   before attention, upsample after. Attention cost ∝ N²: 441² vs 1681² = **14.5× cheaper**. This
   brings the T-block back to ~25 M score-MACs/board — *cheaper than an R-block* — so RRTRRT becomes
   ~**1.0–1.1× trunk FLOPs** and the throughput hit nearly vanishes. It also keeps the paper's
   "attention over board positions" semantics (just at a coarser global resolution, which is exactly
   what "global knowledge" wants). 21×21 still covers the full crop; relative-position bias is cheap
   here (§3.4 option 2). **This is the single highest-leverage decision.**
2. **Flash/SDPA + full 1681, RRTRRT only.** Accept the ~35% throughput hit, keep 2 T-blocks, never
   materialize N². Most faithful to "exactly as the paper"; viable if you're willing to pay it.
3. **Windowed / local attention** (Swin-style 2D windows, e.g. 7×7 or hex-shaped) with periodic
   shifted windows for cross-window flow. Cost ∝ N·window² (linear in cells). More faithful to "full
   resolution" than downsampling but more code, and arguably less "global" than the paper intends.
4. **Sparse candidate-token attention (the hexgt approach).** Attend only over occupied stones +
   candidate cells (~250 tokens) rather than all 1681. This is *proven in-repo* (hexgt) and very
   cheap, but it is a **departure** from "attention over board cells as tokens" — it's a different
   architecture. Keep as a fallback if dense attention proves too slow even downsampled.

**Recommendation:** start with **(1) downsample2 (21×21)** for the T-blocks. It is the only option
that gives you the global-attention benefit *without* a throughput regression versus the live run,
and it degrades gracefully to (2) full-resolution by a single config flip (`token_grid="full"`) once
you've decided the quality is worth the speed.

### 4.5.1 A note on Hexo's sparsity (why downsampling is benign here)
Hexo's board is a *sparse infinite* board; the 41×41 is a **crop** centered on the action. Early and
mid-game, the vast majority of the 1681 cells are empty and far from any stone. Full per-cell
attention spends almost all of its N² budget on empty-empty pairs. A downsampled (or candidate)
token set loses very little signal because the informative cells are few. This is a structural reason
to expect downsampled attention to retain most of the paper's benefit on Hexo specifically.

---

## 5. Effort breakdown (S / M / L)

| Work item | Size | Notes / risk |
|---|---|---|
| Fork scaffolding (copy dir, rename ids, pyproject entry point, reuse `_rust.dense_cnn`) | **S–M** | Pure mechanical; hexgnn is the template. Bigger dir than hexgnn but no native work if reusing dense_cnn's submodule. |
| `TransformerBlock` module | **S** | ~40 lines, adapted from hexgt's `GraphTransformerLayer`. Must use SDPA. |
| R/T interleave from `block_pattern` config | **S** | Replace homogeneous `nn.Sequential` build with a per-char loop; add `R3(RRT)` expander + validation. |
| Positional encoding | **S** (abs) / **M** (relative on downsampled grid) | Abs-2D embedding first; relative-bias if matching paper. |
| Token-grid downsample/upsample adapters | **S–M** | AvgPool+interpolate (S) or learned strided conv (M). The throughput lever. |
| R↔T channel/token reshape | **S** | No projection (embed dim = C); pure flatten/reshape. |
| Config + plugin wiring (`parse_restnet_config`, `build_model`, dataclass keys) | **S** | Add 4–5 keys; follow fail-fast convention. |
| Pretrain/eval/selfplay wiring | **S** | Arch-agnostic; inherited from dense_cnn unchanged. New launch script + config + isolated `runs/` dir (copy `_rl_train.py` pattern). |
| Tests | **M** | New: T-block shape/equivariance smoke, pattern parser, forward-throughput smoke. **D6 augmentation interaction with attention needs a correctness test** (the trunk is no longer fully D6-equivariant; the dense_cnn pipeline relies on D6 *augmentation at training time*, which still works, but verify attention doesn't break any symmetry assumption baked into samples). |
| **Throughput validation + mitigation tuning** | **M–L** | The real work: measure pos/s, choose token-grid strategy, confirm no OOM, decide on TRT. |
| **Training to a comparable checkpoint** | **L (open-ended)** | Dwarfs the engineering. From-scratch interleaved net; the paper trains 100k–150k steps. Our self-play budget + the throughput hit make this the dominant cost. |

### Hard parts & risks (ranked)
1. **Attention cost / throughput at 1681 tokens** (§4) — *the* risk. Mitigated by token-grid
   downsampling; must be measured, not assumed.
2. **Loss of the TensorRT FP16 path** for the attention trunk → stuck on slower torch backend,
   compounding the slowdown vs the conv-only live run (§4.4 caveat).
3. **Training stability / convergence of a from-scratch interleaved net.** The paper reports no
   special init/warmup, but their LRs are large (0.02–0.1) and tuned per game; our pipeline's LR and
   our self-play data distribution differ. Pre-norm + LayerNorm (hexgt's choice) is the safe default;
   budget for some LR/warmup tuning.
4. **Narrow attention (C=64, 4 heads → head dim 16)** underutilizes GPU and may make SDPA's real
   speed worse than the FLOP estimate. Consider 8 heads or a wider channel for the T-blocks only.
5. **Positional-encoding faithfulness vs flash-attention** trade (§3.4) — relative bias fights
   no-materialization; downsampled grid resolves it.

---

## 6. Verdict & recommended first config

**Overall difficulty:** the *fork + architecture* is **Easy–Moderate (~1–2 focused weeks)** of
engineering. The *throughput engineering* is the Moderate part and the *training to a competitive
checkpoint* is the open-ended Large part that dominates calendar time. There is **no Rust work** if
you reuse `hexo_models._rust.dense_cnn` read-only.

**Central risk:** attention cost at the 41×41 / 1681-token crop. Everything else is mechanical. The
risk is real but **bounded and solvable** by downsampling the T-block token grid, which on Hexo's
sparse board costs little signal.

**Is it worth trying for Hexo?** Yes. The paper's **19×19 Hex** result (50.4%→58.0% from interleaving
T-blocks into a 10R trunk) is the most relevant external evidence available, and connection-game
global reasoning is exactly Hexo's weak spot for a pure conv trunk. The fork is additive and
non-disruptive, so the downside is bounded.

**Recommended first config (concrete):**
- **Fork as a standalone additive package** (`packages/dense_cnn_restnet/`, Layout A), **reusing
  `_rust.dense_cnn`** — zero native rebuild, zero risk to `dense_cnn_rl_main1`.
- **Trunk: `RRTRRT`** at the **current 64 channels** (matches the live run's width; 2 T-blocks is the
  paper's proven 6-block sweet spot and keeps the cost manageable). Provide `block_pattern` config so
  `R3(RRT)` (`"RRRTRRTRRT"`, 10 blocks) is one config flip away for a later, deeper run.
- **Attention:** `F.scaled_dot_product_attention`, **4 heads** (try 8 if head-dim-16 underutilizes),
  **MLP ratio 2**, **pre-norm LayerNorm**, ReLU — i.e. hexgt's `GraphTransformerLayer` self-attn half.
- **Token-grid strategy:** **`downsample2`** — AvgPool 41×41→21×21 (441 tokens) for the T-blocks,
  `F.interpolate` back to 41×41. ~14.5× cheaper than full attention; keeps the trunk roughly
  throughput-neutral vs the live conv-only run. Relative-position bias (Shaw 2018) on the 21×21 grid
  is cheap, so you can match the paper's positional encoding here rather than fall back to absolute.
  Keep `token_grid="full"` available to A/B the quality/speed trade once it trains.
- **Validate forward pos/s on a CPU/dev smoke and a short warm-GPU probe before any real launch**,
  into an isolated `runs/dense_cnn_restnet_*` dir, with `require_sealbot` eval wired exactly as
  dense_cnn does. Do not contend with the live run for GPU.

**One-line recommendation:** Fork is cheap and safe; build RRTRRT@64 with SDPA self-attention over a
2×-downsampled 21×21 token grid, prove the throughput on a smoke run, then decide between staying
downsampled (fast) and going full-1681 or R3(RRT)-deep (more faithful to the paper) based on measured
SealBot strength.

---

## Appendix A — Sources
- Paper: *Bridging Local and Global Knowledge via Transformer in Board Games*, Ju, Wu, Shih, Wu,
  IJCAI 2025. [arXiv:2410.05347](https://arxiv.org/abs/2410.05347) ·
  [HTML](https://arxiv.org/html/2410.05347) ·
  [project page](https://rlg.iis.sinica.edu.tw/papers/restnet/) ·
  [IJCAI proceedings](https://www.ijcai.org/proceedings/2025/828)
- In-repo grounding (read-only):
  `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/architecture.py` (trunk),
  `config.py`, `constants.py`, `plugin.py`, `inference.py`, `performance.py`
  (`MCTS_EVAL_CHUNK_STATES = 1024`);
  `packages/hexo_models/dense_cnn/rust/src/{lib.rs,mcts_eval.rs,encoding.rs}` (arch-agnostic Rust);
  `packages/hexo_models/hexgt/python/hexo_models/hexgt/architecture.py` (`GraphTransformerLayer`
  reference block);
  `packages/hexgnn/` + `packages/hexo_models/rust/src/lib.rs` (fork template);
  `configs/dense_cnn_rl_main1.toml` (live config: 64ch×10, 512 visits, 256 active);
  `analysis/throughput_understanding.md` (≈38→90 search pos/s; forward ≈70% of wall; real leaf batch
  mean≈99/p95 228); `CLAUDE.md` (≥128 searched pos/s calibration target for the 64ch/4block baseline).

## Appendix B — Implementation checklist (when/if green-lit)
1. `cp -r packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn packages/dense_cnn_restnet/python/dense_cnn_restnet` (+ rename namespace).
2. Rename identifiers (`Model1Network`→`RestnetNetwork`, `DenseCNNPlugin`→…, `parse_model1_config`→…, config dataclasses, `__all__`, plugin `name`, `model_family`).
3. Point `rust_bridge.py` at `getattr(_rust, "dense_cnn")` (reuse; **no lib.rs edit**).
4. Add `TransformerBlock` (SDPA) + pattern-driven trunk build + token-grid down/up adapters in `architecture.py`.
5. Add config keys `block_pattern`, `attention_heads`, `mlp_ratio`, `token_grid` to `RestnetArchitectureConfig` + parser validation.
6. `pyproject.toml` with `dense_cnn_restnet = "dense_cnn_restnet.plugin:get_plugin"`; `pip install -e`.
7. `configs/dense_cnn_restnet_main1.toml` (RRTRRT@64, downsample2, 512 visits) + `scripts/_rl_train_dense_cnn_restnet.py` (copy `_rl_train.py`, isolated `runs/` dir).
8. Tests: pattern parser, T-block forward shape, throughput smoke, D6/sample-pipeline correctness.
9. CPU smoke → short warm-GPU pos/s probe → decide token-grid strategy → real launch.
