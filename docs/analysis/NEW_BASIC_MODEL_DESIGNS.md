# New Basic Model Designs for Hexo

**Date:** 2026-06-05
**Status:** Fresh design exploration — clean-slate baselines

## Scope and method

This document collects **five independent, from-scratch model architectures** for
the game Hexo, each a deliberately *simple, basic baseline* rather than a tuned or
maximal system. It is a forward-looking design exercise, not a post-mortem.

How it was produced:

1. **Objective analysis (design-only).** Two read-only agents extracted purely
   factual design information (input representation, board handling, architecture,
   heads, MCTS interface, symmetry) directly from the source of the two existing
   model families in `packages/hexo_models/`. They were explicitly forbidden from
   reading or referencing any analysis docs, notes, handoffs, or any framing about
   the models' problems, failures, regressions, or performance. **All such material
   was deliberately excluded** so the designs below are uncontaminated by past
   narratives.
2. **Minimal brief.** Those analyses were compressed into a single
   architecture-agnostic design brief (below) — the *only* context the designers
   received. It names no existing model and prescribes no architecture.
3. **Five independent designs.** Five agents each designed one architecture family
   — GNN, CNN, Transformer, ResTNet, and a Hybrid — from the brief alone, with no
   access to the codebase. **Infinite/unbounded-board support was a mandatory,
   first-class requirement for every design.**

Each design is about **how the model works** — representation, architecture,
information flow — not code, files, or APIs.

---

## Shared design brief (the only context given to designers)

> **THE BOARD.** Hexo is played on a sparse, effectively unbounded hexagonal board.
> Each cell has six neighbors (hex adjacency). A position is a set of placed stones,
> each owned by one of two players, located at integer hex coordinates. There is no
> fixed bounding box: the playable region grows outward from existing stones as the
> game develops, so any model must treat the board size as variable and potentially
> large in every direction. Most of the infinite board is empty; only cells on or
> near existing stones are relevant.
>
> **AVAILABLE INPUT INFORMATION** (generic, not a fixed encoding): stone occupancy
> (mine / opponent / empty), side to move and game-phase / move-count context,
> recent-move / history signals, legality of moves, and tactical / threat-like
> signals (near-complete lines, contested regions, immediate threats).
>
> **THE I/O CONTRACT.** For each evaluated position the model must output (1) a
> policy — prior probabilities over exactly the legal moves of that position,
> aligned one-to-one with the legal move set — and (2) a value — a single scalar in
> [-1, 1] from the side-to-move's perspective. These are consumed by a PUCT-style
> MCTS that evaluates many positions per forward pass, in batches of varying size,
> with widely varying per-position legal-move counts.
>
> **TRAINING SIGNAL.** Self-play in the AlphaZero/KataGo style, using policy targets
> (search visit distributions) and value targets (game/search outcomes), with
> optional auxiliary targets. The hex board carries a D6 dihedral symmetry (6
> rotations × 2 reflections), usable as augmentation or built into the model.
>
> **HARD INVARIANTS.** Handle arbitrary-size, unbounded positions of any stone
> count; policy output must correspond exactly to the legal moves; value is a single
> scalar in [-1, 1] from the side-to-move's perspective; keep the design simple and
> small — a clean, minimal baseline.

---

## Comparison at a glance

| # | Family | Name | Core idea | Unbounded-board handling | ~Params |
|---|--------|------|-----------|--------------------------|---------|
| 1 | **GNN** | HexGraphNet (HGN) | Graph over stones + their empty 1-hop halo; 4 rounds of hex message passing | No grid at all; graph = live region only; translation-invariant by graph identity | ~120–150K |
| 2 | **CNN** | HexCrop-CNN | Residual hex-conv tower on a fixed window cropped & recentered on the active region | Crop-and-recenter ROI into fixed hex window; tile if oversize; translation-equivariant convs | ~0.3–0.5M |
| 3 | **Transformer** | HexAxial-T | One token per relevant cell; full attention with relative axial-offset bias | Sparse tokenization (no board materialized) + purely relative geometry → exact translation invariance | ~1M |
| 4 | **ResTNet** | HexoResTNet-Lite | Interleaved hex-conv (local) and self-attention (global) blocks, RRTRRT | Active set = stones + radius-r halo; local hex frame; conv on patch, attention on tokens | ~0.5–1.5M |
| 5 | **Hybrid** | HexLocal-Glance | Local hex-CNN feeds a sparse global self-attention pass over active cells | Relative-reindexed finite buffer for CNN; set attention over active cells; both size-agnostic | ~0.5–2M |

**Common threads across all five (convergent design choices):**

- **Sparsity is exploited, not fought.** Every design represents only the *live
  region* (stones plus a small empty halo / frontier), so cost scales with stone
  count, never with the infinite board's spatial extent.
- **Relative geometry, no origin.** None feed absolute coordinates; geometry enters
  via hex adjacency, relative axial offsets, or graph identity — giving translation
  invariance, the correct symmetry for an unbounded board.
- **Per-cell policy, gathered to legal moves.** Every design emits one logit per
  relevant cell and softmaxes over exactly the legal set, satisfying the
  one-to-one policy contract with no fixed-size action vocabulary.
- **Pooled scalar value with `tanh`**, side-to-move-relative (encoded by always
  presenting the mover as "me").
- **D6 symmetry via data augmentation** in every baseline (equivariant layers noted
  only as a future upgrade), keeping each model simple and small.
- **Segmented / masked batching** to evaluate many variable-size positions per
  PUCT forward pass.

The designs differ mainly in **how far information travels in one pass**: the GNN
and CNN are local (receptive field bounded by depth), while the Transformer,
ResTNet, and Hybrid add explicit global reasoning via attention — at an O(N²)
token cost that the sparse representation keeps affordable.

---

## 1. GNN — HexGraphNet (HGN)

> *A minimal hex message-passing baseline.*

**One-liner.** A small graph neural network that builds one node per relevant cell
(stones plus their empty neighborhood), connects them by hex adjacency, runs a
handful of message-passing rounds, and reads policy logits directly off the
empty-cell nodes plus a pooled value scalar — so the graph spans exactly the live
region of the board and nothing more.

### Input representation

The position is converted into a graph G = (V, E) whose extent is the position's own
live region, never a fixed grid.

**Node set V.** Start from the set of occupied cells. Add the one-hop empty halo:
every empty cell that is hex-adjacent to at least one stone. The union of (all
stones) ∪ (their empty neighbors) is the node set. Every legal move is, by
construction, a placement on an empty cell touching the live region, so every
legal-move target is guaranteed to be a node — this is what lets the policy align
one-to-one with legal moves. (If the rules ever allow a legal move farther from
existing stones, the halo radius is simply set to cover it; one hop is the natural
choice for a contact/line game.)

**Node features (per node, ~12–16 dims), all from the side-to-move's perspective:**
- occupancy one-hot: {mine, opponent, empty} (3 dims)
- is-legal-move flag (1 dim, set on empty nodes that are playable)
- recency: a scalar in [0,1] for how recently this cell was played, plus a 1-dim
  flag for "the single most recent move" (2 dims)
- local tactical scalars from the available threat-like signals: "part of a
  near-complete line for me", "...for opponent", "contested" (3 dims)
- degree / local density: occupied-neighbor counts for me and opponent, normalized
  (2 dims)
- a constant bias channel = 1 (1 dim)

**Edge features (~6–8 dims).** Each undirected hex edge carries a 6-dim one-hot of
which of the six axial directions it points (built so reflections/rotations permute
these channels, supporting D6); optionally a 1-dim "both endpoints same-owner" flag.
Direction-typed edges are what let messages distinguish lines from blobs.

**Global context.** Side-to-move is baked into per-node perspective; move-count /
game-phase scalars (normalized move number, coarse early/mid/late one-hot) feed the
value head.

### Board & infinite handling

Handled structurally, not by padding.

- **No grid, ever.** No H×W tensor and no bounding box — only the graph, which
  contains exactly stones plus their immediate empty halo. 8 stones → ~8 + ≤48 empty
  nodes; 800 stones → proportionally larger. Cost is O(|V| + |E|), with |E| ≤ 3|V|
  because hex degree is six. The infinite empty ocean is never represented.
- **Translation invariance for free.** Nodes are addressed by graph identity, not
  grid coordinates, so there is no learned position embedding tied to an origin.
  Coordinates are used only to determine adjacency and the 6-direction edge type.
  Sliding the position anywhere yields an isomorphic graph → identical output.
- **Growth is automatic.** New stones and halos simply appear as new nodes/edges;
  no shape or parameter depends on a maximum board size. The same weights process a
  20-node and a 5000-node graph identically.
- **Batching variable sizes.** Disjoint-union batching: concatenate all positions'
  nodes, tag each with a position id, run message passing over the block-diagonal
  adjacency (no cross-position edges), and segment all pooling/softmax by position
  id. No padding waste.

### Architecture & information flow

A deliberately plain message-passing stack — no attention, no fancy aggregators.

1. **Node encoder.** Shared 2-layer MLP → hidden embedding h⁰ of width d = 64. Edge
   features encoded once by a tiny linear layer to width d_e = 16.
2. **Message-passing rounds (T = 4),** each with its own weights:
   - *Message:* for edge u→v, m = MLP_msg([h_u, edge_feat(u→v)]) — the edge
     direction type is what lets a node detect lines vs. blobs.
   - *Aggregate:* each node sums incoming messages (degree ≤ 6 bounds this).
   - *Update:* h_v ← h_v + MLP_upd([h_v, aggregated]) with LayerNorm — residual +
     norm keeps 4 rounds stable.
   Four rounds give a receptive field of radius 4 hexes — enough to see
   near-complete lines and immediate threats, small enough to stay cheap and to
   generalize across regions (shared weights → a tactic learned anywhere transfers
   everywhere).
3. **Two readouts** consume the final embeddings h^T.

Total depth is shallow (2 + 4 + heads), width 64 — a handful of small MLPs.

### Policy & value heads

- **Policy (per-node, naturally aligned).** Take only nodes flagged as legal moves;
  a shared 2-layer MLP maps each to a single logit; segmented softmax *within each
  position* over its legal nodes → the prior. One legal node per legal move means
  one-to-one alignment with zero masking gymnastics, at any move count.
- **Value (graph-level).** Segmented mean + max pool over all node embeddings
  (2·d), concatenated with the small global context, → 2-layer MLP → `tanh` ∈
  [-1, 1], from the mover's perspective. Mean captures balance; max captures the
  single decisive feature (e.g. a winning threat).
- **D6 symmetry** via augmentation: the 12 transforms permute coordinates and the 6
  edge-direction / 6-neighbor-degree channels consistently; graph topology is
  invariant, so only directional channels need permuting. Optional later upgrade:
  test-time averaging over the 12 transforms.

### Why it fits Hexo

Hexo is intrinsically a graph problem: stones on a sparse infinite hex lattice where
only the neighborhood of existing stones matters. A GNN over exactly that
neighborhood is the most direct possible match — sparsity is the whole point,
unboundedness is structural (no origin, no max size, no padding), local tactics map
to 4-hop message passing with shared weights, the per-legal-node policy gives exact
alignment at any count, and D6 symmetry is cheap because it maps to permutations of
the six edge-direction channels.

### Simplicity & size

Width d = 64, edge width 16, T = 4 rounds, a 2-layer encoder and two 2-layer heads.
Rough budget: encoder ≈ 5K; each MP round ≈ 25K ×4 ≈ 100K; policy head ≈ 4K; value
head ≈ 9K. **Total ~120K–150K parameters** — tiny, fast to train, easy to debug. It
stays basic by choosing the plainest option at every fork (sum/mean instead of
attention, residual+LayerNorm instead of gating, one-hop halo, augmentation instead
of equivariance, single global pool for value).

### Tradeoffs & risks

- **Receptive field bounded by T.** 4 rounds ≈ 4 hexes; long-range relationships
  (ladders/connections spanning many cells) are invisible in one pass. Fix: more
  rounds, or longer-range "virtual" edges between same-owner stones (complicates the
  baseline).
- **Mean/max value pooling is coarse** — a tiny far-corner deciding threat may be
  underweighted.
- **Halo radius assumption** must match the real legality rules; widening it grows
  |V|.
- **No absolute spatial anchoring** — correct for a truly unbounded board, a
  mismatch if special regions existed.
- **Sum aggregation + variable degree** can drift; LayerNorm controls it, mean is
  the safe fallback.
- **Augmentation-only D6** gives approximate, not exact, symmetry.

### Information flow

```
Position (stones on infinite hex lattice, side-to-move)
        |
        v
Build live-region graph:  V = stones ∪ empty 1-hop halo
                          E = hex adjacency (6 dirs, direction-typed)
        |
        v
Node features [mine/opp/empty, legal?, recency, threats, degree]
Edge features [6-dir one-hot]      (all from mover's perspective; D6-augmented)
        |
        v
   +--------------------+
   |  Node encoder MLP  |  -> h0  (width 64)
   +--------------------+
        |
        v
  ====  x4  message-passing rounds  ====
   for each edge u->v:  m = MLP_msg([h_u, edge])
   aggregate at v:      a = sum_u m
   update:              h_v += MLP_upd([h_v, a]); LayerNorm
  =====================================
        |
        v   h^T  (one embedding per node)
        |
   +----+--------------------------+
   |                               |
   v (legal nodes only)            v (all nodes of position)
 MLP -> 1 logit/node          mean-pool ++ max-pool ++ context
   |                               |
 segmented softmax              MLP -> tanh
 within position                   |
   |                               v
   v                          VALUE  scalar in [-1,1]
 POLICY  (1 prior per legal move, exact 1:1)

(batches = disjoint union of many such graphs; all pools/softmax
 are segmented per position -> no padding, any size, any move count)
```

---

## 2. CNN — HexCrop-CNN

> *Centered-window hex-equivariant baseline.*

**One-liner.** A small residual CNN run on a fixed-size hexagonal window cropped and
recentered around the active region of the unbounded board, producing a per-cell
policy that is gathered to the legal-move set plus a single pooled value scalar.

### Input representation

Each position is encoded as a stack of feature planes over a fixed-size local hex
grid (the "window"), using the **axial hex coordinate system** (q, r) where each
cell's six neighbors are at the six axial offsets. Convolutions use a kernel shaped
to the hex neighborhood (a 7-tap kernel: center + 6 neighbors; optionally a 19-tap
2-ring kernel).

Feature planes (~10 channels), all from the side-to-move's perspective:
- Plane 0/1: current player's / opponent's stones (1/0).
- Plane 2: empty-and-legal cells (the playable set inside the window).
- Plane 3: recency — a single decayed scalar per cell (e.g. exp(-age/τ) over the
  last K stones), encoding move order without one plane per ply.
- Plane 4: the most-recent move (one-hot).
- Planes 5–6: cheap threat-like signals — per color, a normalized "near-complete
  line through this cell" derived from occupancy along hex line directions
  (optional).
- Plane 7: a constant in-window mask (1 inside the real cropped region, 0 in
  padding), so the network can tell true empties from off-window padding.
- Planes 8–9: two scalar broadcast planes — side-to-move parity and normalized
  move-count / game-phase.

Side-to-move is handled by *always* placing the mover in plane 0, so the network
never needs a "whose turn" branch and value is automatically mover-relative.

### Board & infinite handling

By **crop-and-recenter into a fixed window**, exploiting that only cells on or near
stones (and one ring of empties) are relevant.

1. Compute the **region of interest (ROI):** the bounding hex region containing all
   stones plus a margin of M empty rings (M ≈ 3–4) so every legal move and its
   local tactics are inside.
2. **Recenter** the ROI at the window origin by subtracting the ROI centroid (axial
   coords). Because features are relative occupancy, this translation is
   information-preserving; the network is translation-equivariant, so absolute
   coordinates never matter.
3. If the ROI fits the fixed window (a hex region of radius R, e.g. R ≈ 20), encode
   it directly and **zero-pad** the unused border, with plane 7 marking padding.
   Padded cells read as off-board/empty/illegal, so edge convolutions behave
   consistently and policy never proposes padded cells.
4. If the ROI is *larger* than the window (rare late-game sprawl), **tile:** slide
   overlapping crops over the active region, each as a batch item, each owning the
   legal moves whose cells fall in its interior. Model size stays fixed; cost grows
   linearly with active area, not with the infinite board.

Convolutions are translation-equivariant, so recentering is "free." The only edge
effect is at the padding boundary, made explicit by the mask; padding never leaks
into a legal move because policy is gathered only at legal-cell positions.

### Architecture & information flow

A minimal residual hex-CNN — a stripped-down AlphaZero tower on hex adjacency:

- **Stem:** one hex-conv (7-tap) from ~10 planes to C ≈ 64 channels, BN, ReLU.
- **Trunk:** B ≈ 4–6 residual blocks, each [hex-conv → BN → ReLU → hex-conv → BN] +
  skip + ReLU, all "same"-padded (mask prevents padding bleed), so resolution is
  preserved end-to-end — every cell keeps a one-to-one board correspondence, which
  is what lets the policy head emit a per-cell prior.

~4–6 blocks at 64 channels give a receptive field of ~9–13 hex rings — spanning
local tactics and medium-range line threats while staying small. Information
propagates purely locally (each layer mixes a cell with its 6 neighbors; stacking
grows the field linearly); the broadcast scalar planes inject the only global
context. Inductive bias: "Hexo is dominated by local hex structure."

### Policy & value heads

- **Policy (per-cell, then gather).** A hex-conv → BN → ReLU, then a 1-tap conv to
  **1 logit per cell** (a dense logit map). To get priors over exactly the legal
  moves, **gather** the logits at the cell coordinates of the position's legal moves,
  then softmax over only that gathered set. One-to-one alignment, zero probability
  on illegal/padded cells, no fixed-size action space; per-position varying move
  counts handled by per-position gather indices.
- **Value (pooled scalar).** A hex-conv to a few channels, BN, ReLU, then **masked
  global average pool** over real cells → small FC → ReLU → single unit → `tanh` ∈
  [-1, 1], mover-relative. Masked pooling keeps padding from diluting the value.
- **Optional auxiliary head:** a pooled score-margin or ownership-map predictor
  (KataGo-style), omittable for the minimal baseline.
- **D6 symmetry** via data augmentation (the 12 transforms of window/policy/value);
  optionally average over a few transforms at inference.

### Why it fits Hexo

Hex adjacency is native (the 7-tap kernel matches the six-neighbor topology
exactly); locality matches the game (near-complete lines, immediate threats are
local structures); per-cell policy is the right shape for a placement game and the
gather-to-legal step satisfies exact alignment without a global vocabulary; varying
legal-move counts are handled by per-position gather indices — exactly what a batched
PUCT MCTS needs.

### Simplicity & size

Stem + ~5 residual blocks at 64 channels. Per hex-conv ≈ 7 × 64 × 64 ≈ 29K; two per
block × 5 ≈ 290K; plus stem and heads → **~0.3–0.5M parameters.** A deliberately
small single-tower baseline: one plane encoder, one residual trunk, two thin heads.
No attention, recurrence, graph machinery, global tokens, or multi-scale pyramid —
just standard conv/BN/ReLU/residual plus a hex-shaped kernel and a gather-at-legal
policy. Fixed window → fixed shapes, predictable memory.

### Tradeoffs & risks

- **Window/ROI sizing** is the main risk: too-small margin clips a relevant cell;
  too-large wastes compute. Tiling handles oversize boards but adds seam bookkeeping
  (each legal move must belong to exactly one tile's interior).
- **Bounded receptive field** (~10 rings): very long-range relationships are not
  directly modeled; broadcast planes and value pooling only partially compensate.
- **Padding edge effects:** cells near the boundary see less context.
- **D6 as augmentation, not built-in** → not exactly equivariant.
- **Cheap threat planes are hand-derived** prior knowledge; droppable if they hurt.
- **Tiling cost grows with active area** (still linear in stones).

### Information flow

```
Unbounded sparse board (stones at integer hex coords)
        |
        v
[ROI = bbox(stones) + margin M empty rings]
        |  recenter at centroid (translation-equivariant)
        v
+---------------------------------------------------+
| Fixed hex window (radius R), ~10 feature planes   |
|  mover / opp / empty-legal / recency / last-move  |
|  threats / padding-mask / phase / parity          |
|  (if ROI > window: tile into overlapping crops)   |
+---------------------------------------------------+
        |
        v
   [Stem hex-conv 7-tap -> BN -> ReLU]   (C=64)
        |
        v
   [Residual block] x B (~5)             (hex-conv,BN,ReLU,skip)
        |  spatial resolution preserved
        |-------------------------+
        v                         v
 POLICY HEAD                 VALUE HEAD
 hexconv->BN->ReLU           hexconv->BN->ReLU
 1x1 conv -> 1 logit/cell    masked global avg pool
        |                    -> FC -> ReLU -> 1 unit
 gather at legal-move cells  -> tanh
        |                         |
 softmax over legal set           v
        |                    value in [-1, 1]
        v                    (mover's perspective)
 priors aligned 1:1 with legal moves
        |
        v
   PUCT MCTS (batched, varying legal-move counts)
```

---

## 3. Transformer — HexAxial-T

> *A stone-token hex transformer.*

**One-liner.** A minimal transformer that tokenizes the sparse hex board as one
token per relevant cell, encodes geometry with relative axial offsets in attention,
runs a few full-attention blocks for global reasoning, and reads out a per-cell
policy over legal moves plus a pooled scalar value.

### Input representation

**One token per relevant cell.** The board is never materialized. Tokens come from
the *region of interest* (ROI): all occupied cells plus every empty cell hex-adjacent
to at least one stone (the frontier / legal-move candidates). S stones → on the
order of S to a few×S tokens — never the whole plane.

Each token (a cell) carries a small feature vector embedded by a shared linear layer
into model dimension d (e.g. d = 128):
- **Occupancy (3-way, mover-relative):** my-stone / opponent-stone / empty.
- **Legality flag:** 1 if placing here is legal — tells the network candidate moves
  and is reused to mask the policy.
- **Recency scalar(s):** normalized "moves-ago" for the most recent k stones.
- **Local tactical bits (optional):** flags like "part of a near-complete line",
  "contested".

**Global context** (side to move, move count / game phase) enters two ways: folded
into the mover-relative occupancy convention, and as a single learned **global
[CLS]-style token** whose initial embedding is conditioned on phase/move-count via a
tiny MLP. A position = {global token} ∪ {one token per ROI cell}.

### Board & infinite handling

By **never materializing the board** and using **purely relative geometry**, so
absolute coordinates and board size are irrelevant.

1. **Sparse tokenization.** Only ROI cells become tokens; the transformer is
   set-based and length-agnostic, accepting any token count with no padding to a
   fixed grid and no max coordinate.
2. **Relative axial positional encoding.** Cells have axial coords (q, r), never fed
   absolutely. Geometry enters attention as a function of the *difference* (Δq, Δr)
   between a pair: a learned relative-position bias added to the attention logits,
   looked up in a table indexed by (Δq, Δr) clamped to radius R (e.g. ≤ 4 hexes),
   with offsets beyond R collapsing to a "far" bucket plus a smooth hex-distance
   scalar. Translating the whole position leaves all pairwise offsets unchanged →
   **exact translation invariance.**
3. **Distance-aware far interactions.** Distant stones still interact via full
   attention, with the relative bias degrading smoothly with hex distance.
4. **Output scales with the board.** Per-token policy readout → output size matches
   the legal-move count automatically; no fixed-size policy plane.

Arbitrary stone counts, arbitrary extent, and arbitrary translation are all handled
by construction.

### Architecture & information flow

A clean encoder-only transformer, deliberately small.

- **Size:** d = 128, h = 4 heads, L = 4 blocks, MLP expansion 4× (hidden 512).
- **Per block (pre-norm):** LayerNorm → multi-head self-attention with the relative
  axial bias added to attention logits → residual; LayerNorm → 2-layer MLP (GELU) →
  residual.
- **Full attention over all tokens.** With O(stones) tokens this is affordable and
  gives true global reasoning — any cell attends to any other in one hop, so a threat
  on one side can relate to a defensive resource on the other. The relative axial
  bias makes that global attention geometry-aware. (Degrades gracefully to
  local/windowed attention + global token if a position has very many tokens.)

Flow: embed cells → tokens and phase → global token; L blocks of full self-attention
with relative-offset bias (early blocks favor near cells via the bias, later blocks
integrate long-range structure through chained attention and the global token); final
representations feed two lightweight heads. The network is permutation-invariant over
the token set (geometry carried only by the relative bias), matching the unordered
nature of a stone set.

### Policy & value heads

- **Policy (per-cell, exactly the legal moves).** Every ROI token with legality flag
  = 1 is scored by a shared 2-layer MLP → one logit per legal cell; non-legal tokens
  excluded (masked to −∞). A single softmax over the legal-move tokens yields the
  prior — one-to-one with legal moves by construction, count adapts per position.
- **Value (single scalar, mover-relative).** The final embedding of the global
  [CLS]-style token (which has attended over the whole position) → small 2-layer MLP
  → single unit → `tanh` ∈ [-1, 1].
- **Batching variable sizes.** Pack positions with a block-diagonal attention mask
  so tokens attend only within their own position; each position's global token reads
  only its cells; per-position softmax gives independent policies.
- **D6 symmetry** via augmentation: apply one of the 12 hex symmetries to the (q, r)
  offsets during training — cheap, since geometry is relative.

### Why it fits Hexo

Sparse + unbounded → tokenizing only the ROI means cost scales with the actual game,
no max size anywhere. No natural origin → relative axial bias gives exact translation
invariance (a cropped-grid CNN must instead pick an origin and crop). Hex adjacency
is first-class via the axial-offset table. Variable legal-move counts ↔ variable
policy size for free. Full attention lets a defensive resource attend to a far threat
in one hop — important for a line-forming game. D6 augmentation is a trivial offset
transform.

### Simplicity & size

d = 128, L = 4, h = 4, MLP 4× → ~0.8–1.2M params in the trunk, plus tiny embedder /
heads and a small relative-bias table → **~1M parameters total.** Stays a baseline:
one token type, one standard pre-norm encoder, two small heads; the only
Hexo-specific piece is the relative axial bias table; full attention removes
windowing/neighbor-graph/sparse-kernel logic; symmetry by augmentation. Scale later
by raising only d and L.

### Tradeoffs & risks

- **Full attention is O(N²)** in token count — fine for hundreds of tokens, but a
  very large endgame is quadratic. Mitigation (out of baseline): hex-distance
  windowed attention + global token.
- **ROI definition matters** — assumes useful moves live near stones; widen the
  frontier if not. The legality flag must be authoritative.
- **Relative-bias clamp radius R** trades table size against fine long-range
  distinctions.
- **Translation invariance vs. absolute cues** — intentionally no absolute position.
- **Augmentation, not built-in equivariance** — approximate symmetry, mild early
  inconsistency.
- **Set/permutation invariance** conveys move ordering only through the recency
  feature.

### Information flow

```
Position on infinite hex board (only stones + frontier matter)
        │
        ▼
  Build ROI tokens (one per relevant cell)            + 1 global token
  features: occ(me/opp/empty), legal?, recency, tactics    (phase, move#)
        │                                                      │
        ▼                                                      ▼
  Linear token embedder ──► token vectors (d)        MLP ──► global token (d)
        │                                                      │
        └───────────────► [ T0 , T1 , ... , Tn , G ] ◄────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 │   L x Transformer block            │
                 │   LN → MHSA (+ relative axial       │
                 │        bias from (Δq,Δr), full attn)│
                 │   → residual                        │
                 │   LN → MLP(GELU) → residual         │
                 └─────────────────┬─────────────────┘
                                   │
        ┌──────────────────────────┴──────────────────────────┐
        ▼                                                       ▼
  legal tokens only                                      global token G
  shared MLP → 1 logit each                              MLP → tanh
  softmax over legal set                                       │
        │                                                       ▼
        ▼                                                  value ∈ [-1,1]
  policy: prior over exactly the legal moves           (side-to-move)
```

---

## 4. ResTNet — HexoResTNet-Lite

> *Interleaved ResNet + Transformer (after the
> [ResTNet paper](https://rlg.iis.sinica.edu.tw/papers/restnet)).*

**One-liner.** A small ResTNet baseline that crops a per-position window of active
hex cells, runs an interleaved stack of hex-residual-conv blocks (local) and
self-attention blocks (global) over the live cell tokens, and reads out a per-cell
policy aligned to legal moves plus a pooled scalar value.

### Input representation

A position is a small set of feature planes over a **finite window of cells**; each
in-window cell is a token with a feature vector; empty/off-window cells are not
represented. Per cell, ~10–12 channels:

- **Occupancy (3 binary):** current-player stone, opponent stone, empty — always
  mover-relative (swap stone planes by side to move), making value sign and policy
  perspective-correct.
- **Legality (1 binary):** the channel the policy head reads from, guaranteeing
  one-to-one alignment.
- **Recency (2 scalars):** decaying "recently played at/adjacent" for each side, plus
  a folded-in "last move" bit.
- **Tactical hints (2–3 scalars):** cheap engine signals ("completes/extends a
  near-line for me", "...for opponent", contested) — optional but informative.
- **Frontier/empty-but-relevant (1 binary):** empty cells adjacent to any stone.

Global scalars (side-to-move, move count / phase, stone count, window size) are
broadcast as extra constant channels on every token. No fixed grid index is fed; a
cell knows only its features and relative geometry — keeping the representation
translation-invariant.

### Board & infinite handling

The infinite board is never materialized.

1. **ROI = stones plus a fixed-radius halo.** Collect all occupied cells, add every
   empty cell within hex-distance r (e.g. 2–3) of any stone. This active set is
   exactly the legally relevant region (all legal moves are in the frontier halo, all
   tactics near stones). Size ∝ stone count, not spatial extent.
2. **Local hex coordinates.** Translate the active set into a local frame (subtract
   the centroid in axial coords); only *relative* offsets are used, so a position
   that drifts a million cells produces identical input. Unbounded for free — no max
   coordinate, no padding-to-fixed-grid.
3. **Two compatible layouts feed the two block types.** Conv blocks operate on a
   small dense axial patch tightly bounding the active set (off-active cells masked
   to zero); attention blocks operate on the token list of active cells only (tens to
   a few hundred), so attention is O(active²) over a small set.
4. **Variable batch sizes:** conv patches padded to batch-max with a validity mask;
   attention uses a key-padding mask; both heads read only real cells.
5. **D6 symmetry** via data augmentation — randomly map each position and its policy
   target through one of the 12 D6 elements on the local hex coordinates.

### Architecture & information flow

A single shared width C (e.g. 64). A linear stem embeds each cell's features to C.

**Block stack — "RRTRRT" (the paper's small-board optimum), kept to 2 repeats:**

```
stem -> R R T R R T -> heads
```

- **R = hex residual conv block.** Two 6-neighbor hex convolutions (3×3 axial,
  masked to the hex kernel) with norm + activation and a residual skip, on the dense
  axial patch — captures local stone shapes, lines, threats.
- **T = transformer block.** Standard pre-norm self-attention + small MLP + residual,
  on the active-cell token list — captures long-range reasoning (distant ladders,
  global line races, whole-board balance) beyond conv's small receptive field.

**Local↔global hand-off.** Between an R group and a T block, gather the conv patch's
active cells into the token list (scatter/gather by local coordinate); after the T
block, scatter tokens back into patch positions. Both views share the same C-dim
per-cell state, so information flows freely both ways at every transition.

**Relative hex positional bias for attention** — a learned bias depending only on the
relative hex offset bucket (by distance and direction, with one shared "far" bucket).
Translation-invariant and D6-friendly; this is what lets the transformer reason about
geometry on the infinite board.

Total depth is shallow (6 blocks), width small — scale later by adding RRT repeats.

### Policy & value heads

- **Policy (per-cell, legality-masked).** A tiny 2-layer MLP → one logit per cell;
  keep only logits of legal-flagged cells; softmax over exactly that set → priors
  one-to-one with legal moves, no fixed-size vector. Illegal/padded cells get −inf.
- **Value (pooled scalar).** Masked mean-pool (and max-pool, concatenated) over
  active-cell representations → small MLP → `tanh` ∈ [-1, 1], mover-relative. Pooling
  over real cells only makes value invariant to active-set size and padding.
- **Optional auxiliary head (off by default):** ownership/score-margin from the same
  pooled feature.

### Why it fits Hexo

Hexo is line/threat oriented but unbounded. Local stone shapes and immediate threats
are what residual hex convolutions detect cheaply; whole-board races, distant
blocking, and long near-complete lines are what self-attention's global field
handles. ResTNet's interleaving gives both in one small network — and the paper shows
this specifically helps long-sequence/ladder-like patterns pure-conv AlphaZero nets
miss, which is precisely Hexo's lines. Sparsity matches token attention (only active
cells become tokens, so attention stays cheap even as stones spread). Translation
invariance is mandatory and free here via relative-offset conv + relative-hex-bias
attention. The paper's small-board winner RRTRRT is a good fit because Hexo
positions, viewed as active sets, are effectively small/medium boards.

### Simplicity & size

C = 64, depth 6 (RRTRRT) — 4 conv + 2 attention blocks, attention with 4 heads and
~2× MLP. **~0.5–1.5M parameters** — a tiny AlphaZero-scale net. One shared feature
width across both block types and layouts, so a single hyperparameter governs
capacity; scale up by repeating RRT or raising C. No equivariant layers, no
multi-resolution pyramid, no recurrent history, no learned region proposal — symmetry
is augmentation, history is a 2-scalar recency feature, ROI is a fixed-radius rule.
Per-position cost scales with stone count, not board extent.

### Tradeoffs & risks

- **Fixed halo radius can clip relevant cells** — a tactically important empty cell
  just beyond r is dropped. Pick r from Hexo's actual line/threat reach (r = 2–3);
  validate against the engine's legal-move and threat outputs (main correctness
  risk).
- **Attention is O(active²)** — late-game many-stone positions cost more; fine for a
  baseline (hundreds of tokens), may need sparsification later.
- **Augmentation-only symmetry** → small D6 inconsistencies until well trained, a
  little wasted capacity.
- **Relative-bias buckets and recency τ** are extra, un-range-validated knobs.
- **Variable active-set padding wastes some compute;** bucket by size when batching.
- **Tactical-hint features couple the model to the engine;** droppable for a purer
  baseline at some strength cost.

### Information flow

```
Infinite sparse hex board (stones at integer hex coords, no bounds)
        |
        |  collect occupied cells + empty cells within hex-radius r  (the "active set")
        |  translate to LOCAL hex frame (centroid-relative; relative offsets only)
        v
  +-----------------------------+
  |  ACTIVE-SET FEATURES        |  ~10-12 ch/cell: occ(me/opp/empty),
  |  (size ~ #stones, sparse)   |  legality, recency, tactical hints, frontier,
  +-----------------------------+  + broadcast globals (side, move#, phase)
        |
        v   linear stem -> C=64 per cell
   [ two synchronized views of the SAME per-cell state ]
        |
   patch view (dense axial, masked)        token view (active cells only)
        |                                          |
        v                                          v
  R --- R --(gather)-->                       T (self-attn + MLP,
  hex conv x2, residual                       relative-hex-offset bias,
        ^                                       key-padding mask)
        |<--------------------(scatter)-----------|
        v
  R --- R --(gather)--> T  ... (RRTRRT, 6 blocks total)
        |
        v   per active cell: C-dim vector
   +----------------------+         +---------------------------+
   | POLICY head (per cell)|        | VALUE head                |
   | MLP->1 logit/cell     |        | masked mean+max pool ->    |
   | keep LEGAL cells only  |        | MLP -> tanh               |
   | softmax over legal set |        +---------------------------+
   v                                  v
 priors aligned 1:1 to legal moves   scalar value in [-1,1] (side-to-move)
        \________________  ________________/
                         \/
              consumed by PUCT MCTS (batched, varying sizes)
```

---

## 5. Hybrid — HexLocal-Glance

> *Local hex-CNN feeding a sparse global self-attention pass (CNN + Transformer).*

**One-liner.** A minimal hybrid that runs a hex-aware CNN over a per-position local
crop centered on the active region, then a tiny global self-attention pass across
only the occupied/frontier cells, so local shape and long-range threats both inform
per-move policy and a single value.

### Input representation

A position is a sparse set of **active cells**: every occupied cell, plus every empty
cell within hex-distance R (small fixed radius, e.g. R = 2) of any stone. Empty cells
beyond R of all stones are ignored — provably irrelevant, since legal moves and
tactics occur on or adjacent to existing structure. The legal-move set is a subset of
the active cells.

Each active cell carries a small per-cell feature vector (binary or normalized
scalars):
- Occupancy: 3 flags (current player's stone / opponent's stone / empty).
- Legality: 1 flag (is this a legal move).
- Recency: 1–2 scalars (decayed "how recently a stone was placed here", plus a most-
  recent-move flag).
- Tactical hints: 2–4 scalars (near-complete lines through the cell for me / for
  opponent).
- Global broadcast features appended identically to every cell: side-to-move flag,
  normalized move-count / game-phase.

A position = (list of active cells with integer hex coords, per-cell feature vectors,
and which cells are legal moves). No fixed bounding box; the representation scales
with stone count, not board extent.

### Board & infinite handling

The whole board is never materialized — two complementary tricks, one per mechanism:

1. **CNN half:** no global fixed window. Translate coordinates to be relative — take
   the centroid (or bounding box) of the active cells and re-index them into a local,
   dynamically-sized buffer just large enough for the active region plus a 1-cell
   halo. The CNN is fully convolutional and the active region is O(stone count), so
   the buffer is always finite and translation-invariant. Hex adjacency uses the
   axial trick: store cells in a 2D array by axial (q, r) and use a 6-neighbor
   convolution kernel (a masked 3×3 with the two non-hex corners zeroed). Cells
   outside the active set are masked to zero and excluded from outputs.
2. **Attention half:** no grid at all — it operates directly on the variable-length
   list of active cells (tokens), so arbitrary size is native (permutation-
   equivariant, length-agnostic). Relative hex offsets between cells are injected as a
   relative positional bias so long-range geometry is preserved without absolute
   coordinates.

Batching different active-set sizes: pad to the per-batch max with a mask; masked
cells contribute nothing to convolutions, attention, or outputs. The same model
evaluates a 5-stone opening and a 300-stone midgame unchanged.

### Architecture & information flow

Two stacked stages, deliberately shallow.

- **Stage A — Local hex-CNN (local feature extraction).** Per-cell features laid into
  the axial buffer, passed through ~3–4 hex-masked conv + ReLU + residual blocks (64
  channels). Each block mixes across the 6 hex neighbors, so after k blocks every
  cell sees its k-ring neighborhood — the cheap, weight-shared, translation-invariant
  extractor of local shape (lines forming, eyes, contested clusters, immediate
  threats). Output: a 64-dim embedding per active cell.
- **Stage B — Sparse global attention (long-range reasoning).** The CNN embeddings
  are gathered into the variable-length token list (occupied + frontier cells only —
  far fewer than a dense board). One or two standard multi-head self-attention + MLP
  blocks, with a relative-hex positional bias, let any cell attend to any other —
  where distant interactions resolve (a threat on one side influencing a defensive
  move's priority on the other). Restricting attention to the active set is what makes
  global reasoning affordable on an unbounded board.

**Division of labor / connection.** CNN = dense local pattern recognition with strong
spatial bias and weight sharing; attention = sparse, content-based long-range mixing
with no distance limit. They connect by a simple gather (grid cells → tokens); a
residual add keeps Stage-A local features available to the heads after global mixing.
~4 CNN blocks + ~2 attention blocks at 64 channels.

### Policy & value heads

Both heads read the final per-cell embeddings (now locally and globally informed).

- **Policy:** a shared small MLP → one logit per active cell; select only logits of
  legal-flagged cells and softmax over exactly that subset → policy over precisely
  the legal move set, one-to-one, no fixed-size vocabulary. Illegal/empty/padded
  cells masked before softmax.
- **Value:** mean-pool (over real, non-masked active cells) → one position vector,
  concatenate global broadcast features (side-to-move, move-count) → 2-layer MLP →
  `tanh` ∈ [-1, 1], mover-relative.
- **Optional auxiliary head (off by default):** per-cell ownership/territory from the
  same trunk.

### Why it fits Hexo

Hexo is geometric and local at the stone level (lines, eyes, six-neighbor shape) but
strategic and long-range at the position level (threats anywhere can matter). A pure
CNN handles locality but reaches distant cells only through many layers; a pure
transformer handles range but is weak on cheap local shape and expensive over many
tokens. Splitting the work — CNN for local hex shape, attention for sparse long-range
interaction over only the active set — captures both with a small model. Both halves
are size-agnostic (fully convolutional + set attention), honoring the unbounded-board
invariant end to end. The per-legal-cell policy matches the variable-move I/O
contract exactly; masked padded batching suits PUCT's many-position forward passes.
D6 symmetry is simple data augmentation (the relative-coordinate representation makes
transforms trivial).

### Simplicity & size

6–12 input channels per cell; 64-channel trunk; ~4 hex-CNN residual blocks (~6 conv
layers) + ~2 attention blocks (~4 heads); two tiny MLP heads. **~0.5–2M parameters.**
Stays basic: one trunk, two well-understood mechanisms stacked once, no progressive
widening, no multi-resolution pyramids, no learned symmetry layers, no exotic graph
machinery, a single optional auxiliary head left off. Reuses standard building blocks
(masked conv, standard self-attention, masked softmax, mean-pool + tanh).

### Tradeoffs & risks

- **Active-region radius R** is a hyperparameter: too small clips relevant frontier
  moves (and R must be ≥ large enough to include every legal move — bounded below by
  the rules); too large wastes compute. Mitigation: define the active set from
  legality + a small halo so legal moves are guaranteed included.
- **Attention is O(N²)** in active-cell count; dense late-game positions make
  attention the bottleneck. Mitigation: frontier-only token set; later restrict to
  stones + top-k threat cells.
- **The CNN's axial buffer can become sparse/large** if stones spread thin (masked
  waste); bounded by the active bounding box, usually compact.
- **Rotation/reflection invariance relies on augmentation,** not architecture — early
  training may be less sample-efficient on symmetry.
- **Gather between grid and token list** adds indexing complexity; masked variable-
  size batching is fiddlier than fixed-grid models.
- **Mean-pool value is crude** — may dilute a decisive local feature in large
  positions; an attention-pool could replace it later.

### Information flow

```
Position (sparse, unbounded)
  = occupied stones + empty cells within radius R of any stone
        |
        v
Per-cell features [occ(3), legal, recency, tactics, +global broadcast]
        |
   relative (q,r) axial re-indexing  (translation-invariant, finite buffer)
        |
        v
 +-------------------------------+
 | STAGE A: hex-masked CNN       |  local shape
 |  ~4 residual blocks, 64ch     |  (6-neighbor kernel)
 +-------------------------------+
        | per-cell embeddings
   gather grid-cells -> token list (active set only)
        v
 +-------------------------------+
 | STAGE B: self-attention       |  long-range threats
 |  ~2 blocks + relative-hex bias |  (set-based, size-agnostic)
 +-------------------------------+
        | globally-informed per-cell embeddings
        +-----------------------------+
        |                             |
        v                             v
  POLICY head                    VALUE head
  MLP -> 1 logit/cell            mean-pool active cells
  mask to legal cells            + global feats -> MLP
  softmax over legal set         -> tanh
        |                             |
        v                             v
  priors over exactly           scalar in [-1,1]
  this position's legal moves    (side-to-move POV)
```

---

## Closing notes

All five designs independently converged on the same core answer to the unbounded
board — **represent only the live region with relative geometry** — and on the same
answer to the variable I/O contract — **per-cell logits gathered/softmaxed over the
legal set, plus a pooled `tanh` value.** They diverge on the reasoning range packed
into a single forward pass:

- **Local-only baselines** (GNN, CNN) are the smallest and simplest; their receptive
  field is bounded by depth, so very long-range strategy needs more layers or extra
  edges/tiles.
- **Global-reasoning baselines** (Transformer, ResTNet, Hybrid) add attention for
  one-hop long-range interaction, at an O(N²) token cost the sparse representation
  keeps tractable.

Each is intended as a *clean starting point* to implement and train from scratch,
with an obvious scaling path (more rounds/blocks/width, or built-in D6 equivariance)
once a baseline works.
