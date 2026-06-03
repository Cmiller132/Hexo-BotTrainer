# hexgt Value-Head / Trunk Redesign — Deep Design Investigation

Design / analysis only. **No code, config, or model files are changed by this
document, and nothing here runs training or touches any live/stopped run.**

This goes deep on **one question the owner raised**: the Phase-2 value head reads
`[SIDE hub | mean-pool | max-pool]` over the node embeddings, and the owner finds
mean+max *"insufficient — a fixed, lossy, non-learnable aggregation."* The goal is a
value head that genuinely **reasons** over both the GNN's local/typed features and
the transformer's whole-board attention, the way dense_cnn "reasons with global
pooling." This doc evaluates better readouts (A–D), a trunk-fusion option (B/E),
directly answers the owner's three questions, and gives a ranked recommendation that
**composes with** the soft-Z / threat-feature / TSS / ownership-head work already
specified in the two companion docs:

- [`HEXGT_ARCH_DESIGN_EXPLORATION.md`](HEXGT_ARCH_DESIGN_EXPLORATION.md) — the
  value-readout-bottleneck finding (Gap B/D), the ranked recs (soft-Z 0a, ownership
  0b, global pooled readout Rank 1, graded threat features Rank 2, KL weighting §3),
  hot tokens (§5), and the §6 prior art (KataGo global pooling, ResTNet).
- [`HEXGT_TSS_AND_SOFT_VALUE_DESIGN.md`](HEXGT_TSS_AND_SOFT_VALUE_DESIGN.md) — TSS on
  the engine-confirmed `count ≥ 4` forcing model and the soft-Z target detail.

Every hexgt-side claim is cited to a file:line in the hexgt worktree
`E:\Hexo-BotTrainer-hexgt` (READ-ONLY) or the dense_cnn main tree
`E:\Hexo-BotTrainer`. External claims are cited to papers, with fetched-vs-reasoned
flagged at the end.

---

## 0. The single most important finding: the readout the owner is questioning is ALREADY SHIPPED, and it is mean+max

The companion docs describe the global-pooled readout as a **proposal** ("Rank 1").
It is no longer a proposal — it is **live code**. The value head's input is exactly
`[SIDE | mean | max]`:

```
# architecture.py:42-43
VALUE_READOUT_MULT = 3   # Value readout = [SIDE hub | mean-pool | max-pool], each token_dim wide.

# architecture.py:236-240   value_head first Linear reads 3*token_dim
self.value_head = nn.Sequential(
    nn.Linear(VALUE_READOUT_MULT * self.token_dim, self.token_dim),
    nn.ReLU(inplace=True),
    nn.Linear(self.token_dim, VALUE_BINS),
)

# architecture.py:306-311   the concat that feeds it
def _value_readout(self, batch, node_emb):
    side = self._graph_readout(batch, node_emb)            # SIDE hub row
    mean_pool, max_pool = self._global_pool(batch, node_emb)
    return torch.cat([side, mean_pool, max_pool], dim=-1)  # (G, 3*D)
```

`_global_pool` (`architecture.py:284-304`) is segment **mean** (`index_add_` ÷
counts) and segment **max** (`index_reduce_(..., "amax")`) over **all** node
embeddings per graph. The graft that put it there is fully built and tested:
`expand_value_readout_columns` (`architecture.py:387-428`) widens a pre-pool
checkpoint's `value_head.0.weight` from `(D, D)` to `(D, 3D)`, copying the old SIDE
weight into the leading block and **zeroing the mean/max blocks** so the first step
after resume is byte-identical (test
`test_hexgt_value_readout.py::test_expand_value_readout_gives_identical_output`,
lines 142-179). The RL resume path calls it (`scripts/_rl_train.py:262, 281`), and
permutation-invariance (⇒ D6) of the pool is unit-tested
(`test_hexgt_value_readout.py:118-137`).

**So the owner's instinct is being voiced against a head that is currently
mean+max, and the live RL run is already absorbing it via zero-init graft.** The
right framing of the question is therefore *sharper* than "should we add pooling":
it is **"is fixed mean+max the right pooling, or do we need a learnable, queryable
readout (and/or a trunk that conditions on global context the way KataGo does)?"**
That is exactly what §1–§4 below answer.

Two important code corrections to the prior docs follow from this:

- **CORRECTION 1 (status).** Companion Rank 1 is described as a partial-checkpoint
  *proposal*. It is **implemented and live**; any new readout work is now an
  *evolution of* the shipped `_value_readout`, not a greenfield add. The validation
  framing "value should stop pinning at +0.8 once pooling is wired" must account for
  the run **already having the pool**.
- **CORRECTION 2 (the dense_cnn analogy is imperfect).** The companion docs say the
  CNN value head "pools over the entire board." It does **not** pool — `ValueBinnedHead`
  is `Conv(C→1)` then `Linear(BOARD_AREA=1681 → 64 → 65)`
  (`dense_cnn/architecture.py:143-159`). That `Linear` is a **learnable, position-
  specific weighted readout of every cell** — far more expressive than a symmetric
  mean+max. dense_cnn does **not** use mean/max/std pooling in its value head at all
  (it can afford a position-keyed dense layer because the board is a fixed 41×41
  raster). This sharpens the owner's point: hexgt replaced a *learnable per-position
  readout* (which the CNN has, by virtue of a fixed grid) with a *fixed symmetric
  pool* (which is all a permutation-invariant graph can do **without an attention
  readout**). The honest analog of "reason like dense_cnn with global pooling" on a
  permutation-invariant graph is **not** mean+max — it is a **learnable attention
  pool (PMA)** and/or **KataGo-style global-context fusion in the trunk**.

---

## 1. What hexgt's value path actually is today (read from code)

**Trunk.** Shared input projection `node_in: Linear(32→168) → ReLU → Linear(168→168)`
(`architecture.py:212-216`); then `gnn_layers = 3` relational message-passing layers
(`architecture.py:217-219`, `constants.py:175 DEFAULT_GNN_LAYERS = 3`); then
`ctx_layers = 3` transformer layers (`architecture.py:220-222`,
`constants.py:176 DEFAULT_CTX_LAYERS = 3`), `attention_heads = 4`, `token_dim = 168`,
`ffn_dim = 336` (`constants.py:174-178`). The live config matches
(`configs/hexgt_model2.toml:22-26`). **So the prior docs' "3 GNN + 3 transformer" is
correct, confirmed.**

**Message passing** (`RelationalMessagePassing`, `architecture.py:58-100`): typed
`m_{j→i} = relu(W_type · h_j + edge_proj(attr))`, aggregated by **mean over incoming
edges** (`agg = index_add_(...) / counts`, lines 97-99), residual + LayerNorm
(line 100). Edge types: ADJACENCY, STONE_WINDOW, CANDIDATE_WINDOW, RECENCY, and
**CONTEXT** = SIDE-hub ↔ every node (`constants.py:29-34`).

**Transformer** (`GraphTransformerLayer`, `architecture.py:147-183`): per-graph
**context self-attention** over `{side, stone, window}` tokens (lines 170-174), then
**candidate→context cross-attention** (candidates query context, lines 177-182).
**Confirmed asymmetry:** context tokens (incl. SIDE) update **only** via context
self-attention; they never cross-attend to candidates. The SIDE hub can attend to
window/stone tokens, but a candidate's value-relevant content reaches SIDE only by
(i) the GNN CONTEXT edge (a *diluted mean*, line 99) or (ii) being a window/stone
token in the context set.

**Heads** (`_heads`, `architecture.py:313-327`):
- `policy` / `opp_policy`: per-candidate `Linear(168→1)` on `cand_emb` (lines
  314-318, 324) — a short, dedicated, per-move path. **This is why offense works.**
- `value`: `value_head` over `_value_readout = [SIDE | mean | max]` (lines 316, 319).
- `stvalue_<h>`: still read the **SIDE-hub only** readout (`graph_emb`, lines
  322-326) — the STV heads did **not** get the pool. (Minor: they still bottleneck
  through one token; relevant in §6.)

**Diagnosis recap (from the companion docs, re-grounded).** The measured failure is
**defensive value miscalibration**: value ≈ +0.8 in the plies before losing 8/8 to
SealBot, and a same-board optimism sum `v(A)+v(B) ≈ +0.82`. Two structural facts in
the code make this *specifically defensive*: (a) offense has a dedicated per-candidate
path (policy head), defense (a global "I am losing") has only the long
opp-stone→window→SIDE / pool path; (b) the GNN's CONTEXT aggregation is a **mean**
(line 99) that dilutes a single decisive opponent window into the average.

---

## 2. Is mean+max a REAL limitation? — the math, honestly

This is the owner's question 1. I argue from representational capacity, not vibes.

### 2.1 What a fixed symmetric pool *can* represent

Let the post-trunk node embeddings of a graph be a set `{h_1, …, h_N}`, each in
`R^168`. The readout sees `[h_side, mean_i h_i, max_i h_i]`.

- **mean** is the sufficient statistic for "average board character" — material
  balance, overall tempo. Good for *quiet* positions.
- **max** (per channel) is an **OR / "does there exist a node with a large value on
  channel c"** detector. This is genuinely powerful for one class of defensive
  signal: **"is there ANY opponent ≥4 window?"** If some channel `c` of a WINDOW
  node's embedding encodes "I am a live opponent ≥4 threat," then `max_i h_i[c]`
  fires iff such a window exists anywhere on the board — a translation-free,
  count-free existence detector. So the companion-doc claim that max-pool "surfaces
  the single most dangerous window structurally" is **correct for the existence
  question**, and it is a real reason the shipped pool should already help over
  SIDE-only. (Per-channel max also loses *which* node fired and cannot combine two
  channels from *different* nodes — see 2.2.)

So: **mean+max is not nothing.** For "am I materially ahead" (mean) and "does a lone
catastrophic threat exist" (per-channel max), it is adequate, and combined with
soft-Z (recalibrated label) + the v3 `F_CAND_OPP_THREAT` feature
(`constants.py:132-135`, the count-4-inclusive must-answer flag) + the TSS leaf
override, a lot of the +0.8-before-loss error may already be addressed. **Honest
caveat: the shipped pool + v3 features + soft-Z may move the metric enough that a
fancier readout is not the highest-value next step.** This must be measured before
investing in A–D (see §7 validation — ablate readout last).

### 2.2 What a fixed symmetric pool *cannot* represent — and why defense needs it

Here is the precise capacity gap, and it is real:

1. **Per-channel max cannot do cross-node, cross-channel reasoning.** Defense in
   Connect6/Hexo is rarely "does a threat exist" — it is **"weigh THIS opponent
   threat against MY counter-threat / MY ability to block it."** That is a function of
   *two different nodes' features jointly* (opp window A's urgency **and** my
   tempo/own-win-this-turn). `max_i h_i[c]` takes the max **independently per
   channel** — the value at channel `c1` (opp threat) and channel `c2` (my counter)
   can come from **different nodes**, and the readout has **no way to know they came
   from different nodes** or to compute a node-wise interaction like
   `Σ_i f(opp_threat_i, my_answer_i)`. A symmetric pool destroys the *binding*
   between channels within a node. The CNN dodges this because its value `Linear`
   reads all 1681 cells with **distinct weights**, so it can learn
   "weight cell p's threat channel against cell q's defense channel"
   (`dense_cnn/architecture.py:152-155`). hexgt's mean+max **cannot**, by
   construction.

2. **mean dilutes; max saturates.** mean over N nodes (N often 100s here, candidate
   radius 3 ∪ active windows) washes out one decisive window — the exact §1.4-Gap-B
   dilution. max is the opposite failure: once *any* node lights a channel, adding a
   **second** independent threat (the classic "double threat you can't block both")
   does **not change the max** — `max(big, big) = big`. So **mean+max literally
   cannot distinguish a single blockable threat from an unblockable double threat**
   if both saturate the same channel. Double-threat-loss is *the* defensive failure
   mode, and it is precisely a **count/conjunction** that neither mean (diluted) nor
   per-channel max (saturated) represents. This is the strongest single argument that
   the readout is a real cap.

3. **Non-learnable.** mean/max have **zero parameters** and a **fixed** weighting
   (uniform / argmax). They cannot *learn to look for* the dangerous window — they
   pool whatever the trunk happens to put on each channel. The value head's only
   freedom is the `Linear(3D→D)` *after* pooling, which sees an already-collapsed
   summary. If the trunk hasn't already isolated "danger" onto a clean channel, the
   readout cannot recover it.

**Verdict on question 1 (honest).** Mean+max is a **partial** limitation. It is
*adequate* for material balance and single-threat existence (and that is most of the
board most of the time, which is why the model is already competitive). It is
**genuinely incapable** of the two things defense most needs: (a) **node-wise
cross-channel weighing** of "this threat vs my answer," and (b) **counting /
conjunction** of independent threats (double-threat). These are not edge cases —
they are the named loss mode (value +0.8 then lose). So the cap is **real for the
specific failure**, *but* it is not the only fix and may not be the first one to pay
off. A **learnable, queryable readout (PMA)** removes (a) and (b) directly: a seed
query can attend to *the* opponent threat and, with a value-side cross-attention,
combine it with the mover's counter — the cross-node, cross-channel interaction a
symmetric pool cannot express.

---

## 3. The options, evaluated

For each: **feasibility**, **D6-invariance** (must hold — confirm permutation
invariance over the node set), **drop-in zero-init graft** onto the live checkpoint,
**param/compute cost**, and **expected effect on defensive blindness / value
optimism**. The standing D6 fact (companion §5.5, `constants.py:103-106`): D6 maps
windows→windows, owners→owners, the cell→its image **bijectively**; node embeddings
are D6-invariant by construction (no positional encoding). So **any operation that is
permutation-invariant over the node set is D6-invariant**, and any per-node operation
is D6-equivariant. This is the test every option below must pass.

### Option A — Attention-pooling readout (PMA / attentional graph pooling)

**Change.** Replace (or augment) the fixed mean+max in `_value_readout` with
**Pooling by Multihead Attention (PMA)**: `k` learnable seed query vectors
`S ∈ R^{k×D}` multihead-attend over the graph's node embeddings, producing `k` pooled
vectors that concatenate into the value-head input. Per Lee et al. 2019,
`PMA_k(Z) = MAB(S, rFF(Z))` — the seeds are learnable queries, the set elements are
keys/values; it is permutation-invariant and **strictly generalizes mean/sum/max**
(a seed can learn uniform attention = mean; a sharp seed = soft-max-pool), and **k>1
seeds capture distinct aspects/clusters** of the set. Empirically PMA beats mean/max
pooling on set tasks (Set Transformer §experiments). The graph-NN literature calls
the k=1 form **global attention pooling** (Li et al. GGNN: a softmax gate over nodes
× a value projection) and the iterative form **Set2Set** (Vinyals; order-invariant
LSTM readout).

**The Hexo-specific design** (this is the leverage): use **k = 3 (or 4) seeds with
assigned roles**, so the readout *queries for the defensive sub-structure* the
symmetric pool cannot isolate:
- seed 0 → "overall balance" (free to learn ≈ mean),
- seed 1 → "most dangerous OPPONENT threat" (learns to attend to opp ≥4 window /
  `F_CAND_OPP_THREAT` nodes),
- seed 2 → "my best counter / own win-this-turn" (own count-5 / `F_CAND_WIN_NOW_OWN`),
- (optional seed 3 → "conjunction of independent opp threats" — the double-threat
  signal max-pool saturates on; a seed with multi-head attention *can* spread mass
  over several threat nodes and a downstream Linear can count them).

This directly attacks §2.2 (a) and (b): seeds 1+2 give the value head **cross-node,
cross-channel weighing** (their outputs are combined by the value `Linear`), and a
multi-head seed over several threat nodes gives a **soft count** rather than a
saturating max.

- **Feasibility.** High. PMA is a single `nn.MultiheadAttention` with a learned
  `(k, D)` query parameter, applied per graph over the **padded node set** — and the
  per-graph padded-attention machinery **already exists** (`_AttentionLayout` /
  `build_attention_layout`, `architecture.py:103-144`; the transformer already runs
  batched padded MHA). Add a `ctx_index`-style layout over **all** nodes (or reuse
  `node_graph` with a new padded gather), run one batched MHA with the seed query,
  scatter the `k` outputs into the readout. ~30–50 lines, mirrors the existing
  transformer layer.
- **D6.** **Safe.** PMA attention is symmetric over the key set (no positional
  encoding), so it is permutation-invariant ⇒ D6-invariant — the same property the
  existing context self-attention already relies on and that
  `test_hexgt_value_readout.py:118-137` checks for the current pool. The seed queries
  are graph-independent learned constants (D6 acts on nodes, not on the seeds).
- **Graft.** **Clean, by the *same* surgery already shipped.** Append the PMA output
  to the existing `[side | mean | max]` (→ `(3+k)·D` input) so the value head's first
  `Linear` widens; extend `expand_value_readout_columns` (`architecture.py:387-428`)
  to zero the **new** PMA blocks at resume (the SIDE/mean/max blocks keep their
  trained weights) ⇒ **byte-identical first step, no cold start**, then it learns the
  attention pool from zero. The seed parameters initialize fresh and contribute 0 on
  step 0 because their output block is zeroed in the head. **This is the cleanest
  evolution of the shipped code.**
- **Cost.** Small. `k·D` seed params (≈ 3·168 = 504) + one MHA's projections
  (`4·D·D` ≈ 113K) ≈ **~115K params** (under 6% of 2.07M). One extra batched MHA per
  forward over the node set — comparable to one transformer layer's context attn,
  which the model already runs 3 of. Negligible throughput hit (self-play is
  featurize-bound, per the perf memory).
- **Expected effect.** **Directly targets the defensive cap (§2.2 a, b).** Removes
  the cross-node weighing and double-threat-counting limitation a symmetric pool
  cannot express. **Best readout-only option**; well-grounded that it *can* represent
  the missing function, empirical that it *will* fix the metric.

### Option B — KataGo-style SE global-context fusion INTO the trunk

**Change.** This is **not a readout change — it is a trunk change**, and per the
KataGo evidence it is likely the **highest-leverage** option for "reason like
dense_cnn with global pooling." KataGo inserts a **global-pooling bias** structure
into **2–3 residual blocks** (and the policy/value heads): it pools each channel to
**mean, mean×board-width, and max** (3c values), runs them through a fully-connected
layer, and **adds the result channelwise as a per-channel bias** to the next conv
layer — Squeeze-and-Excitation style — so that **every subsequent local computation
is conditioned on global board context** ("conditioning on board context outside the
perceptual radius"). Ablating global pooling cost KataGo **1.60×** learning
efficiency (KataGo paper, ablations).

**hexgt analog.** After GNN layer `ℓ` (and/or before a transformer layer), compute a
per-graph pooled context `g = MLP([mean | max | std]_over_nodes)` and **add it back
to every node embedding** of that graph (`h_i ← h_i + broadcast(g_{graph(i)})`), a
SE-style channelwise bias. Then the *next* GNN/transformer layer computes local
messages **already aware** of the whole-board summary — so a WINDOW node's update can
encode "I am a ≥4 threat **and the board is otherwise quiet so this decides it**,"
which a post-hoc readout can never reconstruct because the trunk discarded it.

- **Feasibility.** Moderate. `_global_pool` **already computes mean+max per graph**
  (`architecture.py:284-304`); add `std` (one more `index_add_` of squares), an MLP
  `Linear(3D→D)`, and a broadcast-add inside `_encode_nodes` between layers
  (`architecture.py:262-270`). The broadcast index is `node_graph` (already in
  scope).
- **D6.** **Safe.** The pooled `g` is permutation-invariant per graph; broadcasting
  the **same** `g` to every node of a graph is permutation-equivariant (it adds an
  identical vector regardless of node order). D6 commutes with it. Add a unit test
  mirroring `test_global_pool_is_permutation_invariant`.
- **Graft.** **Clean zero-init graft.** Make the fusion an *additive residual* whose
  MLP **output projection is zero-initialized** (or gated by a zero-init scalar `α`,
  `h_i ← h_i + α·MLP(...)` with `α=0`). At resume the fusion contributes 0 ⇒ forward
  is byte-identical to the current checkpoint, then `α`/MLP learn from zero. Same
  philosophy as `expand_value_readout_columns` and `zero_init_expanded_feature_columns`
  (`architecture.py:357-428`). Resume guard extends `_validate_stv_resume_load`
  (`_rl_train.py:55-64`) to allow the new fusion params missing.
- **Cost.** Moderate. One MLP per fused layer (`Linear(3D→D)` ≈ 85K) × 1–2 fusion
  points ≈ **~85–170K params**. One extra pool+broadcast per fused layer; cheap.
- **Expected effect.** **Likely the highest-leverage for the owner's literal goal**
  ("local reasoning that is globally aware THROUGHOUT, like the CNN"). It is the only
  option that fixes the trunk-side root cause: today the trunk computes node features
  with **no whole-board conditioning** until the very end, so the readout (any
  readout) is pooling features that were never computed with global awareness. **This
  is the part a fancier readout alone cannot replace** (see §5, the trunk-vs-readout
  honesty). Strongly grounded by KataGo's measured 1.60× ablation.

### Option C — Dual-stream readout (pool GNN node embeddings AND transformer outputs)

**Change.** The owner's literal phrasing — "reason over BOTH the graph and the
transformer." Today `_encode_nodes` returns **only the post-transformer** embeddings
(`architecture.py:262-270`); the GNN's pre-transformer node states are discarded. A
dual-stream readout would pool/attend over **both** the post-GNN embeddings (local,
typed, message-passed) **and** the post-transformer embeddings (whole-board attention
context) and concatenate, so value explicitly integrates local-typed + global-attn
representations.

- **Feasibility.** Easy mechanically (return `h_gnn` alongside `h_final` from
  `_encode_nodes`, pool each, concat). The *question* is whether it adds signal over
  A/B: the transformer is residual, so `h_final` already contains `h_gnn`'s
  information *plus* attention — pooling both is partly redundant. It mainly helps if
  the transformer **washes out** a local feature the GNN had isolated. With only 3
  transformer layers and residual connections, that washout is modest, so C's
  marginal value over A is **uncertain**.
- **D6 / Graft / Cost.** Same as A (permutation-invariant pools/PMA over each stream;
  zero-init the new block in the head; ~2× the pool params, still small).
- **Expected effect.** Modest and **partly redundant with A**. Best treated as a
  *variant of A* (run the PMA seeds over the concatenation of `h_gnn` and `h_final`,
  or give some seeds the GNN stream and others the transformer stream) rather than a
  separate intervention. **Folds into A.**

### Option D — Value/CLS query token with dedicated cross-attention

**Change.** Add a dedicated **value token** (a learned embedding, or reuse the SIDE
hub) that, in **one or more extra cross-attention layers**, attends over **all**
tokens (context **and** candidates) to build the value summary by attention rather
than post-hoc pooling. This specifically **fixes the confirmed asymmetry** (§1: SIDE
never cross-attends to candidates): a value token that cross-attends to candidates
can pull in "the decisive move's refutation" that today only reaches SIDE via the
diluted GNN CONTEXT mean.

- **Feasibility.** Moderate. It is essentially **PMA with k=1 seed attending over
  context ∪ candidates**, optionally stacked for 1–2 layers. Mechanically it is A
  with the key set extended to include candidate tokens (the
  `cand_index`/`ctx_index` split already exists, `architecture.py:139-144`). The
  difference from A is *which tokens it attends over* (D includes candidates) and
  *depth* (D may stack layers, making it a small value-specific transformer).
- **D6 / Graft / Cost.** Same safe story as A (permutation-invariant attention,
  zero-init the value-token contribution, small params). A stacked version costs one
  MHA+FFN per extra layer (~340K each).
- **Expected effect.** Strong, and it uniquely closes the SIDE←candidate asymmetry.
  But it **overlaps heavily with A**: a PMA seed *is* a value query token; the only
  real addition is (i) attending over candidates too and (ii) optional depth.
  **Recommendation: realize D as "A's seeds attend over context ∪ candidates," and
  add depth only if A's single-pass version underperforms.** Don't build D and A as
  separate modules.

### Option E — Trunk depth/width adequacy (is the TRUNK the real bottleneck?)

**This is the honesty check the owner asked for, and it is load-bearing.** With **3
GNN + 3 transformer layers**, is there even enough computed local+global information
for *any* readout to reason over?

- **Message-passing hop count vs threat-propagation distance.** A *developing*
  double-threat loss relates **two** opponent windows through a shared candidate cell:
  opp-stone → (STONE_WINDOW) → window A → (CANDIDATE_WINDOW) → shared cell →
  (CANDIDATE_WINDOW) → window B → (CONTEXT) → SIDE. That is **~3–4 GNN hops** before
  the transformer runs, and the GNN is **only 3 layers** (`constants.py:175`). The
  SIDE CONTEXT edge is a 1-hop shortcut to everything, **but only as a diluted mean**
  (`architecture.py:99`). So the *structured two-window relationship that defines the
  loss can be truncated by GNN depth*, and **no readout can pool information the
  trunk never computed.** The transformer's global self-attention partially rescues
  this (it is not hop-limited — any context token can attend to any other in one
  layer), but **candidates are not in the context self-attention set** (§1), so the
  candidate→candidate relationship that carries a double-threat is **never directly
  attended** — it only flows through window tokens.
- **Width.** At `token_dim=168` (~2.07M params, matched to the 96×8 CNN,
  `constants.py:171-173`), raw capacity is comparable to the CNN; width is **not**
  the likely bottleneck. **Depth and connectivity are.**
- **The interaction (flagged, per the brief).** **If E is the real bottleneck, a
  fancier readout (A/D) alone will not help** — it will pool a trunk that never bound
  the two windows together. **This is the single most important caveat in the doc.**
  The mitigations, cheapest first:
  - **B (global-context fusion)** is *also* a partial fix for E: a pooled global
    context broadcast after GNN layer 1 gives every later local update a 1-step view
    of the whole board, shortening the effective propagation distance. This is why
    **B composes with A** and is arguably prerequisite.
  - **+1–2 GNN layers** (`gnn_layers` 3→4/5, a config/`__init__` change,
    `architecture.py:194-195`) lets the two-window relationship complete before
    readout. Partial checkpoint break (new layer re-inits; resume via the existing
    `strict=False` path, `_rl_train.py:263`); D6-safe (message passing is
    permutation-equivariant).
  - **Let the value token (D) attend over candidates** so candidate↔candidate
    double-threat structure reaches the value path without needing more GNN hops.

**Verdict on E.** The trunk is a **plausible co-bottleneck**, specifically via
(i) GNN depth = 3 < the ~4-hop double-threat path and (ii) candidates being excluded
from context self-attention. **The validation plan MUST ablate readout-vs-trunk
(§7)** so the team does not spend effort on a readout that a thin trunk starves.

---

## 4. Directly answering the owner

**Q1 — Is mean+max a REAL limitation?** *Partially, but the real part is exactly the
failure mode.* (§2) It is adequate for material balance (mean) and "does a lone
threat exist" (per-channel max), which is most positions — which is why the model is
already competitive and why soft-Z + the v3 threat features + TSS may move the metric
without touching the readout. It is **genuinely incapable** of the two things defense
most needs: **(a)** node-wise **cross-channel weighing** ("this opponent threat
*versus* my counter"), because per-channel max takes each channel from possibly
different nodes and destroys the within-node binding; and **(b)** **counting
independent threats** (the double-threat loss), because mean dilutes it and max
saturates (`max(big, big)=big`). dense_cnn avoids this not by pooling but by a
**learnable per-position `Linear` over all 1681 cells** (`dense_cnn/architecture.py:152-155`)
— so the honest analog of its power on a permutation-invariant graph is a **learnable
attention pool (PMA)**, not mean+max. **So yes, the cap is real for the diagnosed
failure — but it is not the only lever and likely not the first to pay off.**

**Q2 — Is "reason over BOTH the graph and the transformer" feasible?** *Yes, and
cheaply.* (§3 A, B, C) Two concrete forms, both D6-safe and zero-init graftable onto
the live checkpoint: **(B)** fuse a pooled global context back into the trunk so
local GNN/transformer updates are globally aware *throughout* (the KataGo recipe, the
truest match to "reason like the CNN"); and/or **(A/C)** a learnable PMA readout whose
seeds attend over the node set (optionally over the GNN stream *and* the transformer
stream). Both reuse machinery already in the file (`_global_pool`,
`_AttentionLayout`, the zero-init graft helpers). Feasible is not the question; *which
combination* is.

**Q3 — Is the framing correct?** *Mostly, with two corrections.* (§0) (i) The
framing assumes the head is "just SIDE+mean+max and that's the whole problem" — but
the mean+max pool is **already shipped and being grafted in**, so the next step is
*learnable/queryable* pooling, not *adding* pooling. (ii) The deeper risk the framing
underweights is **the trunk (E)**: a 3-layer GNN with candidates excluded from
context self-attention may not *compute* the double-threat relationship that any
readout would need to pool. **"Make the value head reason over both" is right, but
the binding insight is that the readout can only reason over what the trunk
computed** — so the strongest version of the owner's goal is **trunk fusion (B) +
learnable readout (A)** together, not a readout alone.

---

## 5. Recommendation (ranked, composed with prior work)

The diagnosed failure is **defensive value miscalibration**, and the code shows
**three distinct roots** that need **three distinct fixes** (this is the crux the
companion docs already establish, and it holds): the **label** (soft-Z), the
**explicit tactical signal** (TSS / threat features), and the **value
representation** (this doc). The value-representation fix itself splits into
**readout** (A) and **trunk** (B/E). The honest ordering:

### Recommended combination (the headline)

> **KataGo-style global-context fusion in the trunk (B) + a PMA attention-pooling
> readout (A) with role-assigned seeds, built as one zero-init graft onto the value
> path, on top of the already-shipped soft-Z + v3 threat features + TSS work.**

Why this pair: **B fixes the trunk** so local reasoning is globally aware throughout
(the owner's literal "reason like the CNN with global pooling," and the part a
readout cannot replace, §5/E); **A fixes the readout** so the value head can do the
**cross-node weighing and threat-counting** that mean+max provably cannot (§2.2).
They are complementary — B makes the trunk produce a better-conditioned
representation; A reasons over it learnably. Neither alone is sufficient (B with a
mean+max readout still can't weigh threat-vs-counter; A over an un-fused 3-hop trunk
still pools un-bound windows).

### Ranking

| Rank | Change | Targets | Cost | D6 | Graft | Grounding |
|---|---|---|---|---|---|---|
| **V1** | **B — trunk global-context fusion (SE-style, 1–2 points)** | trunk computes features WITHOUT global awareness (the deepest root) | moderate (~85–170K) | safe (perm-equivariant broadcast) | clean (zero-init α/MLP) | **strong** (KataGo 1.60× ablation) |
| **V2** | **A — PMA readout, k=3–4 role seeds, appended to [side\|mean\|max]** | cross-node weighing + threat counting mean+max can't do (§2.2) | small (~115K) | safe (perm-invariant attn) | clean (extend `expand_value_readout_columns`) | **strong** (Set Transformer; generalizes mean/max) |
| **V3** | **E-lite — +1 GNN layer AND/OR let A's seeds attend over candidates (=D)** | trunk depth < double-threat hop count; SIDE←candidate asymmetry | small–moderate | safe | partial (new layer re-inits, `strict=False`) | grounded hop-count; empirical depth |
| — | C (dual-stream) | local+global integration | small | safe | clean | **fold into A** (run seeds over both streams) — not separate |
| — | D (value/CLS token) | SIDE←candidate asymmetry | moderate | safe | clean | **realize as A-over-candidates** — not separate |

**V1 before V2** is deliberate and is the doc's most consequential ranking call: per
the E analysis, **a learnable readout over a trunk that never computed the relationship
is pooling noise.** Fuse first (or together). If the team can only ship one, ship
**B** — it is the truest match to the owner's goal and the only one that fixes the
trunk root.

### Honest grounded-vs-speculative split

- **Well-grounded:** mean+max's specific incapacities (§2.2, capacity argument); PMA
  generalizes mean/max and is permutation-invariant (Set Transformer, fetched); KataGo
  global-pooling-into-trunk gives a measured 1.60× and is the canonical "globally
  aware local reasoning" (KataGo, fetched); the trunk hop-count vs double-threat path
  (read from code); all D6 and graft mechanics (the helpers and tests exist in code).
- **Speculative (hypotheses to validate, not facts):** that A/B *will* move the
  defensive metric (the *capacity to represent* the function is proven; that training
  *learns* it on this pipeline is empirical); the exact number of fusion points
  (1 vs 2) and seeds (3 vs 4); whether E (depth) is a true co-bottleneck or the
  transformer's global attention already compensates. **Measure, don't assume.**

---

## 6. D6-invariance, the graft recipe onto e42, budget — for the recommendation

**D6-invariance (must hold — confirmed).** Every recommended op is either
**permutation-invariant over the node set** (A's seed attention, B's per-graph pool)
or **permutation-equivariant per node** (B's broadcast-add of the same `g` to every
node; an extra GNN layer). D6 acts on hexgt only by **relabeling nodes**
(windows→windows, cells→images, bijectively — `constants.py:103-106`), so any
permutation-invariant readout commutes with D6 and any per-node op is D6-equivariant.
This is the **same property** the shipped mean+max pool relies on and that
`test_hexgt_value_readout.py::test_global_pool_is_permutation_invariant` (lines
118-137) already enforces. **Gate the new modules on an extended version of that test
+ the end-to-end `test_hexgt_d6.py` equivariance test before any training** (the
discipline that catches the D6-poisoning failure class the project has hit before).

**Zero-init graft onto the live checkpoint (e42-class), reproducing current behavior
then learning.** The pieces all exist; assemble them so the first post-resume forward
is byte-identical:
1. **A (PMA):** append the `k·D` PMA output **after** `[side|mean|max]` in
   `_value_readout` (→ `(3+k)·D`). Extend `expand_value_readout_columns`
   (`architecture.py:387-428`) to widen `value_head.0.weight` from `3D` to `(3+k)D`,
   copying the trained `3D` block and **zeroing the new PMA block** ⇒ PMA contributes
   0 on step 0. Seed params init fresh (irrelevant on step 0 since their output is
   zeroed in the head). **No cold start.**
2. **B (fusion):** add the fusion as a residual with a **zero-initialized gate `α`
   (or zero-init MLP output)** so `h_i ← h_i + α·MLP([mean|max|std])` is identity at
   `α=0` ⇒ forward unchanged at resume, then `α`/MLP learn. Mirrors
   `zero_init_expanded_feature_columns`'s philosophy (`architecture.py:357-384`).
3. **Resume guard:** extend `_validate_stv_resume_load` (`_rl_train.py:55-64`) to
   allow the new `value_pma.*` / `trunk_fusion.*` keys to be missing from an old
   checkpoint (exactly as it already tolerates `short_term_value_heads.*`), and call
   the expand helper before `load_state_dict(..., strict=False)` (as
   `_rl_train.py:262-263, 281-282` already do for the value readout). **The graft
   pattern is proven; this is additive.**
4. **STV heads:** consider routing the STV heads through the same new readout (today
   they read SIDE-only, `architecture.py:322-326`) — optional, but it removes a
   residual single-hub bottleneck on the auxiliary value signal at no extra module
   cost. Zero-init graftable the same way.

**Param/compute budget.** B ≈ 85–170K, A ≈ 115K, V3 (one GNN layer) ≈ `D·D·5` (5 edge
types) ≈ 235K. **Total well under +0.5M on a 2.07M model (<25%)**, and the dominant
self-play cost is featurization, not the forward (per the perf memory), so throughput
impact is minor. Calibration (`configs/hexgt_model2.toml:105-113`) re-tunes batch
sizes automatically.

---

## 7. Validation plan (and the load-bearing readout-vs-trunk ablation)

Judge by these, **not train loss** (a documented artifact pitfall in this project).
Reuse the harness the companion docs specify and the `root_value` traces self-play
already logs (`selfplay.py`).

1. **Same-board `v(A)+v(B)` sum (primary optimism metric).** Re-measure on a fixed
   probe set from both sides; target ≈ +0.82 → 0. *Note (CORRECTION 1): the run
   already has the mean+max pool, so the baseline for this experiment is the
   pooled-readout model, not a SIDE-only one — measure the current run's sum first as
   the true baseline.*
2. **Calibration on opponent-hot / defensive slices.** Slice held-out self-play to
   positions with an active opponent ≥4 window (label cheaply with the Part-1 TSS
   detector / the `F_CAND_OPP_THREAT` flag, `constants.py:132`); measure value-head
   CE/Brier and the reliability curve. This is the slice A is *designed* to fix.
3. **The 8/8-lost-game value trace.** Re-run the forensic probe on the SealBot loss
   games; final-ply value should drop toward the loss instead of pinning near +0.8.
4. **H2H** vs dense_cnn e24 via `run_head_to_head` (`player.py` / `evaluation.py`) —
   the integrative judge.
5. **THE ablation that decides readout-vs-trunk (per the brief).** Run four arms from
   the live checkpoint (all zero-init graft, a few RL epochs each):
   - (i) current (mean+max readout, no fusion) — baseline,
   - (ii) **+A only** (PMA readout, no trunk fusion),
   - (iii) **+B only** (trunk fusion, mean+max readout),
   - (iv) **+A+B**.
   **If (iii) ≫ (ii) on the defensive slice, the TRUNK (E) was the bottleneck and a
   fancier readout alone was never going to help** — confirming §5/E and re-ordering
   future effort toward depth/fusion. If (ii) ≈ (iv) ≫ (iii), the readout was the
   bottleneck. This ablation is the cheapest way to avoid building the wrong thing.
6. **D6 / parity gate (before any training).** Extend
   `test_hexgt_value_readout.py`'s permutation-invariance test to the new modules and
   confirm `test_hexgt_d6.py` still passes for all 12 elements.

---

## 8. How this composes with the already-done / planned work

| Prior work | Relationship to A/B (this doc) |
|---|---|
| **soft-Z value target** (companion 0a; `samples.py:198`) | **Orthogonal, both needed.** soft-Z recalibrates the *label* (fixes +0.82 saturation); A/B fix the *representation* the recalibrated label flows through. soft-Z is cheaper and should land first; a recalibrated target still benefits from a readout that can see the threat. |
| **Shipped mean+max readout** (`architecture.py:236-311`) | **A *extends* it** (append PMA, keep `[side\|mean\|max]` as the zero-init base). Not a replacement — the graft *requires* the existing block to stay for byte-identical resume. |
| **v3 threat features** `F_CAND_WIN_NOW_OWN` / `F_CAND_OPP_THREAT` (`constants.py:129-135`) | **Synergistic input to A's seeds.** The "opp-threat" seed learns to attend to nodes carrying `F_CAND_OPP_THREAT`; better features make the attention target cleaner. A makes the *value head* use the signal the features make *available*. |
| **TSS injection + leaf override** (companion PART 1) | **Complementary, different layer.** TSS guarantees the must-block is *searched* and gives a *hard* 1-ply override; A/B fix the *net's own* value when no proof exists (multi-ply, quiet positions). TSS handles the sharp 1-ply tactic; A/B handle the integrated judgment. |
| **Ownership aux head** (companion 0b) | **Strongly complementary with B.** Ownership *forces the trunk to localize* where the game is decided; B *broadcasts that localized context globally*. Together they make the trunk both localize and globally condition — the two halves of "reason like the CNN." If only one trunk change is built, B; if budget allows, B+ownership. |
| **Policy-surprise KL weighting** (companion §3) | **Orthogonal training-emphasis lever.** Trains hardest on the defensive blind spots; A/B give the net the *capacity* to not be blind there. Neither removes the other. |
| **ResTNet / D6-safe conv bias** (companion §6.4) | **The structural Model-2.x option beyond A/B.** If A+B+E still leave a local-pattern gap, the axis-shared 1-D conv (companion path B/C) adds the conv inductive bias D6-safely. Deferred behind the cheaper graftable fixes here. |

**Bottom line:** soft-Z (label) + this doc's B+A (representation: trunk + readout) +
TSS (tactical) are **three complementary tracks**, all D6-safe and all zero-init
graftable onto the live checkpoint, attacking the same defensive miscalibration from
the three roots the code exposes. Land soft-Z first (cheapest, already specified),
then B+A as the value-representation upgrade, with the §7.5 ablation deciding whether
trunk or readout was the binding constraint.

---

## Sources — fetched vs. reasoned-about

- **FETCHED (full text via ar5iv HTML):** Lee et al., *Set Transformer* (arXiv
  1810.00825) — PMA definition `PMA_k(Z)=MAB(S, rFF(Z))`, k learnable seeds, generalizes
  mean/max, permutation-invariance, k-seed roles. KataGo / Wu 2019 (arXiv 1902.10565)
  — global pooling = {mean, mean×board-width, max} → FC → **channelwise bias into the
  next conv**, in 2–3 residual blocks + policy/value heads; ablation **1.60×**;
  auxiliary ownership/score targets help. ResTNet (arXiv 2410.05347, via search) —
  conv+transformer interleave cut a Go threat blind-spot **70.44%→23.91%**, +win-rate
  across Go/Hex.
- **FETCHED (abstract/secondary only — reasoned from the well-documented method):**
  The Set Transformer / KataGo **publisher PDFs** returned binary/abstract-only; the
  precise mechanics above are from the **ar5iv HTML full text**, which I treat as
  authoritative. GNN readouts (Li et al. GGNN global attention pooling; Vinyals
  Set2Set; multi-level attention pooling arXiv 2103.01488) — from survey snippets,
  reasoned about, not fetched in full; used only as corroborating prior art, not for
  load-bearing claims.
- **CODE (read directly, cited by file:line):** hexgt worktree
  `E:\Hexo-BotTrainer-hexgt`: `architecture.py` (full), `constants.py`, `losses.py`,
  `configs/hexgt_model2.toml`, `scripts/_rl_train.py`, `tests/test_hexgt_value_readout.py`;
  dense_cnn main tree `E:\Hexo-BotTrainer`: `dense_cnn/architecture.py`.

## Code-citation corrections to the prior docs

1. **The global-pooled `[SIDE | mean | max]` value readout is SHIPPED, not a
   proposal.** Companion `HEXGT_ARCH_DESIGN_EXPLORATION.md` Rank 1 and
   `HEXGT_TSS_AND_SOFT_VALUE_DESIGN.md` PART 2 describe it as a future change; it is
   live (`architecture.py:42-43, 236-311`), with a tested zero-init graft
   (`expand_value_readout_columns`, `architecture.py:387-428`;
   `test_hexgt_value_readout.py`) wired into the RL resume (`_rl_train.py:262, 281`).
2. **dense_cnn's value head does NOT pool — it is a learnable per-position `Linear`
   over all 1681 cells.** Companion docs repeatedly call it "global pooling"
   (e.g. §1.2, §1.4 Gap B). `ValueBinnedHead` is `Conv(C→1)` → `Linear(BOARD_AREA→64→65)`
   (`dense_cnn/architecture.py:143-159`) — a position-keyed dense readout, strictly
   more expressive than a symmetric mean/max pool. This *strengthens* the owner's
   point and reframes the honest CNN analog as a **learnable attention pool (PMA)**,
   not mean+max.
3. **The STV (short-term-value) heads still read the SIDE hub only** — they did NOT
   receive the pooled readout (`architecture.py:322-326`). The companion docs imply
   all value heads read the pool; only the **main** value head does.
4. **Layer counts and live widening confirmed accurate:** 3 GNN + 3 transformer
   (`constants.py:175-176`, `configs/hexgt_model2.toml:23-24`), `widening_max_children
   = 96` (`configs/hexgt_model2.toml:86`) — the prior docs are correct here.
