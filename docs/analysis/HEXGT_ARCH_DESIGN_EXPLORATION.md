# hexgt Architecture & Training-Weighting Design Exploration

Design/analysis only. No code, config, or model files are changed by this document.
Every claim below is cited to a file path (most in the hexgt worktree
`E:\Hexo-BotTrainer-hexgt`, some in the dense_cnn main tree `E:\Hexo-BotTrainer`).

---

## Headline summary

**Most likely representational gap.** hexgt's value signal is read out from a
**single SIDE hub node** (`architecture.py::HexgtNetwork._graph_readout`, which
gathers only `NODE_TYPE_SIDE` rows), fed by a body that is **3 GNN message-passing
layers + 3 transformer layers** (`constants.py::DEFAULT_GNN_LAYERS = 3`,
`DEFAULT_CTX_LAYERS = 3`). The opponent's developing threats live in
**WINDOW nodes** (count-3/4/5 active windows) and in the *graph structure* of which
empty cells lie on which lines. For the value head to "see" a developing
open-four, threat evidence has to propagate from opponent stones → window hubs →
candidate cells and then be **selected by the transformer into the SIDE token**.
Two things make that fragile and under-powered for *defense* specifically:

1. **Threat severity is encoded too coarsely.** The only window-danger signals are
   one-hot count buckets `{3,4,5}` (`features.rs`, slots `F_WIN_COUNT_ONEHOT`,
   `F_CAND_COMPLETE_OPP`, `F_CAND_OPP_WIN{3,4,5}`). There is **no graded
   "open-ended-ness" / urgency** feature: a count-4 window with *two* open ends
   (an immediate forced win-threat) looks identical to a count-4 window with one
   open end (often harmless). A CNN doesn't need this feature because it can *grow
   the pattern itself* across a translation-equivariant receptive field; the GNN
   is handed only the count and must reconstruct urgency from structure it may not
   reach in 3 hops.

2. **The whole-board value comes through one token and a shallow attention path.**
   A *global* "am I losing?" judgement is exactly the kind of board-wide integration
   a candidate-centric, hub-readout design is weakest at. The CNN value head pools
   over the **entire 41×41 board** (`dense_cnn/architecture.py::ValueBinnedHead`,
   `Conv→Linear(BOARD_AREA→…)`); hexgt's value head sees one 168-d vector that the
   transformer had to assemble. When the SIDE token fails to attend to the decisive
   opponent window, the value head is confidently wrong — which is *precisely* the
   reported failure (value ≈ +0.8 right before losing 8/8 to SealBot).

These produce **value-head overconfidence about defense** much more readily than
overconfidence about offense, because your own attack is visible in *your own
candidate* features (the policy head's job, which works), whereas the opponent's
attack must be integrated globally into the value readout (the part that is thin).

**Top recommendations (ranked, detail in §2, §4, and §6).** *The external prior-art
review in §6 promotes two **value-OUTPUT** fixes to share the top tier, because they
attack the failure — value miscalibration / lack of spatial localization — even more
directly than the readout plumbing, and both are externally **validated** (not just
plausible):*

0a. **Soft-Z value targets (NEW, §6.1)** — blend the hard ±1 outcome with the
   bootstrapped MCTS root value when forming the **main `value` target**
   (`samples.py::finalize_game_samples`, currently hard `_winner_value`). Validated to
   train faster / play stronger than vanilla AlphaZero (A0C/A0GB, 10.1007/s00521-021-05928-5).
   Recalibrates *the exact target the value head learns* — the most direct attack on
   "value ≈ +0.8 right before losing." Tiny, near-zero-risk, D6-safe, reuses the EMA
   machinery hexgt already has for the STV heads. **Shares Rank 1.**

0b. **Ownership auxiliary head (NEW, §6.2)** — a KataGo-style per-cell control head on
   the GNN cell-node embeddings, regularizing the trunk and *forcing spatial
   localization* of where the game is decided. Ablating ownership hurt KataGo's learning
   (arXiv 1902.10565). Zero-init graftable onto e42, D6-equivariant per-node head.
   Attacks the defensive blindness at its root (the trunk never has to localize threats
   today). **Shares Rank 1 / just below the readout fix.**

1. **Stronger whole-board value readout** — replace/augment the single-SIDE-hub
   value readout with a global pooled readout over all nodes (mean+max over
   window/stone/candidate tokens), so the global "is the opponent winning"
   judgement does not bottleneck through one attention-assembled token. *Small,
   drop-in, D6-safe, highest leverage-to-risk.* *(Still the top **architectural**
   fix; §6 pairs it with 0a/0b as the top **output/target** fixes.)*

2. **Policy-surprise (KL) sample weighting** — KataGo-style upweighting of
   positions where search disagreed most with the prior. Defensive blind spots are
   *by construction* high-surprise (the prior thought it was safe; search found the
   refutation). hexgt has forced playouts but **no** surprise weighting today
   (`_rl_train.py`, `losses.py`), and — critically — hexgt currently **discards the
   root prior before writing shards** (`compact_io.py` restores
   `root_prior_policy=()`), so this needs a small data-path change. *Targets the
   defense failure directly; medium cost because of the prior-persistence gap.*

3. **Richer / graded threat features + deeper threat propagation** — add
   open-end/urgency features on window and candidate nodes (cheap, in `features.rs`)
   and/or add 1–2 GNN layers so opponent threats propagate one more hop. *Medium
   cost; attacks the coarse-threat root cause; D6-safe.*

A spatial/CNN board-encoder bolted onto the graph (§2, option D) is the most
"complete" fix for the user's representational hypothesis but is the **highest
cost and highest risk** (it threatens the D6-invariance-by-construction property
that the equivariance test currently guarantees), so it is recommended only if
(1)–(3) fail to move the defense metric. *Update (§6.4): ResTNet (arXiv 2410.05347)
gives empirical support for the conv-bias hypothesis (it cut Go's circular-pattern
threat blind-spot 70%→24%), and §6.4 identifies a **D6-respecting** path (group-/
hex-equivariant conv, or conv along the 3 line-axes with 6-fold weight sharing) that
sidesteps the square-crop D6 break. This **raises the CNN-hybrid above "last resort"
to a deferred-but-promising structural Model-2.x move** — but it stays deferred behind
the validated, cheaper 0a/0b/Rank-1 fixes.*

**Addendum — "hot tokens" (§5).** A later pass explored an explicit representation
for *immediately decisive* cells ("hot" cells: own-win-now, own-forced-win-next,
opponent-must-block). The finding (detail in §5): **opponent-hot detection is the
defensive-blindness signal in its most concentrated form**, and it is cheap to
compute in the Rust window walk already present in `candidates.rs`/`features.rs`.
The recommended form is a **two-tier graft**: (a) two opponent-hot / own-hot
**candidate features** in the reserved slots `[30:32)` (rides Rank 2's existing
zero-init feature-expansion path, no cold start, D6-safe), plus (b) a **value-readout
attention bias** (§5.2-iii) that forces the value path / SIDE hub to attend to any
opponent-hot token — which is the most *targeted* version of Rank 1, guaranteeing a
must-block threat reaches the value head structurally rather than by winning a
single-token attention competition. Hot tokens are a **sharper, decisive subset** of
Rank 2's graded threat features wired into Rank 1's readout fix — a complement, not a
substitute. It ranks **just below Rank 1 and above the CNN hybrid (Rank 4)**.

---

## 1. Representational adequacy: GNN+transformer vs CNN for Hexo

### 1.1 What Hexo's value function actually requires

Hexo is won by completing a length-6 line (`candidates.rs`: `WIN = 6`,
`window_tokens` count buckets 3/4/5). The decisive facts are **line geometry**:
which empty cells complete or extend which windows, and whether a window is
*open* (completable) at one or both ends. Defense = recognizing the opponent's
windows that are about to become forcing (open-four, double-three, 5-with-open-end)
and that *you cannot stop all of them*. This is inherently:

- **translation-invariant** (the same threat pattern matters anywhere on the board), and
- **multi-window / global** (a loss is often "two simultaneous threats you can't both block").

### 1.2 Why a CNN is a natural fit (the baseline)

dense_cnn encodes the board as 13 dense planes
(`dense_cnn/constants.py`: own/opp stones, legal, recency, and notably
`PLANE_OPPONENT_HOT`, `PLANE_OWN_HOT`, `PLANE_OPPONENT_LAST_TURN`,
`PLANE_CENTER_DISTANCE`) over a fixed 41×41 crop, processed by
`HexConv2d` blocks whose 3×3 kernels are **masked to the 6 axial hex neighbors**
(`dense_cnn/architecture.py::HexConv2d`, masking corners `(0,0)` and `(2,2)`).
Properties that suit Hexo:

- **Translation equivariance**: every conv filter that learns "open-four along an
  axis" applies identically everywhere. A line threat is recognized regardless of
  board location with one shared filter.
- **Receptive field grows with depth**: with `DEFAULT_BLOCKS = 6` gated residual
  blocks of 3×3 hex convs, the receptive field is wide enough to span a length-6
  line *and its surroundings* and to relate **multiple nearby threats** spatially.
- **Global value pooling**: `ValueBinnedHead` does `Conv(C→1)` then
  `Linear(BOARD_AREA→64→65)` — the value head literally sees **every cell**, so a
  far-away decisive opponent window cannot be "forgotten."

So a CNN gets line-pattern recognition (equivariance) and whole-board integration
(deep RF + dense value pool) essentially for free. That is the bar.

### 1.3 What hexgt does, precisely (read from code)

**Node types** (`candidates.rs`, `constants.py`): SIDE (1 hub), STONE, CANDIDATE
(empty cells in active windows ∪ n=3-radius minus dead cells), WINDOW (active
count-3/4/5 windows of both colors).

**Message passing** (`architecture.py::RelationalMessagePassing`, stacked
`DEFAULT_GNN_LAYERS = 3`): typed mean-aggregation `m_{j→i}=relu(W_type h_j +
edge_proj(attr))`, residual+LayerNorm. Edge types (`candidates.rs`): ADJACENCY
(hex-distance-1), STONE_WINDOW, CANDIDATE_WINDOW, RECENCY (stone chain), CONTEXT
(SIDE hub ↔ every node). Lines/co-linearity are **routed through window hubs**, not
as same-axis cliques (`candidates.rs` header; `constants.py` §6.3 note).

**Transformer** (`architecture.py::GraphTransformerLayer`, stacked
`DEFAULT_CTX_LAYERS = 3`, `DEFAULT_ATTENTION_HEADS = 4`, `token_dim = 168`,
`ffn_dim = 336`): per-graph **context self-attention** over {side, stone, window}
tokens, then **candidate→context cross-attention** (candidates query the context).
Note the asymmetry: candidates attend *to* context, but context tokens (including
SIDE) are updated only by **context self-attention** — they do **not** cross-attend
to candidates.

**Value readout** (`architecture.py::_graph_readout` + `_heads`): value and all
stvalue heads read **only the SIDE hub node embedding** (`NODE_TYPE_SIDE` row),
one 168-d vector per graph. The policy head reads **per-candidate** embeddings.

**D6 handling** (`constants.py` feature-layout note, `features.rs`): all features
are D6-invariant (no raw axial coords, no axis labels; geometry carried by edge
structure + invariant hex-distance), so the model is D6-invariant *by
construction* and the equivariance test passes with **no augmentation**
(`expand.py` docstring). This is a real asset to protect.

### 1.4 The candidate gaps, mapped to the defensive-blindness symptom

For each gap: G = well-grounded in code, S = more speculative.

#### Gap A — Threat severity/urgency is too coarse (G, strongest feature-level cause)

The only opponent-threat features are:
- WINDOW nodes: `F_WIN_COUNT_ONEHOT` (3/4/5 one-hot) + `F_WIN_EMPTY_CELLS`
  (`features.rs`).
- CANDIDATE nodes: `F_CAND_COMPLETE_OPP` (completes a count-5 opp window),
  `F_CAND_NWIN_OPP`, and the v2 splits `F_CAND_OPP_WIN{3,4,5}` (per-count window
  counts through the cell) (`features.rs`, `constants.py`).

What's **missing**: any notion of **open-endedness / forcing-ness**. In line games
the danger of a window is dominated by how many ways it can be completed and
whether the opponent has an *unstoppable* continuation (open-four = two winning
completions; double-three; 5-with-open-end). A count-4 opp window appears the same
whether it is dead-blockable (one empty cell) or an open-four (forced loss).
`empty_count` (`window_tokens` in `candidates.rs`) partially proxies this but is
not surfaced as a graded "this is forcing" signal, and there is **no count of
distinct opponent winning replies**.

*How this becomes value overconfidence about defense:* the value head must infer
"the opponent has an unstoppable threat" from raw count buckets. Early/mid-game,
counts 3/4 are common and usually *not* losing, so the head learns count-4-is-fine
on average and is **blind to the specific count-4 that is an open-four**. It
predicts +0.8 while a forced loss is one move away.

#### Gap B — Single-hub value readout under-integrates the board (G, strongest architectural cause)

Value comes through **one SIDE token** (`_graph_readout`). The SIDE node is 1 hop
from everything via CONTEXT edges in the GNN, but a *single mean-aggregated* GNN
update over potentially hundreds of CONTEXT neighbors **dilutes** any one decisive
opponent window into the average. The transformer is supposed to fix this by
letting SIDE *attend* to the dangerous window — but SIDE is a **context token**,
updated only by **context self-attention** (`GraphTransformerLayer.forward`: the
`ctx_attn` over {side,stone,window}), so it can attend to window tokens; with 4
heads and 3 layers, reliably and sharply selecting *the one* losing window out of
many is a hard attention problem, and any failure shows up directly as a
mis-calibrated global value.

*How this becomes value overconfidence about defense:* offense is read out
**per-candidate** (the policy head sees each move's own embedding directly), so the
attack signal has a short, dedicated path and the policy works. Defense (a *global*
"I am losing") has only the long path opponent-stone→window→SIDE-attention. When
that path drops the decisive window, value is confidently wrong. This asymmetry
explains why the failure is specifically **defensive** overconfidence.

#### Gap C — Threat propagation may be too shallow (G/S)

Count the hops needed for the value head to register a *developing* (not yet
count-5) opponent threat that spans relationships between **two** opponent windows
(the typical "double threat" loss):

- opp stone → (STONE_WINDOW) → window A
- window A → (CANDIDATE_WINDOW) → shared candidate cell → (CANDIDATE_WINDOW) →
  window B
- window B → (CONTEXT) → SIDE

That is ~3–4 GNN hops *before* the transformer even runs, and the GNN is only
**3 layers** (`DEFAULT_GNN_LAYERS = 3`). The SIDE hub's CONTEXT edge gives a
1-hop shortcut to every node, but only as a **diluted mean**. So the *structured*
two-window relationship that defines many losses can be **truncated** by GNN depth;
the transformer then has to recover it from incompletely-propagated embeddings.
This is grounded (the hop count and layer count are in code) but "is 3 too few?"
is empirical, hence G/S.

#### Gap D — Candidate-centric design loses whole-board context for value (G)

The architecture's richest pathway is the **candidate** pathway (per-candidate
cross-attention, per-candidate policy/opp_policy heads). The CONTEXT/value pathway
is comparatively thin (one hub, no candidate→SIDE write-back). The design is
optimized for "which move is best" (policy) and under-invests in "what is the
global outcome" (value). A *calibrated global value* needs whole-board pooling,
which §1.2 shows the CNN has and hexgt lacks.

#### Gap E — Capacity (width/depth/heads) (S)

At `token_dim=168`, 4 heads, 3+3 layers (~2.07M params, matched to the 96×8 CNN by
design, `constants.py`), raw capacity is comparable to the CNN it ties/beats. So
*global* capacity is probably **not** the bottleneck; the issue is **where** the
capacity is spent (candidate pathway vs value pathway, Gap B/D) and **what features
feed it** (Gap A). Adding width everywhere is the least targeted lever.

#### Gap F — D6-invariance vs spatial bias (S, the user's hypothesis, nuanced)

The user hypothesizes the GNN+transformer "cannot represent the spatial/geometric
structure as well as a CNN." Partially supported: the CNN's *translation
equivariance with a growing receptive field* is a genuinely better inductive bias
for repeating line patterns than a 3-hop GNN with coarse threat features. **But**
the hexgt graph *does* encode geometry losslessly in principle (adjacency + window
membership + invariant hex-distance), and it adds a property the CNN lacks: exact
**D6 invariance by construction**. So the gap is not "geometry is unrepresentable"
— it is "geometry-derived **threat urgency** is under-featurized (Gap A) and
**global value** is under-pooled (Gap B/D)." The fix is therefore better targeted
at readout+features than at wholesale CNN-ification.

---

## 2. Architecture-change proposals, ranked

Conventions: **Cost** = small tweak / moderate / structural. **Drop-in** = whether
the existing checkpoint + pipeline (`expand.py`, `collate.py`, `mcts_eval.rs`,
`trainer.py`) keep working without a schema break. **D6** = effect on the
invariance-by-construction guarantee (`expand.py`, the equivariance test).

### Rank 1 — Stronger whole-board value readout (Gap B, D)

**Change.** Replace the SIDE-only value readout with a **global pooled readout**:
in `_graph_readout`, additionally pool the post-transformer embeddings of all nodes
per graph (segment **mean** and **max** over node_graph; optionally restricted to
{stone, window, candidate} or weighted by node type), concatenate with the SIDE
token, and feed the wider vector to `value_head` (and stvalue heads). Mean captures
"overall balance," max captures "the single most dangerous window," which is
exactly the defensive signal currently diluted.

**Rationale.** Directly fixes Gap B/D: the decisive opponent window no longer has
to win a single-token attention competition to reach the value head; a **max-pool**
surfaces it structurally. This mirrors the CNN's whole-board value pool
(`dense_cnn/architecture.py::ValueBinnedHead`) without abandoning the graph.

**Cost.** Small→moderate. Segment mean/max over `node_graph` is the same primitive
already used (`architecture.py::_graph_slices`, `_padded_index`, and `index_add_`
in `RelationalMessagePassing`). `value_head`'s input width changes
(168 → e.g. 168·3), so the value/stvalue head first `Linear` grows — a **partial**
checkpoint break (trunk + policy load exactly; value heads re-init). That is the
exact same surgery already supported for the stvalue graft
(`_rl_train.py::_validate_stv_resume_load`, `strict=False`), so it is low-risk to
implement; just extend the resume guard to allow `value_head.*` to be missing.

**Validate.** (a) Re-run the forensic probe: value prediction in the 3 plies before
a loss should stop pinning at +0.8 — measure mean predicted value on
known-lost positions (the `example_games` traces in `selfplay.py` already log
`root_value` per move). (b) H2H vs dense_cnn e24 win-rate (the existing
`run_eval`/`run_head_to_head` harness). (c) Value-head calibration: Brier/CE of
predicted vs realized outcome on a held-out self-play epoch.

**Drop-in.** Yes for trunk/policy/search (`forward_policy_value` value path
unchanged except readout); value heads re-init. **D6.** Safe — mean/max segment
pooling is permutation-invariant, so D6-invariance by construction is preserved
(the pooling commutes with node relabeling).

### Rank 2 — Graded threat / open-end features (Gap A)

**Change.** Add D6-invariant features (in `features.rs` + `constants.py`,
reserved slots `[30:32)` plus one or two more if the vector is widened — note the
zero-init layer-expansion path already exists for exactly this,
`architecture.py::zero_init_expanded_feature_columns`, `NEW_FEATURE_SLOTS_V2`):
- **WINDOW node:** `open_ends ∈ {0,1,2}` (one-hot) — whether the window can be
  extended at one/both ends into a longer line (computable from the board scan
  already in `candidates.rs::has_open_window`); and a **"forcing" flag** (count==5
  with ≥1 empty, or count==4 with both ends open).
- **CANDIDATE node:** `n_opp_winning_replies` (how many distinct opponent windows
  this cell *fails to block*) and `creates_double_threat` (own).

**Rationale.** Hands the model the urgency signal it currently must reconstruct,
attacking Gap A at the source. Especially `forcing` and `n_opp_winning_replies`
give the value head a near-linear feature for "I am about to lose."

**Cost.** Moderate. Pure-Rust feature computation (`features.rs`) + matching Python
constants + the byte-parity test (`test_hexgt_feature_buffer.py`, referenced in
`features.rs` header) must be updated together — this is the documented "update
both halves" discipline (`CLAUDE.md`). No architecture change.

**Validate.** Ablate features on/off across a few RL epochs; track defense metric
(value calibration on lost positions) and H2H. The zero-init expansion means a live
checkpoint absorbs the new slots with **no cold start** (per `_rl_train.py` resume
path), so the ablation is cheap to run from the current weights.

**Drop-in.** Yes (zero-init layer-expansion is the designed mechanism). **D6.**
Safe — open-end counts, forcing flags, and winning-reply counts are all preserved
by every D6 element (windows/owners/counts map bijectively; same argument as the v2
features in `constants.py`).

### Rank 3 — Deeper / longer-range threat propagation (Gap C)

**Change (cheapest first):**
- **3a:** Increase `gnn_layers` 3→4 or 5 (`HexgtNetwork.__init__`, default in
  `constants.py`). One extra hop lets the two-window relationship in Gap C complete
  before readout.
- **3b:** Add a **candidate→SIDE (or window→SIDE) write-back** in
  `GraphTransformerLayer`: currently SIDE only self-attends within context; let the
  SIDE/context tokens *also* cross-attend to candidates (a second cross-attn), so
  the value token can pull in the decisive move's refutation. (Closes the
  context←candidate asymmetry noted in §1.3.)

**Rationale.** Targets Gap C/B propagation directly without changing the data path.

**Cost.** 3a small (a hyperparameter; partial checkpoint break — the new GNN layer
re-inits, trunk-below loads). 3b moderate (new attention sublayer; partial break).

**Validate.** Same defense-calibration + H2H as Rank 1; specifically measure on
*double-threat* lost positions whether value detects the loss earlier.

**Drop-in.** Partial (extra layer/sublayer re-inits; resume via `strict=False`
already supported). **D6.** Safe — attention is over per-graph token sets and uses
no positional encoding (positions are D6-variant and absent), so adding
layers/cross-attn keeps permutation/D6 invariance.

### Rank 4 — Spatial / CNN board-encoder hybrid (Gap F, the user's hypothesis)

**Change.** Add a small **CNN board-encoder** (a few `HexConv2d` blocks on a dense
crop, à la dense_cnn) whose per-cell features are **gathered onto STONE/CANDIDATE
nodes** (by cell coordinate) and concatenated to their node features before the
GNN. This injects translation-equivariant, growing-receptive-field line-pattern
recognition into the graph.

**Rationale.** Most directly tests/satisfies the user's "the GNN can't represent
spatial structure like a CNN" hypothesis: it *literally adds the CNN's inductive
bias* as a feature source while keeping the graph's dynamic candidate/value
machinery.

**Cost.** Structural. New dense-crop encoder (needs a board crop + cell→node gather
in `features.rs`/`expand.py`/`mcts_eval.rs`), new params, new throughput profile
(self-play is featurization-bound per the memory notes), and a new
Python/Rust contract to keep byte-identical.

**D6 — the catch.** A square dense crop is **NOT closed under hex D6** (this is a
documented, already-painful failure class: `compact_io.py` D6-coverage guard;
the "96x8 D6 square-crop crash" memory). A CNN board-encoder would **break
D6-invariance-by-construction**, forcing a return to D6 *augmentation* (the very
thing hexgt was designed to avoid, `expand.py` docstring). That is a real
regression in a known-fragile area (the dense_cnn D6 augmentation bug poisoned a
model).

**Validate.** Only worth it if Ranks 1–3 fail to fix defense. If pursued, gate hard
on an equivariance/augmentation-parity test before any training.

**Drop-in.** No (new encoder + data contract). **D6.** Breaks the guarantee —
highest risk. **Recommend deferring.**

### Rank 5 — Width/heads capacity scaling (Gap E)

**Change.** Increase `token_dim`/`attention_heads`/`ffn_dim` (`constants.py`).

**Rationale.** Weakest-targeted: §1.4 Gap E argues capacity is roughly matched and
the problem is *where* capacity is spent. Useful only as a secondary knob *after*
the value readout/features are fixed (more heads help the SIDE token select the
right window — but Rank 1 removes that need).

**Cost/D6/Drop-in.** Moderate; full re-train (most params change shape); D6 safe.
**Lowest priority.**

---

## 3. Policy-surprise (KL) sample weighting — concrete design

### 3.1 Why this attacks the defense failure directly

A defensive blind spot is, definitionally, a position where the **prior** (the raw
policy net) thought the position was fine / a particular move was unnecessary, but
**search** (visits) discovered the refutation and shifted mass to the defensive
move. That is a **high `KL(visits ‖ prior)`** position. Upweighting high-KL samples
makes the network train hardest **exactly on the positions where its prior was most
wrong about danger** — the surprise signal *is* the "you were blind here" signal.
This is the KataGo frequency-weight idea and dense_cnn already uses it
(`dense_cnn/replay.py::materialize_policy_surprise_rows`,
`dense_cnn/README.md` §"Policy surprise"). hexgt has forced playouts
(`_rl_train.py --forced-playout-k 2.0`) but **no surprise weighting**
(`losses.py`, `_rl_train.py`).

### 3.2 The metric

Per sample, surprise `s_i = KL(π_visits ‖ π_prior)` over the candidate set, where
`π_visits` is the (temperature-free) normalized visit distribution and `π_prior`
the normalized **root prior** (post-Dirichlet-noise or pre-noise — prefer
**pre-noise** prior so injected exploration noise is not counted as "surprise";
see §3.5). This is exactly `dense_cnn/replay.py::_policy_kl(sample.policy,
sample.root_prior_policy)`.

### 3.3 Per-sample weight (reuse the proven mapping)

Mirror `materialize_policy_surprise_rows` (`dense_cnn/replay.py:149`):

```
n            = #samples in the game
surprise_tot = Σ s_i
if surprise_tot > 0:
    w_i = min(max_weight, uniform_fraction + (1 - uniform_fraction) * n * s_i / surprise_tot)
else:
    w_i = 1.0
```

with `uniform_fraction = 0.5`, `max_weight = 8.0` (the dense_cnn defaults,
`dense_cnn/config.py:87`). The uniform floor guarantees every position is still
trained; the cap bounds the variance from one freak-surprise position. Per-game
normalization keeps weights comparable across games of different length.

### 3.4 WHERE it plugs in — and the blocking data-path gap (cite the functions)

There are **two** viable integration points. The honest finding is that the clean
one is blocked by a persistence gap:

**The gap.** hexgt self-play has the prior in hand — `selfplay.py` captures
`prior_pairs = list(search.root_prior_policy)` and passes it to
`sample_from_state(..., root_prior_policy=prior_pairs)`. **But** the shard writer
it reuses, `dense_cnn.compact_io.write_compact_shard`, **does not persist the root
prior**, and the reader `read_compact_shard` explicitly restores
`root_prior_policy=()` (see `compact_io.py` docstring: *"The root prior policy,
policy_surprise and frequency_weight are intentionally dropped"* and the reader's
`root_prior_policy=(),` line). dense_cnn gets away with this because it
**materializes surprise rows BEFORE writing** (`dense_cnn/selfplay.py:324`
`materialize_policy_surprise_rows(...)`). hexgt's `selfplay.py` does **not** call
that function, so its priors are simply lost at shard write.

**Option A — duplicate at self-play write (mirror dense_cnn, recommended).**
In `selfplay.py::run_selfplay_games`, after `finalize_game_samples(...)` and
before `write_compact_shard(...)`, call the dense_cnn
`materialize_policy_surprise_rows(finalized, seed=..., uniform_fraction=...,
max_weight=...)`. Since the finalized samples still carry `root_prior_policy`
(set by `sample_from_state`), the KL is computable there, and duplication needs no
schema change (it just writes more/fewer rows). This is the **lowest-risk** path
because it reuses dense_cnn's tested, proven implementation verbatim and keeps the
compact schema untouched.
- *Caveat:* row **duplication** inflates shard size and interacts with the
  recency-weighted replay sampler in `_rl_train.py::build_replay_window` /
  `epoch_positions` (which counts `searched_positions` / `num_rows`). Duplicated
  rows would over-count positions in the pool-cap accounting. Mitigate by counting
  *raw* (pre-duplication) rows for `epoch_positions` (dense_cnn already separates
  `raw_rows` vs `effective_rows` in the write result), or prefer Option B.

**Option B — loss reweighting (no duplication; needs prior persisted).**
Carry a **per-sample scalar weight** (or the prior, to compute KL at expand time)
through the pipeline and multiply it into the loss:
1. Persist the prior **or** a precomputed `frequency_weight` in the shard
   (`compact_io.write_compact_shard` — add a `pol_prior_*`/`freq_weight` column;
   small schema add, bump `COMPACT_SCHEMA_VERSION`). Note this is a **shared**
   dense_cnn file used by the live run, so prefer an *additive, optional* column
   read defensively (absent ⇒ weight 1.0) to avoid disturbing dense_cnn.
2. Thread the weight through `expand.py::build_training_batch` into `targets`
   (alongside `policy`, `value`), as a `(G,)` per-graph weight.
3. In `losses.py::hexgt_loss`, multiply each graph's **policy** (and optionally
   **value**) per-graph CE term by `w_g` before the mean. Concretely:
   `segment_softmax_cross_entropy` currently returns `per_segment[positive].sum()
   / denom`; add an optional `segment_weight` so it returns the **weighted** mean
   `Σ w_g·CE_g / Σ w_g`. `binned_value_loss` already supports a `mask`; generalize
   that mask to a float weight (it already does `(per_item*mask).sum()/mask.sum()`
   — passing the surprise weight as the "mask" gives weighted value loss for free).
4. `trainer.py::optimizer_step` passes `targets` straight to `hexgt_loss`, so no
   trainer change beyond plumbing the weight key.

Option B is cleaner (no data inflation, exact weighting, no replay-count
distortion) but touches the **shared** compact schema and the loss functions.
Option A is faster to land and reuses proven code. **Recommendation: start with
Option A** (validate the hypothesis with minimal new code), move to **Option B** if
duplication variance or replay-count distortion proves problematic.

### 3.5 Interactions

- **Recency weighting (0.9/epoch).** Independent and complementary:
  `_rl_train.py::epoch_recency_weight` weights **whole epochs** at the *sampling*
  level (which shards to draw), while surprise weights **individual positions**
  within a shard. They compose multiplicatively (recent + surprising = trained
  most). No conflict; just ensure `epoch_positions` counts raw rows if Option A is
  used (see §3.4 caveat).
- **STV heads.** Apply the surprise weight to **policy** and the **main value**
  head; be cautious applying it to the **short-term-value** heads (`STV_HORIZONS =
  (4,12,24)`), whose targets are EMA look-aheads (`_rl_train.py` comment) and whose
  weight is already deliberately tuned small (`--short-term-value-weight 0.10`).
  Simplest: weight policy+value, leave STV at its mask-only behavior. (In Option A
  duplication, all heads get duplicated together — acceptable, but worth watching
  the STV vs main balance the run already tunes.)
- **Forced playouts (k=2.0).** Forced playouts *inflate* visits on under-explored
  moves, which **changes `π_visits`** and therefore the KL. This is fine and even
  synergistic (forced playouts surface refutations → higher surprise on exactly the
  defensive moves), but it means the prior used for KL should be the **pre-noise,
  pre-forced** root prior (`search.root_prior_policy` as captured in `selfplay.py`)
  so surprise measures genuine prior-vs-search disagreement, not the noise/forcing
  you injected.

### 3.6 Validation

- **Defense calibration** (primary): on held-out self-play, value-head CE/Brier vs
  realized outcome, sliced to the last K plies before terminal — surprise weighting
  should reduce the +0.8-before-loss overconfidence.
- **H2H** vs dense_cnn e24 and (if available) SealBot via the existing
  `run_eval`/`run_head_to_head`.
- **Surprise telemetry**: log `policy_surprise_mean` / `frequency_weight_mean`
  (dense_cnn already returns these from `materialize_policy_surprise_rows`) per
  epoch to confirm the weighting is active and not saturating the `max_weight` cap.

---

## 4. Synthesized recommendation

The reported failure is **value-head overconfidence about defense**, and the
code points at two structural causes (single-hub value readout, §1.4 Gap B/D; coarse
threat features, §1.4 Gap A) plus a training-signal cause (no surprise weighting,
§3). The user's "GNN can't do spatial structure" hypothesis is **partly** right but
the targeted root causes are readout + features, not the graph paradigm itself —
so the highest-leverage/lowest-risk first moves keep the graph and avoid the
D6-breaking CNN hybrid.

**Do these three first, in this order:**

1. **Rank 1 — global pooled value readout (architecture).** Highest leverage,
   small/moderate cost, **D6-safe**, drop-in for search, reuses the existing
   `strict=False` partial-load surgery. Most directly removes the single-token
   bottleneck that produces *global* value blindness. *Well-grounded.*

2. **Rank 2 (§2) graded threat features.** Cheap (Rust feature + parity test),
   **D6-safe**, and — crucially — absorbed by a **live checkpoint with no cold
   start** via the existing zero-init layer-expansion (`_rl_train.py` resume,
   `zero_init_expanded_feature_columns`). Hands the value head the urgency signal it
   currently lacks. *Well-grounded that the feature is missing; empirical that it
   fixes the metric.*

3. **Policy-surprise weighting (§3), via Option A** (mirror dense_cnn's
   `materialize_policy_surprise_rows` at self-play write). Directly targets the
   defensive blind spots (high-KL positions = "you were blind here"). *Mechanism
   well-grounded; note the prior-persistence gap (§3.4) is the one real
   implementation cost.*

**Suggested validation order.** Establish a **defense-calibration metric** first
(value CE/Brier on the last K plies before a loss, from the `root_value` traces the
self-play loop already records). Then land items in cheapest-to-validate order:
(3) surprise weighting Option A (no architecture change, fastest A/B), then (2)
features (zero-init resume, no cold start), then (1) value readout (partial
checkpoint surgery). Judge each by the defense metric **and** H2H — train-loss is
an unreliable judge here (a known measurement-artifact pitfall in this project:
the dense_cnn "rising loss" was an artifact; judge by eval).

**Deferred / honest caveats.**
- **Rank 4 (CNN board-encoder)** is the most complete answer to the user's literal
  hypothesis but is **structural, throughput-affecting, and breaks
  D6-invariance-by-construction** (square crop not D6-closed — a documented,
  already-painful failure class). Pursue **only** if (1)–(3) fail to move defense,
  and gate it behind an equivariance/augmentation-parity test.
- **Rank 3 (deeper GNN / context←candidate cross-attn)** is a good cheap follow-up
  if (1)+(2) help but double-threat detection still lags — it specifically targets
  the multi-window propagation depth (Gap C), which is grounded in the hop-count vs
  3-layer analysis but empirically uncertain.
- **Rank 5 (width/heads)** is the least-targeted lever; defer until readout and
  features are fixed.
- All "fixes the metric" claims are **hypotheses to validate**; only the *presence*
  of each gap (hub readout, coarse features, missing surprise weighting, prior not
  persisted) is established directly from the code cited above.

---

## 5. "Hot tokens" — explicit representation of immediately-decisive cells

This section explores the idea of giving hexgt a **dedicated, explicit
representation for cells that are decisive *this move or next*** — "hot" cells —
so that a win or a must-block can never be lost in the diffuse path to the value
head. It builds directly on §1.4 Gap A (threat severity too coarse) and Gap B/D
(single-hub value readout under-integrates the board). Everything below is grounded
in the Rust window walk that already exists in
`candidates.rs`/`features.rs` (hexgt worktree) and the engine window API
(`hexo_engine/rust/src/tactics.rs`).

### 5.1 Definition & taxonomy of "hot" cells

Hexo is won by completing a **length-6 line** (`candidates.rs` header,
`tactics.rs::WINDOW_LEN`; a window is `is_win_for(player)` when
`count(player) == 6`). A window is **active** for exactly one player when it holds
that player's stones and none of the opponent's (`tactics.rs::active_player`,
`is_active`); `empty_mask`/`empty_cells` and `count(player)` are already computed
per window. "Hot" is defined entirely in terms of these length-6 windows, so it is a
pure function of board geometry (important for D6 — see §5.5).

A candidate cell `c` (every hot cell is necessarily an empty candidate — it is a
*move*) is classified as:

- **(a) Immediate-win cell (OWN-hot, tier W).** Playing `c` completes a six. In
  window terms: there exists an active OWN window through `c` with `count(own) == 5`
  and `c` is the single empty cell (`empty_mask.count_ones() == 1`). This is *already
  partially surfaced*: `features.rs` sets `F_CAND_COMPLETE_OWN` when a count-5 own
  window routes to the candidate (`F_CAND_COMPLETE_OWN = 1.0` at
  `features.rs` in the `cnt == 5 && own == 0` branch). So tier-W own-hot is the one
  hot class hexgt *can* already see — and, per §1.4, offense is read out
  per-candidate and works. The symmetric opponent case `F_CAND_COMPLETE_OPP`
  (count-5 opp window through `c`) is the **must-block-now** cell.

- **(b) Forced-win-next cell (OWN-hot, tier F).** Playing `c` does not win this
  move but creates a position where OWN has an **unstoppable** continuation —
  classically an **open-ended four**: a count-4 line with an empty completing cell
  at *both* ends, so the opponent cannot block both sixes. Equivalently, after
  playing `c` there exist **two distinct empty cells each of which completes a six
  for OWN** (a double-four / four-with-two-ways, or two separate count-5 windows
  sharing no blocker). This is the "forcing" notion §2 Rank 2 already proposes as an
  `open_ends`/`forcing` feature; "hot tokens" makes it a *named, discrete* class
  rather than a continuous severity scalar.

- **(c) Must-block cell (OPPONENT-hot).** The OPPONENT wins *next turn* if we do not
  play `c` now. Two sub-cases mirroring (a)/(b) from the opponent's side:
  - **opp immediate threat:** an active OPP window with `count(opp) == 5` whose lone
    empty cell is `c` — if we don't take `c` (or otherwise kill the window), the
    opponent completes the six next move. (This is exactly the cell tagged by
    `F_CAND_COMPLETE_OPP` today.)
  - **opp forced four:** the opponent has an open-ended four (or double-four) and
    `c` is one of the cells that, if unaddressed, lets them complete an
    un-blockable six. In the open-four case *no single `c` saves us* — that is a
    lost position, and the right behavior is for the **value head** to know it (the
    8/8 SealBot loss), not for the policy to find a phantom defense.

**OWN-hot vs OPPONENT-hot is the crucial axis.** OWN-hot cells are *winning* moves;
the policy head already handles those well via the per-candidate path (§1.4 Gap D).
**OPPONENT-hot cells ARE the defensive-blindness signal** the whole doc is chasing:
they are precisely the positions where "I am about to lose" must reach the value
head, and §1.4 Gap B/D shows that path is the thin one. So the high-value half of
"hot tokens" is the **opponent-hot** half, and it should be wired toward the *value
readout*, not (only) the policy head.

### 5.2 Representation options in hexgt (trade-offs)

#### (i) Extra per-candidate node features — *cheapest, rides the existing graft path*

Add 1–2 D6-invariant scalars on CANDIDATE nodes in the **reserved slots `[30:32)`**
(`constants.py` line "`# [30:32) reserved.`"; `constants.rs` has the matching
`NODE_FEATURE_DIM = 32` with slots filled through index 29):
- `F_CAND_OPP_HOT` ∈ {0,1} (or graded {0, .5, 1}): `c` is opponent-hot — an opp
  count-5 window has its lone empty at `c` (immediate must-block) or `c` is on an
  opp open-four.
- `F_CAND_OWN_HOT`: `c` is own-hot tier-W or tier-F.

These are written in the **same loop** that already builds candidate tactical
features from candidate↔window edges (`features.rs`, the
`FT_EDGE_CANDIDATE_WINDOW` walk that sets `F_CAND_COMPLETE_OPP`/`F_CAND_NWIN_OPP`).
The tier-W/immediate cases need *no new pass* — they are a stricter predicate on the
windows already visited there (count==5 ∧ `empty_mask.count_ones()==1`). Tier-F
(open-four) needs the small extra check in §5.4.

*Trade-off:* this is the lowest-risk option and **does not by itself fix the readout
bottleneck** — an opponent-hot *candidate* feature still has to propagate
candidate → window/context → SIDE to influence value, the very path §1.4 Gap B says
is thin. It strengthens the *policy's* must-block reflex and gives the value head a
sharper input, but the value-side guarantee comes from (ii)/(iii).

**Where it plugs in:** `features.rs` (write the two slots), `constants.rs` +
`constants.py` (name slots 30/31, extend `NEW_FEATURE_SLOTS_V2` → a `_V3` tuple,
bump `FEATURE_SCHEMA_VERSION` 2→3), and the byte-parity test. No `architecture.py`
change. Absorbed by the live checkpoint via
`zero_init_expanded_feature_columns` (it zeroes exactly the new `node_in` input
columns; the function already takes the slot tuple as an argument).

#### (ii) A dedicated "hot" token TYPE in the graph — *targets the readout, schema change*

Add a fifth node type (e.g. `NODE_TYPE_HOT = 4`, bumping `NUM_NODE_TYPES` 4→5 in
both `constants.py`/`constants.rs` and widening the `F_TYPE_ONEHOT` block). A hot
token is materialized per decisive cell, carrying owner (own/opp) and tier
(W/F) features, and is connected by CONTEXT edges to the SIDE hub (the
`EDGE_CONTEXT` hub-to-all wiring in `candidates.rs::build_graph` already adds
`side ↔ every node`, so a new node type is automatically context-linked) and by
ADJACENCY/CANDIDATE_WINDOW edges to its cell and windows.

Because hot tokens are **context tokens** (not candidates), they enter the
transformer's `ctx_attn` set (`architecture.py::build_attention_layout` puts every
non-candidate node into `ctx_index`), so the SIDE hub can self-attend to them in
`GraphTransformerLayer.forward`. This *raises the prior* that the decisive window is
selected — there are now few, salient hot tokens competing for SIDE's attention
instead of one window buried among many count-3/4 windows.

*Trade-off:* a new node type is a **schema + arch change** (`NUM_NODE_TYPES`,
`F_TYPE_ONEHOT` width, the `node_in` projection's *expectation* of the type one-hot,
the parity test, and `mcts_eval.rs`/`expand.py` graph construction). The `node_in`
input width stays 32, so the projection matrix does **not** change shape — the new
type just activates a one-hot column that was previously always 0 for index 3, plus
a *new* index 4 which **does not exist** in a 4-wide one-hot. That means
`F_TYPE_ONEHOT` must widen from `[0:4)` to `[0:5)`, shifting every downstream slot —
a full feature-layout reshuffle, i.e. **not** a clean zero-init graft. Higher cost
than (i), and it still relies on attention *selecting* the hot token (softer
guarantee than (iii)).

#### (iii) Attention bias forcing the value readout onto opponent-hot tokens — *strongest value-side guarantee*

This is the option that ties most directly to the §1.4 Gap B finding. Two concrete
sub-forms:

- **(iii-a) Hot-biased value pooling (combine with Rank 1).** Rank 1 already
  proposes replacing the SIDE-only `_graph_readout` with global mean+max pooling.
  Extend it so the readout **additionally pools over the opponent-hot token/feature
  set specifically** — e.g. a third pooled vector `max over {nodes flagged
  opponent-hot}` concatenated into the value head input. A max over the opponent-hot
  set is, by construction, "the single most dangerous immediate threat," which is
  exactly the defensive quantity §1.4 says is diluted. If there are no opponent-hot
  cells the pool is a zero vector (safe). This is a **few lines in `_graph_readout`**
  (it already has `node_type`, `node_graph`, and `node_emb`; add a boolean mask from
  either the hot node type (ii) or a threshold on the opponent-hot feature column
  (i), then a segment max). It is the cheapest way to get a *structural* guarantee.

- **(iii-b) Attention bias into the SIDE/context self-attention.** Add an additive
  attention bias (or a dedicated extra attention head) in
  `GraphTransformerLayer.ctx_attn` so the SIDE query receives a positive bias on
  keys that are opponent-hot. `nn.MultiheadAttention` as currently used
  (`architecture.py`, `key_padding_mask=layout.ctx_pad`) does not expose a per-key
  additive bias directly; this needs either an `attn_mask` carrying `+β` on
  (side-query, hot-key) pairs or a swap to a hand-rolled attention that adds a bias
  term. Heavier than (iii-a) and partially redundant with it.

**Why (iii) is the point.** §1.4 Gap B: "a must-block threat must win a single-token
attention competition to reach the value head." (iii-a) removes the competition: a
**max-pool over opponent-hot tokens is a guaranteed, gradient-friendly channel** from
the decisive cell straight into the value head input, independent of whether the
transformer's 4 heads happened to select it. This is the most defensively-targeted
intervention in the whole doc.

### 5.3 Interaction with the three existing recommendations

| | Rank 1 (global pooled value readout) | Rank 2 (graded threat features) | §3 (policy-surprise KL weighting) |
|---|---|---|---|
| **Hot tokens (i) features** | independent; feeds richer inputs into whatever readout Rank 1 builds | **sharper special case** of Rank 2 — a discrete decisive subset of the same continuous severity signal | complementary; opponent-hot positions are *exactly* the high-KL positions §3 upweights |
| **Hot tokens (ii) type** | enables a hot-specific readout in Rank 1 | superset wiring of Rank 2's features onto a token | complementary |
| **Hot tokens (iii) readout bias** | **a more targeted version of Rank 1** (pool the decisive subset, not just everything) | uses Rank 2/§5.1's detection as the mask | complementary |

Concretely:

- **vs Rank 2 (graded threat features).** Hot tokens are a **sharper, decisive,
  defensively-targeted subset** of graded threat features, not a substitute. Rank 2
  is *continuous and board-wide* (every window gets an open-ends/forcing severity);
  hot tokens are a *discrete, binary, "this loses now" subset* explicitly wired to
  the readout (iii). Overlap is real: Rank 2's `forcing` flag (count==5, or count==4
  both ends open) **is** the own-hot/opp-hot predicate. The difference is *where it
  goes*: Rank 2 drops it into the candidate/window feature soup; hot tokens (iii)
  route the *opponent* half into the value pool. **Recommendation: implement them
  together** — compute the forcing/open-end predicate once (§5.4), expose the graded
  version on windows (Rank 2) *and* the binary opponent-hot version wired to the
  value readout (hot tokens iii). Doing hot tokens reduces the *marginal* value of
  the opponent-side of Rank 2 (the readout wiring is the part that matters for
  defense), but Rank 2's own-side and graded mid-game severity still add signal.

- **vs Rank 1 (global pooled value readout).** Form (iii-a) **is** Rank 1 with a
  decisive-subset pool added. They should be built in the same `_graph_readout`
  edit. Hot tokens make Rank 1 strictly more targeted at almost zero extra cost.

- **vs §3 (policy-surprise weighting).** Fully complementary and mutually
  reinforcing: opponent-hot positions where the prior under-weighted the must-block
  are *by construction* the high-`KL(visits‖prior)` positions §3 upweights. Hot
  tokens fix the *representation* (the net can see the threat); surprise weighting
  fixes the *training emphasis* (the net trains hardest on the threats it missed).
  Neither removes the need for the other.

**Where it ranks.** Among {Rank 1, Rank 2, §3, Rank 4 (CNN hybrid)}: hot tokens in
form **(i)+(iii-a)** rank **just below Rank 1 and above Rank 4**, and arguably *as
part of* Rank 1 (since iii-a is an extension of it). It is more targeted than Rank 2
for the *defensive* failure specifically, and far cheaper / lower-risk than the
D6-breaking CNN hybrid. It does **not** reduce the need for §3 (orthogonal training
signal) and only partially reduces Rank 2 (its opponent-side severity).

### 5.4 Computation in Rust (cost)

All detection lives in the window walk already present. The engine gives, per
length-6 window: `count(player)`, `empty_mask` / `empty_cells`, `is_win_for`,
`active_player` (`tactics.rs:119-208`), and `WindowKey::{intersects, touches,
contains}` for relating two windows (`tactics.rs:80-107`).

- **Immediate hot (tiers W and opp-immediate) — zero extra passes.** In the existing
  `features.rs` candidate↔window loop (the `FT_EDGE_CANDIDATE_WINDOW` branch that
  already reads `node_wcount`), a candidate is **own-immediate-hot** iff it routes to
  an own window with `count == 5` and that window's `empty_count == 1` (one blocker —
  this cell). `empty_count` is already on the window node (`node_wempty`, set from
  `entry.empty_mask().count_ones()` in `candidates.rs::window_tokens`). The opponent
  case is the same with `own == 1`. So opp-immediate-hot is computable **in the loop
  that already exists**, reading fields already materialized — *no new board scan*.

- **Forced-four (tier F) — one cheap window-pair check.** "Open-ended four" =
  a count-4 active window with empties at *both* extension cells, OR two distinct
  count-5 windows for the same player whose blockers differ (double-four). Detecting
  it needs, for each count-4/5 window, a look at whether its empties extend the line
  on both sides. The board scan in `candidates.rs::has_open_window` already walks the
  six windows through a cell along each axis (`tactics.rs` axis vectors); the
  open-end test is the same kind of axis walk one cell beyond the window ends. Cost
  is **O(active windows × 6)** — the same order as the window enumeration already
  performed in `window_tokens`, run once per position. Double-four via
  `WindowKey::intersects`/`touches` is O(windows²) over the *active* windows only
  (typically a handful), which is negligible.

- **Net cost.** The immediate tier is free (rides the existing loop). The forced tier
  adds one bounded pass proportional to the active-window count, which is already
  walked. This matches the doc's standing claim (§2 Rank 2) that open-end/forcing
  features are "cheap, in `features.rs`."

### 5.5 D6-safety, drop-in checkpoint compatibility, validation — per option

**D6-safety.** Hot-ness is a pure function of board geometry: a cell is hot iff a
length-6 window through it satisfies a count/empty predicate, and **D6 maps windows
to windows, owners to owners, counts to counts, and the cell to its image**
bijectively (the same argument `constants.py` makes for the v2 window-count
features). A hot cell therefore maps to a hot cell of the same owner/tier under every
D6 element. So:
- **(i) features** — D6-invariant scalars on candidate nodes; preserved exactly,
  same as the existing `F_CAND_*` family. **Safe.**
- **(ii) hot token type** — node types are permutation-equivariant labels; D6
  relabels which cell is hot but the *set* of hot tokens is preserved, and the GNN +
  per-graph attention are permutation-equivariant (`architecture.py` docstring;
  no positional encoding). **Safe** (the *type-onehot reshuffle* is a schema cost,
  not a D6 cost).
- **(iii) readout bias / pool** — segment max/mean over the opponent-hot mask is
  permutation-invariant and the mask is D6-invariant, so pooling **commutes with D6
  relabeling** (same argument the doc makes for Rank 1's mean/max pool). An additive
  attention bias keyed on the D6-invariant hot flag is likewise invariant. **Safe.**

All three preserve the **D6-invariance-by-construction** guarantee — unlike the CNN
hybrid (Rank 4), which breaks it.

**Drop-in onto the live epoch-42 checkpoint.**
- **(i) features:** clean **zero-init graft, no cold start** — exactly the
  `zero_init_expanded_feature_columns` path (`architecture.py:310-337`). Put the two
  new flags in reserved slots `[30:32)`, add them to a `NEW_FEATURE_SLOTS_V3` tuple,
  bump `FEATURE_SCHEMA_VERSION` 2→3, and the RL resume zeroes those `node_in` columns
  once so the post-resume forward is byte-identical to e42, then learns them. The
  mechanism is already proven for v2 (the six `F_CAND_*_WIN{3,4,5}` + dist/second
  slots).
- **(iii-a) readout pool:** **partial checkpoint break, same surgery as Rank 1** —
  the value/stvalue head's first `Linear` input width grows (token_dim → token_dim·k),
  so those heads re-init while trunk+policy load exactly via the existing
  `strict=False` resume (`_rl_train.py::_validate_stv_resume_load`, per §2 Rank 1).
  No cold start for the *body*; only the value heads warm up. If built *with* Rank 1,
  it is the *same* head surgery done once.
- **(ii) hot token type:** **not a clean graft** — widening `F_TYPE_ONEHOT` to `[0:5)`
  shifts the downstream feature layout, changing what every `node_in` column means,
  so the input projection must be re-mapped (or the layout re-versioned with a column
  permutation). This is the only option that disturbs trained `node_in` weights.
  Prefer (i)+(iii-a) for the live run; reserve (ii) for a from-scratch Model-2.x.

**Validation (the defensive metric this is for).** Reuse the doc's standing
defense-calibration harness:
1. **Opponent-hot calibration.** Slice held-out self-play to positions where ≥1
   opponent-hot cell exists (cheap to label with the same detector). On those
   positions, measure value-head CE/Brier vs realized outcome and the rate of
   "confidently-safe-then-lost" (`root_value > +0.5` within K plies of a loss, from
   the `root_value` traces `selfplay.py` already logs). Hot tokens should sharply cut
   the over-confidence on this slice specifically — that is the targeted claim.
2. **The 8/8-lost trace.** Re-run the forensic probe (§2 Rank 1 validation): on the
   known SealBot loss games, confirm the value prediction in the final plies drops
   toward the loss instead of pinning near +0.8 once opponent-hot pooling is wired.
3. **H2H** vs dense_cnn e24 (and SealBot if a path is available) via the existing
   `run_eval`/`run_head_to_head` — the integrative judge, since train-loss is an
   unreliable judge in this project (the documented "rising loss" artifact).
4. **Detector parity / D6 test.** Because hot-ness feeds the model, gate it on the
   existing Rust↔Python byte-parity test (`tests/test_hexgt_feature_buffer.py`) and
   the equivariance test (a hot cell's image is hot under all 12 D6 elements) before
   any training — the same discipline that catches the D6-augmentation poisoning
   failure class.

### 5.6 Section verdict

"Hot tokens" is **worth doing**, in the form **(i) two reserved-slot opponent-hot /
own-hot candidate features (zero-init graft) + (iii-a) an opponent-hot max-pool added
to the Rank 1 global value readout**. That pairing is the most *defensively targeted*
intervention available: it is the concentrated, decisive core of Rank 2's threat
features (§5.3) routed structurally into the value head (the Rank 1 fix to Gap B),
computed for free in the Rust window walk (§5.4), D6-safe and live-graftable (§5.5).
It ranks **just below / folded into Rank 1**, above Rank 2 for the defensive failure
specifically, and far above the deferred CNN hybrid (Rank 4). It does **not** displace
§3 policy-surprise weighting (orthogonal training-signal lever). Skip the dedicated
hot **node type** (ii) for the live run — its feature-layout reshuffle forfeits the
clean zero-init graft for no gain over (i)+(iii-a).

---

## 6. Related work / grounding (external prior art, reconciled)

This section folds four external prior-art items into the recommendations above and
reconciles each against the existing ranking — where it plugs into hexgt code (cited
by file path in the worktree `E:\Hexo-BotTrainer-hexgt`), its D6-safety, rough cost,
and whether it changes the ranking. **The net effect:** two externally-*validated*
value-OUTPUT fixes (soft-Z targets §6.1, ownership aux head §6.2) move **up** to share
the top tier with the Rank 1 readout fix, because they attack the failure
(value miscalibration / no spatial localization) more directly and at lower risk than
any plumbing change; the "hot tokens" §5 idea gains a name and a domain pedigree
(threat-space search §6.3); and the deferred CNN hybrid (Rank 4) is upgraded from
"last resort" to "deferred-but-promising" with a concrete D6-respecting path (§6.4).

### 6.0 The current hexgt value-target baseline (what these fixes change)

Grounding first, so the reconciliations are concrete. hexgt forms its **main value
target as the hard game outcome**: `samples.py::finalize_game_samples` sets
`value=_winner_value(winner, player)` (= +1 / −1 / 0,
`dense_cnn/.../samples.py:318`), and `losses.py::binned_value_loss` trains the 65-bin
value head against that scalar (soft-binned, but the *scalar* is the raw outcome). The
**only** bootstrapped-value signal today is the auxiliary **short-term-value (STV)
heads** (`STV_HORIZONS=(4,12,24)`, `_rl_train.py:48`): `_short_term_value_targets`
(`samples.py:337`) builds, per horizon `h`, an **EMA of future root MCTS values** with
decay `λ=h/(h+1)` — a perspective-corrected look-ahead of the *search's* value
estimate. They are weighted small (`--short-term-value-weight 0.10/0.25`) and read,
like the main value, from the SIDE hub via `_graph_readout` (`architecture.py:256`).
Critically, **dense_cnn has neither a soft value target nor an ownership head** either
(`dense_cnn/architecture.py` heads = policy / value-binned / opp-policy / STV only; its
main value target is also hard `_winner_value`). So §6.1 and §6.2 are *new to both
models*, not hexgt catch-up.

### 6.1 Soft value targets — soft-Z / A0C / A0GB (10.1007/s00521-021-05928-5)

**Summary.** "Value targets in off-policy AlphaZero" (Willemsen, Baier, Kaisers; NCAA
2021) studies replacing AlphaZero's hard final-outcome value target `z=±1` with targets
that blend the outcome and the **bootstrapped MCTS search value** `Q`. Variants include
**A0C** (train toward the search value), **soft-Z** (a convex blend
`(1−γ)·z + γ·Q`-style target), and **A0GB** (a greedy-backup value computed from the
tree). On Connect-Four and Breakthrough these targets **trained faster and produced
stronger play** than vanilla hard-`z` AlphaZero — because the final outcome is a
high-variance, off-policy-noisy label (the rest of the game was played by a weaker/older
policy with exploration noise), whereas the MCTS value at the position is a
lower-variance, on-position estimate. This is the **validated** form of value
recalibration.

**Why it targets the exact failure.** The reported hexgt failure is *value
miscalibration* — `+0.8` right before an 8/8 loss. A hard `z` target says "this position
was a win" only because the game was *eventually* won/lost many plies later, under a
different (noisier) policy; it teaches the value head almost nothing locally calibrated
about *this* position's danger. A soft-Z target pulls the label toward the search's own
`Q` at that node, which already integrated the refutation MCTS found — directly
recalibrating the number the value head is asked to reproduce. This is a *target* fix,
complementary to Rank 1's *readout* fix and §3's *emphasis* fix.

**Where it plugs in (cite).** Cleanly, in **one place**: the main-value assignment in
`samples.py::finalize_game_samples` (line 198, `value=_winner_value(winner, player)`).
hexgt already carries the needed bootstrapped value per decision — `pending` /
`decisions` is the `(player, sample, root_value)` triple (`samples.py:168-185`), and
`root_value` is the MCTS root estimate from that position (the same field
`_short_term_value_targets` consumes at `samples.py:354-356`, already
perspective-corrected). So soft-Z is literally:
`value = (1−γ)·_winner_value(winner, player) + γ·root_value_in_player_perspective`,
with `γ` a config scalar (~0.3–0.5 per the paper's regime). **No new data plumbed, no
schema change, no new search output** — `root_value` is already in hand at finalize
time. `losses.py::binned_value_loss` is unchanged (the soft scalar still goes through
`scalar_to_binned_target`, which already produces a two-bin soft target from any scalar
in `[−1,1]`).

**Interaction with the STV heads.** Soft-Z and STV are **the same idea applied to
different heads**, and they compose cleanly: STV already trains *auxiliary* heads on
EMA-of-future-`root_value`; soft-Z applies a *single-step* bootstrap (this position's own
`root_value`) to the **main** value head. They do not conflict — soft-Z is, in effect,
"give the main value head a horizon-0 bootstrap too." One caution: if `γ` is large and
the early-RL `root_value` is itself badly miscalibrated (the very disease), soft-Z could
slow the correction; mitigate by **annealing `γ` low→moderate** as the value head
calibrates, or keep `γ≈0.3` so the hard outcome still anchors. Because the STV machinery
proves the `root_value` field is reliable enough to learn from, the risk is low.

**D6 / cost / drop-in.** **D6-safe** (value targets are scalars; D6 acts only on board
geometry, not on the outcome/`Q` scalar). **Cost: trivial** — a few lines in one
function plus a config scalar; no Rust change, no parity test, no head reshape.
**Drop-in onto e42: yes, fully** — it changes only the *target* a future epoch trains
toward; weights load identically, no graft, no cold start. The forensic
"value-before-loss" probe (§2 Rank 1 validation, the `root_value` traces in
`selfplay.py`) measures it directly.

**Ranking effect.** **Promotes to share Rank 1 (as "Rank 0a").** It is the
*highest-leverage-to-risk single change in the whole document*: it attacks value
miscalibration at the source (the label), is externally validated, costs a few lines,
breaks nothing, and grafts onto the live checkpoint with no cold start. It should be the
**first thing tried**, ahead of even the readout surgery (which is a partial checkpoint
break). It does not displace Rank 1 — a recalibrated target still benefits from a readout
that can actually *see* the threat — but it is at least co-equal and cheaper to land.

### 6.2 KataGo auxiliary subcomponent targets — ownership + score (arXiv 1902.10565)

**Summary.** KataGo (Wu 2019) augments the AlphaZero value head with **auxiliary
prediction targets**: per-point **ownership** (which player ultimately controls each
board point) and **final score** (plus score distribution). These are pure auxiliary
losses — they do not change how moves are chosen — yet **ablating them measurably slowed
learning**. The mechanism: forcing the trunk to predict *where* the board is won/lost
regularizes the shared representation and **localizes** the value signal spatially,
rather than letting a single global value head collapse everything to one scalar.

**Why it targets the exact failure.** hexgt's defensive blindness is, per §1.4 Gap B/D,
a *lack of spatial localization in the value path*: the trunk is never asked *where* the
danger is, only "what is the global outcome," and that global judgement bottlenecks
through one SIDE token. An ownership head forces every cell-node embedding to carry "is
this region going to be controlled by me or the opponent" — exactly the localized,
board-wide control signal whose absence lets the value head ignore the decisive opponent
region. It is the **structural/regularization** analog of soft-Z's *target* fix.

**Where it attaches (cite).** Onto the **per-cell node embeddings**, parallel to the
policy head — **not** the graph readout. In `architecture.py::_heads` the policy head
already reads per-candidate embeddings (`cand_emb = node_emb.index_select(0,
candidate_index)`, `architecture.py:269-273`). An ownership head is a sibling
`nn.Linear(token_dim, 1|3)` applied to the **STONE ∪ CANDIDATE** node rows (node types 1
and 2, `constants.py:21-22`), emitting a per-cell control logit (own / opp / neutral, or
a signed scalar). The target: terminal board control **back-propagated to every
training position** — at finalize time (`samples.py::finalize_game_samples`, the same
place soft-Z plugs in), the terminal stone ownership of each cell that *exists as a node*
in that position is known from the engine's final state; for non-terminal cells one can
use a softened/zeroed target or restrict the loss to stones+occupied-line cells (KataGo
uses a discounted/softened ownership for unsettled points). The loss is a per-node
segmented CE/BCE, mirroring `losses.py::segment_softmax_cross_entropy`'s per-graph
segmenting over `candidate_graph` — here segmented over the STONE/CANDIDATE node set.

**D6-safety — the one real constraint.** A per-cell head **must be D6-equivariant**: the
ownership prediction for a cell must map to the ownership of that cell's image under
every D6 element. This is satisfied **for free** here because (a) the head is a shared
`Linear` applied identically to every node embedding (permutation-equivariant), and (b)
the node embeddings are already D6-invariant-by-construction (no positional encoding;
geometry via edge structure — §1.3, `expand.py` docstring). So a per-node ownership
output is D6-**equivariant** by the same argument that makes the policy head D6-safe (the
policy head is *also* a per-candidate `Linear`, and it is the property the equivariance
test already guarantees). The **target** is likewise D6-equivariant: terminal control
maps cell→image bijectively under D6 (the §5.5 / `constants.py` window-mapping argument
applied to occupancy). **Safe**, with the discipline that the equivariance test must
cover the new head.

**Cost / drop-in.** **Cost: moderate** — new head params + a new target column produced
at finalize (terminal control per node) + plumbing through `expand.py`/`collate.py`/
`losses.py` (a new `ownership`/`ownership_mask` batch key and a new loss component in
`hexgt_loss`, exactly parallel to how `stvalue_*` keys are looped in
`losses.py:202-205`). No Rust change is *required* (control is derivable in Python from
the finalized terminal state, like `_winner_value`), though precomputing it in Rust
alongside the window walk is an option. **Drop-in onto e42: yes, by zero-init graft** —
a new head is new params; load the trunk/policy/value with `strict=False` (the existing
`_validate_stv_resume_load` surgery, `_rl_train.py:51`, already grafts fresh STV heads
this way) and let the ownership head warm up as an aux loss. The trunk forward is
unchanged at resume; only the new aux gradient flows. **No cold start for the body.**

**Ranking effect.** **Promotes to the top tier (as "Rank 0b"), at or just below the
Rank 1 readout fix.** It is externally validated (ablation-confirmed in KataGo), attacks
the *spatial-localization* root cause that Rank 1 only partially addresses (Rank 1 pools
the trunk better; ownership *makes the trunk produce a better-localized representation to
pool*), is D6-equivariant for free, and grafts onto the live checkpoint. It is more
costly than soft-Z (new head + target plumbing vs. a one-line target blend) and slightly
more speculative for Hexo specifically (KataGo's ownership is well-defined on a settled
Go board; Hexo's "control" of empty line-cells needs a target definition choice), so it
ranks **just behind soft-Z** but **co-equal with / just below Rank 1**. Soft-Z + ownership
together are the "fix the value output" pair; Rank 1 is the "fix the value plumbing"
companion — all three are mutually reinforcing and all three are now the recommended top
tier.

### 6.3 Threat-Space Search / VCF / VCDT for Connect6 & six-in-a-row

**Summary.** In the *k*-in-a-row family (Connect6, Gomoku/Renju, six-in-a-row), the
domain-standard way to handle decisive tactics is **explicit threat enumeration**:
**Threat-Space Search (TSS)**, **Victory-by-Continuous-Fours (VCF)**, and
dependency-/threat-based search (the "Dependency-Based Search for Connect6" line, and
recent "deep learning approaches to Connect6"). Rather than hoping a value net
*notices* a forcing sequence, the engine *enumerates* the forcing moves (fours, open
threes, double-threats) and proves a win/loss is forced. The lesson: **forcing threats
are reliably handled by explicit must-block/must-respond machinery, not by diffuse value
estimation** — precisely hexgt's weak point.

**Reconciliation with §5 ("hot tokens").** §5's "hot tokens" are the **neural-network-side
analog of TSS/VCF**: an *opponent-hot* cell (§5.1c — an opp count-5 window with its lone
empty at `c`, or an opp open-four) is exactly a TSS "must-block" / VCF-threat cell, and an
*own-hot* tier-F cell (§5.1b) is a VCF-style forcing-win cell. §5 already proposes
detecting these in the Rust window walk (`features.rs`/`candidates.rs`, §5.4) and wiring
the opponent-half into the value readout (§5.2-iii). This prior art **validates the
direction** and supplies the vocabulary, but also points at a **stronger option §5 only
gestured at: feed the tactical check into the SEARCH, not only the features.** Two forms:

- **(search-side) A forcing-move check in MCTS.** A bounded TSS/VCF probe at expansion
  (or as a root filter): if the opponent has an immediate win-threat (opp count-5 with one
  empty), the search *must* consider the block; if `own` has a forced win, prune to it.
  This is a **classic strength multiplier** for *k*-in-a-row and is independent of the
  net — but it is a change to the Rust MCTS (`mcts_eval.rs`/search), heavier than features,
  and it interacts with the existing `forced_playout_k=2.0` setting (§5, `_rl_train.py`).
  Forced playouts already *inflate visits on under-explored moves* (§3.5); a TSS forcing
  check is a sharper, **predicate-driven** version of the same impulse — it would
  guarantee the must-block is searched, where forced playouts only make it *more likely*.
  Care: stacking a hard TSS forcing check on top of `forced_playout_k` risks
  over-determining the visit distribution (which then distorts the policy/KL targets, §3);
  if both are on, the forcing check should bias *expansion/selection*, not be double-counted
  in the visit-count target.
- **(feature-side) the §5 plan, unchanged.** The cheapest, lowest-risk form remains §5's
  opponent-hot feature + readout pool — it is TSS's *detection* without TSS's *search*.

**Ranking effect.** **No change to the ranking; it strengthens the case for §5 and adds an
explicitly-deferred search-side option.** "Hot tokens" (§5) keeps its place (just below
Rank 1, folded into it via iii-a). The TSS-in-search variant is recorded as a
**deferred** alternative: higher ceiling (provably correct tactics) but a Rust-MCTS change
that interacts with forced playouts and the KL target — pursue only if the feature/readout
form (§5) underperforms on the opponent-hot calibration slice (§5.5).

### 6.4 ResTNet — interleaved residual-conv + transformer blocks (arXiv 2410.05347)

**Summary.** "Bridging Local and Global Knowledge via Transformer in Board Games"
(ResTNet) interleaves **residual convolution blocks** with **transformer blocks** in a
single trunk. The headline empirical result: adding the conv inductive bias to a
transformer **cut a Go threat blind-spot — the circular/ladder-style "circular pattern"
vulnerability — from ~70% down to ~24%**. This is direct, *empirical* support for the
user's hypothesis that a pure global-attention model under-represents *local geometric
patterns* (lines/threats) that a conv captures cheaply, and that the fix is to **add the
conv bias back**, not to abandon attention.

**Re-evaluation of the Rank 4 "CNN hybrid breaks D6" deferral.** The doc's Rank 4
deferral (§2, §4) is correct *as stated* — a **square dense crop** is not closed under hex
D6 (the documented "96x8 D6 square-crop crash" failure class, `compact_io.py` D6-coverage
guard), so a naïve dense-CNN encoder forfeits hexgt's invariance-by-construction asset.
ResTNet does **not** invalidate that specific objection. **But** ResTNet's evidence makes
the *goal* (local conv bias) worth more, and the D6 objection is specifically about the
*square-crop conv*, not about *local conv bias in general*. There **are** D6-respecting
ways to inject local pattern bias:

- **(path A) Hex group-equivariant convolution.** Use a **G-CNN / steerable conv on the
  hex lattice** whose filters are constrained to be equivariant under the 6-fold rotation
  + reflection group (D6). This keeps translation+rotation equivariance *by construction*
  — the conv weights are shared across the 6 axial directions and their mirrors — so it
  adds local-pattern bias **without** breaking D6. dense_cnn already takes a half-step
  here: `HexConv2d` masks the 3×3 kernel to the 6 axial hex neighbors
  (`dense_cnn/architecture.py::HexConv2d`); a full D6-equivariant version additionally
  *ties* the 6 directional weights under rotation. Cost: a custom equivariant conv layer;
  it still needs a board raster, so the **crop-closure problem persists** unless combined
  with path C.
- **(path B) Conv along the 3 line-axes with 6-fold weight sharing — the natural Hexo
  fit.** Hexo's decisive structure is **length-6 lines along 3 axes** (`candidates.rs`,
  `tactics.rs` axis vectors). A **1-D convolution *along each axis*** with weights **shared
  across all 3 axes and both directions** is D6-invariant by construction (D6 permutes the
  axes/directions, and weight-sharing makes the operation commute with that permutation).
  This is *exactly* a learned generalization of the window-count features: instead of
  hand-coded count-{3,4,5}, a small 1-D conv over the stones-along-a-line sequence learns
  "open four," "split three," "gap pattern" as filters — the ResTNet local-pattern bias,
  specialized to Hexo's geometry, **without any square crop**. It is the most principled
  D6-safe conv path.
- **(path C) Convolve along the window-node chains (graph-native).** hexgt already has
  WINDOW nodes and CANDIDATE_WINDOW/STONE_WINDOW edges (`candidates.rs`). A conv-like
  operator that slides along the **ordered cell sequence of a window** (the 6 cells of a
  length-6 window, in line order) and shares weights across all windows is a graph-native
  1-D conv that respects D6 (windows map to windows, the within-window order is a line
  order that D6 preserves up to reversal — handle with a symmetric/reversible filter). This
  injects local line-pattern bias **entirely inside the existing graph**, with no raster
  and no crop — the lowest-D6-risk path, and the most architecturally compatible.

**Ranking effect.** **Rank 4 moves up from "last resort / pursue only if 1–3 fail" to a
"deferred-but-promising structural Model-2.x option," with the concrete D6-respecting path
being (path B) axis-shared 1-D conv or (path C) window-chain conv** (both D6-safe), rather
than the D6-breaking square-crop encoder originally described. It stays **deferred** — it
is structural, throughput-affecting, needs a from-scratch or major-graft retrain, and is
**more speculative** than the validated 0a/0b fixes (no Hexo-specific evidence yet; the
D6-equivariant-conv path is a design sketch, not proven code). The honest ordering is:
land the validated, cheap, graftable fixes first (soft-Z 0a, ownership 0b, readout Rank 1,
features Rank 2/§5, KL §3); treat the D6-safe conv-bias trunk (paths B/C) as the leading
**structural** experiment *after* those, ahead of the abandoned square-crop hybrid.

### 6.5 Net effect on the ranking (summary)

| Item | Source | Targets | Ranking move | Validated? |
|---|---|---|---|---|
| **Soft-Z value target (0a)** | 10.1007/s00521-021-05928-5 | value-target miscalibration | **UP — shares Rank 1; try first (cheapest, graftable)** | **Yes** (Connect-Four/Breakthrough) |
| **Ownership aux head (0b)** | arXiv 1902.10565 | no spatial localization in value | **UP — top tier, at/just below Rank 1** | **Yes** (KataGo ablation) |
| **TSS/VCF threats** | Connect6 / six-in-a-row lit. | forcing-threat handling | no change — strengthens §5; adds deferred search-side option | Yes (domain-standard) |
| **ResTNet conv-bias** | arXiv 2410.05347 | local line-pattern under-representation | **UP — Rank 4 last-resort → deferred-but-promising, via D6-safe conv (paths B/C)** | conv-bias yes; *D6-safe path for hexgt is speculative* |

The two **value-OUTPUT** fixes (0a, 0b) are the headline change: they attack the failure
— value miscalibration and missing spatial localization — more directly and at lower risk
than the readout plumbing, and both are externally validated, so they join (and 0a
arguably leads) the top tier. The architectural fixes (Rank 1 readout, §5 hot tokens) and
the training-emphasis fix (§3 KL) are unchanged and complementary. The CNN-hybrid is no
longer a dead end but a deferred structural option with a real D6-safe path.
