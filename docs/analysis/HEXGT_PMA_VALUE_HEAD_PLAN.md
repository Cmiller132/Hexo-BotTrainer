# hexgt PMA Value-Head Readout — Design + Test Plan

Design / analysis only. **No code, config, or model files are changed by this
document; nothing here runs training or touches any run.** Code citations are to
the hexgt worktree `E:\Hexo-BotTrainer-hexgt` (READ-ONLY). Builds on the prior
committed analysis `HEXGT_VALUE_HEAD_REDESIGN.md` (reused, not re-derived).

---

## Headline

- **Recommended seed count: k = 1**, with **H = 4-head** attention (H = the
  model's existing transformer head count, `attention_heads=4` —
  `architecture.py:196`, `constants.py:177`). Per Set Transformer (Lee et al.
  2019, arXiv 1810.00825), PMA pools a set with k learnable seed queries via
  multihead attention; the paper's default is **k=1** for set→single-output
  tasks, and k>1 is for set-*valued* outputs (clustering / amortized inference).
  A hexgt value head emits **one** output (a 65-bin value distribution,
  `VALUE_BINS`), so the principled default is k=1.
- **How we know k:** k is a small hyperparameter, decided **empirically**, not
  analytically. We START at k=1 and run a short **ablation over k ∈ {1, 2, 4}**,
  measuring value calibration (below). Go higher only if the ablation shows a
  real gain — "k doesn't help" is itself the signal (see Caveat).
- **Why k=1 is already strong:** a single seed with H=4-head attention attends
  with 4 different patterns and **concatenates** them, so k=1 is already
  substantially more expressive than fixed mean+max — it can *learn* a soft,
  cross-node, cross-channel weighting (e.g. "this opponent threat vs my
  counter") that a symmetric pool cannot represent. Extra seeds mainly add
  capacity for *multiple distinct pooled summaries*, which is often redundant for
  a scalar value.

---

## What the value head is today (verified)

The value head reads `[SIDE hub | mean-pool | max-pool]`, each `token_dim=168`
wide (`VALUE_READOUT_MULT=3`, `architecture.py:43`):

```
# architecture.py:306-311
def _value_readout(self, batch, node_emb):
    side = self._graph_readout(batch, node_emb)          # SIDE hub row
    mean_pool, max_pool = self._global_pool(batch, node_emb)
    return torch.cat([side, mean_pool, max_pool], dim=-1)  # (G, 3*D)
# architecture.py:236-240 — value_head first Linear reads 3*token_dim
self.value_head = nn.Sequential(
    nn.Linear(VALUE_READOUT_MULT * self.token_dim, self.token_dim),
    nn.ReLU(inplace=True),
    nn.Linear(self.token_dim, VALUE_BINS),
)
```

**Critical data-flow fact (established, re-verified):** `_encode_nodes`
(`architecture.py:252-270`) runs **GNN first, then the transformer over the GNN
output** (`for layer in self.gnn` at :262-263 feeding `for layer in
self.transformer` at :268-269), and every transformer update is **residual**
(`ctx = norm(ctx + a)` at :172, `:180`). So the **post-transformer node
embeddings already integrate local typed-GNN features AND global attention.**
The diagnosed problem is **defensive value miscalibration** (same-board
`v(A)+v(B) ≈ +0.82` optimism). The "role-assigned seeds" idea from the prior doc
is **dropped** (seeds are LEARNED, not hand-assigned) unless the ablation
justifies otherwise. SE-trunk-fusion is explicitly **out of scope** here — PMA
only.

---

## Minimal design (against the real `architecture.py`)

**The PMA block.** One Set-Transformer PMA = a single MAB(seed, set):
`PMA_k(Z) = MAB(S, Z)` where `S ∈ R^{k×D}` are k learnable seed queries and Z is
the node set (keys/values). Concretely one `nn.MultiheadAttention(D, heads=4)`
with a learned `(k, D)` query parameter, run per graph over the **padded node
set**. The paper's optional rFF + LayerNorm inside the MAB may be included only
if it stays trivial; the **smallest sound** form (seed-query MHA, no extra FFN)
is the default for k=1.

**What set to pool — pool the context-transformer OUTPUT tokens (the embeddings
`_encode_nodes` already returns).** Justification: because the transformer
*consumes* the GNN output and is residual (above), those tokens already carry
local + global signal — pooling them is sufficient *representationally* and is
the simplest sound choice. Pooling the pre-transformer GNN states or
SIDE/candidate tokens separately is an option only if an ablation shows a gap;
it is **not** the default (it largely re-reads what the post-transformer
embeddings already hold).

**Mechanically it reuses existing machinery.** The per-graph padded-attention
layout already exists (`_AttentionLayout` / `build_attention_layout`,
`architecture.py:103-144`); the transformer already runs batched padded MHA
(`GraphTransformerLayer`, `:147-183`). PMA adds one analogous batched MHA with
the seed as query over a padded gather of **all** nodes (mirror `_padded_index`,
`:122-136`), then scatters the k pooled vectors into the readout. ~30-50 lines.

**Replace vs augment — REPLACE mean+max with the PMA vector, KEEP the SIDE
token.** Recommended readout: `[SIDE | PMA_k]` (width `(1+k)·D`). Rationale: the
PMA pooled vector is a strict, learnable generalization of mean+max (a seed can
learn uniform attention ≈ mean, or a sharp attention ≈ soft-max), so carrying
mean+max alongside is redundant; keeping the cheap SIDE token preserves the
dedicated whole-board hub the graft already trained. (An *augment* variant —
`[SIDE | mean | max | PMA_k]` — is the fallback if the ablation shows replacing
mean+max regresses; it is strictly safer for graft but wider.)

**Output → existing value MLP.** The k pooled vectors are flattened/concatenated
(`k·D`) and fed to the **existing** `value_head` MLP (`:236-240`), with its first
`Linear` widened from `3·D` to `(1+k)·D` (replace) or `(3+k)·D` (augment). No new
MLP.

**D6-safety.** PMA attention is permutation-invariant over the key set (no
positional encoding; the seeds are graph-independent learned constants, and D6
acts on nodes not seeds) ⇒ **D6-invariant**, the same property the existing pool
relies on and that `test_hexgt_value_readout.py::test_global_pool_is_permutation_invariant`
(`:118`) enforces. Add an analogous permutation-invariance unit test for the PMA
output before any training; keep the `test_hexgt_d6.py` parity gate.

**Zero-init graft (bonus, not required — a short re-fit is acceptable).** The
shipped surgery `expand_value_readout_columns` (`architecture.py:387-428`) widens
the value head's first `Linear` in place: old weight into the leading block,
**zeros** in the new blocks → byte-identical first step, then it learns from
zero. For PMA, the same recipe applies: keep the SIDE block's trained weight, and
**zero the PMA block** of `value_head.0.weight` at resume so the model drops the
module in producing identical output, then learns the attention pool from zero.
(For the *augment* variant this is the exact same shape of edit as the shipped
helper. For *replace*, the mean/max columns are dropped and the PMA column is
zeroed.) Because a full retrain is acceptable, graft-exactness is a convenience,
not a hard requirement.

**Param / compute cost — SMALL.** k=1: seed `(1, 168)` ≈ 168 params + one MHA's
4 projections (`4·D·D ≈ 4·168² ≈ 113K`) ≈ **~115K params (<6% of the 2.07M
model)**. One extra batched MHA over the node set per forward — comparable to one
of the 3 transformer layers' context attention the model already runs.
Throughput hit is negligible (self-play is featurize-bound per prior perf work).

---

## Test plan (simple — one change at a time)

**Ablation: k ∈ {1, 2, 4}.** Short runs from the same checkpoint, PMA-replace
readout, identical everything else. Start k=1; escalate only if k=2/4 measurably
beats it.

**Metric battery (same for every k):**
1. **Same-board optimism sum:** `v(A) + v(B)` over a fixed position set → should
   move from +0.82 toward 0.
2. **Calibration / Brier on opponent-hot defensive positions:** the slice where
   the opponent has a live ≥4 threat — PMA's targeted slice. Lower Brier /
   better reliability.
3. **The 8/8-lost-game value trace:** value should stop pinning near +0.8 in the
   plies before the loss.
4. **(Optional) H2H** vs dense_cnn e24 at matched visits (`run_head_to_head`) as
   an end-to-end sanity check.

**Gate before training any arch change:** the PMA permutation-invariance unit
test + `test_hexgt_d6.py` must pass.

**Decision rule:** pick the smallest k whose metrics are not beaten by a larger
k beyond noise. Default expectation: **k=1 wins or ties** — ship k=1.

---

## One honest caveat

The trunk is only 3 GNN + 3 transformer layers. If it is **too shallow to have
computed the cross-window threat relationships** (e.g. a developing
double-threat), then PMA pools *incomplete* information and cannot recover what
the trunk never represented. **Watch the ablation for "k doesn't help" across all
of {1,2,4}** — that flat result is the signal that the binding constraint is the
trunk (or the distribution/label causes diagnosed in `HEXGT_DEFENSE_DEEP_ANALYSIS.md`),
not the readout operator. This plan stays PMA-only as requested; if PMA is flat,
the next lever is the trunk-side work, evaluated separately.
