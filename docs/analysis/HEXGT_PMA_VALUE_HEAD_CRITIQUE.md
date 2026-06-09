# Critical review: hexgt PMA value head

Scope: read-only analysis of `PMAValuePool` and the value head in
`packages/hexo_models/hexgt/python/hexo_models/hexgt/architecture.py`. No code
was changed. Two questions from the owner, answered against the actual code and
Set-Transformer (Lee et al. 2019, arXiv:1810.00825) practice.

---

## Headline verdicts

- **Q1 (residual + LayerNorm + FFN inside PMA):** Mostly *not worth it*. Every
  one of these is largely **absorbable by the parameters we already have**
  (the seed constants, the q/k/v in-projection, the out-projection, and the
  downstream `Linear→ReLU→Linear` value MLP). The FFN/rFF are redundant with the
  trunk + downstream MLP — **skip**. Residual and LayerNorm are cheap but their
  expected gain here is **marginal**, not the textbook "stability win" — A/B only
  if you're retraining anyway. The owner's instinct that FFN/rFF are redundant is
  **correct**; the residual+LN "mild win" framing is **slightly optimistic**.

- **Q2 (is SIDE redundant because PMA already attends over it?):** **Premise
  CONFIRMED by the code.** SIDE *is* in the PMA's attended node set. So the
  separate `[SIDE | PMA]` concat is a **skip connection, not new information** —
  the owner is right that "PMA already takes SIDE into account." **But it is a
  legitimate, cheap design choice, not a mistake**, and the git/history evidence
  argues *for* keeping it. Recommendation: **keep it; low priority to test
  dropping.**

---

## The code facts

### What PMA pools over (the crux of Q2)

`_value_readout` (architecture.py:364) builds the value input as:

```python
side   = self._graph_readout(batch, node_emb)              # the SIDE hub node, per graph
pooled = self.value_pool(node_emb, batch["node_graph"], num_graphs)
return torch.cat([side, pooled], dim=-1)                    # [SIDE | PMA_k]
```

`PMAValuePool.forward` (architecture.py:237) receives **`node_emb` — the full
packed node set — with no node-type filter**. The keys/values are
`F.linear(node_emb, …)` over *every* row; the only grouping is `node_graph`,
which scopes the softmax per graph. Nothing excludes the SIDE node.

And the SIDE node is unambiguously part of that set: `GraphTransformerLayer`'s
docstring (architecture.py:145) states **"Context tokens = {side, stone, window}
nodes"**, and `_graph_readout` (architecture.py:352) pulls SIDE out of the same
`node_emb` via `node_type == NODE_TYPE_SIDE`.

**Therefore: the SIDE hub embedding is one of the keys/values the PMA seeds
attend over.** Concatenating it again is a skip path around the pool, carrying no
information the pool didn't already see.

### How heavily processed the keys are (bears on Q1's rFF claim)

Before PMA ever runs, every node passes through:

- **3× `RelationalMessagePassing`** — each ends in `LayerNorm(h + out_proj(agg))`
  (architecture.py:95): residual + LN per layer.
- **3× `GraphTransformerLayer`** — each context node gets full MAB-style
  treatment: `norm_ctx1(ctx + attn)` then `norm_ctx2(ctx + ffn_ctx(ctx))`
  (architecture.py:167–168), i.e. residual + LN + a per-node FFN (`Linear→ReLU→
  Linear`, dim→ffn_dim→dim).

So the keys/values are *already* the output of six residual+LN blocks, three of
which include a per-node FFN. An `rFF(Z)` on the keys inside PMA is piling a
seventh per-node MLP on top.

### PMA's current internals

`PMAValuePool` is the bare attention core: learned `seeds`, the q/k/v
in-projection (`self.attn.in_proj_weight`), a varlen segment-softmax, and
`self.attn.out_proj`. **No residual, no LN, no FFN.** Output is `(G, k*dim)`,
concatenated with SIDE and fed to:

```python
value_head = Linear((1+k)*dim, dim) → ReLU → Linear(dim, VALUE_BINS)   # k=2 default
```

That downstream block is itself an FFN/rFF over the readout.

### History (bears on Q2)

`expand_value_readout_columns` (architecture.py:446) documents the lineage: the
value head originally read **SIDE only** (`Linear(dim, dim)`), and pooling
(`mean|max`, now PMA) was **added later** as a zero-initialized expansion. So
the empirically-validated baseline was *SIDE alone carries the readout*; the pool
is the augmentation, not the other way round. That is the opposite of the framing
"is SIDE redundant given the pool" — historically SIDE was the proven signal.

---

## Q1 — per-component verdict

Canonical PMA is `PMA_k(Z) = MAB(S, rFF(Z))` with
`MAB(X,Y) = LN(H + rFF(H))`, `H = LN(X + Multihead(X,Y,Y))`. That unpacks into
four things the current code omits. Taking each *in our context*:

| Component | Worth it here? | Why |
|---|---|---|
| **Residual `S + attn(S,Z)`** | **No / marginal** | In canonical PMA the residual is on the **query**, which here is the *learned, graph-independent* `seeds`. Adding a constant vector to every graph's pooled output is **fully absorbable by the downstream `Linear`'s bias** — zero representational gain. The only residual argument is gradient flow, and PMA is a *single* attention layer, not a deep stack, so that benefit is weak. Cheap, but expect ~nothing. |
| **LayerNorm on the pooled output** | **Marginal — best of the bunch** | Mild conditioning win: SIDE enters the concat post-LN (transformer ends in LN), but `pooled` is post-`out_proj`, **un-normalized**, so the two concat blocks aren't on the same scale. LN would standardize that. But the pooled value is already a convex (softmax) combination of ~unit-scale `v` vectors, so the mismatch is modest and the downstream `Linear` can rescale per-column anyway. Cheap (2·dim params); only A/B-worthy if retraining. |
| **FFN/rFF on the pooled output** | **No — redundant** | This is exactly what the downstream `value_head` (`Linear→ReLU→Linear`) already is. Adding an in-PMA FFN just stacks two MLPs back-to-back. Adds real params for negligible expressivity the head doesn't already have. **Skip.** |
| **rFF on the keys `rFF(Z)`** | **No — redundant** | The keys are already the output of 3 GNN + 3 transformer blocks (three with per-node FFNs), and the q/k/v in-projection already applies a learned linear to K/V. A seventh per-node MLP buys essentially nothing. **Skip.** |

**Q1 bottom line.** The owner's read is essentially right: the FFN and rFF are
redundant given the trunk and the existing downstream value MLP — don't add them.
Where I'd push back on the owner's own framing: **residual + LN are not a
reliable "stability win" here either** — both are largely absorbable by the bias
and the downstream MLP that already exist, so the honest expectation is *marginal
to nil*. They're cheap enough to throw into an A/B if a retrain is already
planned, but I would not bill them as improvements. If you only have appetite for
one experiment, test **LayerNorm on the pooled block** (the scale-consistency
argument is the only one with a concrete mechanism) and drop the rest.

All four are **behavior-changing** (new params / new init), so none can be
validated without retraining — there is no "free" version.

---

## Q2 — per-component verdict

**Premise: CONFIRMED.** SIDE is in the PMA's attended set (evidence above). The
separate concat is a **skip connection**, not extra information.

So is it redundant? Critically:

- **It is not new information** — true. A sufficiently expressive pool *could* in
  principle reproduce the SIDE signal (one seed learning a near-delta attention on
  the SIDE node). So in the limit, yes, redundant.
- **But the skip is a defensible, cheap guarantee, not a bug.** With only `k=2`
  seeds and a softmax over potentially hundreds of nodes, the pool can easily
  **dilute or ignore** the SIDE hub. The direct concat *guarantees* the
  whole-board hub reaches the value head **undiluted**. The SIDE node is also the
  trunk's privileged global aggregator (it's the readout for `_graph_readout`,
  feeding the STV aux heads), so it's the one node specifically shaped to be a
  whole-board summary — exactly the thing you'd want to hand the value head
  directly.
- **History reinforces keeping it:** the value head was *originally SIDE-only*
  and worked; pooling was added on top. Dropping SIDE now would be removing the
  proven signal in favor of the newer addition — the riskier direction.

**Q2 bottom line.** The owner's instinct ("PMA already takes SIDE into account")
is **factually correct**, and that makes the concat a skip rather than a second
information source. But "redundant" ≠ "wrong": a cheap skip that guarantees the
privileged global token reaches the readout undiluted is a legitimate design
choice, and the historical evidence favors it. **Recommendation: keep the
`[SIDE | PMA]` concat.** Dropping SIDE is a valid but **low-priority** experiment;
if tested it is behavior-changing (and note `expand_value_readout_columns` /
the `value_head.0.weight` width machinery would need corresponding handling), and
I would not expect it to be a free win.

---

## Recommended experiments (all require retraining)

Ordered by expected value-per-effort. None is a confident win; all are A/B-grade.

1. **LayerNorm on the pooled block before concat.** The one component with a
   concrete mechanism (scale consistency between the post-LN SIDE block and the
   un-normalized pooled block). Cheapest meaningful test.
2. **Drop the SIDE concat** (value head reads PMA-only). Tests whether the pool
   has actually subsumed SIDE. Expect neutral-to-slightly-worse; do it only to
   confirm, and budget for the checkpoint-width / expander handling.
3. *(Do not bother)* residual on seeds, in-PMA FFN, rFF on keys — redundant with
   existing params/trunk; low expected gain for added parameters.

Keep these mutually exclusive in the A/B matrix — stacking them confounds the
already-small effects you're trying to measure.
