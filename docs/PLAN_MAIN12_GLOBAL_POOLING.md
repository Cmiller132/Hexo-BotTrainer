# PLAN — main_12: KataGo-style global pooling in the trunk (`CC(GP)A`)

Status: DRAFT / proposal. Author: architecture spike. Date: 2026-07-08.

## 0. Review corrections (v2 — 2026-07-08)

Reconciles two opus reviews: KataGo fidelity (grounded verbatim in KataGo's
`python/katago/train/model_pytorch.py` — `KataGPool`, `KataValueHeadGPool`,
`KataConvAndGPool`, `ResBlock`, `modelconfigs.py` — + Wu 2019 arXiv:1902.10565)
and hexfield integration (grounded in model.py/checkpoints.py + BOTH eval loaders).
Locked: SCOPE = trunk gpool + token-write + value-head gpool (policy head OUT);
RUN = warm-start branch from `main_10 ep20` (main_11 = control). These corrections
SUPERSEDE the v1 sections they name.

**Design-defining:**
- **The faithful-vs-graftable dichotomy was FALSE (§3.2 rewritten).** KataGo funds
  the pool from a DEDICATED PARALLEL conv over EXTRA channels — the trunk/residual
  width never changes — and adds the projected pooled vector to the regular
  channels BEFORE the block's second conv. In hexfield: keep `conv1:C→C` and
  `conv2:C→C` (both graft byte-identically) + NEW parallel `conv1g:C→c_pool` (own
  LN+ReLU) → pool → zero-init `linear_g:3·c_pool→C` added before `conv2`. This is
  simultaneously fully faithful AND graft-clean (new keys only; zero-init ⇒
  numerically identical to main_11 at ep0). NO surgery. Variants (a)/(b) dropped.
- **Value head uses KataGo's VALUE triple, not the trunk triple (§3.5).** KataGo's
  value pool has **NO max**; its third channel is a variance-centered
  QUADRATIC-in-width mean: `(mean, mean·(√N−N0)/S, mean·((√N−N0)²−VAR)/S²)`,
  `VAR=var(√N)`. Trunk gpool + token-write keep the max triple `(mean, mean·size,
  max)` [KataGPool]. Two distinct pool functions.
- **c_pool = 32**, not 64. KataGo's direct analogue (b10c128) uses 32 at width 128
  (= C/4); 64 was 2× any KataGo config. 64 is only an over-provisioned upper bound.

**Correctness (both reviews):**
- Serve dtype: cast the pooled vector to the stream dtype on the FC **input**
  (`linear_g(pooled.to(y.dtype))`), reduction stays fp32. Output-cast raises a
  dtype error on the fp16 `SERVE_HALF` path.
- Max: `masked_fill(~m, 0.0)` (pooled input is post-ReLU ≥0), not `-inf` — identical
  amax, no NaN on an all-pad row.
- Toggle-off parity: gate the `_pooled` fp32 change behind `VALUE_GPOOL`; applying
  it unconditionally makes `GPOOL_CONV_IDS=""` NOT byte-identical to main_11 on the
  serve path and confounds the arm-1 A/B.
- Size numerics: clamp `size=(√N−N0)/S` to `[-3,3]` (N=1 opening is a large-magnitude
  outlier); clamp `S ≥ ε`.
- Drop `ls_pool` — zero-init `linear_g` already no-ops the branch and KataGo has no
  such per-channel scale.

**Cross-arch / persistence (REQUIRED — silent-corruption class):**
- Persist the resolved gpool config (`conv_ids, c_pool, token_write, N0, S, VAR`)
  in the checkpoint **`meta`/`extra`** dict AND N0/S/VAR as buffers — not fully
  recoverable from state-dict shapes, and env-only N0/S/VAR silently MIS-SCALE a
  main_12 net loaded in any other process.
- **TWO** foreign inference paths, not one: `model.infer_net_kwargs_from_state_dict`
  (eval_arena) AND `debug_infer.py`'s own `_infer_hexfield_arch` (dashboard worker).
  Both must learn gpool or the dashboard drops gpool weights and serves wrong
  outputs; the meta-read is the clean single fix.
- `GPoolConvBlock` needs the **fused conv+LN serve branch** (mirror `ConvBlock`
  L406-427), with pool + `linear_g` bias (+ token-write) BETWEEN the two fused calls.
- Validate `GPOOL_CONV_IDS` ⊂ `[0, layout.count("C"))` at import (conv# 0-5, NOT
  layout position 0-8).

**Value lever (ablation arm 2b):** hexfield's residual skip is NEVER LayerNorm'd
(LNs live on the branch), so pooling post-LN strips the absolute board "level" — the
"who's-ahead" signal the value hypothesis chases. KataGo pools post-BN, which keeps
a batch-relative level; this is a real LN-vs-BN divergence. Add a raw-residual-stream
mean pool to the value input as an ablation — keep OUT of the faithful arm-1 baseline.

**Placement:** `CC(GP)A` (`GPOOL_CONV_IDS=1,3,5`, 3/6) is at the TOP of KataGo's
"two or three blocks at regular intervals" range — kept as primary per the chosen
design. The more-canonical KataGo density is `3,5` (2/6, matches b6c96; KataGo never
gpools block 0), the cheaper alternative arm.

## 1. Purpose & hypothesis

**Hypothesis:** hexfield's value head is partly *trunk-limited* — the network has
only three global mixing points (the attention blocks at trunk positions 2/5/8),
and the summary tokens that carry value are frozen through the conv stretches
between them. Whole-board aggregates that value needs (who is ahead, global
threat density, region size) are not available to the conv blocks and are only
weakly recoverable by softmax attention.

**Change:** add KataGo-style **global-pooling residual blocks** to the conv trunk,
toggleable, in the `CC(GP)A` arrangement (the second conv of each `CC` group
becomes a global-pooling block). This is the closest well-tested design to copy
(KataGo, Wu 2019) and the cheapest global mechanism (O(N), not O(N²) attention).

**Non-goals / controls:** the search + training regime stays **byte-identical to
main_11** (Gumbel completedQ sharp target, 256 visits, 100% PCR, de-contaminated
search). The *only* deliberate degree of freedom is the trunk architecture, so any
delta is attributable to global pooling.

See also: [[main10-regression-diagnosis]] (value/policy-target context),
`docs/specs/hexfield_model_spec.md`, `docs/PLAN_katago_replay_buffer_port.md`.

## 2. Current architecture recap (what we are modifying)

- Trunk `CCACCACCA` (env `HEXFIELD_TRUNK`), `c=128`, 2 heads (`head_dim=64`).
- `ConvBlock` (`model.py:388`): post-activation residual, two `HexNodeConv`
  (7-tap hex conv) with LayerNorm, `LayerScale` on the residual branch.
- Conv blocks see **cells only** (`x`). The 8 summary **tokens** are split off
  after each attention block and held **frozen** through the following `CC`
  stretch (`model.py:932-933`), re-joined only at the next attention block.
- Value head reads `cat(tokens[0], tokens[1], pooled)` where `pooled` is the
  masked **mean** of the LN_final cells (`_pooled`, `model.py:938-942`,
  `_value_input`, `model.py:1012`).

Conv indices vs layout positions:

```
layout:  C  C  A  C  C  A  C  C  A
pos:     0  1  2  3  4  5  6  7  8
conv#:   0  1     2  3     4  5
attn#:         0        1        2
```

## 3. Architecture change

### 3.1 The pooling primitive (`kata_global_pool`)

KataGo pools a channel group into **(mean, size-scaled mean, max)** per channel.
hexfield's "board size" is the **live-cell count** `mask.sum(1)` — which, unlike
Go, grows within a game as the legal halo spreads, so the size channel is *more*
informative here. Reductions in fp32 (a mean/max over hundreds of nodes loses
precision under fp16 autocast; today's `_pooled` reduces in `cells.dtype` — fix
that too).

```python
# model.py — module-level helper
def kata_global_pool(p, mask):                       # p: (B, Npad, Cp)
    m = mask.unsqueeze(-1)
    counts = mask.sum(1, keepdim=True).clamp(min=1).float()      # (B,1) live cells
    mean  = (p.float() * m).sum(1) / counts                      # (B, Cp)
    size  = (counts.sqrt() - GP_SIZE_N0) / GP_SIZE_SCALE         # ~ KataGo "width"
    smean = mean * size                                          # size-scaled mean
    pmax  = p.float().masked_fill(~m, float("-inf")).amax(1)     # (B, Cp)
    return torch.cat([mean, smean, pmax], dim=-1)                # (B, 3*Cp)
```

`GP_SIZE_N0` / `GP_SIZE_SCALE`: derive from data, do **not** guess. Over a sample
of main_11 shards, compute the live-cell count per row (`mask.sum`), take
`GP_SIZE_N0 = median(sqrt(N))`, `GP_SIZE_SCALE = std(sqrt(N))` (fallback 1.0).
`sqrt(N)` because `N` is area-like and KataGo's factor is linear in *width*.

### 3.2 The global-pooling block (`GPoolConvBlock`) — faithful & graft-clean

ONE design (v1 variants a/b dropped — see §0). KataGo's defining structure: a
DEDICATED PARALLEL pool conv over EXTRA channels, its pooled vector projected and
added to the regular channels BEFORE the block's second conv, trunk width
invariant. Realized in hexfield's post-activation idiom with `conv1`/`conv2` left at
`C→C` (graft byte-identically from main_11) and `conv1g`/`lng`/`linear_g` new
(zero-init `linear_g` ⇒ numerically identical to main_11 at ep0, NO surgery):

```python
class GPoolConvBlock(nn.Module):
    """KataGo global-pooling residual block, hexfield post-act idiom. Dedicated
    parallel pool conv (conv1g) over EXTRA c_pool channels; pooled bias added to the
    regular channels before conv2; trunk width invariant. conv1/conv2 stay C->C
    (graft); conv1g/lng/linear_g new, linear_g zero-init => no-op at ep0."""

    def __init__(self, channels, c_pool, *, token_write=False, n_tokens=NUM_TOKENS):
        super().__init__()
        # Regular path — identical to ConvBlock (grafts byte-identically).
        self.conv1, self.ln1 = HexNodeConv(channels, channels), nn.LayerNorm(channels)
        self.conv2, self.ln2 = HexNodeConv(channels, channels), nn.LayerNorm(channels)
        self.ls = LayerScale(channels)
        # Dedicated parallel pool branch (KataGo conv1g/normg/actg) over EXTRA chans.
        self.conv1g = HexNodeConv(channels, c_pool)
        self.lng = nn.LayerNorm(c_pool)
        self.linear_g = nn.Linear(3 * c_pool, channels)
        nn.init.zeros_(self.linear_g.weight); nn.init.zeros_(self.linear_g.bias)
        self.c_pool, self.token_write = c_pool, token_write
        self.token_coupled = token_write
        if token_write:
            self.n_tokens = n_tokens
            self.token_fc = nn.Linear(3 * c_pool, n_tokens * channels)
            nn.init.zeros_(self.token_fc.weight); nn.init.zeros_(self.token_fc.bias)
        # Size scaling travels WITH the checkpoint (buffer + meta), never eval-env.
        self.register_buffer("gp_size", torch.tensor([GP_SIZE_N0, GP_SIZE_SCALE]))

    def _pool(self, g, mask):                              # KataGPool triple, fp32
        m = mask.unsqueeze(-1)
        counts = mask.sum(1, keepdim=True).clamp(min=1).float()
        n0, s = self.gp_size[0], self.gp_size[1].clamp(min=1e-3)
        mean = (g.float() * m).sum(1) / counts
        size = ((counts.sqrt() - n0) / s).clamp(-3.0, 3.0)
        pmax = g.float().masked_fill(~m, 0.0).amax(1)      # post-ReLU >= 0
        return torch.cat([mean, mean * size, pmax], dim=-1)          # (B, 3*c_pool)

    def forward(self, x, gather_idx, mask, tokens=None):
        m = mask.unsqueeze(-1)
        y = F.relu(self.ln1(self.conv1(x, gather_idx, mask))) * m           # regular
        g = F.relu(self.lng(self.conv1g(x, gather_idx, mask))) * m          # pool branch
        pooled = self._pool(g, mask)                                       # (B,3*c_pool)
        y = y + self.linear_g(pooled.to(y.dtype)).unsqueeze(1) * m         # bias before conv2
        if self.token_write and tokens is not None:
            upd = self.token_fc(pooled.to(tokens.dtype))
            tokens = tokens + upd.reshape(tokens.shape[0], self.n_tokens, -1)
        y = self.ln2(self.conv2(y, gather_idx, mask)) * m
        out = F.relu(x + self.ls(y))
        return (out, tokens) if self.token_coupled else out
    # SERVE: also implement the fused conv+LN branch (mirror ConvBlock L406-427),
    # running _pool + linear_g bias (+ token_write) BETWEEN the two fused() calls.
```

`conv1g` reads the block input `x` in parallel with `conv1` (KataGo's
`conv1r`/`conv1g` both read the block input). The pooled bias is added to the
regular activation before `conv2`. `token_write` (all 8 tokens, zero-init) rides
the same pooled vector; the trunk loop threads tokens only into `token_coupled`
blocks. The size scaling is a persistent buffer (§0) so foreign loaders get it.

### 3.3 Placement — `CC(GP)A` (gpool at conv indices 1, 3, 5)

The second conv of each `CC` group becomes a `GPoolConvBlock`:

```
C  C(GP)  A   C  C(GP)  A   C  C(GP)  A
```

Rationale (why this over the alternatives that were considered):

- **`CC(GP)A` (chosen).** Each attention block — the consumer that feeds the
  value-carrying tokens — receives cells that already carry the board-wide
  summary. GP computes the aggregate; attention routes it relationally
  (complementary, not redundant). Also places a gpool *before the first
  attention block*, covering the "purely local until A0" gap.
- `C(GP)CA` (gpool at conv 0/2/4): more downstream-conv coverage per gpool, but
  conv 0 pools the raw stem output (weak aggregate).
- Sparse `CC A · CC(GP)A · CC(GP)A` (conv 3/5 only): closest to KataGo's actual
  density (~2 of 6 blocks). Cheaper; a good **first arm** before committing to 3.
- Standalone `...A (GP)` after attention: least KataGo-faithful (gpool is a block
  *variant*, not a post-attention stage) and worst timing (attention just
  globalized the cells). Rejected.

Encoding: keep `HEXFIELD_TRUNK=CCACCACCA` unchanged and select gpool positions
with a new list `HEXFIELD_GPOOL_CONV_IDS="1,3,5"` (empty ⇒ feature OFF ⇒ arch
identical to main_11). This avoids touching layout validation / `KNOWN_TRUNK_LAYOUTS`
and makes the toggle trivial.

### 3.4 Does GP interact with the summary tokens? (direct answer)

**No — not by default.** As placed inside a `ConvBlock`, GP pools **cells only**
and biases **cells only**. The 8 summary tokens live *outside* the conv blocks
(split off and frozen through the `CC` stretch), so the default gpool never reads
or writes them. The tokens benefit only **indirectly and downstream**: the enriched
cells are read by the next attention block, where the tokens attend to them.

**Is token interaction a reasonable addition? Yes — and it is arguably the most
value-relevant extension.** The tokens are the dedicated value/aux carriers and
are updated only 3× (at the attention blocks), frozen the rest of the trunk.
Letting the gpool refresh them mid-trunk gives them mean/max/**count** information
that softmax attention cannot easily compute (especially region size). Conceptually
this treats the summary tokens as the **virtual/global nodes** they already are
(cf. the virtual-node / GraphGPS framing) and lets them participate in the pooling.

Two directions, each a **separate toggle**, default **OFF** for arm 1 so the
KataGo-faithful cell-gpool effect is isolated first:

- **`token_write` (GP → tokens), the valuable direction.** Each of the 8 tokens
  gets a *bespoke, learned* additive update from the board-wide aggregate at each
  gpool block: `token_fc: Linear(3·c_pool → NUM_TOKENS·C)` → reshape `(B,8,C)` →
  `tokens += ls_tok(update)`, **zero-init** (`ls_tok`), so it is a no-op at graft
  and grows in. Because the tokens are partitioned across heads (value 0/1, stv
  2/3, ml 4/5), each head's tokens can read the aspect of the aggregate it needs.
  This makes the gpool block a cheap, orderless cell→token "global refresh"
  inserted into the conv stretches where no token update otherwise happens.
- **`token_read` (tokens → GP), optional/minor.** Condition the *cell* bias on the
  current summary state by concatenating `tokens.mean(1)` into the pool vector
  before `pool_fc`. Lower value (tokens are already global — somewhat circular).

Plumbing: token-coupled gpool blocks take `tokens=` and return `(x, tokens)`; the
trunk loop threads tokens only for those blocks (plain blocks keep the
`x = block(...)` signature). See §6.

### 3.5 Value-head global pooling (IN SCOPE; KataGo VALUE triple, not the trunk one)

KataGo's value scalar is produced from a global-pooled vector, but its head pool
(`KataValueHeadGPool`) is NOT the trunk triple: **no max**, and its third channel is
a variance-centered QUADRATIC-in-width mean (paper: `mean·((b−b_avg)²−σ²)/100`).
Faithful hexfield value pool:

```python
def kata_value_pool(cells, mask, n0, s, var):        # NO max; quadratic-in-width
    m = mask.unsqueeze(-1)
    counts = mask.sum(1, keepdim=True).clamp(min=1).float()
    mean = (cells.float() * m).sum(1) / counts
    off  = counts.sqrt() - n0
    lin  = (off / s).clamp(-3, 3)
    quad = ((off * off - var) / (s * s)).clamp(-3, 3)          # variance-centered
    return torch.cat([mean, mean * lin, mean * quad], dim=-1)  # (B, 3C)
```

- Value/aux/ml input: `cat(tok0, tok1, kata_value_pool(cells))` = 5C →
  `value_reduction: Linear(5C→C)` (likewise `aux_reduction`, `ml_reduction`).
  hexfield keeps the tokens KataGo lacks, so this is a superset of KataGo's value pool.
- **Graft remap surgery (mandatory):** the tolerant loader would reinit the reshaped
  reduction (a *wrecked* value head, not a no-op). Order the input
  `cat(tok0, tok1, mean, …)` so `mean` lands in the old `pooled` slot `[2C:3C]`; then
  `W[:,:3C]=old_W`, `W[:,3C:]=0` reproduces main_10's value output bit-for-bit and
  learns the width channels in. NEVER route this through the plain warm start.
- Persist `n0/s/var` (buffer + meta). Toggle `HEXFIELD_VALUE_GPOOL`; gating the
  `_pooled`→`kata_value_pool` swap behind it is also what keeps trunk-only arms
  byte-identical (§0).

## 4. Config & toggles

Arch is set by **service-file env** (read once at import in `constants.py`),
matching the existing convention (`hexfield-supervisor-11.service`). New env:

| Env var | Meaning | Default (OFF) |
|---|---|---|
| `HEXFIELD_GPOOL_CONV_IDS` | comma list of conv indices that become gpool blocks | `""` (off) |
| `HEXFIELD_GPOOL_CHANNELS` | `c_pool` (pooled channel slice / group width) | `0` |
| `HEXFIELD_GPOOL_TOKEN_WRITE` | GP updates the summary tokens | `0` |
| `HEXFIELD_GPOOL_TOKEN_READ` | GP cell-bias conditioned on token summary | `0` |
| `HEXFIELD_GPOOL_SIZE_N0` | size-channel center (`median sqrt(N)`) | data-derived |
| `HEXFIELD_GPOOL_SIZE_SCALE` | size-channel scale (`std sqrt(N)`) | `1.0` |
| `HEXFIELD_GPOOL_FAITHFUL_SPLIT` | use channel-split variant (b) instead of (a) | `0` |
| `HEXFIELD_VALUE_GPOOL` | value/aux/ml head global pooling | `0` |

main_12 service env (delta from main_11 — arm 1, cell-gpool only):

```ini
Environment=HEXFIELD_TRUNK=CCACCACCA          # unchanged
Environment=HEXFIELD_GPOOL_CONV_IDS=1,3,5     # CC(GP)A
Environment=HEXFIELD_GPOOL_CHANNELS=64        # c_pool = C/2
Environment=HEXFIELD_GPOOL_SIZE_N0=<from data>
Environment=HEXFIELD_GPOOL_SIZE_SCALE=<from data>
# token coupling + value-head gpool OFF for arm 1
```

toml (`configs/hexfield_main_12.toml`) = **copy of main_11** with only:

```toml
[run]
name = "hexfield_main_12"
output_dir = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_12"

[checkpoint]
# A/B BRANCH POINT: warm-start from the SAME point main_11 started from
# (main_10 ep20), so main_11's own recorded curve is the control arm — no re-run.
# Tolerant loader; the zero-init gpool params are "missing" and keep their no-op
# init, so at ep0 the net is numerically identical to main_11's start.
initialize_from = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_10/checkpoints/epoch_000020.pt"
```

Everything else (selfplay/training/eval, Gumbel target, 256 visits, 100% PCR) is
identical to main_11 so architecture is the only variable.

## 5. Warm-start / checkpoint strategy

- **Recommended for the A/B verdict: branch from main_11's own start point
  (`main_10 ep20`)** using variant (a) full-channel gpool — NOT a late main_11
  checkpoint. The tolerant `initialize_from` path (`checkpoints.py:84-107,141`,
  `strict=False`) transfers every existing weight; the new `pool_fc`/`ls_pool`
  (+ `token_fc`/`ls_tok` if enabled) are "missing" and keep their zero-init. At
  ep0 the net is **numerically identical to main_11's start**, so the trunk
  co-adapts to the global channel from the beginning (the fair test of the
  architecture) and **main_11 itself is the control** (its curve already exists).
  Arm 1 adds keys only — **no checkpoint surgery**.
- Alternative: **cold main_12** with variant (b) faithful channel-split (loses
  `conv2` on the 3 gpool blocks under grafting anyway) — only if we want KataGo
  purity over continuity.
- **Value-head gpool** needs the manual remap surgery in §3.5 (not the auto
  loader) to preserve the trained reduction.

### Cross-arch loading (eval anchors / dashboard) — required

main_12 checkpoints carry new keys, so foreign-process loaders (eval anchors,
the dashboard debug worker) must rebuild the exact gpool arch off the state dict:

- Extend `infer_net_kwargs_from_state_dict` (`model.py:551`) to detect gpool ids:
  `gpool_conv_ids = sorted(i for i in conv_ids if f"conv_blocks.{i}.pool_fc.weight" in sd)`,
  and `token_write` from the presence of `conv_blocks.{i}.token_fc.weight`, and
  `c_pool` from `pool_fc.weight` shape. Pass to the constructor.
- Confirm the strict `load_into` (`checkpoints.py:56-66`) passes once the net is
  built with the inferred gpool config (bidirectional key equality holds).
- `eval_arena` strict-load + `KNOWN_TRUNK_LAYOUTS` (`model.py:544`): layout string
  is unchanged (still `(6,3)` ⇒ `CCACCACCA`), so no new entry is needed; gpool is
  an orthogonal per-conv attribute inferred separately.

## 6. Implementation checklist

- [ ] `constants.py`: read `HEXFIELD_GPOOL_*` env; expose `GPOOL_CONV_IDS`,
      `GPOOL_CHANNELS`, `GPOOL_TOKEN_WRITE/READ`, `GP_SIZE_N0`, `GP_SIZE_SCALE`,
      `GPOOL_FAITHFUL_SPLIT`, `VALUE_GPOOL`.
- [ ] `model.py`: add `kata_global_pool`; add `GPoolConvBlock` (variants a/b);
      in `HexfieldNet.__init__` build `GPoolConvBlock` for conv indices in
      `GPOOL_CONV_IDS`, else `ConvBlock`; constructor kwargs mirror env (for
      cross-arch rebuild).
- [ ] `model.py` `trunk()`: thread tokens through token-coupled gpool blocks
      (`if getattr(blk, "token_coupled", False): x, tokens = blk(..., tokens=tokens)`).
- [ ] `model.py` heads: if `VALUE_GPOOL`, swap `_pooled`→`kata_global_pool` and
      widen `value_reduction`/`aux_reduction`/`ml_reduction` to `5C→C`.
- [ ] `model.py` `_pooled`: fp32 reduction fix regardless of toggle.
- [ ] `infer_net_kwargs_from_state_dict`: detect gpool ids / c_pool / token_write.
- [ ] Serve fused path: in `GPoolConvBlock`, run the pool **between** the two fused
      `_hex_conv_ln` calls (do NOT fold pooling into the fused kernel); keep the
      two fused conv+LN calls intact.
- [ ] `configs/hexfield_main_12.toml` (copy of main_11 + §4 deltas).
- [ ] `scripts/systemd/hexfield-supervisor-12.service` (copy of -11 + §4 env).
- [ ] Value-head remap surgery script (§3.5) if `VALUE_GPOOL` on a graft.
- [ ] Data script to compute `GP_SIZE_N0` / `GP_SIZE_SCALE` from main_11 shards.
- [ ] Parity/unit test: with `GPOOL_CONV_IDS=""` the net is byte-identical to
      main_11 (toggle-off is a true no-op); with gpool on + zero-init, forward
      output equals the plain net at step 0 (graft no-op).

## 7. Experimental arms (A/B)

Control = **main_11 itself** — it started from `main_10 ep20`, so branching the
treatment from that SAME point makes main_11's recorded eval curve the control (no
re-run). Two strategies:

- **Branch-from-ep0 (recommended for the verdict).** `initialize_from` = `main_10
  ep20` (main_11's own start) + byte-identical config + gpool env. At ep0 both nets
  are numerically identical (gpool zero-init = no-op); with the same `seed` the
  first epoch's self-play is identical, then the runs diverge purely as the gpool
  params learn. Fair test — the trunk co-adapts to the global channel from the
  start. Costs a full trajectory for the treatment (control is free). Arm 1 needs
  **no checkpoint surgery** (gpool only ADDS keys).
- **Late-graft probe (cheap, weaker).** `initialize_from` = latest main_11
  checkpoint: a few epochs on the current best, but tests "does gpool help an
  already-settled net?" — the no-op-init branch must carve a role in a net that
  won't reorganize, so it under-tests the architecture. A quick screen, not the
  verdict.

Arms (graft from the branch point above), same data regime:

1. **Arm 1 — cell-gpool `CC(GP)A`** (`GPOOL_CONV_IDS=1,3,5`, token coupling off,
   value-gpool off). Isolates the KataGo-faithful trunk change.
2. **Arm 2 — + `token_write`.** Adds the summary-token refresh (§3.4).
3. **Arm 3 — + `VALUE_GPOOL`.** Adds the head-side pooling triple (§3.5).
4. (Optional) **Arm 0 — sparse** (`GPOOL_CONV_IDS=3,5`) as a cheaper first probe.

Run arms sequentially (or as separate run dirs) so each adds exactly one DOF over
the previous. Zero-init guarantees each arm starts identical to its predecessor.

## 8. Success metrics

Judge against the value diagnostics established earlier (baseline them on the
main_11 checkpoint first):

- **Strix win% / pentanomial ladder** vs the same anchors (primary; multi-stage
  eval already configured, every 5 epochs).
- **Search-vs-net value gap** shrinks (raw net root value vs 256-visit
  completedQ) — the most direct "value got better" signal in this regime.
- **Value calibration** tightens in the 20–80-ply midgame bucket (reliability by
  ply, à la `head_audit._BUCKETS`).
- **`grad_norm_trunk_conv`** carries more of the value gradient (already logged,
  `trainer.py:870-873`); `loss_value` / `loss_cell_q` / `loss_stvalue_*` trend.

**Kill criterion:** if arm 1 does not move the search-vs-net gap or Strix win% over
~10-15 epochs, value is *target-limited* (the Gumbel signal), not trunk-limited —
stop and do not escalate to arms 2/3 or the GraphGPS rebuild.

## 9. Numerics & serve correctness

- Pool reductions in **fp32**; `masked_fill(-inf)` for max; divide mean by clamped
  live counts (never zero — the opening always has ≥1 cell).
- Pad rows receive the broadcast bias but are re-zeroed by the `* m` epilogue —
  consistent with the "re-apply mask after every parameter-carrying op" invariant
  (`model.py` docstring).
- Serve fused conv+LN path: pooling sits between the two fused kernels (cheap
  reduction + small GEMM + broadcast add); do not attempt to fuse it in.
- `LayerScale(init=0.0)` on `ls_pool`/`ls_tok` for graft no-op; the standard
  residual `ls` keeps its `1e-4` init.

## 10. Risks & rollback

- **Redundancy with attention** for the cell bias right before an attention block —
  mitigated: the unique contribution is the mean/max/**size** aggregate attention
  can't compute; measure, don't assume.
- **Serve throughput**: extra reduction + GEMM per gpool block on the no-grad
  serve path. Small vs the convs, but confirm live serve rate doesn't regress
  (the Triton attention/conv kernels are untouched).
- **Rollback**: set `HEXFIELD_GPOOL_CONV_IDS=""` → arch reverts to main_11 exactly;
  the toggle-off parity test guarantees this is a true no-op.

## 11. Decisions (resolved by the v2 reviews + user)

1. Block design → the single faithful + graft-clean `GPoolConvBlock` (§0/§3.2);
   variants (a)/(b) dropped. NO surgery for the trunk block.
2. `c_pool` → **32** (KataGo b10c128 analogue); 64 is an over-provisioned upper bound.
3. Placement → `1,3,5` (CC(GP)A, chosen); `3,5` is the more-canonical/cheaper
   alternative arm.
4. Scope → trunk gpool + token-write + value-head gpool bundled in main_12 (user
   choice), toggles kept independent for later per-component ablation.
5. Run → warm-start from `main_10 ep20`; main_11 = control.
6. Open ablation → raw-residual "who's-ahead" level pool (§0 value lever) as arm 2b.
7. Value-head gpool → KataGo VALUE triple (no max, quadratic-width), NOT the trunk
   triple; mandatory reduction remap surgery (§3.5).
