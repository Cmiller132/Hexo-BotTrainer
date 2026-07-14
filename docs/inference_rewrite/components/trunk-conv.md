# trunk-conv

I have complete ground truth. Now I'll write my deliverable.

The key design decision for my component: **the trunk stays in the padded `(B, Npad, ·)` layout**. Per the spec (§C3) the v1 dense scatter remains the default; ragged-direct is "a later optimization gated separately." The attention kernels (hexflash/flex) accept padded `(B,H,S,Dh)` tensors plus `coords`/`seq_mask` — they do not require the conv path to go ragged. So HexNodeConv's gather+GEMM is unchanged. What I must change in model.py is:

1. Thread `coords`/`seq_mask`/`bias_table` to the attention when impl∈{hexflash,flex}, skipping `build_attn_bias` entirely under no-grad.
2. Preserve every pad-inertness invariant (the `*m` re-zeroing in `AttnBlock`, the conv mask, int32 coords for the kernel).

Let me deliver the full changed code as text.

---

# DELIVERABLE — model.py conv/trunk adaptation for the hexflash/flex attention layout

## Summary of layout decision

**Layout = PADDED, unchanged.** Per spec §C3, the v1 dense `(B, Npad, ·)` scatter remains the default and "ragged-direct is a later optimization gated separately." `HexNodeConv`'s gather+GEMM operates on `(B, Npad, Cin)` and is **bit-identical** in either attention regime — it never touches `coords` or the bias. Therefore:

- **`HexNodeConv` is UNCHANGED.** No edits. Its pad-inertness (`out * mask.unsqueeze(-1)` at `model.py:120`, with the appended zero row at index `Npad`) is exactly what the kernels rely on downstream.
- **`ConvBlock` is UNCHANGED.**
- The only edits are the *threading* of `coords`/`seq_mask`/`bias_table` from `trunk` → `AttnBlock` → `RelPosAttention` so the inference path can route to `hexflash_attention`/`flex_attention_relpos` and **skip `build_attn_bias` entirely** under `no_grad`.

This is the minimal change that satisfies my component charter ("adapt HexNodeConv + the trunk to the chosen layout, preserving pad-inertness and parity"): the chosen layout keeps the conv path as-is, and the trunk changes are purely additive routing that leave the `sdpa`/`materialized` paths byte-identical.

## Parity assertions this code must satisfy (reuse existing gates)

- `test_sdpa_equals_materialized` / `..._fp16_cuda` (`tests/test_hexfield_model.py:295`): with `impl="sdpa"` or `"materialized"`, `trunk` must produce the **same** tensors as before — my edits add branches but leave those two paths on the original `build_attn_bias` codepath, so `maxabsdiff == 0.0` by construction (Tier-1, statically certain).
- Conv oracle (`HexConv2d` M1 test): untouched, `HexNodeConv` unmodified → passes unchanged.
- Tier-2 fp16 oracle extended to `impl∈{hexflash,flex}` runs against this `trunk`.

---

## EDIT 1 — `RelPosAttention.forward`: add hexflash/flex branches

Replace `RelPosAttention.forward` (`model.py:158-175`) with the overload below. The `q/k/v` projection, `scale`, and `out_proj` are **bit-identical** to the SDPA path (spec hard rule for Implementer 3); only the score+softmax+@v core swaps kernel. The new path takes `coords`/`seq_mask` instead of a prebuilt bias.

```python
    def forward(
        self,
        seq: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        *,
        coords: torch.Tensor | None = None,
        seq_mask: torch.Tensor | None = None,
        bias_table: torch.Tensor | None = None,
        exact_lut: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, s, c = seq.shape
        h, d = self.heads, self.head_dim
        # q/k/v projection + scale + out_proj are IDENTICAL across all impls;
        # only the score+softmax+@v core changes kernel. (spec C1 hard rule)
        q = self.q_proj(seq).reshape(b, s, h, d).transpose(1, 2)
        k = self.k_proj(seq).reshape(b, s, h, d).transpose(1, 2)
        v = self.v_proj(seq).reshape(b, s, h, d).transpose(1, 2)

        if self.impl in ("hexflash", "flex"):
            # Kernel reconstructs the rel-pos bias row in-kernel from the model's
            # OWN _exact_lut + bias_table; never materializes (B,H,S,S). coords
            # and seq_mask are threaded from the trunk; build_attn_bias is skipped.
            from .hexflash import flex_attention_relpos, hexflash_attention

            fn = hexflash_attention if self.impl == "hexflash" else flex_attention_relpos
            out = fn(
                q,
                k,
                v,
                coords,
                bias_table,
                seq_mask,
                exact_lut,
                self.scale,
                NUM_TOKENS,
            )
            out = out.transpose(1, 2).reshape(b, s, c)
            return self.out_proj(out)

        # Match the bias dtype to q under autocast: a dtype mismatch silently
        # drops sdpa to the slow math fallback. -3.0e4 stays finite in fp16.
        attn_bias = attn_bias.to(q.dtype)
        if self.impl == "sdpa":
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        elif self.impl == "materialized":
            scores = (q @ k.transpose(-2, -1)) * self.scale + attn_bias
            out = torch.softmax(scores, dim=-1) @ v
        else:  # pragma: no cover - config validation
            raise ValueError(f"unknown attention impl: {self.impl}")
        out = out.transpose(1, 2).reshape(b, s, c)
        return self.out_proj(out)
```

Notes:
- `coords` here is the `(B, Npad, 2)` int32 tensor; the kernel internally maps cell slot `i` (sequence index `i ≥ NUM_TOKENS`) to `coords[:, i - NUM_TOKENS]`. Token slots `< NUM_TOKENS` never read `coords` (token-class rows selected by slot, per `model.py:293-295`). This matches `build_attn_bias`'s slot convention exactly.
- The `from .hexflash import …` is function-local so model.py has **no module-level dependency** on hexflash (keeps `sdpa`/`materialized` training importable even if Triton is absent), per spec "no model imports beyond constants" working the other direction.

## EDIT 2 — `AttnBlock.forward`: thread coords/seq_mask through

Replace `AttnBlock.forward` (`model.py:189-195`). The `seq_mask.unsqueeze(-1)` re-zero (`*m`) — the pad-QUERY inertness invariant (`model.py:193`) — is **kept verbatim**.

```python
    def forward(
        self,
        seq: torch.Tensor,
        attn_bias: torch.Tensor | None,
        seq_mask: torch.Tensor,
        *,
        coords: torch.Tensor | None = None,
        bias_table: torch.Tensor | None = None,
        exact_lut: torch.Tensor | None = None,
    ) -> torch.Tensor:
        m = seq_mask.unsqueeze(-1)
        # Pad-QUERY rows are re-zeroed here by *m REGARDLESS of impl, so the
        # kernel is free to emit garbage in pad-query rows (it does — those rows
        # are never read). This is the invariant that makes any Npad bit-identical.
        seq = seq + self.attn(
            self.ln1(seq),
            attn_bias,
            coords=coords,
            seq_mask=seq_mask,
            bias_table=bias_table,
            exact_lut=exact_lut,
        ) * m
        seq = seq + self.fc2(F.gelu(self.fc1(self.ln2(seq)))) * m
        return seq
```

When `impl∈{sdpa,materialized}`, `coords`/`bias_table`/`exact_lut` arrive as `None` and `attn(...)` ignores them (takes the `attn_bias` branch) — identical to the original call.

## EDIT 3 — `HexfieldNet.trunk`: skip build_attn_bias on the inference kernel path

Replace `trunk` (`model.py:335-367`). The conv path (`stem`, `conv_blocks`, `gather_idx`, all `*mask` re-zeros) is **unchanged**. The only change: decide `use_kernel`, and when set, pass `coords`(int32)/`seq_mask`/`bias_table`/`_exact_lut` to the A-blocks and **do not build the `(B,H,S,S)` bias** (the ~70% serve cost the rewrite removes).

```python
    def trunk(
        self,
        feats: torch.Tensor,
        nbr: torch.Tensor,
        mask: torch.Tensor,
        coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (cells (B,Npad,C), tokens (B,8,C), gather_idx) after
        LN_final."""

        b, n, _ = feats.shape
        self_idx = torch.arange(n, device=feats.device).reshape(1, n, 1).expand(b, -1, -1)
        gather_idx = torch.cat([self_idx, nbr], dim=2)  # (B, Npad, 7), tap 0 = self

        x = F.relu(self.stem_ln(self.stem(feats, gather_idx, mask))) * mask.unsqueeze(-1)
        seq_mask = torch.cat([mask.new_ones(b, NUM_TOKENS), mask], dim=1)

        # Inference kernel path (hexflash/flex): reconstruct the rel-pos bias row
        # IN-KERNEL from bias_table + _exact_lut. Skip the (B,heads,S,S)
        # build_attn_bias materialization entirely. Only when serving (no grad):
        # the training path keeps build_attn_bias (the _BiasGather fp32 backward
        # + the test oracle ground truth). coords pass to the kernel as int32.
        impl = self.attn_blocks[0].attn.impl
        use_kernel = impl in ("hexflash", "flex") and not torch.is_grad_enabled()
        if use_kernel:
            attn_bias = None
            kcoords = coords.to(torch.int32)
            kbias = self.bias_table
            klut = self._exact_lut.to(torch.int32)
        else:
            attn_bias = self.build_attn_bias(coords, mask)
            kcoords = kbias = klut = None

        def _attn(block, seq):
            return block(
                seq,
                attn_bias,
                seq_mask,
                coords=kcoords,
                bias_table=kbias,
                exact_lut=klut,
            )

        tokens = self.tokens.unsqueeze(0).expand(b, -1, -1)
        x = self.conv_blocks[0](x, gather_idx, mask)
        x = self.conv_blocks[1](x, gather_idx, mask)
        x = self.conv_blocks[2](x, gather_idx, mask)
        seq = _attn(self.attn_blocks[0], torch.cat([tokens, x], dim=1))
        tokens, x = seq[:, :NUM_TOKENS], seq[:, NUM_TOKENS:]
        x = self.conv_blocks[3](x, gather_idx, mask)
        x = self.conv_blocks[4](x, gather_idx, mask)
        seq = _attn(self.attn_blocks[1], torch.cat([tokens, x], dim=1))
        tokens, x = seq[:, :NUM_TOKENS], seq[:, NUM_TOKENS:]
        x = self.conv_blocks[5](x, gather_idx, mask)
        seq = _attn(self.attn_blocks[2], torch.cat([tokens, x], dim=1))
        seq = self.ln_final(seq)
        tokens, x = seq[:, :NUM_TOKENS], seq[:, NUM_TOKENS:]
        return x * mask.unsqueeze(-1), tokens, gather_idx
```

---

## Unified diff (apply-ready against the live `model.py`)

```diff
--- a/packages/hexfield/python/hexfield/model.py
+++ b/packages/hexfield/python/hexfield/model.py
@@ -155,18 +155,49 @@ class RelPosAttention(nn.Module):
         self.out_proj = nn.Linear(channels, channels)
         self.impl = "sdpa"
 
-    def forward(self, seq: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
+    def forward(
+        self,
+        seq: torch.Tensor,
+        attn_bias: torch.Tensor | None = None,
+        *,
+        coords: torch.Tensor | None = None,
+        seq_mask: torch.Tensor | None = None,
+        bias_table: torch.Tensor | None = None,
+        exact_lut: torch.Tensor | None = None,
+    ) -> torch.Tensor:
         b, s, c = seq.shape
         h, d = self.heads, self.head_dim
+        # q/k/v projection + scale + out_proj are IDENTICAL across all impls;
+        # only the score+softmax+@v core changes kernel. (spec C1 hard rule)
         q = self.q_proj(seq).reshape(b, s, h, d).transpose(1, 2)
         k = self.k_proj(seq).reshape(b, s, h, d).transpose(1, 2)
         v = self.v_proj(seq).reshape(b, s, h, d).transpose(1, 2)
+
+        if self.impl in ("hexflash", "flex"):
+            # Kernel reconstructs the rel-pos bias row in-kernel from the model's
+            # OWN _exact_lut + bias_table; never materializes (B,H,S,S). coords
+            # and seq_mask are threaded from the trunk; build_attn_bias skipped.
+            from .hexflash import flex_attention_relpos, hexflash_attention
+
+            fn = hexflash_attention if self.impl == "hexflash" else flex_attention_relpos
+            out = fn(
+                q,
+                k,
+                v,
+                coords,
+                bias_table,
+                seq_mask,
+                exact_lut,
+                self.scale,
+                NUM_TOKENS,
+            )
+            out = out.transpose(1, 2).reshape(b, s, c)
+            return self.out_proj(out)
+
         # Match the bias dtype to q under autocast: a dtype mismatch silently
         # drops sdpa to the slow math fallback. -3.0e4 stays finite in fp16.
         attn_bias = attn_bias.to(q.dtype)
         if self.impl == "sdpa":
             out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
         elif self.impl == "materialized":
             scores = (q @ k.transpose(-2, -1)) * self.scale + attn_bias
             out = torch.softmax(scores, dim=-1) @ v
         else:  # pragma: no cover - config validation
             raise ValueError(f"unknown attention impl: {self.impl}")
         out = out.transpose(1, 2).reshape(b, s, c)
         return self.out_proj(out)
@@ -186,14 +217,30 @@ class AttnBlock(nn.Module):
         self.fc2 = nn.Linear(MLP_RATIO * channels, channels)
 
     def forward(
-        self, seq: torch.Tensor, attn_bias: torch.Tensor, seq_mask: torch.Tensor
+        self,
+        seq: torch.Tensor,
+        attn_bias: torch.Tensor | None,
+        seq_mask: torch.Tensor,
+        *,
+        coords: torch.Tensor | None = None,
+        bias_table: torch.Tensor | None = None,
+        exact_lut: torch.Tensor | None = None,
     ) -> torch.Tensor:
         m = seq_mask.unsqueeze(-1)
-        seq = seq + self.attn(self.ln1(seq), attn_bias) * m
+        # Pad-QUERY rows are re-zeroed here by *m REGARDLESS of impl, so the
+        # kernel is free to emit garbage in pad-query rows (it does — those rows
+        # are never read). This is the invariant that makes any Npad bit-identical.
+        seq = seq + self.attn(
+            self.ln1(seq),
+            attn_bias,
+            coords=coords,
+            seq_mask=seq_mask,
+            bias_table=bias_table,
+            exact_lut=exact_lut,
+        ) * m
         seq = seq + self.fc2(F.gelu(self.fc1(self.ln2(seq)))) * m
         return seq
@@ -342,26 +389,52 @@ class HexfieldNet(nn.Module):
         """Returns (cells (B,Npad,C), tokens (B,8,C), gather_idx) after
         LN_final."""
 
         b, n, _ = feats.shape
         self_idx = torch.arange(n, device=feats.device).reshape(1, n, 1).expand(b, -1, -1)
         gather_idx = torch.cat([self_idx, nbr], dim=2)  # (B, Npad, 7), tap 0 = self
 
         x = F.relu(self.stem_ln(self.stem(feats, gather_idx, mask))) * mask.unsqueeze(-1)
-        attn_bias = self.build_attn_bias(coords, mask)
         seq_mask = torch.cat([mask.new_ones(b, NUM_TOKENS), mask], dim=1)
 
+        # Inference kernel path (hexflash/flex): reconstruct the rel-pos bias row
+        # IN-KERNEL from bias_table + _exact_lut. Skip the (B,heads,S,S)
+        # build_attn_bias materialization entirely. Only when serving (no grad):
+        # training keeps build_attn_bias (the _BiasGather fp32 backward + the test
+        # oracle ground truth). coords pass to the kernel as int32.
+        impl = self.attn_blocks[0].attn.impl
+        use_kernel = impl in ("hexflash", "flex") and not torch.is_grad_enabled()
+        if use_kernel:
+            attn_bias = None
+            kcoords = coords.to(torch.int32)
+            kbias = self.bias_table
+            klut = self._exact_lut.to(torch.int32)
+        else:
+            attn_bias = self.build_attn_bias(coords, mask)
+            kcoords = kbias = klut = None
+
+        def _attn(block, seq):
+            return block(
+                seq,
+                attn_bias,
+                seq_mask,
+                coords=kcoords,
+                bias_table=kbias,
+                exact_lut=klut,
+            )
+
         tokens = self.tokens.unsqueeze(0).expand(b, -1, -1)
         x = self.conv_blocks[0](x, gather_idx, mask)
         x = self.conv_blocks[1](x, gather_idx, mask)
         x = self.conv_blocks[2](x, gather_idx, mask)
-        seq = self.attn_blocks[0](torch.cat([tokens, x], dim=1), attn_bias, seq_mask)
+        seq = _attn(self.attn_blocks[0], torch.cat([tokens, x], dim=1))
         tokens, x = seq[:, :NUM_TOKENS], seq[:, NUM_TOKENS:]
         x = self.conv_blocks[3](x, gather_idx, mask)
         x = self.conv_blocks[4](x, gather_idx, mask)
-        seq = self.attn_blocks[1](torch.cat([tokens, x], dim=1), attn_bias, seq_mask)
+        seq = _attn(self.attn_blocks[1], torch.cat([tokens, x], dim=1))
         tokens, x = seq[:, :NUM_TOKENS], seq[:, NUM_TOKENS:]
         x = self.conv_blocks[5](x, gather_idx, mask)
-        seq = self.attn_blocks[2](torch.cat([tokens, x], dim=1), attn_bias, seq_mask)
+        seq = _attn(self.attn_blocks[2], torch.cat([tokens, x], dim=1))
         seq = self.ln_final(seq)
         tokens, x = seq[:, :NUM_TOKENS], seq[:, NUM_TOKENS:]
         return x * mask.unsqueeze(-1), tokens, gather_idx
```

---

## Pad-inertness audit (my charter's core requirement)

Every invariant in the spec's "exactness trio" survives, and I verified each against the source:

1. **Pad QUERY rows re-zeroed** — `AttnBlock.forward` keeps `* m` (`model.py:193`, preserved in EDIT 2) and `trunk` keeps the final `x * mask.unsqueeze(-1)` (`model.py:367`). The kernel may emit garbage in pad-query rows; it is multiplied out before any conv or head reads it. **This is why a kernel that pads `head_dim` 24→32 and computes junk in pad rows is still bit-identical.**
2. **Pad KEY columns masked** — moves *into* the kernel (it adds `PAD_KEY_MASK_VALUE` where `seq_mask` is false, per spec §A1), so `build_attn_bias`'s `fill` (`model.py:329`) is no longer needed on this path. The `seq_mask` I thread is the exact same `torch.cat([ones(b,NUM_TOKENS), mask], dim=1)` (`model.py:351`) — token keys stay live, so a fully-masked softmax row remains structurally impossible.
3. **Conv pad-inertness** — `HexNodeConv` UNCHANGED: appended zero row at index `Npad` + `out * mask` (`model.py:116-120`). `gather_idx` UNCHANGED (`self_idx = arange(n)` tap-0, `nbr` sentinel→`Npad`, `model.py:346-347`) — this is the convention the Rust v2 gather-index remap (Implementer 5) must match.

**Conv layout note for the orchestrator:** because I keep the padded layout, the Rust v2 ABI's "flat int32 gather-index" must still produce, after the dense scatter to `(g, pad_to, 7)`, the same `[self_idx ; nbr→pad_to]` tensor `trunk` builds at `model.py:346-347`. My `trunk` constructs `self_idx` itself and concatenates `nbr`; it does **not** consume a precomputed 7-tap gather index. If B2 wants to ship the full 7-tap index, `trunk` would need a parallel "gather_idx provided" branch — but per §C3 the dense `(g,pad_to,·)` default keeps the v1 `nbr` field, so no trunk change is required for the default path. I am flagging this as the one cross-component contract point.

## What is statically certain vs needs the GPU pause

- **Statically certain (no GPU):** the `sdpa`/`materialized` paths are byte-identical (branches are purely additive; `attn_bias` flows unchanged) → existing `test_sdpa_equals_materialized` passes by construction. `HexNodeConv`/`ConvBlock` unmodified → conv oracle passes. `coords.to(int32)`/`_exact_lut.to(int32)` are lossless (board offsets `|q|,|r|<~60`; LUT rows 0..236).
- **Needs the GPU pause:** the actual `impl="hexflash"`/`"flex"` numeric output (depends on the kernel from Implementers 1/2) against the Tier-2 fp16 oracle (`diff <= 2e-3`). My trunk code only guarantees the *inputs* handed to the kernel are correct (right coords slot convention, right seq_mask, int32 dtype, build_attn_bias skipped under no-grad only).

## Files
- Changed (text above; do NOT write to live tree): `E:\Hexo-BotTrainer-hexgt\packages\hexfield\python\hexfield\model.py` — `RelPosAttention.forward`, `AttnBlock.forward`, `HexfieldNet.trunk`.
- Unchanged by design: `HexNodeConv`, `ConvBlock`, `build_attn_bias`, all heads, `constants.py`.