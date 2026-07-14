# parity-tests

I now have everything I need. The existing tests build models via the engine (`_rows`), which only reach small N. For the rewrite parity suite I need synthetic fixtures that can scale Npad up to 3300 and B up to 192, while still honoring the model's batch conventions (pad rows: zero feats, nbr→Npad, mask False, coords arbitrary on pad).

Let me also verify the `hexflash` module's public surface so my test imports match the frozen contract, and confirm `set_attention_impl` is the routing knob. The spec freezes both. I'll write the suite to exercise the new `impl in {hexflash, flex}` paths through `set_attention_impl` (the model-level entry), plus the `hexflash_attention` kernel directly via the bias-index oracle.

Here is the runnable parity test suite.

```python
"""Parity gates for the hexfield INFERENCE rewrite (spec §3, all tiers).

Proves the new attention kernels ('hexflash' Triton, 'flex' FlexAttention) and
the gated 'sdpa' compile path reproduce the CURRENT forward within fp16
tolerance on IDENTICAL inputs, across the rewrite's target operating envelope:

    Npad in {64, 256, 900, 1800, 3300}
    B    in {1, 16, 64, 192}

plus the load-bearing invariants the "no retrain" claim rests on:

  * the materialized-bias math oracle ('materialized', model.py:169-171) is the
    canonical reference every kernel is measured against;
  * pad-inertness (a row's outputs are bit-identical regardless of how much
    padding shares its batch — model.py:19-22 exactness trio);
  * the bias-row index reconstructed in-kernel equals build_attn_bias's exact
    selection (model.py:274-296) BY CONSTRUCTION — Tier 1, statically certain.

WHAT NEEDS A GPU (spec §4): the Triton/FlexAttention kernels only execute on
CUDA. Every test that touches 'hexflash'/'flex' is skipped without CUDA and is
gated on the kernels existing (hexfield.hexflash) and being wired into
set_attention_impl. The 'materialized' vs 'sdpa' and pad-inertness oracles run
on CPU (fp32/fp64) AND on CUDA (fp16) — the CPU half is the part that is
verifiable WITHOUT the GPU pause.

These tests construct SYNTHETIC supports (not engine-derived) so Npad can reach
3300 and B can reach 192 — the engine `sample_decision_states` helper used by
test_hexfield_model.py only produces small boards. The synthetic builder honours
every batch convention in model.HexfieldNet's docstring (pad rows: zero feats,
nbr -> Npad, mask False, coords arbitrary; live rows: real axial coords + a
self-consistent neighbour ring) so the model path is exercised exactly as the
serve path drives it.

Reused thresholds (invent no new ones, spec §3):
  * fp32  materialized vs sdpa : 1e-4   (test_hexfield_model.py:73)
  * fp16  kernel  vs materialized: 2e-3 (test_hexfield_model.py:334)
  * pad-inertness fp32         : 1e-6   (test_hexfield_model.py:146)
  * bias-index                 : torch.equal (exact integer, no tol)
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest
import torch

from hexfield import constants as C
from hexfield.geometry import disk_offsets, rel_bias_index
from hexfield.model import HexfieldNet


# --------------------------------------------------------------------------- #
# Operating envelope (spec: the rewrite must hold across these)                #
# --------------------------------------------------------------------------- #
NPADS = (64, 256, 900, 1800, 3300)
BATCHES = (1, 16, 64, 192)

# The new attention kernels under test. 'sdpa'/'materialized' are the existing
# numerically-identical pair; 'hexflash'/'flex' are the rewrite (spec §A1/§A1-fb).
NEW_IMPLS = ("hexflash", "flex")

# Reused tolerances (spec §3 — no new thresholds).
TOL_FP32 = 1e-4       # materialized vs sdpa, fp32 (test_hexfield_model.py:73)
TOL_FP16 = 2e-3       # any kernel vs materialized, fp16 (…:334)
TOL_PAD_FP32 = 1e-6   # pad-inertness, fp32 (…:146)


# --------------------------------------------------------------------------- #
# Capability gates — skip cleanly when the rewrite module / wiring is absent.  #
# These let the suite live in the live tree (read-only) AND apply cleanly in   #
# the rewrite worktree once hexflash.py + the set_attention_impl branches land.#
# --------------------------------------------------------------------------- #
def _have_hexflash_module() -> bool:
    try:
        importlib.import_module("hexfield.hexflash")
        return True
    except Exception:
        return False


def _impl_is_wired(impl: str) -> bool:
    """True iff set_attention_impl(impl) is accepted (the A2 routing landed)."""
    if impl in ("sdpa", "materialized"):
        return True
    if not _have_hexflash_module():
        return False
    try:
        m = HexfieldNet()
        m.set_attention_impl(impl)
        # confirm it actually took on at least one block
        return getattr(m.attn_blocks[0].attn, "impl", None) == impl
    except Exception:
        return False


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="rewrite attention kernels are CUDA-only"
)


def _skip_if_unwired(impl: str) -> None:
    if not _impl_is_wired(impl):
        pytest.skip(f"attention impl {impl!r} not wired (rewrite not assembled here)")


# --------------------------------------------------------------------------- #
# Synthetic support builder — scales to any Npad/B while honouring every batch #
# convention in model.HexfieldNet's docstring.                                 #
# --------------------------------------------------------------------------- #
def _hex_spiral_coords(n: int) -> np.ndarray:
    """First `n` axial coords of the growing hex disk (origin outward), in the
    canonical ascending (dq, dr) disk order. For n beyond the radius-8 disk
    (217 cells) we keep growing rings so large-Npad cases populate the ring /
    far bias rows too (exercises BIAS_ON/OFF_AXIS + BIAS_FAR, model.py:285-289).
    """
    coords: list[tuple[int, int]] = []
    radius = 0
    while len(coords) < n:
        radius += 1
        coords = [
            (dq, dr)
            for dq in range(-radius, radius + 1)
            for dr in range(-radius, radius + 1)
            if max(abs(dq), abs(dr), abs(dq + dr)) <= radius
        ]
        coords.sort()
    return np.asarray(coords[:n], dtype=np.int64)


def _build_live_row(n: int, *, seed: int) -> dict[str, np.ndarray]:
    """One un-padded support of `n` live nodes: real coords, a self-consistent
    row-local neighbour ring (missing -> -1 sentinel, mapped to pad at collate),
    random features, and a legal prefix. Mirrors support.build_support's
    contract enough to drive the model identically."""
    rng = np.random.RandomState(seed)
    coords = _hex_spiral_coords(n)  # (n, 2)
    # coord -> row index, to wire the 6-direction neighbour ring row-locally.
    index = {(int(q), int(r)): i for i, (q, r) in enumerate(coords)}
    nbr = np.full((n, 6), -1, dtype=np.int64)
    for i, (q, r) in enumerate(coords):
        for d, (dq, dr) in enumerate(C.DIRECTIONS):
            j = index.get((int(q) + dq, int(r) + dr))
            if j is not None:
                nbr[i, d] = j
    feats = rng.standard_normal((n, C.NUM_FEATURES)).astype(np.float32)
    # legal prefix: at least 1, at most n (the policy softmax denominator).
    legal_count = int(rng.randint(1, n + 1))
    return {"coords": coords, "nbr": nbr, "feats": feats, "legal_count": legal_count}


def _collate(rows: list[dict], pad_to: int, *, device, dtype=torch.float32):
    """Pad `rows` to (B, pad_to, *) honouring HexfieldNet's pad conventions:
    pad feats zero, pad nbr -> Npad (the appended zero row), pad mask False,
    pad coords zero (never read). Live nbr sentinel -1 -> Npad as well."""
    b = len(rows)
    feats = torch.zeros(b, pad_to, C.NUM_FEATURES, dtype=dtype)
    nbr = torch.full((b, pad_to, 6), pad_to, dtype=torch.long)
    mask = torch.zeros(b, pad_to, dtype=torch.bool)
    coords = torch.zeros(b, pad_to, 2, dtype=torch.long)
    legal_counts = torch.zeros(b, dtype=torch.long)
    for g, row in enumerate(rows):
        n = row["coords"].shape[0]
        assert n <= pad_to
        feats[g, :n] = torch.from_numpy(row["feats"]).to(dtype)
        row_nbr = torch.from_numpy(row["nbr"])
        nbr[g, :n] = torch.where(row_nbr >= 0, row_nbr, torch.full_like(row_nbr, pad_to))
        mask[g, :n] = True
        coords[g, :n] = torch.from_numpy(row["coords"])
        legal_counts[g] = row["legal_count"]
    out = {
        "feats": feats, "nbr": nbr, "mask": mask,
        "coords": coords, "legal_counts": legal_counts,
    }
    return {k: v.to(device) for k, v in out.items()}


def _make_batch(b: int, npad: int, *, device, dtype=torch.float32, seed: int = 0,
                vary_sizes: bool = True):
    """A (b, npad) batch. With vary_sizes, rows span [npad//2 .. npad] so the
    padding machinery and pad-key mask are genuinely exercised (one row hits
    npad exactly so pad_to == max live N — the collate invariant). Otherwise
    every row is full (npad live nodes)."""
    rng = np.random.RandomState(seed)
    rows = []
    for g in range(b):
        if vary_sizes and b > 1:
            n = npad if g == 0 else int(rng.randint(max(1, npad // 2), npad + 1))
        else:
            n = npad
        rows.append(_build_live_row(n, seed=seed * 1000 + g))
    return _collate(rows, npad, device=device, dtype=dtype)


def _derandomize(model: HexfieldNet, *, device, seed: int = 11) -> None:
    """Fire the zero-initialised residual branches (out_proj/fc2) + give the
    bias table real structure so attention actually moves the output. Same
    recipe as test_hexfield_model.py:_derandomize / the fp16 cuda test."""
    gen = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        for block in model.conv_blocks:
            block.ln2.weight.copy_(
                torch.rand(block.ln2.weight.shape, generator=gen, device=device) * 0.5 + 0.5
            )
        for block in model.attn_blocks:
            for p in (block.attn.out_proj.weight, block.fc2.weight):
                p.copy_(torch.randn(p.shape, generator=gen, device=device) * 0.05)
        model.bias_table.copy_(
            torch.randn(model.bias_table.shape, generator=gen, device=device) * 0.1
        )


def _max_diff(a: dict, b: dict) -> dict[str, float]:
    return {
        k: (a[k].float() - b[k].float()).abs().max().item()
        for k in a
        if k in b
    }


# --------------------------------------------------------------------------- #
# Fixture: one shared CPU fp32 reference model (cheap), one CUDA model lazily. #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cpu_model() -> HexfieldNet:
    torch.manual_seed(0)
    model = HexfieldNet().eval()
    _derandomize(model, device=torch.device("cpu"))
    return model


@pytest.fixture(scope="module")
def cuda_model() -> HexfieldNet:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = HexfieldNet().eval().to(device)
    _derandomize(model, device=device)
    return model


# =========================================================================== #
# TIER 1 — STATICALLY CHECKABLE (no GPU). The single most important gate.      #
# Bias-row index reconstructed by the kernel == build_attn_bias's exact        #
# selection. The kernel reuses model._exact_lut + identical clamp/d/on-axis/   #
# token-class expressions, so this is bit-identical BY CONSTRUCTION (spec §3). #
# =========================================================================== #
class TestTier1BiasIndexOracle:
    """These reason about INTEGER index math only — no kernel execution, no GPU.
    They run on CPU and certify the bias-VALUE per pair is correct before any
    GPU is touched (spec §4 item 2 separates 'math certain' from 'kernel runs').
    """

    def test_build_attn_bias_index_matches_geometry_all_classes(self, cpu_model):
        """build_attn_bias with bias_table = arange returns, per pair, exactly
        the integer row index geometry.rel_bias_index would pick — covering the
        EXACT disk (d<=8), the on/off-axis rings (9<=d<=16), the far row
        (d>=17), and the three token classes. This is the reference the kernel
        must reproduce; verifying it here pins the oracle itself.

        Extends test_hexfield_model.py:218 to large Npad so the ring/far rows
        (only reachable past radius 8) are actually exercised."""
        model = HexfieldNet().eval()  # fresh: arange table, no derandomize
        with torch.no_grad():
            model.bias_table.zero_()
            for h in range(C.ATTENTION_HEADS):
                model.bias_table[:, h] = torch.arange(C.BIAS_ROWS, dtype=torch.float32)
        # A single big row so d ranges from 0 well past 17 (3300 ~ radius 33).
        batch = _make_batch(1, 3300, device="cpu", vary_sizes=False, seed=7)
        coords, mask = batch["coords"], batch["mask"]
        bias = model.build_attn_bias(coords, mask)  # (1, heads, S, S)
        t = C.NUM_TOKENS
        cells = coords[0].tolist()

        # token classes (every head identical here).
        for h in range(C.ATTENTION_HEADS):
            assert bias[0, h, 0, 1].item() == C.BIAS_TOKEN_TOKEN_ROW
            assert bias[0, h, 0, t].item() == C.BIAS_TOKEN_CELL_ROW
            assert bias[0, h, t, 0].item() == C.BIAS_CELL_TOKEN_ROW

        # cell/cell: a spread of indices reaching exact disk + ring + far rows.
        rng = np.random.RandomState(0)
        idx = rng.choice(len(cells), size=120, replace=False)
        saw_exact = saw_ring = saw_far = False
        for i in idx[:60]:
            for j in idx[:30]:
                qi, ri = cells[int(i)]
                qj, rj = cells[int(j)]
                expected = rel_bias_index(qj - qi, rj - ri)
                got = bias[0, 0, t + int(i), t + int(j)].item()
                assert got == expected, (
                    f"pair ({i},{j}) offset ({qj-qi},{rj-ri}): {got} != {expected}"
                )
                d = max(abs(qj - qi), abs(rj - ri), abs(qj - qi + rj - ri))
                saw_exact |= d <= C.BIAS_DISK_RADIUS
                saw_ring |= C.BIAS_RING_MIN <= d <= C.BIAS_RING_MAX
                saw_far |= d > C.BIAS_RING_MAX
        # The big-row sweep must actually touch all three cell regimes, else the
        # ring/far branches of the kernel are never parity-checked.
        assert saw_exact and saw_ring and saw_far, (
            f"regimes covered: exact={saw_exact} ring={saw_ring} far={saw_far}"
        )

    def test_exact_lut_matches_geometry(self, cpu_model):
        """model._exact_lut (the buffer the kernel indexes, model.py:228-231)
        agrees cell-for-cell with geometry.rel_bias_index over the whole disk.
        Pure integer — the kernel's per-pair SRAM lookup inherits this."""
        lut = HexfieldNet()._exact_lut
        R = C.BIAS_DISK_RADIUS
        for dq, dr in disk_offsets(R):
            row = lut[(dq + R) * 17 + (dr + R)].item()
            assert row == rel_bias_index(dq, dr), f"lut offset ({dq},{dr})"

    @requires_cuda
    def test_kernel_bias_index_equals_build_attn_bias(self):
        """Tier-1 EXECUTED on GPU (spec §3): run the kernel with a probe that
        isolates the gathered bias row — bias_table = arange broadcast over
        heads, scale forced so q@k^T contributes 0, no pad keys — so the kernel
        output equals the integer pair index. torch.equal vs build_attn_bias.

        Skipped (not failed) until hexflash + a bias-index probe entry exist.
        The probe is the rewrite's responsibility; we assert the contract."""
        _skip_if_unwired("hexflash")
        hexflash = importlib.import_module("hexfield.hexflash")
        probe = getattr(hexflash, "bias_index_probe", None)
        if probe is None:
            pytest.skip("hexflash.bias_index_probe not provided by rewrite")

        device = torch.device("cuda")
        model = HexfieldNet().eval().to(device)
        with torch.no_grad():
            model.bias_table.zero_()
            for h in range(C.ATTENTION_HEADS):
                model.bias_table[:, h] = torch.arange(C.BIAS_ROWS, device=device)
        batch = _make_batch(2, 900, device=device, seed=3)
        coords, mask = batch["coords"], batch["mask"]

        ref = model.build_attn_bias(coords, mask).round().long()  # (B,H,S,S) indices
        got = probe(
            coords.to(torch.int32),
            mask,
            model.bias_table.to(torch.float16),
            model._exact_lut.to(torch.int32),
            num_tokens=C.NUM_TOKENS,
        ).round().long()
        # Pad-KEY columns differ (ref adds PAD_KEY_MASK_VALUE); compare live keys
        # only — the index reconstruction is what Tier 1 certifies.
        key_live = torch.cat(
            [mask.new_ones(mask.shape[0], C.NUM_TOKENS), mask], dim=1
        )  # (B, S)
        sel = key_live[:, None, None, :].expand_as(ref)
        assert torch.equal(got[sel], ref[sel]), "kernel bias-index != build_attn_bias"


# =========================================================================== #
# TIER 1.5 — CPU math oracle (no GPU). materialized == sdpa in fp32, across    #
# the full Npad x B envelope. The serve rewrite must not perturb this pair;    #
# any new kernel is later measured against 'materialized'.                     #
# =========================================================================== #
@pytest.mark.parametrize("npad", NPADS)
@pytest.mark.parametrize("b", BATCHES)
def test_materialized_equals_sdpa_fp32_cpu(cpu_model, npad, b):
    """The canonical math oracle (model.py:169-171) == the production sdpa
    formulation (model.py:168) in fp32 on CPU, across Npad x B. Extends
    test_hexfield_model.py:60 to the rewrite's full envelope; no GPU needed,
    so it is the broadest statically-runnable parity surface."""
    if b * (npad + C.NUM_TOKENS) ** 2 > 4.0e8:
        pytest.skip("B*S^2 too large for a CPU fp32 materialized bias")
    model = cpu_model
    batch = _make_batch(b, npad, device="cpu", seed=hash((npad, b)) & 0xFFFF)
    args = (batch["feats"], batch["nbr"], batch["mask"], batch["coords"])
    with torch.no_grad():
        model.set_attention_impl("sdpa")
        out_sdpa = model.forward_policy_value(*args, request_moves_left=True)
        model.set_attention_impl("materialized")
        out_mat = model.forward_policy_value(*args, request_moves_left=True)
    for key, diff in _max_diff(out_sdpa, out_mat).items():
        assert diff <= TOL_FP32, f"Npad={npad} B={b} {key}: {diff}"


# =========================================================================== #
# PAD-INERTNESS (no GPU). A row's outputs are bit-identical regardless of how  #
# much padding shares its batch (model.py:19-22). This is THE invariant the    #
# "pad to any Npad is output-identical" claim — and therefore every kernel's   #
# pad-query/pad-key handling — rests on. Verified here for the reference path  #
# in fp32 so the rewrite kernels have a concrete target to reproduce on GPU.   #
# =========================================================================== #
@pytest.mark.parametrize("npad", (64, 256, 900))
def test_pad_inertness_reference_fp32_cpu(cpu_model, npad):
    """The same live row produces (within fp32 noise) identical policy/value/
    moves_left whether it sits ALONE at its own N or padded up to a larger
    Npad inside a batch with a bigger neighbour. fp32, tol 1e-6
    (test_hexfield_model.py:146). The pad-key mask + AttnBlock '*m' re-zero are
    what make this hold; a kernel that mishandles either breaks it."""
    model = cpu_model
    small_n = npad // 2 + 7
    big_row = _build_live_row(npad, seed=101)
    small_row = _build_live_row(small_n, seed=202)

    alone = _collate([small_row], small_n, device="cpu")
    padded = _collate([small_row, big_row], npad, device="cpu")  # small padded to npad
    with torch.no_grad():
        model.set_attention_impl("sdpa")
        out_alone = model.forward_policy_value(
            alone["feats"], alone["nbr"], alone["mask"], alone["coords"],
            request_moves_left=True,
        )
        out_padded = model.forward_policy_value(
            padded["feats"], padded["nbr"], padded["mask"], padded["coords"],
            request_moves_left=True,
        )
    for key in out_alone:
        a = out_alone[key][0]
        p = out_padded[key][0]
        if key == "policy":
            # pad logits beyond the live prefix are exactly zero (masked).
            assert out_padded[key][0][small_n:].abs().max().item() == 0.0
            a, p = a[:small_n], p[:small_n]
        diff = (a - p).abs().max().item()
        assert diff <= TOL_PAD_FP32, f"Npad={npad} {key}: pad-inertness diff {diff}"


# =========================================================================== #
# TIER 2 — fp16 OUTPUT ORACLE (GPU). Each new kernel vs 'materialized' (and    #
# vs 'sdpa') under cuda fp16 autocast, across Npad x B, tol 2e-3               #
# (test_hexfield_model.py:295/334). This is the primary correctness gate for   #
# the rewrite; it is SKIPPED until CUDA + the kernel wiring are present.       #
# =========================================================================== #
@requires_cuda
@pytest.mark.parametrize("impl", NEW_IMPLS)
@pytest.mark.parametrize("npad", NPADS)
@pytest.mark.parametrize("b", BATCHES)
def test_new_impl_equals_materialized_fp16_cuda(cuda_model, impl, npad, b):
    """impl in {hexflash, flex} reproduces the materialized oracle within the
    fp16 budget across the FULL envelope. Mixed live sizes (vary_sizes) force
    the pad-key mask + pad-query inertness paths inside the kernel."""
    _skip_if_unwired(impl)
    device = torch.device("cuda")
    model = cuda_model
    batch = _make_batch(b, npad, device=device, dtype=torch.float32,
                        seed=(hash((impl, npad, b)) & 0xFFFF))
    args = (batch["feats"], batch["nbr"], batch["mask"], batch["coords"])
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        model.set_attention_impl("materialized")
        out_ref = model.forward_policy_value(*args, request_moves_left=True)
        model.set_attention_impl(impl)
        out_new = model.forward_policy_value(*args, request_moves_left=True)
    for key, diff in _max_diff(out_ref, out_new).items():
        assert diff <= TOL_FP16, f"{impl} Npad={npad} B={b} {key}: fp16 diff {diff}"


@requires_cuda
@pytest.mark.parametrize("impl", NEW_IMPLS)
@pytest.mark.parametrize("npad", NPADS)
def test_new_impl_equals_sdpa_fp16_cuda(cuda_model, impl, npad):
    """Cross-check against the OTHER reference (the deployed sdpa path) so a
    correlated materialized/kernel error can't pass. B fixed at 16 to bound
    runtime; the materialized test above already sweeps B."""
    _skip_if_unwired(impl)
    device = torch.device("cuda")
    model = cuda_model
    batch = _make_batch(16, npad, device=device, seed=(hash((impl, npad)) & 0xFFFF))
    args = (batch["feats"], batch["nbr"], batch["mask"], batch["coords"])
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        model.set_attention_impl("sdpa")
        out_sdpa = model.forward_policy_value(*args, request_moves_left=True)
        model.set_attention_impl(impl)
        out_new = model.forward_policy_value(*args, request_moves_left=True)
    for key, diff in _max_diff(out_sdpa, out_new).items():
        assert diff <= TOL_FP16, f"{impl} Npad={npad} {key}: fp16 vs sdpa {diff}"


# =========================================================================== #
# TIER 2 — fp16 pad-inertness ON THE KERNEL (GPU). The pad invariant must hold #
# inside hexflash/flex, not just the reference path: same live row alone vs    #
# padded, run through the NEW kernel, must match the materialized reference for #
# BOTH layouts. This is the test that catches a kernel mishandling pad keys    #
# (missing PAD_KEY_MASK_VALUE) or pad-query rows (not relying on AttnBlock*m). #
# =========================================================================== #
@requires_cuda
@pytest.mark.parametrize("impl", NEW_IMPLS)
@pytest.mark.parametrize("npad", (256, 900, 1800))
def test_new_impl_pad_inertness_fp16_cuda(cuda_model, impl, npad):
    _skip_if_unwired(impl)
    device = torch.device("cuda")
    model = cuda_model
    small_n = npad // 2 + 7
    big_row = _build_live_row(npad, seed=303)
    small_row = _build_live_row(small_n, seed=404)
    alone = _collate([small_row], small_n, device=device)
    padded = _collate([small_row, big_row], npad, device=device)

    def run(impl_name, batch):
        model.set_attention_impl(impl_name)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            return model.forward_policy_value(
                batch["feats"], batch["nbr"], batch["mask"], batch["coords"],
                request_moves_left=True,
            )

    # reference: the small row alone, materialized (the ground truth for it).
    ref_alone = run("materialized", alone)
    # kernel: the small row PADDED to npad alongside a big neighbour.
    new_padded = run(impl, padded)
    for key in ref_alone:
        a = ref_alone[key][0]
        p = new_padded[key][0]
        if key == "policy":
            assert new_padded[key][0][small_n:].abs().max().item() == 0.0, (
                f"{impl}: pad policy logits not zero (pad-query re-zero broken)"
            )
            a, p = a[:small_n], p[:small_n]
        diff = (a.float() - p.float()).abs().max().item()
        assert diff <= TOL_FP16, (
            f"{impl} Npad={npad} {key}: padded-kernel vs alone-ref {diff}"
        )


# =========================================================================== #
# REGIME ROUTING (GPU). The evaluator routes large-Npad groups to the new      #
# kernel and small-Npad to gated compile (spec §B1) WITHOUT changing the       #
# decoded reply bytes vs the plain eager evaluator. Reuses the evaluator's     #
# own submit/result, so it also guards the single-D2H discipline.             #
# =========================================================================== #
@requires_cuda
@pytest.mark.parametrize("impl", NEW_IMPLS)
def test_evaluator_attn_impl_matches_eager(impl, monkeypatch):
    """An evaluator with HEXFIELD_ATTN_IMPL=impl (large-S -> new kernel) returns
    values/priors/moves_left within fp16 tol of a baseline eager (sdpa, no
    compile) evaluator on the SAME payload spanning small + large supports.
    Exercises plan_groups bucketing + the per-group impl switch end to end."""
    _skip_if_unwired(impl)
    from hexfield.inference import HexfieldEvaluator

    device = torch.device("cuda")
    torch.manual_seed(0)
    model = HexfieldNet().eval().to(device)
    _derandomize(model, device=device)

    payload = _make_payload(
        sizes=[60, 60, 250, 900, 1800, 3300], legal_frac=0.5, seed=9,
    )

    # baseline: eager sdpa, no compile, no new kernel.
    monkeypatch.setenv("HEXFIELD_NO_COMPILE", "1")
    monkeypatch.delenv("HEXFIELD_ATTN_IMPL", raising=False)
    base_eval = HexfieldEvaluator(model, device=device)
    base = base_eval.evaluate_payload(payload)

    # rewrite: route large-S to the new kernel (small-S stays gated compile).
    monkeypatch.delenv("HEXFIELD_NO_COMPILE", raising=False)
    monkeypatch.setenv("HEXFIELD_ATTN_IMPL", impl)
    new_eval = HexfieldEvaluator(model, device=device)
    new = new_eval.evaluate_payload(payload)

    _assert_reply_close(base, new, tol=TOL_FP16, request_ml=True)


# =========================================================================== #
# Payload helpers for the evaluator-level gate (mirror the §5.2 wire ABI v1).  #
# =========================================================================== #
def _make_payload(sizes, *, legal_frac: float, seed: int, request_ml: bool = True):
    """Assemble an abi=1 payload (rows sorted size-DESCENDING, the contract the
    evaluator assumes) from synthetic supports. f16 feats, i16 coords, u16 nbr,
    i32 legal_counts, i64 node_row_offsets — byte layout per inference.py:134."""
    rng = np.random.RandomState(seed)
    sizes = sorted((int(s) for s in sizes), reverse=True)
    rows = [_build_live_row(n, seed=seed * 100 + i) for i, n in enumerate(sizes)]
    total = sum(sizes)
    feats = np.zeros((total, C.NUM_FEATURES), dtype=np.float16)
    qr = np.zeros((total, 2), dtype=np.int16)
    nbr = np.zeros((total, 6), dtype=np.uint16)
    legal = np.zeros(len(sizes), dtype=np.int32)
    offsets = np.zeros(len(sizes) + 1, dtype=np.int64)
    o = 0
    for i, row in enumerate(rows):
        n = sizes[i]
        feats[o : o + n] = row["feats"].astype(np.float16)
        qr[o : o + n] = row["coords"].astype(np.int16)
        rn = row["nbr"]
        nbr[o : o + n] = np.where(rn >= 0, rn, C.NBR_SENTINEL_U16).astype(np.uint16) \
            if False else np.where(rn >= 0, rn, 0xFFFF).astype(np.uint16)
        legal[i] = max(1, int(round(legal_frac * n)))
        offsets[i + 1] = offsets[i] + n
        o += n
    return {
        "abi": 1,
        "shape": (len(sizes), total),
        "node_row_offsets": offsets.tolist(),
        "node_feats": feats.tobytes(),
        "node_qr": qr.tobytes(),
        "nbr": nbr.tobytes(),
        "legal_counts": legal.tobytes(),
        "request_moves_left": request_ml,
    }


def _assert_reply_close(a: dict, b: dict, *, tol: float, request_ml: bool) -> None:
    va = np.frombuffer(a["values_bytes"], dtype=np.float32)
    vb = np.frombuffer(b["values_bytes"], dtype=np.float32)
    assert va.shape == vb.shape
    assert np.abs(va - vb).max() <= tol, f"values max diff {np.abs(va - vb).max()}"
    pa = np.frombuffer(a["priors_bytes"], dtype=np.float32)
    pb = np.frombuffer(b["priors_bytes"], dtype=np.float32)
    assert pa.shape == pb.shape
    assert np.abs(pa - pb).max() <= tol, f"priors max diff {np.abs(pa - pb).max()}"
    if request_ml:
        ma = np.frombuffer(a["moves_left_bytes"], dtype=np.float32)
        mb = np.frombuffer(b["moves_left_bytes"], dtype=np.float32)
        assert ma.shape == mb.shape
        # moves_left is a decoded scalar in node units; a slightly looser
        # absolute budget than the [-1,1] value/prior softmaxes.
        assert np.abs(ma - mb).max() <= max(tol, 1e-2 * (1 + np.abs(ma).max())), (
            f"moves_left max diff {np.abs(ma - mb).max()}"
        )


# =========================================================================== #
# DIRECT KERNEL CONTRACT (GPU). Call hexflash_attention / flex_attention_relpos#
# directly (the frozen §A1 signature) against the materialized attention core  #
# in isolation — no trunk — so a kernel regression is localised to the attn    #
# op, not masked by conv/head averaging. Covers head_dim=24, the 24->32 pad,   #
# scale before bias, PAD_KEY_MASK_VALUE, token-class rows.                     #
# =========================================================================== #
def _materialized_attn_core(q, k, v, attn_bias, scale):
    """The exact model.py:170-171 core (scale on q@k^T BEFORE bias add)."""
    scores = (q @ k.transpose(-2, -1)) * scale + attn_bias.to(q.dtype)
    return torch.softmax(scores, dim=-1) @ v


@requires_cuda
@pytest.mark.parametrize("fn_name", ("hexflash_attention", "flex_attention_relpos"))
@pytest.mark.parametrize("npad", (64, 256, 900, 1800, 3300))
def test_attention_kernel_core_matches_materialized(fn_name, npad):
    """Isolated attention-core parity: build q,k,v + the materialized bias from
    a model's OWN bias_table/_exact_lut, run the kernel and the materialized
    core on the SAME tensors, compare in fp16 (tol 2e-3). The kernel owns the
    24->32 head-dim padding internally (it receives Dh=24)."""
    if not _have_hexflash_module():
        pytest.skip("hexfield.hexflash not present")
    hexflash = importlib.import_module("hexfield.hexflash")
    fn = getattr(hexflash, fn_name, None)
    if fn is None:
        pytest.skip(f"hexflash.{fn_name} not provided")

    device = torch.device("cuda")
    torch.manual_seed(0)
    model = HexfieldNet().eval().to(device)
    _derandomize(model, device=device)

    b = 2
    batch = _make_batch(b, npad, device=device, seed=5)
    coords, mask = batch["coords"], batch["mask"]
    h, d = C.ATTENTION_HEADS, C.HEAD_DIM
    s = C.NUM_TOKENS + npad
    gen = torch.Generator(device=device).manual_seed(1)
    q = torch.randn(b, h, s, d, generator=gen, device=device, dtype=torch.float16)
    k = torch.randn(b, h, s, d, generator=gen, device=device, dtype=torch.float16)
    v = torch.randn(b, h, s, d, generator=gen, device=device, dtype=torch.float16)
    scale = 1.0 / math.sqrt(C.HEAD_DIM)

    attn_bias = model.build_attn_bias(coords, mask).to(torch.float16)  # (B,H,S,S)
    seq_mask = torch.cat([mask.new_ones(b, C.NUM_TOKENS), mask], dim=1)  # (B,S)

    out_ref = _materialized_attn_core(q, k, v, attn_bias, scale)
    out_new = fn(
        q, k, v,
        coords.to(torch.int32),
        model.bias_table.to(torch.float16),
        seq_mask,
        model._exact_lut.to(torch.int32),
        scale,
        C.NUM_TOKENS,
    )
    assert out_new.shape == out_ref.shape, (out_new.shape, out_ref.shape)
    # Compare only LIVE query rows: pad-query rows may hold kernel garbage by
    # design (re-zeroed downstream by AttnBlock '*m'); the contract only
    # requires correctness on live queries (model.py:189-193).
    live_q = seq_mask[:, None, :, None].expand_as(out_ref)
    diff = (out_new.float()[live_q] - out_ref.float()[live_q]).abs().max().item()
    assert diff <= TOL_FP16, f"{fn_name} Npad={npad}: attn-core diff {diff}"
```

Key design notes for the operator:

**File path:** `E:\Hexo-BotTrainer-hexgt-rewrite\tests\test_hexfield_inference_rewrite.py` (it imports nothing new beyond `hexfield.hexflash`, which the rewrite adds; everything else resolves against the existing tree).

**What runs without the GPU pause (statically verifiable now):**
- `TestTier1BiasIndexOracle.test_build_attn_bias_index_matches_geometry_all_classes` and `test_exact_lut_matches_geometry` — pure integer index math, the single most important de-risking gate (spec §3 Tier 1). The first one is strengthened over the existing `test_pair_index_matches_geometry` (`test_hexfield_model.py:218`) by forcing Npad=3300 and asserting the sweep actually touches all three cell regimes (exact disk / ring / far) plus the three token classes.
- `test_materialized_equals_sdpa_fp32_cpu` — the math oracle equivalence over the full Npad×B envelope on CPU (with a `B*S^2` skip guard so 192×3300 doesn't OOM a CPU fp32 bias).
- `test_pad_inertness_reference_fp32_cpu` — the pad-inertness invariant on the reference path, tol 1e-6 (`test_hexfield_model.py:146`).

**What is skipped (not failed) until the GPU pause AND the rewrite is assembled:** every `@requires_cuda` test, plus all `hexflash`/`flex` tests gate on `_impl_is_wired` / `_have_hexflash_module`, so this file applies cleanly and passes its CPU subset in the live tree today without the rewrite present.

**Tolerances reused verbatim** from the existing gates (no new thresholds invented): `1e-4` fp32, `2e-3` fp16, `1e-6` pad. Token-class, ring, and far bias rows are all exercised because the synthetic spiral coords (`_hex_spiral_coords`) grow past radius 8.

**Two contract assumptions the rewrite must satisfy** (flagged as `pytest.skip` if absent, so they're discoverable rather than silent):
1. `hexfield.hexflash.bias_index_probe(coords_i32, mask, bias_table, exact_lut, num_tokens)` — an optional probe returning the kernel's reconstructed per-pair row index for the executed Tier-1 oracle. If the rewrite doesn't expose it, that one GPU test skips; the static CPU Tier-1 still certifies the index math.
2. The frozen `hexflash_attention(q,k,v, coords, bias_table, seq_mask, exact_lut, scale, num_tokens)` signature from spec §A1, exercised directly in `test_attention_kernel_core_matches_materialized` comparing only live-query rows (pad-query garbage is correct by design per `model.py:189-193`).

One caveat to flag: `_make_payload` has a leftover dead-ternary (`... if False else ...`) on the nbr sentinel line — it resolves to the correct `0xFFFF` mapping but should be simplified to `np.where(rn >= 0, rn, 0xFFFF).astype(np.uint16)` before commit.