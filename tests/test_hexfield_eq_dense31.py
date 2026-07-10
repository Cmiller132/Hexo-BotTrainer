"""Dense 31-tap Design-A CPU investigation gates (plan D1-D5)."""

from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from hexfield_eq import _dense31 as D31
from hexfield_eq import constants as C
from hexfield_eq import equivariant as EQ
from hexfield_eq.batching import collate_rows
from hexfield_eq.constants import DIRECTIONS
from hexfield_eq.features import AXIS_DELTAS, PositionFacts, build_position, build_ray_lengths
from hexfield_eq.geometry import apply_d6, disk_offsets
from hexfield_eq.model import HexfieldNet, infer_net_kwargs_from_state_dict
from scripts.dense31_surgery import convert_checkpoint, convert_state_dict


_REPO = Path(__file__).resolve().parents[1]
RL = C.RAYLEN_SLOTS
eq_only = pytest.mark.skipif(C.GROUP_ORDER != 12, reason="requires D6 tied build")
COVARIANT_HEADS = ("policy", "opp_policy", "soft_policy", "cell_q")
INVARIANT_HEADS = ("value", "stvalue_2", "stvalue_6", "stvalue_16", "moves_left")
_AXIS_VECS = tuple(AXIS_DELTAS[k] for k in ("Q", "R", "QR"))


def _facts(seed: int, n_stones: int = 8) -> PositionFacts:
    rng = random.Random(seed)
    cells = [(0, 0)]
    seen = set(cells)
    while len(cells) < n_stones:
        q, r = cells[rng.randrange(len(cells))]
        dq, dr = rng.choice(DIRECTIONS)
        step = rng.randint(1, 3)
        cand = (q + dq * step, r + dr * step)
        if cand not in seen:
            seen.add(cand)
            cells.append(cand)
    return PositionFacts(
        records=tuple((q, r, i % 2, i) for i, (q, r) in enumerate(cells)),
        current_player=seed % 2,
        phase="SecondStone",
        first_stone=cells[0],
    )


def _batch(seeds=(3, 4), n_stones=8):
    rows, rays = [], []
    for seed in seeds:
        facts = _facts(seed, n_stones)
        support, feats = build_position(facts)
        rows.append((support, feats))
        rays.append(build_ray_lengths(facts, support))
    return collate_rows(rows, raylen=rays)


def _args(batch):
    return batch["feats"], batch["nbr"], batch["mask"], batch["coords"]


def _randomize(model: HexfieldNet, seed: int, scale: float = 0.08) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape, generator=gen) * scale)


def _disk_board(radius=3):
    cells = disk_offsets(radius)
    n = len(cells)
    at = {cell: i for i, cell in enumerate(cells)}
    nbr = torch.full((1, n, 6), n, dtype=torch.long)
    for i, (q, r) in enumerate(cells):
        for d, (dq, dr) in enumerate(DIRECTIONS):
            nbr[0, i, d] = at.get((q + dq, r + dr), n)
    coords = torch.tensor([[list(cell) for cell in cells]], dtype=torch.long)
    mask = torch.ones(1, n, dtype=torch.bool)
    sig = [
        torch.tensor([at[apply_d6(g, *cell)] for cell in cells]) for g in range(12)
    ]
    return n, nbr, coords, mask, sig


def _slot_perm(g: int) -> list[int]:
    perm = [0] * RL
    for ai, (dq, dr) in enumerate(_AXIS_VECS):
        for di, sign in ((0, 1), (1, -1)):
            target = apply_d6(g, sign * dq, sign * dr)
            for aj, (aq, ar) in enumerate(_AXIS_VECS):
                if target == (aq, ar):
                    dst = aj * 2
                    break
                if target == (-aq, -ar):
                    dst = aj * 2 + 1
                    break
            for side in range(2):
                perm[side * 6 + ai * 2 + di] = side * 6 + dst
    return perm


def _transform_raylen(raylen, g, sig):
    perm = _slot_perm(g)
    inv = [0] * RL
    for src, dst in enumerate(perm):
        inv[dst] = src
    out = torch.zeros_like(raylen)
    out[0, sig] = raylen[0][:, inv]
    return out


def test_gather_index31_is_shellwise_7tap_action() -> None:
    idx7 = EQ.conv_gather_index()
    idx31 = EQ.conv_gather_index31()
    assert idx31.shape == (31, 12, 12)
    assert torch.equal(idx31[0], idx7[0])
    for shell in range(5):
        got = idx31[1 + shell * 6 : 1 + (shell + 1) * 6]
        assert torch.equal(got, idx7[1:] + shell * 6 * 12)


@eq_only
def test_t3_full_net_dense31_equivariance_all_d6() -> None:
    n, nbr, coords, mask, sig = _disk_board()
    model = HexfieldNet(trunk_layout="CCACCA", raytap="dense31").eval()
    _randomize(model, 2)
    gen = torch.Generator().manual_seed(21)
    feats = torch.randn(1, n, C.NUM_FEATURES, generator=gen)
    raylen = torch.randint(0, C.RAY_REACH + 1, (1, n, RL), generator=gen).to(
        torch.uint8
    )
    rin = EQ._in_rep_matrix()
    with torch.no_grad():
        base = model(feats, nbr, mask, coords, raylen=raylen)
        for g in range(12):
            fg = torch.zeros_like(feats)
            fg[0, sig[g]] = feats[0] @ rin[g].T
            got = model(
                fg, nbr, mask, coords,
                raylen=_transform_raylen(raylen, g, sig[g]),
            )
            for head in COVARIANT_HEADS:
                torch.testing.assert_close(
                    got[head][0].index_select(0, sig[g]), base[head][0],
                    atol=1e-4, rtol=0, msg=f"{head} g={g}",
                )
            for head in INVARIANT_HEADS:
                torch.testing.assert_close(
                    got[head], base[head], atol=1e-4, rtol=0,
                    msg=f"{head} g={g}",
                )


def test_t4_fresh_init_matches_baseline_and_far_shells_are_live() -> None:
    batch = _batch(seeds=(3, 4))
    torch.manual_seed(0)
    base = HexfieldNet(trunk_layout="CCA").eval()
    torch.manual_seed(0)
    dense = HexfieldNet(trunk_layout="CCA", raytap="dense31").eval()
    with torch.no_grad():
        out_base = base(*_args(batch))
        out_dense = dense(*_args(batch), raylen=batch["raylen"])
    for key in out_base:
        torch.testing.assert_close(out_dense[key], out_base[key], atol=1e-6, rtol=0)

    with torch.no_grad():
        dense.conv_blocks[0].ls.gamma.fill_(1.0)
        before = dense(*_args(batch), raylen=batch["raylen"])["policy"]
        far = dense.conv_blocks[0].conv1.w_base[7:]
        far.copy_(
            torch.randn(
                far.shape,
                generator=torch.Generator().manual_seed(44),
                dtype=far.dtype,
            )
            * 0.25
        )
        after = dense(*_args(batch), raylen=batch["raylen"])["policy"]
    assert not torch.allclose(after, before, atol=1e-6)


@eq_only
def test_surgery_all_distance_fold_matches_and_k1_only_does_not() -> None:
    batch = _batch(seeds=(6, 7), n_stones=10)
    source = HexfieldNet(trunk_layout="CCA", raytap="both").eval()
    _randomize(source, 9)
    with torch.no_grad():
        for block in source.conv_blocks:
            for conv in (block.conv1, block.conv2):
                conv.alpha.copy_(
                    torch.linspace(0.35, 1.15, C.RAY_REACH).unsqueeze(1)
                    * torch.linspace(0.7, 1.3, conv.alpha.shape[1]).unsqueeze(0)
                )
        expected = source(*_args(batch), raylen=batch["raylen"])

    payload = {"model": source.state_dict(), "meta": source.arch_meta(), "optimizer": {}}
    converted = convert_checkpoint(payload)
    assert converted["meta"]["raytap"] == "dense31"
    assert converted["optimizer"] is None
    kwargs = infer_net_kwargs_from_state_dict(converted["model"], converted["meta"])
    assert kwargs["raytap"] == "dense31"
    dense = HexfieldNet(**kwargs).eval()
    dense.load_state_dict(converted["model"], strict=True)
    with torch.no_grad():
        got = dense(*_args(batch), raylen=batch["raylen"])
    for key in expected:
        torch.testing.assert_close(got[key], expected[key], atol=1e-5, rtol=1e-5)

    broken_sd = convert_state_dict(source.state_dict())
    with torch.no_grad():
        for key, value in broken_sd.items():
            if key.endswith(".w_base") and value.shape[0] == 31:
                value[7:].zero_()
    broken = HexfieldNet(trunk_layout="CCA", raytap="dense31").eval()
    broken.load_state_dict(broken_sd, strict=True)
    with torch.no_grad():
        bad = broken(*_args(batch), raylen=batch["raylen"])
    assert not torch.allclose(bad["policy"], expected["policy"], atol=1e-5, rtol=1e-5)


def _fn_inputs(seed=0, *, b=2, n=7, c=4, cout=5, dtype=torch.float32):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(b, n, c, generator=gen, dtype=dtype)
    idx = torch.randint(0, n + 1, (b, n, 6, C.RAY_REACH), generator=gen)
    reach = torch.randint(
        0, C.RAY_REACH + 1, (b, n, 2, 6), generator=gen
    ).to(torch.uint8)
    weight = torch.randn(31, c, cout, generator=gen, dtype=dtype)
    bias = torch.randn(cout, generator=gen, dtype=dtype)
    mask = torch.rand(b, n, generator=gen) > 0.25
    return x, idx, reach, weight, bias, mask


def test_t8_function_matches_naive_outputs_and_gradients() -> None:
    x, idx, reach, weight, bias, mask = _fn_inputs()
    xa, wa, ba = (v.clone().requires_grad_(True) for v in (x, weight, bias))
    xb, wb, bb = (v.clone().requires_grad_(True) for v in (x, weight, bias))
    out_fn = D31.dense31_conv(xa, idx, reach, wa, ba, mask, 2)
    out_nv = D31.dense31_conv_naive(xb, idx, reach, wb, bb, mask, 2)
    assert torch.equal(out_fn, out_nv)
    grad = torch.randn_like(out_fn)
    out_fn.backward(grad)
    out_nv.backward(grad)
    for got, ref in ((xa.grad, xb.grad), (wa.grad, wb.grad), (ba.grad, bb.grad)):
        scale = ref.abs().max().clamp(min=1e-12)
        assert float((got - ref).abs().max() / scale) <= 1e-5


def test_t8_function_gradcheck_float64() -> None:
    x, idx, reach, weight, bias, mask = _fn_inputs(
        1, b=1, n=3, c=2, cout=2, dtype=torch.float64
    )

    def fn(x_, weight_, bias_):
        return D31.dense31_conv(x_, idx, reach, weight_, bias_, mask, 2)

    assert torch.autograd.gradcheck(
        fn,
        (x.requires_grad_(), weight.requires_grad_(), bias.requires_grad_()),
        fast_mode=True,
    )


def test_t8_function_saves_no_gathered_intermediate() -> None:
    x, idx, reach, weight, bias, mask = _fn_inputs(2)
    out = D31.dense31_conv(
        x.requires_grad_(), idx, reach, weight.requires_grad_(),
        bias.requires_grad_(), mask, 2,
    )
    node = out.grad_fn
    assert "Dense31Conv" in type(node).__name__
    forbidden = {(x.shape[0], x.shape[1], 31 * x.shape[2]),
                 (x.shape[0], x.shape[1], 31, x.shape[2])}
    assert len(node.saved_tensors) == 5
    assert all(tuple(t.shape) not in forbidden for t in node.saved_tensors)


@pytest.mark.parametrize("mode", ["0", "conv2", "both"])
def test_t6_existing_mode_key_sets_remain_disciplined(mode: str) -> None:
    state = HexfieldNet(trunk_layout="CCA", raytap=mode).state_dict()
    c1 = any(k.endswith(".conv1.alpha") for k in state)
    c2 = any(k.endswith(".conv2.alpha") for k in state)
    assert (c1, c2) == ((mode == "both"), (mode in ("conv2", "both")))
    assert all(v.shape[0] == 7 for k, v in state.items() if k.endswith(".w_base"))
    assert infer_net_kwargs_from_state_dict(state, {})["raytap"] == mode


def test_t6_dense31_meta_and_shape_fallback_round_trip() -> None:
    model = HexfieldNet(trunk_layout="CCA", raytap="dense31")
    state = model.state_dict()
    assert not any(k.endswith(".alpha") for k in state)
    trunk_w = [v for k, v in state.items() if k.startswith("conv_blocks.") and k.endswith(".w_base")]
    assert trunk_w and all(v.shape[0] == 31 for v in trunk_w)
    assert model.arch_meta()["raytap"] == "dense31"
    for meta in (model.arch_meta(), {}):
        kwargs = infer_net_kwargs_from_state_dict(state, meta)
        assert kwargs["raytap"] == "dense31"
        # State-dict fallback identifies the dense31 mode from tap shape; the
        # pre-existing loader does not infer a non-default trunk ordering from
        # key counts alone, so retain that known constructor fact here.
        kwargs.setdefault("trunk_layout", "CCA")
        rebuilt = HexfieldNet(**kwargs)
        rebuilt.load_state_dict(state, strict=True)


def test_t6_dense31_env_is_accepted() -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["HEXFIELD_EQ_RAYTAP"] = "dense31"
    env["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")
    code = (
        "import torch; "
        "assert not torch.cuda.is_available(); "
        "from hexfield_eq.constants import RAYTAP; "
        "from hexfield_eq.model import HexfieldNet; "
        "m=HexfieldNet(trunk_layout='CCA'); "
        "assert RAYTAP == m._raytap == 'dense31'; "
        "assert m.conv_blocks[0].conv1.w_base.shape[0] == 31"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=_REPO, env=env,
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr


def test_dense31_w_base_optimizer_and_grad_group_classification() -> None:
    try:
        from hexfield_eq.prefit import make_optimizer
        from hexfield_eq.trainer import HexfieldTrainer
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"training import chain unavailable: {exc}")
    from types import SimpleNamespace

    model = HexfieldNet(trunk_layout="CCA", raytap="dense31")
    dense_params = [
        (name, param)
        for name, param in model.named_parameters()
        if name.startswith("conv_blocks.") and name.endswith(".w_base")
    ]
    opt = make_optimizer(model)
    decay = {
        id(param)
        for group in opt.param_groups
        if group["weight_decay"] != 0
        for param in group["params"]
    }
    trunk = {
        id(param)
        for param in HexfieldTrainer._build_grad_norm_groups(
            SimpleNamespace(model=model)
        )["trunk_conv"]
    }
    assert dense_params
    for name, param in dense_params:
        assert id(param) in decay, f"{name} should use ordinary weight decay"
        assert id(param) in trunk, f"{name} should be trunk_conv"
