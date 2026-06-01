#!/usr/bin/env python3
"""Correctness gate for FP16 *input transport*.

Compares the production forward on real encoded positions with f32 input planes vs
the same planes roundtripped through f16 (what the proposed Rust f16-transport +
on-device upcast would feed). Both go through the identical (TRT FP16) forward, so
this isolates ONLY the input-precision effect. Adopt f16 transport only if policy
argmax over legal moves is ~unchanged and value error is tiny.
"""

from __future__ import annotations

import os
import numpy as np
import torch

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id

from hexo_models.dense_cnn.architecture import Model1Network
from hexo_models.dense_cnn.inference import DenseCNNInference, _to_inference_device
from hexo_models.dense_cnn import rust_bridge
from hexo_models.dense_cnn.losses import decode_binned_value
from hexo_models.dense_cnn.constants import INPUT_CHANNELS


def diverged_states(n: int, plies: int):
    games = [engine.new_game(seed=4000 + i) for i in range(n)]
    rng = np.random.default_rng(0)
    for _ in range(plies):
        for i, st in enumerate(games):
            if engine.terminal(st) is not None:
                continue
            legal = engine.legal_action_ids(st)
            if not legal:
                continue
            a = int(legal[int(rng.integers(len(legal)))])
            engine.apply_action(st, engine.PlacementAction(unpack_coord_id(a)))
    return [st for st in games if engine.terminal(st) is None]


def main() -> None:
    torch.manual_seed(0)
    net = Model1Network(in_channels=INPUT_CHANNELS, channels=64, blocks=4,
                        short_term_value_horizons=(1, 4, 8))
    use_trt = os.environ.get("HEXO_TRT", "").strip() in ("1", "true", "True")
    inf = DenseCNNInference(net, device="cuda", amp=True, return_logits=False,
                            max_batch_size=1024, use_trt=use_trt,
                            bucket_pad_multiple=16, trt_allow_fallback=True)
    print(f"[gate] trt adopted={inf.trt_info.get('adopted')} use_trt={use_trt}")

    states = diverged_states(320, 12)
    payload = rust_bridge.model1_batch_inputs(states)
    shape = tuple(int(x) for x in payload["shape"])
    f32 = torch.frombuffer(bytearray(payload["inputs"]), dtype=torch.float32).reshape(shape)
    legal_rows = [tuple(int(i) for i in row) for row in payload["legal_flat_indices"]]

    dev32 = _to_inference_device(f32, inf.device)
    dev16 = dev32.half().float()  # roundtrip planes through f16 (transport precision)

    with torch.inference_mode():
        o32 = inf._forward_device_inputs(dev32)
        o16 = inf._forward_device_inputs(dev16)

    p32, p16 = o32["policy"].float(), o16["policy"].float()
    v32 = decode_binned_value(o32["value"]).float().cpu()
    v16 = decode_binned_value(o16["value"]).float().cpu()

    # Policy argmax over each row's LEGAL crop cells (what MCTS priors rank on).
    match = 0
    total = 0
    max_legal_logit_err = 0.0
    for r, flats in enumerate(legal_rows):
        if not flats:
            continue
        idx = torch.tensor(flats, dtype=torch.long)
        l32 = p32[r].cpu().index_select(0, idx)
        l16 = p16[r].cpu().index_select(0, idx)
        max_legal_logit_err = max(max_legal_logit_err, float((l32 - l16).abs().max()))
        match += int(torch.argmax(l32).item() == torch.argmax(l16).item())
        total += 1

    val_err = float((v32 - v16).abs().max())
    print(f"[gate] rows={total}")
    print(f"[gate] policy legal-argmax match : {match}/{total} = {match/max(total,1):.4f}")
    print(f"[gate] max |legal policy logit err| : {max_legal_logit_err:.6f}")
    print(f"[gate] max |value err| (decoded) : {val_err:.6f}")
    ok = (match / max(total, 1) >= 0.999) and (val_err <= 1e-3)
    print(f"[gate] VERDICT: {'PASS — f16 input transport is safe' if ok else 'FAIL — keep f32'}")


if __name__ == "__main__":
    main()
