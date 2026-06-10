"""Offline sustained-forward benchmark of the GPU levers on the REAL model.

Compares eager-fp16 vs torch.compile-fp16 vs compile+KV-gather at the production
batch, so we know the available headroom BEFORE bouncing the live run. Read-only:
loads a checkpoint, never writes run state.
"""
from __future__ import annotations
import argparse, copy, sys, time, tomllib
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for pkg in ("dense_cnn_restnet", "hexo_train", "hexo_runner"):
    src = ROOT / "packages" / pkg / "python"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

from dense_cnn_restnet.architecture import RestnetNetwork, RelPosMHSA, optimized_restnet_for_inference
from dense_cnn_restnet.config import parse_model1_config
from dense_cnn_restnet.constants import BOARD_SIZE, INPUT_CHANNELS


def build(cfg):
    a = cfg.architecture
    return RestnetNetwork(
        in_channels=a.input_channels, channels=a.channels, blocks_type=a.blocks_type,
        attention_heads=a.attention_heads, mlp_ratio=a.mlp_ratio,
        embed_kernel_size=a.embed_kernel_size, residual_conv=a.residual_conv,
        attention_impl=a.attention_impl, attention_scope=a.attention_scope, dropout=a.dropout,
        short_term_value_horizons=a.short_term_value_horizons, moves_left_head=a.moves_left_head,
    ).eval()


@torch.inference_mode()
def bench(fwd, x, iters):
    for _ in range(8):
        fwd(x)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fwd(x)
    torch.cuda.synchronize()
    return iters * int(x.shape[0]) / (time.perf_counter() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dense_cnn_restnet_main1.toml")
    ap.add_argument("--checkpoint")
    ap.add_argument("--batch", type=int, default=453)
    ap.add_argument("--iters", type=int, default=60)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    raw = tomllib.loads((ROOT / args.config).read_text())
    cfg = parse_model1_config(raw["model"]["config"])
    model = build(cfg)
    if args.checkpoint:
        p = torch.load(args.checkpoint, map_location="cpu")
        state = p.get("model_state_dict") or p.get("state_dict") or p if isinstance(p, dict) else p
        model.load_state_dict(state, strict=False)

    x = torch.zeros((args.batch, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), device="cuda", dtype=torch.float16)
    x[:, 0, BOARD_SIZE // 2, BOARD_SIZE // 2] = 1
    x = x.to(memory_format=torch.channels_last)

    import torch._dynamo as _dynamo
    _dynamo.config.cache_size_limit = 64
    _dynamo.config.accumulated_cache_size_limit = 128

    def _freeze(m):
        """Mirror the production compile path: freeze the attention bias so
        fullgraph compile has no data-dependent guard."""
        for mod in m.modules():
            if isinstance(mod, RelPosMHSA):
                mod.freeze_inference_bias(dtype=torch.float16, device=torch.device("cuda"))
        return m

    results = {}
    # eager fp16
    eager = optimized_restnet_for_inference(copy.deepcopy(model), attention_kv_gather=False).cuda().half().eval()
    results["eager_fp16"] = bench(eager.forward_policy_value, x, args.iters)
    # eager fp16 + KV-gather
    kv = optimized_restnet_for_inference(copy.deepcopy(model), attention_kv_gather=True).cuda().half().eval()
    results["eager_fp16_kvgather"] = bench(kv.forward_policy_value, x, args.iters)
    # torch.compile fp16 (frozen bias, like production)
    try:
        ec = _freeze(optimized_restnet_for_inference(copy.deepcopy(model), attention_kv_gather=False).cuda().half().eval())
        c = torch.compile(ec.forward_policy_value, mode="reduce-overhead", fullgraph=True, dynamic=False)
        results["compile_fp16"] = bench(c, x, args.iters)
    except Exception as exc:
        results["compile_fp16"] = f"FAILED: {exc!r}"
    # torch.compile fp16 + KV-gather (frozen bias, like production)
    try:
        kc = _freeze(optimized_restnet_for_inference(copy.deepcopy(model), attention_kv_gather=True).cuda().half().eval())
        ck = torch.compile(kc.forward_policy_value, mode="reduce-overhead", fullgraph=True, dynamic=False)
        results["compile_fp16_kvgather"] = bench(ck, x, args.iters)
    except Exception as exc:
        results["compile_fp16_kvgather"] = f"FAILED: {exc!r}"

    base = results["eager_fp16"]
    print(f"batch={args.batch} iters={args.iters} (evals/s, sustained)")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:28s} {v:8.0f}  ({v / base - 1:+.0%} vs eager)")
        else:
            print(f"  {k:28s} {str(v)[:80]}")


if __name__ == "__main__":
    main()
