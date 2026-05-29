"""Shared benchmark + correctness harness for dense_cnn inference-backend tests.

This is the REUSABLE scaffolding for the inference-optimization benchmark cycle
(BF16 / FP16-AMP / torch.compile / TensorRT). It deliberately does NOT bake in
any one backend: a "variant" is just a callable that, given a model and a batch
of inputs on device, returns the dense_cnn output dict (`policy`, `value`, ...).

What it provides
----------------
1. `load_model(...)`            -> builds the 96x6 + P7 Model1Network from
                                    `configs/dense_cnn_model1_target_96x6.toml`
                                    and loads `model_state` STRICT from the
                                    prefit checkpoint.
2. `make_inputs(...)`           -> representative production-shape input batches
                                    (inference bs ~1024 and a single-game bs=1),
                                    REPRESENTATIVE not all-zeros (zeros let cuDNN/
                                    BN take degenerate fast paths and understate
                                    real latency).
3. variants (`Variant`)         -> pluggable forward callables:
                                      - `fp32_reference` (the correctness oracle)
                                      - `fp16_amp`       (the current production path)
                                      - `bf16_amp`       (this branch's deliverable)
4. `time_variant(...)`          -> full warmup (cuDNN autotune + clock ramp) then
                                    many iterations; reports mean / stdev / p50 /
                                    p95 in ms, plus forwards/s and (for the bs=1024
                                    case) positions/s.
5. `compare_to_reference(...)`  -> runs a variant and an FP32 reference on the
                                    SAME inputs and reports max-abs-error of the
                                    policy logits AND the value logits.

Production parity notes (so the variants match what self-play actually runs):
  - Production inference uses `optimized_model1_for_inference` (folds the masked
    HexConvs + BatchNorms into plain convs), channels_last memory format, and
    `torch.autocast` for the FP16 path. The harness mirrors that: it builds the
    folded inference clone once (`build_inference_model`) and every variant runs
    on that same clone, so the only thing that varies between variants is the
    autocast dtype. That isolates the dtype effect, which is the thing we want to
    measure. The FP32 reference also runs on the folded clone so the
    max-abs-error reflects ONLY the dtype change, not conv folding.
  - All inputs are channels_last on CUDA, matching `inference.py`.

Usage (production PYTHONPATH required so we don't import the stale installed copy):
    set PYTHONPATH=E:/Hexo-BotTrainer/packages/hexo_models/python;...   (see scripts/start_model1_training.ps1)
    python -m analysis.inference_backends.bench_harness --smoke       # light smoke (default)
    python -m analysis.inference_backends.bench_harness --full        # full timing sweep (verification agent)
    python -m analysis.inference_backends.bench_harness --variant bf16 --batch 1024 --iters 200
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch

# ---------------------------------------------------------------------------
# Defaults wired to the live target_96x6 run (see NOTES.md).
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = _REPO_ROOT / "configs" / "dense_cnn_model1_target_96x6.toml"
DEFAULT_CHECKPOINT = (
    _REPO_ROOT
    / "runs"
    / "dense_cnn_model1_target_96x6"
    / "checkpoints"
    / "bootstrap_sealbot_prefit.pt"
)
# Production batch shapes (NOTES: evaluator inference bs ~1024 during 256-game
# selfplay; single-game eval is bs=1).
PRODUCTION_BATCH_SHAPES = (1, 1024)


def _load_toml(path: Path) -> dict:
    try:
        import tomllib  # py3.11+
    except ModuleNotFoundError:  # pragma: no cover - py3.10 fallback
        import tomli as tomllib  # type: ignore
    with open(path, "rb") as handle:
        return tomllib.load(handle)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(
    *,
    config_path: Path = DEFAULT_CONFIG,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    device: str = "cuda",
) -> tuple[torch.nn.Module, object]:
    """Build the 96x6 + P7 Model1Network and load `model_state` STRICT.

    Returns (training_model_eval_on_device, parsed_config). The training model
    keeps the explicit HexConv masks; build the folded inference clone with
    `build_inference_model` before timing.
    """

    from hexo_models.dense_cnn.config import parse_model1_config
    from hexo_models.dense_cnn.plugin import DenseCNNPlugin

    raw = _load_toml(config_path)
    model_section = raw.get("model", {}).get("config", {})
    parsed = parse_model1_config(model_section)

    plugin = DenseCNNPlugin()
    model = plugin.build_model(game_spec={}, config=model_section)

    payload = torch.load(checkpoint_path, map_location="cpu")
    if "model_state" not in payload:
        raise KeyError(
            f"checkpoint {checkpoint_path} has no 'model_state' key; keys={list(payload)}"
        )
    # STRICT load — the task requires load model_state strict.
    model.load_state_dict(payload["model_state"], strict=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[load_model] built {parsed.architecture.channels}ch x "
        f"{parsed.architecture.residual_blocks}block P7 model, "
        f"{n_params/1e6:.3f}M params, loaded model_state STRICT from "
        f"{checkpoint_path.name} (epoch={payload.get('epoch')})",
        flush=True,
    )

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        print("[load_model] WARNING: cuda requested but unavailable; using cpu", flush=True)
        resolved = torch.device("cpu")
    model.eval().to(resolved)
    return model, parsed


def build_inference_model(model: torch.nn.Module, device: str = "cuda") -> torch.nn.Module:
    """Return the production folded inference clone on `device` (channels_last on cuda).

    This mirrors `DenseCNNInference`: HexConv masks + BatchNorms are folded into
    plain convs. All variants run on this clone so the only difference between
    them is the autocast dtype.
    """

    from hexo_models.dense_cnn.architecture import optimized_model1_for_inference

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        resolved = torch.device("cpu")
    optimized = optimized_model1_for_inference(model)
    if resolved.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        optimized.to(device=resolved, memory_format=torch.channels_last)
    else:
        optimized.to(resolved)
    optimized.eval()
    return optimized


# ---------------------------------------------------------------------------
# Representative inputs
# ---------------------------------------------------------------------------
def make_inputs(
    batch: int,
    *,
    device: str = "cuda",
    seed: int = 1234,
) -> torch.Tensor:
    """Build a REPRESENTATIVE (N, 13, 41, 41) f32 input batch on device.

    Not all-zeros: zeros let cuDNN winograd / BN take degenerate paths and the
    occupancy/legal planes become unrealistic, which understates real forward
    latency and hides dtype rounding. We synthesize plausible board planes:
    one-hot-ish occupancy across own/opp/empty, a legal mask, plus smooth
    continuous planes (recency / center-distance) so BF16/FP16 rounding is
    actually exercised. channels_last on cuda to match production.
    """

    from hexo_models.dense_cnn.constants import BOARD_SIZE, INPUT_CHANNELS

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        resolved = torch.device("cpu")
    gen = torch.Generator().manual_seed(seed)

    n, c, h, w = batch, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE
    x = torch.zeros((n, c, h, w), dtype=torch.float32)

    # Occupancy: each cell is own(0)/opp(1)/empty(2), roughly 25/25/50.
    occ = torch.randint(0, 4, (n, h, w), generator=gen)  # 0,1 -> stones; 2,3 -> empty
    own = (occ == 0).float()
    opp = (occ == 1).float()
    empty = (occ >= 2).float()
    x[:, 0] = own
    x[:, 1] = opp
    x[:, 2] = empty
    # Legal plane (plane 3): legal where empty, with a little noise.
    x[:, 3] = empty * (torch.rand((n, h, w), generator=gen) > 0.1).float()
    # A few binary flag planes (second placement, first stone, colour, opp-last).
    for plane in (4, 5, 6, 12):
        x[:, plane] = (torch.rand((n, h, w), generator=gen) > 0.95).float()
    # Smooth continuous planes (recency / hot / center distance) in [0,1].
    for plane in (7, 8, 9, 10, 11):
        x[:, plane] = torch.rand((n, h, w), generator=gen)

    x = x.to(resolved)
    if resolved.type == "cuda" and x.ndim == 4:
        x = x.contiguous(memory_format=torch.channels_last)
    return x


# ---------------------------------------------------------------------------
# Pluggable variants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Variant:
    """A pluggable forward path.

    `forward(model, inputs) -> dict[str, Tensor]` runs ONE forward on the given
    inference model and batch (already on device, channels_last). It is the unit
    both the timer and the correctness comparison call. Keep it doing exactly the
    forward (no .cpu(), no decode) so timing reflects only the GPU forward.
    """

    name: str
    forward: Callable[[torch.nn.Module, torch.Tensor], Mapping[str, torch.Tensor]]
    description: str = ""


def _forward_policy_value(model: torch.nn.Module, inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
    """Use the inference-only head path that production search uses, if present."""
    if hasattr(model, "forward_policy_value"):
        return model.forward_policy_value(inputs)
    return model(inputs)


def _make_autocast_variant(name: str, dtype: torch.dtype | None, description: str) -> Variant:
    def forward(model: torch.nn.Module, inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
        device_type = inputs.device.type
        if dtype is None:
            with torch.inference_mode():
                return _forward_policy_value(model, inputs)
        with torch.inference_mode():
            with torch.autocast(device_type=device_type, dtype=dtype):
                return _forward_policy_value(model, inputs)

    return Variant(name=name, forward=forward, description=description)


# The FP32 reference is the correctness oracle: no autocast, full precision.
fp32_reference = _make_autocast_variant(
    "fp32", None, "FP32 reference (correctness oracle, no autocast)"
)
# The current production path: autocast float16 (NOTES: 'FP16/autocast is already on').
fp16_amp = _make_autocast_variant(
    "fp16", torch.float16, "Baseline production: autocast float16"
)
# This branch's deliverable: autocast bfloat16.
bf16_amp = _make_autocast_variant(
    "bf16", torch.bfloat16, "BF16 variant: autocast bfloat16"
)

VARIANTS: dict[str, Variant] = {
    "fp32": fp32_reference,
    "fp16": fp16_amp,
    "bf16": bf16_amp,
}


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TimingResult:
    variant: str
    batch: int
    iters: int
    mean_ms: float
    stdev_ms: float
    p50_ms: float
    p95_ms: float
    forwards_per_s: float
    positions_per_s: float

    def render(self) -> str:
        return (
            f"  {self.variant:>5s}  bs={self.batch:<5d}  "
            f"mean={self.mean_ms:8.3f}ms  std={self.stdev_ms:7.3f}  "
            f"p50={self.p50_ms:8.3f}  p95={self.p95_ms:8.3f}  "
            f"fwd/s={self.forwards_per_s:9.1f}  pos/s={self.positions_per_s:11.1f}"
        )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_variant(
    variant: Variant,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    warmup_iters: int = 30,
    iters: int = 100,
) -> TimingResult:
    """Time one variant with full warmup (cuDNN autotune + clock ramp) then `iters` reps.

    Reports mean/stdev/p50/p95 of per-forward latency (ms), NOT a single number,
    plus forwards/s and positions/s (= batch * forwards/s).
    """

    device = inputs.device
    batch = int(inputs.shape[0])

    # Warmup: lets cuDNN.benchmark autotune the algo for this exact shape and
    # lets the GPU clocks ramp to steady state before any timed iteration.
    with torch.inference_mode():
        for _ in range(warmup_iters):
            variant.forward(model, inputs)
    _sync(device)

    samples: list[float] = []
    with torch.inference_mode():
        for _ in range(iters):
            _sync(device)
            t0 = time.perf_counter()
            variant.forward(model, inputs)
            _sync(device)
            samples.append((time.perf_counter() - t0) * 1000.0)

    samples_sorted = sorted(samples)
    mean_ms = statistics.fmean(samples)
    stdev_ms = statistics.stdev(samples) if len(samples) > 1 else 0.0
    p50_ms = statistics.median(samples)
    p95_idx = min(len(samples_sorted) - 1, int(round(0.95 * (len(samples_sorted) - 1))))
    p95_ms = samples_sorted[p95_idx]
    forwards_per_s = 1000.0 / mean_ms if mean_ms > 0 else float("inf")
    positions_per_s = forwards_per_s * batch
    return TimingResult(
        variant=variant.name,
        batch=batch,
        iters=iters,
        mean_ms=mean_ms,
        stdev_ms=stdev_ms,
        p50_ms=p50_ms,
        p95_ms=p95_ms,
        forwards_per_s=forwards_per_s,
        positions_per_s=positions_per_s,
    )


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CorrectnessResult:
    variant: str
    batch: int
    policy_max_abs_err: float
    value_max_abs_err: float
    policy_argmax_match_frac: float

    def render(self) -> str:
        return (
            f"  {self.variant:>5s}  bs={self.batch:<5d}  "
            f"policy_max_abs_err={self.policy_max_abs_err:.3e}  "
            f"value_max_abs_err={self.value_max_abs_err:.3e}  "
            f"policy_argmax_match={self.policy_argmax_match_frac*100:6.2f}%"
        )


def compare_to_reference(
    variant: Variant,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    reference: Variant = fp32_reference,
) -> CorrectnessResult:
    """Run `variant` and `reference` on the SAME inputs; report max-abs-error.

    Reports max-abs-error of the policy logits AND the value logits (the two
    heads search consumes), plus the fraction of rows whose policy argmax still
    matches the FP32 reference (a search-relevant signal: small logit error that
    flips the top move matters more than the raw magnitude).
    """

    with torch.inference_mode():
        ref_out = reference.forward(model, inputs)
        var_out = variant.forward(model, inputs)

    ref_policy = ref_out["policy"].float()
    var_policy = var_out["policy"].float()
    ref_value = ref_out["value"].float()
    var_value = var_out["value"].float()

    policy_err = (ref_policy - var_policy).abs().max().item()
    value_err = (ref_value - var_value).abs().max().item()
    argmax_match = (ref_policy.argmax(dim=1) == var_policy.argmax(dim=1)).float().mean().item()

    return CorrectnessResult(
        variant=variant.name,
        batch=int(inputs.shape[0]),
        policy_max_abs_err=policy_err,
        value_max_abs_err=value_err,
        policy_argmax_match_frac=argmax_match,
    )


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------
def _resolve_variants(names: Sequence[str]) -> list[Variant]:
    out: list[Variant] = []
    for name in names:
        if name not in VARIANTS:
            raise SystemExit(f"unknown variant {name!r}; known: {sorted(VARIANTS)}")
        out.append(VARIANTS[name])
    return out


def run_smoke(args: argparse.Namespace) -> None:
    """LIGHT smoke check ONLY (no full sweep): load, one baseline + one BF16
    forward on a SMALL batch, print rough BF16-vs-FP32 parity."""

    print("=== SMOKE CHECK (light; full sweep is the verification agent's job) ===", flush=True)
    device = args.device
    model, _ = load_model(device=device)
    inf_model = build_inference_model(model, device=device)

    batch = 8  # small batch — proving it works, not timing it
    inputs = make_inputs(batch, device=device)
    print(f"[smoke] inference model built; inputs shape={tuple(inputs.shape)} "
          f"dtype={inputs.dtype} device={inputs.device}", flush=True)

    # Prove each variant produces sane output on a small batch.
    for variant in (fp32_reference, fp16_amp, bf16_amp):
        with torch.inference_mode():
            out = variant.forward(inf_model, inputs)
        pol = out["policy"]
        val = out["value"]
        print(f"[smoke] {variant.name:>5s} forward OK: policy{tuple(pol.shape)} "
              f"value{tuple(val.shape)} policy_dtype={pol.dtype}", flush=True)

    # Rough parity (the requested smoke deliverable).
    print("[smoke] parity vs FP32 reference (max-abs-error):", flush=True)
    for variant in (fp16_amp, bf16_amp):
        result = compare_to_reference(variant, inf_model, inputs)
        print(result.render(), flush=True)

    # A handful of quick iterations just to confirm timing wiring runs (NOT a
    # benchmark — short warmup/iters so we do not hog the GPU).
    print("[smoke] timing wiring sanity (short, NOT a benchmark):", flush=True)
    for variant in (fp32_reference, bf16_amp):
        timing = time_variant(variant, inf_model, inputs, warmup_iters=5, iters=10)
        print(timing.render(), flush=True)

    print("=== SMOKE CHECK DONE ===", flush=True)


def run_full(args: argparse.Namespace) -> None:
    """FULL timing + correctness sweep. Intended for the verification agent on a
    quiet GPU (long warmup, many iters, all production batch shapes)."""

    print("=== FULL SWEEP ===", flush=True)
    device = args.device
    model, _ = load_model(device=device)
    inf_model = build_inference_model(model, device=device)

    variants = _resolve_variants(args.variants) if args.variants else list(VARIANTS.values())
    batches = args.batches or list(PRODUCTION_BATCH_SHAPES)

    for batch in batches:
        inputs = make_inputs(batch, device=device)
        print(f"\n--- batch {batch} ---", flush=True)
        print("[correctness] max-abs-error vs FP32 reference:", flush=True)
        for variant in variants:
            if variant.name == "fp32":
                continue
            print(compare_to_reference(variant, inf_model, inputs).render(), flush=True)
        print("[timing] mean/stdev/p50/p95 (ms) + throughput:", flush=True)
        for variant in variants:
            print(time_variant(variant, inf_model, inputs,
                               warmup_iters=args.warmup, iters=args.iters).render(), flush=True)
    print("\n=== FULL SWEEP DONE ===", flush=True)


def run_single(args: argparse.Namespace) -> None:
    """Time + check a single variant at given batch(es) — convenience driver."""
    device = args.device
    model, _ = load_model(device=device)
    inf_model = build_inference_model(model, device=device)
    variant = VARIANTS[args.variant]
    batches = args.batches or [args.batch]
    for batch in batches:
        inputs = make_inputs(batch, device=device)
        print(compare_to_reference(variant, inf_model, inputs).render(), flush=True)
        print(time_variant(variant, inf_model, inputs,
                           warmup_iters=args.warmup, iters=args.iters).render(), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="light smoke check (default)")
    parser.add_argument("--full", action="store_true", help="full timing + correctness sweep")
    parser.add_argument("--variant", default=None, choices=sorted(VARIANTS),
                        help="time a single named variant")
    parser.add_argument("--variants", nargs="*", default=None,
                        help="subset of variants for the full sweep")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--batches", nargs="*", type=int, default=None,
                        help="explicit batch sizes (default: production shapes 1 and 1024)")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args(argv)

    if args.full:
        run_full(args)
    elif args.variant:
        run_single(args)
    else:
        run_smoke(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
