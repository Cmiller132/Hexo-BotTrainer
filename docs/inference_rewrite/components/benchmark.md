# benchmark

I now have full ground truth. Here is the benchmark harness as runnable code.

---

`scripts/_hexfield_rewrite_bench.py` (NEW — read-only against the live tree; safe to assemble into the rewrite worktree):

```python
"""Benchmark the INFERENCE-REWRITE stack vs the deployed baseline.

Run in the live torch venv WITH THE RUN STOPPED (GPU free):

    PYTHONPATH=packages/hexfield/python /root/.venvs/hexgt-build/bin/python \
        scripts/_hexfield_rewrite_bench.py [checkpoint.pt] [out.json] [secs_per_cfg]

WHAT IT MEASURES (apples-to-apples, same model weights, same search knobs):
  For each (variant x size-band) it reports the campaign's headline metrics —
  evals/s, pos/s, mean_flush, peak VRAM — PLUS the attention/SDPA backend that
  actually fired (flash / mem-efficient / math / triton-hexflash / flex) so a
  speedup is never mistaken for a silent fallback to a slower kernel.

  Two complementary harnesses, reusing the existing scripts' methodology:
    L1 FORWARD MICRO  (from _hexfield_compile_overlap_test): ABI payloads
        through evaluate_payload at controlled size BANDS (small / mid / large /
        mixed) -> ms/flush + backend. Isolates the attention-kernel win per band
        without scheduler noise. This is where the large-S hexflash gain shows.
    L2 END-TO-END     (from _hexfield_batch_sweep): the real run_continuous
        scheduler at production 512 visits -> evals/s, pos/s, mean_flush, peak
        VRAM, support-size histogram. This is the number the campaign ships.

VARIANTS (each is a HexfieldEvaluator built with a different env env-set; ALL
default to the deployed baseline if the rewrite switches are absent, so this
script runs unmodified against TODAY'S tree as a baseline-only sanity pass):
    baseline   = deployed: gated torch.compile small-S, eager large-S, SDPA bias
    hexflash   = HEXFIELD_ATTN_IMPL=hexflash (large-S Triton; small-S compile)
    flex       = HEXFIELD_ATTN_IMPL=flex     (large-S FlexAttention fallback)
    no-compile = HEXFIELD_NO_COMPILE=1       (control: pure eager, isolates the
                 compile contribution from the attention-kernel contribution)

It runs NO training and writes NO run files (results -> argv[2] or /tmp). It is
read-only w.r.t. the live tree. Math-parity is NOT this script's job — that is
_hexfield_compile_overlap_test.py (fp16 tol) and _hexfield_async_parity.py
(action parity); this is the throughput/VRAM/backend engine that runs AFTER
those gate green.
"""
from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

import hexfield.constants as C
from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction

from hexfield import _rust
from hexfield.geometry import unpack_action_id
from hexfield.inference import HexfieldEvaluator
from hexfield.model import HexfieldNet

CKPT = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] not in ("-", "none") else None
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/hexfield_rewrite_bench.json"
SECONDS_PER_CONFIG = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
VISITS = int(sys.argv[4]) if len(sys.argv) > 4 else 512

NBR_SENTINEL = 0xFFFF
rng = np.random.default_rng(0)
device = torch.device("cuda")

# --- the variants under test. Each maps env-var overrides applied ONLY while the
#     evaluator is constructed (so per-variant compile/impl routing is captured).
#     Unknown env vars are harmless on the current tree -> baseline behaviour.
VARIANTS: dict[str, dict[str, str]] = {
    "baseline":   {"HEXFIELD_NO_COMPILE": "0"},
    "hexflash":   {"HEXFIELD_NO_COMPILE": "0", "HEXFIELD_ATTN_IMPL": "hexflash"},
    "flex":       {"HEXFIELD_NO_COMPILE": "0", "HEXFIELD_ATTN_IMPL": "flex"},
    "no-compile": {"HEXFIELD_NO_COMPILE": "1"},
}
# Restrict via CLI env (comma list) e.g. HEXFIELD_BENCH_VARIANTS=baseline,hexflash
_only = os.environ.get("HEXFIELD_BENCH_VARIANTS")
if _only:
    keep = {v.strip() for v in _only.split(",") if v.strip()}
    VARIANTS = {k: v for k, v in VARIANTS.items() if k in keep}


@contextmanager
def env_overrides(overrides: dict[str, str]):
    saved = {k: os.environ.get(k) for k in overrides}
    try:
        os.environ.update(overrides)
        yield
    finally:
        for k, prev in saved.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


def load_model() -> HexfieldNet:
    model = HexfieldNet()
    if CKPT:
        p = torch.load(CKPT, map_location="cpu", weights_only=False)
        model.load_state_dict(p.get("model", p), strict=True)
        print(f"loaded {CKPT}", flush=True)
    return model.eval()


def build_variant(model: HexfieldNet, name: str) -> tuple[HexfieldEvaluator, dict]:
    """Construct an evaluator under the variant's env. Returns (evaluator, info)
    where info records what was actually configured (so a silently-ignored
    rewrite switch on an un-upgraded tree is visible in the results, not hidden)."""
    with env_overrides(VARIANTS[name]):
        ev = HexfieldEvaluator(model, device=device)
    info = {
        "requested_impl": VARIANTS[name].get("HEXFIELD_ATTN_IMPL", "sdpa"),
        # introspect what the evaluator actually did with the switches:
        "compile_active": bool(getattr(ev, "_use_compile", False)
                               and getattr(ev, "_compiled_fpv", None)
                               is not getattr(ev, "_raw_fpv", None)),
        "compile_max_npad": getattr(ev, "_compile_max_npad", None),
        # the rewrite adds these; absent -> running on the baseline tree:
        "attn_impl_attr": getattr(ev, "_attn_impl", "<absent: baseline tree>"),
        "large_npad_cut": getattr(ev, "_large_npad", "<absent>"),
        "abi_v2": getattr(ev, "_supports_abi2", "<absent>"),
    }
    return ev, info


# --------------------------------------------------------------------------- #
# L1 FORWARD MICRO — controlled size BANDS through the real serve path.
# Bands chosen to straddle the compile cutover (HEXFIELD_COMPILE_MAX_NPAD=512)
# so the small-S-compile vs large-S-kernel split is exercised explicitly.
# --------------------------------------------------------------------------- #
BANDS: dict[str, list[int]] = {
    "small-S(<=512)":  list(rng.integers(40, 480, size=96)),      # compile band
    "mid-S(~prod)":    list(rng.integers(120, 700, size=144)),    # straddles cut
    "large-S(>512)":   list(rng.integers(700, 2200, size=48)),    # kernel band
    "huge-S(>2048)":   list(rng.integers(2200, 3300, size=16)),   # tail
    "mixed-full":      list(rng.integers(40, 3300, size=160)),    # realistic skew
}


def make_payload(sizes, *, request_ml: bool) -> dict:
    """ABI-valid v1 flush payload (rows DESCENDING) — identical construction to
    _hexfield_compile_overlap_test.make_payload, so L1 here and the parity gate
    there feed the model byte-identical inputs."""
    sizes = sorted((int(s) for s in sizes), reverse=True)
    total = sum(sizes)
    feats = rng.standard_normal((total, C.NUM_FEATURES)).astype(np.float16)
    qr = rng.integers(-20, 21, size=(total, 2), dtype=np.int16)
    nbr = np.empty((total, 6), dtype=np.uint16)
    legal_counts = np.empty(len(sizes), dtype=np.int32)
    offsets = [0]
    pos = 0
    for i, n in enumerate(sizes):
        row_nbr = rng.integers(0, n, size=(n, 6)).astype(np.uint16)
        row_nbr[rng.random((n, 6)) < 0.2] = NBR_SENTINEL
        nbr[pos:pos + n] = row_nbr
        legal_counts[i] = max(1, min(n, int(rng.integers(1, n + 1))))
        pos += n
        offsets.append(pos)
    return {
        "abi": 1, "shape": (len(sizes), total),
        "node_feats": feats.tobytes(), "node_qr": qr.tobytes(),
        "node_row_offsets": offsets, "nbr": nbr.tobytes(),
        "legal_counts": legal_counts.tobytes(), "request_moves_left": request_ml,
    }


# kernel-name -> human backend label. Order matters: hexflash/flex first so a
# Triton kernel isn't misread as a generic gemm.
_BACKEND_TAGS = [
    ("hexflash", "triton-hexflash"),       # the new kernel names itself
    ("flex", "flex-attention"),
    ("triton_", "triton"),
    ("flash", "flash-attn"),
    ("efficient_attention", "mem-efficient"),
    ("fmha", "mem-efficient/fmha"),
    ("cutlassf", "mem-efficient/cutlass"),
    ("scaled_dot_product", "sdpa(unfused-math?)"),
]


def detect_backend(ev: HexfieldEvaluator, payload: dict) -> tuple[str, list[str]]:
    """Profile ONE flush and infer the attention backend from kernel names. Also
    returns the top-3 self-CUDA-time op names so a regression is attributable."""
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        ev.evaluate_payload(payload)
        torch.cuda.synchronize()
    rows = prof.key_averages()
    names = " ".join(e.key.lower() for e in rows)
    found = [label for tag, label in _BACKEND_TAGS if tag in names]
    backend = "+".join(dict.fromkeys(found)) or "unknown"
    top = sorted(rows, key=lambda e: e.self_cuda_time_total, reverse=True)[:3]
    top_ops = [f"{e.key[:38]}={e.self_cuda_time_total/1000:.1f}ms" for e in top]
    return backend, top_ops


def bench_band(ev: HexfieldEvaluator, sizes, *, reps=25) -> dict:
    p = make_payload(sizes, request_ml=True)
    for _ in range(6):            # warmup: compiles + autotunes each bucket once
        ev.evaluate_payload(p)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(reps):
        ev.evaluate_payload(p)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / reps * 1000.0
    peak = torch.cuda.max_memory_allocated() / 2**30
    backend, top_ops = detect_backend(ev, p)
    b = int(p["shape"][0])
    return {
        "rows": b, "ms_per_flush": round(ms, 3),
        "states_per_s": round(b / (ms / 1000.0), 0),
        "peak_vram_gib": round(peak, 3),
        "attn_backend": backend, "top_cuda_ops": top_ops,
    }


def run_l1(evaluators: dict[str, tuple[HexfieldEvaluator, dict]]) -> dict:
    print("\n########## L1 FORWARD MICRO (ms/flush, backend, peak VRAM) ##########",
          flush=True)
    out: dict = {}
    for band, sizes in BANDS.items():
        print(f"\n--- band {band} ({len(sizes)} rows) ---", flush=True)
        out[band] = {}
        base_ms = None
        for name, (ev, _info) in evaluators.items():
            try:
                r = bench_band(ev, sizes)
            except torch.cuda.OutOfMemoryError as e:
                r = {"oom": True, "error": str(e)[:160]}
                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                r = {"error": repr(e)[:240]}
            out[band][name] = r
            if name == "baseline" and "ms_per_flush" in r:
                base_ms = r["ms_per_flush"]
            speed = (f" {base_ms / r['ms_per_flush']:.2f}x"
                     if base_ms and "ms_per_flush" in r else "")
            if "ms_per_flush" in r:
                print(f"  {name:11s} {r['ms_per_flush']:8.3f} ms"
                      f"  {r['states_per_s']:7.0f} st/s"
                      f"  vram={r['peak_vram_gib']:.2f}G"
                      f"  [{r['attn_backend']}]{speed}", flush=True)
            else:
                print(f"  {name:11s} {r.get('error', r)}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# L2 END-TO-END — the production scheduler. Lifted verbatim from
# _hexfield_batch_sweep.run_config (same knobs, same metrics) so L2 numbers are
# directly comparable to the campaign's existing sweep results.
# --------------------------------------------------------------------------- #
# (active_games, active_root_limit, flush_target, cache_max_states). vbs FIXED 4.
E2E_CONFIGS = [
    (192, 192, 1024, 262144),   # the deployed operating point (GPU-saturated)
    (256, 256, 2048, 262144),   # headroom probe (VRAM ceiling finder)
]


def run_config(ev: HexfieldEvaluator, active_games, active_root_limit,
               flush_target, cache_max_states) -> dict:
    games = active_games
    states = {k: api.new_game() for k in range(games)}
    plies = {k: 0 for k in range(games)}
    n = {"d": 0}
    support_hist: dict[int, int] = {}
    deadline = time.time() + SECONDS_PER_CONFIG

    def on_move(game_key, payload):
        n["d"] += 1
        st = states[game_key]
        s = len(api.legal_action_ids(st))
        bkt = ((s // 64) + 1) * 64
        support_hist[bkt] = support_hist.get(bkt, 0) + 1
        q, r = unpack_action_id(int(payload["action_id"]))
        res = api.apply_action(st, PlacementAction(AxialCoord(q=q, r=r)))
        plies[game_key] += 1
        if res.terminal or time.time() > deadline:
            return None
        return ("advance", st)

    session = _rust.HexfieldMctsSession(max_states=cache_max_states)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    stats = session.run_continuous(
        list(range(games)), tuple(states.values()), evaluator=ev, on_move=on_move,
        visits=VISITS, c_puct=1.5, base_seed=7, virtual_batch_size=4,
        flush_target=flush_target, active_root_limit=active_root_limit,
        temperature_by_ply=[1.0] * 8 + [0.3] * 400,
        forced_playout_k=2.0, widening_policy_mass=0.95, widening_max_children=96,
        widening_min_children=2, root_dirichlet_total_alpha=10.83,
        root_dirichlet_noise_fraction=0.25, pcr_full_proportion=0.33,
        pcr_fast_visits=128, policy_init_fraction=0.25, policy_init_avg_plies=4.0,
        policy_init_max_plies=8, policy_init_temperature=1.4,
    )
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    return {
        "decisions": n["d"], "seconds": round(dt, 1),
        "pos_per_s": round(n["d"] / dt, 2),
        "evals_per_s": round(stats["flushed_states"] / dt, 0),
        "mean_flush": round(stats["mean_flush_states"], 1),
        "flushed_states": stats["flushed_states"],
        "peak_vram_gib": round(peak, 2),
        "support_hist": dict(sorted(support_hist.items())),
    }


def run_l2(model: HexfieldNet) -> dict:
    print("\n########## L2 END-TO-END (run_continuous @ "
          f"{VISITS} visits, {SECONDS_PER_CONFIG:.0f}s/cfg) ##########", flush=True)
    out: dict = {}
    for ag, arl, ft, cache in E2E_CONFIGS:
        cfg = f"active_games={ag} arl={arl} ft={ft}"
        print(f"\n=== {cfg} ===", flush=True)
        out[cfg] = {}
        base_eps = None
        for name in VARIANTS:
            # Rebuild a fresh evaluator per variant here: run_continuous warms its
            # OWN compile/autotune state, and we want each variant's steady-state.
            ev, _info = build_variant(model, name)
            try:
                r = run_config(ev, ag, arl, ft, cache)
                if name == "baseline":
                    base_eps = r["evals_per_s"]
                spd = (f" {r['evals_per_s'] / base_eps:.2f}x"
                       if base_eps else "")
                print(f"  {name:11s} pos/s={r['pos_per_s']:6.2f}"
                      f"  evals/s={r['evals_per_s']:7.0f}"
                      f"  mean_flush={r['mean_flush']:5.0f}"
                      f"  vram={r['peak_vram_gib']:.2f}G{spd}", flush=True)
            except torch.cuda.OutOfMemoryError as e:
                r = {"oom": True, "error": str(e)[:160]}
                print(f"  {name:11s} OOM", flush=True)
                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                r = {"error": repr(e)[:240]}
                print(f"  {name:11s} ERROR {r['error']}", flush=True)
            out[cfg][name] = r
            del ev
            torch.cuda.empty_cache()
    return out


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available — this benchmark must run on the GPU box.")
        return 2
    torch.manual_seed(0)
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}  "
          f"variants={list(VARIANTS)}", flush=True)
    model = load_model()

    # L1 reuses one evaluator per variant (forward-only, no scheduler state).
    l1_evaluators = {name: build_variant(model, name) for name in VARIANTS}
    print("\n--- variant configuration (what the switches actually did) ---")
    variant_info = {}
    for name, (_ev, info) in l1_evaluators.items():
        variant_info[name] = info
        print(f"  {name:11s} {info}", flush=True)

    l1 = run_l1(l1_evaluators)
    # free L1 evaluators before L2 rebuilds (avoid double VRAM residency)
    del l1_evaluators
    torch.cuda.empty_cache()

    l2 = run_l2(model)

    results = {
        "meta": {
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "ckpt": CKPT, "visits": VISITS,
            "seconds_per_config": SECONDS_PER_CONFIG,
            "variant_info": variant_info,
        },
        "l1_forward_micro": l1,
        "l2_end_to_end": l2,
    }
    Path(OUT).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}", flush=True)

    # headline: hexflash vs baseline on the band/op-point it targets
    try:
        lg = l1["large-S(>512)"]
        if "ms_per_flush" in lg.get("hexflash", {}) and "ms_per_flush" in lg["baseline"]:
            sp = lg["baseline"]["ms_per_flush"] / lg["hexflash"]["ms_per_flush"]
            print(f"HEADLINE L1 large-S: hexflash {sp:.2f}x vs baseline "
                  f"(Amdahl-expected ~1.2-1.5x)", flush=True)
        e2e = l2.get("active_games=192 arl=192 ft=1024", {})
        if "evals_per_s" in e2e.get("hexflash", {}) and "evals_per_s" in e2e.get("baseline", {}):
            sp = e2e["hexflash"]["evals_per_s"] / e2e["baseline"]["evals_per_s"]
            print(f"HEADLINE L2 @192: hexflash {sp:.2f}x evals/s vs baseline", flush=True)
    except Exception:  # noqa: BLE001  -- headline is best-effort
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

## Notes on the harness (file:line justification)

**Why two levels.** The spec's win is band-localized: hexflash only touches the large-S eager tail (Amdahl-capped ~1.2-1.5x per ESTABLISHED FACTS), while Layer B/compile help small-S throughput. A single end-to-end number would average these and hide the kernel effect. **L1** isolates the attention kernel per size-band through the real serve path (`evaluate_payload`, `inference.py:120`); **L2** is the shippable scheduler number, lifted verbatim from `_hexfield_batch_sweep.run_config` (`_hexfield_batch_sweep.py:69-118`) so its evals/s, pos/s, mean_flush, peak VRAM are directly comparable to the campaign's existing sweep.

**Variant construction via env, not code edits.** Each variant toggles env vars only during `HexfieldEvaluator.__init__` (`inference.py:67`), which is where compile gating reads `HEXFIELD_NO_COMPILE` / `HEXFIELD_COMPILE_MAX_NPAD` (`inference.py:94,104`) and where the spec's B1 places `HEXFIELD_ATTN_IMPL` / `HEXFIELD_LARGE_NPAD`. Because those new switches are read with `.get(...)` defaults, **this script runs unmodified on today's tree** — it just collapses to baseline-only and the `variant_info` block prints `<absent: baseline tree>`, making a non-upgraded tree explicit rather than silently reporting fake speedups.

**Backend detection is real, not assumed.** `detect_backend` profiles one flush (same technique as `_hexfield_profile_fwd.py:35-46`) and scans kernel names. The tag table puts `hexflash`/`flex`/`triton_` ahead of generic `gemm`/`flash` so a Triton kernel isn't misreported, and surfaces top-3 self-CUDA ops so a regression is attributable. This directly satisfies the "SDPA/kernel backend used" deliverable.

**VRAM is measured, not assumed.** Spec §4.5 wants the `(B,4,S,S)` bias-transient drop confirmed. Both levels call `reset_peak_memory_stats()`/`max_memory_allocated()` (same as `_hexfield_batch_sweep.py:93,109`), so the hexflash path's lower peak (no materialized bias) shows up directly and tells the operator how far PAIR_CEILING can rise.

**L2 rebuilds per variant** because `run_continuous` warms each evaluator's own compile/autotune state; L1 reuses one evaluator per variant (forward-only). Both free VRAM between phases (`empty_cache`) to avoid double residency on the 12 GB card.

**Scope discipline.** Search knobs are byte-identical to the existing sweep (visits/vbs/PCR/widening/dirichlet/fpu/temperature unchanged), no training, no run-file writes, output to argv[2] or /tmp, read-only against the live tree.

**What it does NOT do (by design):** no math-parity assertion — that is owned by `_hexfield_compile_overlap_test.py` (fp16 tol) and `_hexfield_async_parity.py` (action parity). This is the throughput/VRAM/backend engine that runs *after* those gate green. The Layer B v2-ABI byte-parity and depth-N pipeline are validated by those harnesses (per spec §3 Tier-3); this bench measures their throughput effect via L2 once they land.

**Runtime guard for the operator:** `HEXFIELD_BENCH_VARIANTS=baseline,hexflash` env var restricts the variant set, so the pause-window run can be kept short (e.g. just baseline vs hexflash) without editing the file.

Full file is `E:\Hexo-BotTrainer-hexgt-rewrite\scripts\_hexfield_rewrite_bench.py` when assembled (do NOT write into the live tree `E:\Hexo-BotTrainer-hexgt`).