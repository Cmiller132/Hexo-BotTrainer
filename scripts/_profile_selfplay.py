#!/usr/bin/env python3
"""Exhaustive self-play throughput profiler for dense_cnn Model 1 (64x4).

Answers, with measured numbers, "what exactly is slow in self-play":

  PHASE A  raw forward floor      -- pure GPU compute (TRT/torch), no tree, no
                                     prior-gather. ms/call + states/s vs batch.
  PHASE B  evaluator callback     -- the FULL Python evaluator (tensor-from-bytes,
                                     forward, legal-prior softmax gather, D2H,
                                     serialize). B - A = Python boundary overhead.
  PHASE C  real self-play         -- N divergent games stepped K sequential plies
                                     at 256 visits / vbatch 4 (production params),
                                     steady-state aggregate: time split
                                     (encode/eval/parse/remainder), unique forwards
                                     per position, and the forward batch-size
                                     histogram (the crux).

Run under the WSL venv with TRT on:
  HEXO_TRT=1 HEXO_MCTS_TRACE=1 HEXO_BUCKET_PAD_MULTIPLE=16 \
    /root/.venvs/hexo-bottrainer-wsl/bin/python scripts/_profile_selfplay.py
"""

from __future__ import annotations

import os
import time
import numpy as np
import torch

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id

from hexo_models.dense_cnn.architecture import Model1Network
from hexo_models.dense_cnn.inference import DenseCNNInference, _to_inference_device
from hexo_models.dense_cnn.losses import decode_binned_value
from hexo_models.dense_cnn.mcts import new_mcts_session
from hexo_models.dense_cnn.constants import BOARD_SIZE, BOARD_AREA, INPUT_CHANNELS

CHANNELS = 64
BLOCKS = 4
HORIZONS = (1, 4, 8)
VISITS = 256
VBATCH = 4
N_GAMES = 256
MAX_BATCH = 1024
PAD_MULT = int(os.environ.get("HEXO_BUCKET_PAD_MULTIPLE", "16") or 16)
AVG_LEGAL = 687  # measured live: encoded_legal_actions / encoded_states


def _hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_inference() -> DenseCNNInference:
    net = Model1Network(
        in_channels=INPUT_CHANNELS, channels=CHANNELS, blocks=BLOCKS,
        dropout=0.0, short_term_value_horizons=HORIZONS,
    )
    use_trt = os.environ.get("HEXO_TRT", "").strip() in ("1", "true", "True")
    print(f"[setup] device=cuda channels={CHANNELS} blocks={BLOCKS} "
          f"use_trt={use_trt} pad_mult={PAD_MULT} max_batch={MAX_BATCH}")
    inf = DenseCNNInference(
        net, device="cuda", amp=True, return_logits=False,
        max_batch_size=MAX_BATCH, use_trt=use_trt,
        bucket_pad_multiple=PAD_MULT, trt_allow_fallback=True,
    )
    print(f"[setup] trt adopted={inf.trt_info.get('adopted')} reason={inf.trt_info.get('reason')}")
    return inf


# --------------------------------------------------------------------------- A
def phase_a(inf: DenseCNNInference) -> dict[int, float]:
    _hr("PHASE A -- raw forward floor (pure GPU compute, no tree / no prior gather)")
    sizes = [4, 16, 32, 64, 128, 256, 512, 734, 1024]
    floor_ms: dict[int, float] = {}
    for n in sizes:
        dev = torch.zeros((n, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE),
                          device="cuda", dtype=torch.float32).to(
                              memory_format=torch.channels_last)
        for _ in range(5):
            inf._forward_device_inputs(dev)
        _sync()
        iters = 50
        t0 = time.perf_counter()
        for _ in range(iters):
            inf._forward_device_inputs(dev)
        _sync()
        ms = (time.perf_counter() - t0) / iters * 1e3
        floor_ms[n] = ms
        print(f"  batch={n:5d}  {ms:8.3f} ms/call  {n / (ms/1e3):10.0f} states/s")
    return floor_ms


# --------------------------------------------------------------------------- B
def _make_payload(n: int, legal: int) -> dict:
    legal = min(legal, BOARD_AREA)
    # f16 transport (matches Rust PlaneBuffer); evaluator upcasts on-device.
    planes = np.random.rand(n, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE).astype(np.float16)
    inputs = bytearray(planes.tobytes())
    flats = np.empty(n * legal, dtype=np.int64)
    for r in range(n):
        flats[r * legal:(r + 1) * legal] = np.random.choice(BOARD_AREA, size=legal, replace=False)
    offsets = tuple(int(x) for x in range(0, n * legal + 1, legal))
    return {
        "inputs": inputs,
        "shape": (n, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE),
        "legal_flat_indices_bytes": bytearray(flats.tobytes()),
        "legal_row_offsets": offsets,
    }


def phase_b(inf: DenseCNNInference, floor_ms: dict[int, float]) -> None:
    _hr(f"PHASE B -- full evaluator callback (forward + legal-prior softmax + D2H), "
        f"legal/row={AVG_LEGAL}")
    sizes = [128, 256, 512, 734, 1024]
    for n in sizes:
        payload = _make_payload(n, AVG_LEGAL)
        for _ in range(3):
            inf.evaluate_model1_payload(payload)
        _sync()
        iters = 30
        t0 = time.perf_counter()
        for _ in range(iters):
            inf.evaluate_model1_payload(payload)
        _sync()
        ms = (time.perf_counter() - t0) / iters * 1e3
        floor = floor_ms.get(n)
        overhead = ms - floor if floor is not None else float("nan")
        ov_pct = (overhead / ms * 100.0) if floor is not None else float("nan")
        print(f"  batch={n:5d}  {ms:8.3f} ms/call  {n/(ms/1e3):9.0f} states/s   "
              f"forward~{floor:7.3f}ms  python_overhead~{overhead:7.3f}ms ({ov_pct:4.0f}%)")


# --------------------------------------------------------------------------- D
def phase_d(inf: DenseCNNInference, n: int = 1024, legal: int = AVG_LEGAL) -> None:
    """Stage-by-stage decomposition of evaluate_model1_payload at the production
    batch (1024). Each stage is timed with a cuda.synchronize() barrier so GPU work
    is attributed to its stage, not bled into the next. Reveals what fraction of the
    now-dominant evaluator_seconds is irreducible forward vs trimmable Python glue."""
    _hr(f"PHASE D -- evaluator callback internals at batch {n}, legal/row={legal}")
    payload = _make_payload(n, legal)
    cpu_inputs = torch.frombuffer(payload["inputs"], dtype=torch.float16).reshape(payload["shape"])
    offsets = payload["legal_row_offsets"]
    dev = inf.device

    def timed(fn, iters=40):
        for _ in range(3):
            fn()
        _sync()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        _sync()
        return (time.perf_counter() - t0) / iters * 1e3

    # Pre-stage device tensors reused across timed stages. Forward needs f32; the
    # f16->f32 upcast happens on-device after the (halved) H2D, as in production.
    dev_inputs = _to_inference_device(cpu_inputs, dev).float()
    out = inf._forward_device_inputs(dev_inputs)
    policy_batch, value_batch = out["policy"], out["value"]
    counts = torch.as_tensor([offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)], dtype=torch.long)
    row_ids = torch.repeat_interleave(torch.arange(len(counts), dtype=torch.long), counts)
    row_ids_dev = row_ids.to(dev)
    flat_dev = torch.frombuffer(payload["legal_flat_indices_bytes"], dtype=torch.int64).to(dev)

    def s_h2d():
        return _to_inference_device(cpu_inputs, dev).float()

    def s_fwd():
        return inf._forward_device_inputs(dev_inputs)

    def s_valdecode():
        return decode_binned_value(value_batch).cpu().contiguous()

    def s_gather():
        selected = policy_batch[row_ids_dev, flat_dev]
        mx = torch.full((len(counts),), float("-inf"), dtype=selected.dtype, device=dev)
        mx.scatter_reduce_(0, row_ids_dev, selected, reduce="amax", include_self=True)
        exp = torch.exp(selected - mx[row_ids_dev])
        sm = torch.zeros((len(counts),), dtype=selected.dtype, device=dev)
        sm.scatter_add_(0, row_ids_dev, exp)
        return exp, sm

    exp_t, sum_t = s_gather()

    def s_item_sync():
        return bool((sum_t[counts.to(dev) > 0] <= 0).any().item())

    def s_prior_d2h():
        return (exp_t / sum_t[row_ids_dev]).cpu().contiguous()

    vt = s_valdecode()
    pt = s_prior_d2h()

    def s_serialize():
        return vt.numpy().tobytes(), pt.numpy().tobytes()

    stages = [
        ("H2D input  (f16 .to cuda + upcast f32)", s_h2d),
        ("forward    (TRT policy+value)", s_fwd),
        ("value decode + D2H", s_valdecode),
        ("prior gather/softmax (GPU)", lambda: s_gather()),
        (".item() validation SYNC", s_item_sync),
        ("prior D2H  (exp/sum .cpu)", s_prior_d2h),
        ("serialize  (.numpy().tobytes x2)", s_serialize),
    ]
    total = 0.0
    rows = []
    for name, fn in stages:
        ms = timed(fn)
        total += ms
        rows.append((name, ms))
    for name, ms in rows:
        print(f"  {name:38s} {ms:8.3f} ms  ({ms/total*100:4.1f}%)")
    print(f"  {'SUM':38s} {total:8.3f} ms")
    fwd = next(ms for nm, ms in rows if nm.startswith('forward'))
    non_fwd = total - fwd
    print(f"  forward = {fwd/total*100:.0f}% of callback; trimmable non-forward = {non_fwd:.3f} ms "
          f"({non_fwd/total*100:.0f}%)")


# --------------------------------------------------------------------------- C
class _BatchSpy:
    """Wrap the evaluator callback to record each forward chunk's batch size."""

    def __init__(self, inf: DenseCNNInference) -> None:
        self.inf = inf
        self.real = inf.evaluate_model1_payload
        self.sizes: list[int] = []

    def __enter__(self):
        spy = self

        def wrapped(payload):
            spy.sizes.append(int(payload["shape"][0]))
            return spy.real(payload)

        self.inf.evaluate_model1_payload = wrapped
        return self

    def __exit__(self, *a):
        self.inf.evaluate_model1_payload = self.real

    def reset(self) -> None:
        self.sizes = []

    def histogram(self) -> str:
        if not self.sizes:
            return "(none)"
        arr = np.array(self.sizes)
        buckets = [(1, 8), (8, 32), (32, 128), (128, 384), (384, 768), (768, 1025)]
        out = []
        for lo, hi in buckets:
            mask = (arr >= lo) & (arr < hi)
            c = int(mask.sum())
            if c:
                out.append(f"[{lo:4d},{hi:4d}): {c:5d} calls / {int(arr[mask].sum()):8d} states")
        out.append(f"mean batch = {arr.mean():.1f}  median = {int(np.median(arr))}  "
                    f"calls = {len(arr)}")
        return "\n       ".join(out)


def _run_one(sess, inf, games, vbatch):
    keys = [g["search_key"] for g in games]
    states = [g["state"] for g in games]
    move_temps = [1.0] * len(games)
    return sess.run(
        keys, states, inf,
        visits=VISITS, c_puct=1.5, temperature=1.0, seed=7,
        virtual_batch_size=vbatch, active_root_limit=256,
        root_dirichlet_total_alpha=10.83, root_dirichlet_noise_fraction=0.25,
        root_policy_temperature=1.1, fpu_reduction=0.20, virtual_loss=1.0,
        widening_policy_mass=0.95, widening_max_children=32, widening_min_children=2,
        forced_playout_k=2.0, move_temperatures=move_temps,
    )


def phase_c(inf: DenseCNNInference, vbatch: int, steps: int = 16) -> None:
    _hr(f"PHASE C -- REAL self-play: {N_GAMES} divergent games, {VISITS} visits, "
        f"vbatch={vbatch}, {steps} sequential plies")
    games = [{"search_key": i, "game_id": f"prof-{i:05d}",
              "state": engine.new_game(seed=1000 + i), "actions": []}
             for i in range(N_GAMES)]
    sess = new_mcts_session(max_states=131072)

    agg = {"wall": 0.0, "enc": 0.0, "evs": 0.0, "par": 0.0,
           "payload": 0.0, "dedup": 0.0, "cins": 0.0,
           "chunks": 0, "uniq": 0, "req": 0, "hits": 0, "dups": 0, "pos": 0}
    spy = _BatchSpy(inf)
    warm = max(1, steps // 2)  # ignore early near-opening plies; measure steady state
    with spy:
        for step in range(steps):
            playable = [g for g in games if engine.terminal(g["state"]) is None]
            if not playable:
                break
            measure = step >= warm
            _sync()
            t0 = time.perf_counter()
            searches = _run_one(sess, inf, playable, vbatch)
            _sync()
            wall = time.perf_counter() - t0
            for g, s in zip(playable, searches):
                engine.apply_action(g["state"], engine.PlacementAction(unpack_coord_id(s.action_id)))
                g["actions"].append(s.action_id)
            if not measure:
                spy.reset()
                continue
            ev = dict(dict(dict(searches[0].diagnostics or {}).get("batch", {})).get("evaluation", {}))
            agg["wall"] += wall
            agg["enc"] += float(ev.get("encoding_seconds", 0.0))
            agg["evs"] += float(ev.get("evaluator_seconds", 0.0))
            agg["par"] += float(ev.get("parse_seconds", 0.0))
            agg["payload"] += float(ev.get("payload_seconds", 0.0))
            agg["dedup"] += float(ev.get("dedup_seconds", 0.0))
            agg["cins"] += float(ev.get("cache_insert_seconds", 0.0))
            agg["chunks"] += int(ev.get("evaluator_chunks", 0))
            agg["uniq"] += int(ev.get("unique_states", 0))
            agg["req"] += int(ev.get("requested_states", 0))
            agg["hits"] += int(ev.get("cache_hits", 0))
            agg["dups"] += int(ev.get("duplicate_hits", 0))
            agg["pos"] += len(searches)

    w = max(agg["wall"], 1e-9); pos = agg["pos"]; uniq = agg["uniq"]; chunks = max(agg["chunks"], 1)
    print(f"  STEADY-STATE aggregate over plies {warm}..{steps} (games diverged):")
    print(f"  search wall                : {w:8.3f} s   ({pos/w:6.1f} pos/s, {pos} positions)")
    print(f"  -- time split (serial on GIL/eval thread; select overlaps) --")
    print(f"     encoding_seconds        : {agg['enc']:8.3f} s  ({agg['enc']/w*100:4.1f}%)  Rust state->planes")
    print(f"     evaluator_seconds       : {agg['evs']:8.3f} s  ({agg['evs']/w*100:4.1f}%)  Python callback (fwd+gather+D2H)")
    print(f"     parse_seconds           : {agg['par']:8.3f} s  ({agg['par']/w*100:4.1f}%)  Rust parse+finalize")
    print(f"     payload_seconds         : {agg['payload']:8.3f} s  ({agg['payload']/w*100:4.1f}%)  Rust->Py PyBytes plane copy (~89MB/pass)")
    print(f"     dedup_seconds           : {agg['dedup']:8.3f} s  ({agg['dedup']/w*100:4.1f}%)  cache-check + duplicate coalescing")
    print(f"     cache_insert_seconds    : {agg['cins']:8.3f} s  ({agg['cins']/w*100:4.1f}%)  bounded cache insert/eviction")
    rem = w - agg['enc'] - agg['evs'] - agg['par'] - agg['payload'] - agg['dedup'] - agg['cins']
    print(f"     remainder               : {rem:8.3f} s  ({rem/w*100:4.1f}%)  select/backup/advance/result/hash/Arc/scope")
    print(f"  -- evaluator batching (THE crux) --")
    print(f"     requested leaves        : {agg['req']}")
    print(f"     cache+dup avoided fwd   : {(agg['hits']+agg['dups'])/max(agg['req'],1)*100:4.1f}%")
    print(f"     unique states forwarded : {uniq}  (= {uniq/max(pos,1):.1f} forwards / searched position)")
    print(f"     evaluator chunks (fwd)  : {chunks}  avg {uniq/chunks:6.1f} states/chunk")
    if agg['evs'] > 0:
        print(f"     evaluator throughput    : {uniq/agg['evs']:9.0f} unique states/s   "
              f"(forward floor ~35000; callback floor ~28000)")
    print(f"  -- forward batch-size histogram (steady-state plies) --")
    print(f"       {spy.histogram()}")


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    inf = build_inference()
    floor = phase_a(inf)
    phase_b(inf, floor)
    phase_d(inf)
    phase_c(inf, VBATCH)
    print("\n[done]")


if __name__ == "__main__":
    main()
