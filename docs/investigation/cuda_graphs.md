# OPTION 1 — CUDA Graphs on the small-Npad serve forward

Design agent: `cuda_graphs`. Goal: collapse the hundreds of per-forward kernel
launches in the EARLY/MID launch-bound serve flush into a single graph replay,
without changing the math (fp16-tol parity).

## 1. Why this targets the right bottleneck

Established profiling facts (do not re-derive):

- EARLY/MID flushes are GPU-LAUNCH-bound: `submit_payload` enqueues ~17 ms of
  host-side work while the GPU does only ~11 ms; GPU ~57% idle. Hundreds of
  tiny kernels per forward.
- The forward is `HexfieldNet.forward_policy_value` (inference path), invoked
  once per `plan_groups` group in `_forward_group` (inference.py:209).
- LATE/DEEP flushes are GPU-COMPUTE-bound (GPU ~97%) — graphs help little
  there; consistent with the existing `_compile_max_npad=512` gate.

A CUDA graph captures the entire kernel sequence of one forward at a fixed
shape and replays it with ONE host-side launch instead of hundreds. This is the
canonical fix for a launch-bound forward and is complementary to (or a
replacement for) the existing `torch.compile` path: compile *fuses* kernels
(fewer launches, ~2.4x already), graphs *eliminate the per-launch host cost
entirely* for whatever launches remain. The hypothesis is that even on the
compiled forward, the residual host enqueue is still > GPU compute at small
Npad, so a graph replay closes the remaining GPU-idle gap.

## 2. Structural facts that make capture feasible

The forward is **static-shape and control-flow-free per `(B, Npad,
request_moves_left)`**:

- `forward_policy_value` (model.py:418) -> `trunk` -> `build_attn_bias`.
  `build_attn_bias` INFERENCE branch (model.py:334-348) is taken under
  `torch.no_grad()` (the serve always runs `@torch.no_grad()`), is pure
  tensor ops (clamp / mul-add / gather / permute / broadcast-add), and has NO
  data-dependent control flow.
- `request_moves_left` only toggles whether the `moves_left` head runs
  (model.py:440). It is fixed per run (`divergences.moves_left_utility`,
  search.rs:641), so a graph captured for one value of the flag is valid for
  the whole run. Capture for the run's actual flag value (default: probe it,
  or capture both and pick at replay time).
- `trunk` builds `self_idx = torch.arange(n, ...)` and `gather_idx` inside the
  forward (model.py:363). `arange` over a fixed `n` is captured as constant
  kernels in the graph — fine. The LUT buffers (`_cell_bias_lut`,
  `_exact_lut`) are persistent module buffers at fixed addresses — fine.
- All buffers (`bias_table`, `_cell_bias_lut`, weights) live at fixed device
  addresses; CUDA graphs require fixed addresses + shapes, which holds.

The only thing that varies between calls at a fixed bucket is the **input
tensor contents** (feats / nbr / mask / coords) and the **batch dim B**.

### The batch-dim problem and the chosen resolution

CUDA graphs need a fixed shape, but `g = end - start` (group batch size) varies
per flush (1..~64). Two strategies:

- **(A) Capture per `(Npad, B)`**: one graph per observed `(Npad, B)` pair.
  B ranges 1..~64 and Npad over `<=512/64 = 8` buckets -> up to ~512 graphs.
  Each graph holds static input + output buffers; memory grows with the number
  of live graphs. Too many, and the pad-waste / capture cost explodes.
- **(B) Pad B up to a fixed bucket** (CHOSEN for the prototype): for each Npad
  bucket, capture ONE graph at a fixed `B_cap` (e.g. 64). At replay, copy the
  real `g <= B_cap` rows into the static input buffer's first `g` rows, zero
  the rest (or leave stale — see parity note), replay, and slice the first `g`
  rows out of the static output. Pad rows are **pad-inert** by the model's
  invariant (model.py:11-23): a padded all-zero / masked extra batch row cannot
  affect any real row's output (attention is batched per-row, no cross-batch
  mixing; convs are per-node). So the first `g` output rows are bit-identical to
  a `g`-sized forward. This collapses ~512 graphs to ~8 (one per Npad bucket),
  at the cost of always computing `B_cap` rows. Whether the wasted compute is
  cheaper than the launch overhead it removes is exactly what the bench
  measures.

Strategy (B) is the minimal, memory-bounded design. The prototype validates ONE
representative bucket each: `(B_cap=64, Npad=256)` and `(B_cap=64, Npad=512)`.

## 3. Interaction with torch.compile

`torch.compile` reduce-overhead mode uses CUDA graphs internally, but the serve
gates compile to `Npad<=512` and forces the batch dim dynamic
(`mark_dynamic(t, 0)`, inference.py:266-269) — which DISABLES cudagraphs in
inductor (cudagraphs require static shapes; a dynamic batch dim opts out). So
the current compiled path is NOT already graphed. Two ways to get graphs:

1. **`mode="reduce-overhead"` with a STATIC batch dim** (let inductor capture
   the graph). Requires dropping `mark_dynamic` on dim 0 and instead padding B
   to a fixed bucket — i.e. strategy (B) but via inductor. Cleanest if it works,
   but inductor cudagraph capture is finicky with the existing
   `suppress_errors`/`automatic_dynamic_shapes=False` config and can silently
   fall back.
2. **Manual `torch.cuda.CUDAGraph` capture** of `forward_policy_value` (or the
   eager `_raw_fpv`) at a fixed `(B_cap, Npad)` with static IO buffers (CHOSEN
   for the prototype). Explicit, measurable, no dependence on inductor's
   cudagraph heuristics. We can capture either the eager forward or the compiled
   forward; the prototype captures the **compiled** forward (best of both:
   fused kernels AND zero per-launch host cost) and also measures eager-captured
   as a control.

The prototype keeps the existing compile path untouched and adds capture on top,
so it is a strict superset experiment.

## 4. Minimal prototype (standalone script — NO packages/ changes)

A single script `scripts/_hexfield_cudagraph_proto.py` that:

1. Builds a CUDA `HexfieldNet().eval()`.
2. For each `(B_cap, Npad)` in `{(64,256),(64,512)}`:
   - Allocates STATIC input buffers `s_feats (B,Npad,F) fp32`, `s_nbr
     (B,Npad,6) long`, `s_mask (B,Npad) bool`, `s_coords (B,Npad,2) long` on
     CUDA. These match exactly what `_forward_group` produces (inference.py:232,
     same dtypes) post-H2D.
   - Defines `fwd()` = `forward_policy_value(s_feats, s_nbr, s_mask, s_coords,
     request_moves_left=RML)` under `torch.no_grad()` +
     `autocast(fp16)` — the EXACT serve call context (inference.py:270-277).
   - **Warmup on a side stream** (required by the CUDA graphs API): run `fwd()`
     3x inside `with torch.cuda.stream(side_stream)` after
     `side_stream.wait_stream(current)`, then sync. This forces all lazy
     allocations / autocast cast caches / cuBLAS workspaces to settle so capture
     records a stable kernel set.
   - **Capture**: `g = torch.cuda.CUDAGraph(); with torch.cuda.graph(g):
     out = fwd()`. Keep references to the returned `out` tensors (the graph's
     static OUTPUT buffers — `value`, `policy`, and `moves_left` if RML).
   - **Replay**: `s_feats.copy_(real_feats); ...; g.replay();
     torch.cuda.synchronize()`; read `out["value"]` / `out["policy"]`.
3. Parity: build a real random input at `(g_real, Npad)` with `g_real<=B_cap`
   (e.g. 64 and 40), copy into the first `g_real` rows of the static buffers
   (zero the remaining `B_cap-g_real` rows + set their mask False), replay,
   and compare `out["value"][:g_real]` / `out["policy"][:g_real]` against a
   **direct** `forward_policy_value` over a `(g_real, Npad)` tensor.
   maxabsdiff gate fp16 tol `3e-3` (matches `_hexfield_compile_overlap_test.py`).
4. Also re-run the full async + compile parity gate
   (`_hexfield_compile_overlap_test.py`) unchanged to prove the model/serve math
   itself is untouched (the prototype script imports the model, does not modify
   it).
5. Bench: time `B_cap` direct compiled forward vs graph replay at each shape
   (CUDA-synced, ~100 reps after warmup), report ms and the ratio + the
   per-flush host-enqueue implication.

### Draft prototype (to be created in the VALIDATE phase, not now)

```python
# scripts/_hexfield_cudagraph_proto.py
import sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))
import numpy as np, torch
import hexfield.constants as C
from hexfield.model import HexfieldNet

assert torch.cuda.is_available()
torch.manual_seed(0)
dev = torch.device("cuda")
model = HexfieldNet().eval().to(dev)
RML = False  # the run's divergences.moves_left_utility; capture for its value
F = C.NUM_FEATURES

def make_static(Bc, Np):
    return dict(
        feats=torch.zeros(Bc, Np, F, device=dev),
        nbr=torch.full((Bc, Np, 6), Np, dtype=torch.long, device=dev),
        mask=torch.zeros(Bc, Np, dtype=torch.bool, device=dev),
        coords=torch.zeros(Bc, Np, 2, dtype=torch.long, device=dev),
    )

def capture(Bc, Np):
    s = make_static(Bc, Np)
    def fwd():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            return model.forward_policy_value(
                s["feats"], s["nbr"], s["mask"], s["coords"],
                request_moves_left=RML)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fwd()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fwd()
    return s, g, out

def direct(feats, nbr, mask, coords):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        return model.forward_policy_value(feats, nbr, mask, coords,
                                          request_moves_left=RML)

def rand_inputs(g_real, Np):
    feats = torch.randn(g_real, Np, F, device=dev)
    nbr = torch.randint(0, Np, (g_real, Np, 6), dtype=torch.long, device=dev)
    mask = torch.ones(g_real, Np, dtype=torch.bool, device=dev)
    coords = torch.randint(-20, 21, (g_real, Np, 2), dtype=torch.long, device=dev)
    return feats, nbr, mask, coords

TOL = 3e-3
for Bc, Np in [(64, 256), (64, 512)]:
    s, g, out = capture(Bc, Np)
    for g_real in (Bc, 40):
        feats, nbr, mask, coords = rand_inputs(g_real, Np)
        ref = direct(feats, nbr, mask, coords)
        # load real rows into static buffers, neutralize the pad tail
        s["feats"].zero_();  s["feats"][:g_real].copy_(feats)
        s["nbr"].fill_(Np);  s["nbr"][:g_real].copy_(nbr)
        s["mask"].zero_();   s["mask"][:g_real].copy_(mask)
        s["coords"].zero_(); s["coords"][:g_real].copy_(coords)
        g.replay(); torch.cuda.synchronize()
        dv = float((out["value"][:g_real].float()-ref["value"].float()).abs().max())
        dp = float((out["policy"][:g_real].float()-ref["policy"].float()).abs().max())
        print(f"  PARITY Bc={Bc} Np={Np} g={g_real}: value={dv:.5f} policy={dp:.5f} "
              f"{'PASS' if max(dv,dp)<=TOL else 'FAIL'}")
    # bench: direct(B_cap) vs replay
    feats, nbr, mask, coords = rand_inputs(Bc, Np)
    for _ in range(10): direct(feats, nbr, mask, coords)
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(100): direct(feats, nbr, mask, coords)
    torch.cuda.synchronize(); dms=(time.time()-t0)/100*1000
    s["feats"].copy_(feats); s["nbr"].copy_(nbr); s["mask"].copy_(mask); s["coords"].copy_(coords)
    for _ in range(10): g.replay()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(100): g.replay()
    torch.cuda.synchronize(); rms=(time.time()-t0)/100*1000
    print(f"  BENCH  Bc={Bc} Np={Np}: direct={dms:.3f}ms replay={rms:.3f}ms ({dms/rms:.2f}x)")
print("RESULT: prototype run complete")
```

Note: `direct` above is the EAGER forward for the bench control. To capture the
COMPILED forward instead, build a `torch.compile(model.forward_policy_value)`
exactly as `HexfieldEvaluator.__init__` does (suppress_errors,
automatic_dynamic_shapes=False), warm it at `(B_cap,Np)` static, and capture
that callable. The strongest result is `replay-of-compiled` vs
`compiled-direct`; the bench should report both eager-direct, compiled-direct,
and replay so the launch-overhead delta is isolated.

## 5. Parity strategy (exactly what proves math is unchanged)

1. In-script PARITY (above): graph replay output `[:g_real]` vs direct
   `forward_policy_value` over a true `(g_real, Npad)` tensor — maxabsdiff
   fp16 tol `3e-3` for `value` and `policy` (and `moves_left` if RML), at both
   `(64,256)` and `(64,512)` and at a sub-cap `g=40` (proves pad-inertness of
   the B-padding). This is the load-bearing parity: it proves replay == direct
   forward AND that B-padding does not pollute real rows.
2. The model itself is unmodified, so the existing authoritative gates still
   hold verbatim:
   - `scripts/_hexfield_compile_overlap_test.py` -> must print `RESULT: PASS`.
   - `tests/test_hexfield_model.py` and
     `tests/test_hexfield_continuous_parity.py` -> must pass.
   The prototype touches nothing under `packages/`, so these are unchanged; they
   confirm the baseline serve math we are comparing against is intact.

If/when graphs are promoted into `inference.py` (NOT in this prototype), the
async + action parity gates (`_hexfield_compile_overlap_test.py` and
`_hexfield_async_parity.py`) become the promotion gate.

## 6. Validate recipe (commands; baseline number to beat)

All commands via WSL from the Git-Bash Bash tool. GPU is FREE; do NOT start the
run/supervisor.

Baseline numbers to beat (measure FIRST, same session, same machine):

- Forward ms at the two shapes (current eager + compiled):
  ```
  wsl.exe -- bash -lc 'cd /mnt/e/Hexo-BotTrainer-hexgt && PYTHONPATH=packages/hexfield/python /root/.venvs/hexgt-build/bin/python scripts/_hexfield_profile_fwd.py 64 256'
  wsl.exe -- bash -lc 'cd /mnt/e/Hexo-BotTrainer-hexgt && PYTHONPATH=packages/hexfield/python /root/.venvs/hexgt-build/bin/python scripts/_hexfield_profile_fwd.py 64 512'
  ```
  Record `ms/fwd` for each — this is the per-forward number the graph replay must
  beat at the same shape.
- Serve flush ms (early/mid regimes, the launch-bound ones) — the production
  metric the whole option exists to improve:
  ```
  wsl.exe -- bash -lc 'cd /mnt/e/Hexo-BotTrainer-hexgt && PYTHONPATH=packages/hexfield/python /root/.venvs/hexgt-build/bin/python scripts/_hexfield_serve_profile.py'
  ```
  Record `early(20-260)` and `mid(256-768)` `submit` (host enqueue) and
  `evaluate_payload` ms. submit ~17 ms with GPU ~11 ms is the target gap.

Implement + run the prototype:

1. Create `scripts/_hexfield_cudagraph_proto.py` from the draft in section 4
   (add the compiled-capture control as noted).
2. Run it:
   ```
   wsl.exe -- bash -lc 'cd /mnt/e/Hexo-BotTrainer-hexgt && PYTHONPATH=packages/hexfield/python /root/.venvs/hexgt-build/bin/python scripts/_hexfield_cudagraph_proto.py'
   ```
   Expect: all PARITY lines PASS (maxabsdiff <= 3e-3); BENCH `replay` ms
   meaningfully below `compiled-direct` ms at `(64,256)` and `(64,512)`.

3. Re-run the authoritative parity gate (model unmodified, must still pass):
   ```
   wsl.exe -- bash -lc 'cd /mnt/e/Hexo-BotTrainer-hexgt && /root/.venvs/hexgt-build/bin/python scripts/_hexfield_compile_overlap_test.py'
   ```
   Expect `RESULT: PASS`.

Decision rule: the option is worth promoting into `inference.py` only if
`replay` ms < `compiled-direct` ms by a margin large enough that, multiplied by
the per-flush group count (early/mid flushes have several small-Npad groups),
the modeled `submit` host-enqueue drops below the ~11 ms GPU floor — i.e. the
serve becomes GPU-bound instead of launch-bound. The bench gives replay ms; the
serve_profile baseline gives the submit/GPU split to model against.

## 7. Honest risk assessment

Plausible WIN: the bottleneck is explicitly per-launch host cost; CUDA graphs
are the textbook fix and routinely remove 50-90% of launch overhead for
launch-bound forwards. At small Npad where GPU is ~57% idle, replacing hundreds
of launches with one replay should recover most of that idle time. Expected
forward speedup at `(64,256)`/`(64,512)`: plausibly 1.3-2x on the launch-bound
component, translating to a meaningful early/mid `submit` ms drop.

Risks / why it might NOT pay off:

- **B-padding waste**: strategy (B) always computes `B_cap=64` rows. If real
  early/mid groups are small (g << 64), the graph wastes compute on pad rows.
  The attention cost is `O(B * S^2)`; padding B from g to 64 multiplies GPU
  work by `64/g`. At `Np=256`, `S=264`, that compute may exceed the launch
  overhead saved. Mitigation: capture a few `B_cap` tiers (e.g. 8/16/32/64) and
  pick the smallest `>= g` — bounded graph count (8 Npad x 4 B = 32 graphs).
  The prototype tests only `B_cap=64`; if it wins there it wins more with tiers.
- **Capture fragility**: cuBLAS/cuDNN workspace allocation during capture, or
  any host-side `.item()`/data-dependent op, aborts capture. The forward has
  none (verified: no `.item()` in `forward_policy_value`/`trunk`/
  `build_attn_bias`; the `_BiasGather` autograd path is grad-only and not taken
  under no_grad). autocast cast caches must be warmed on the side stream
  (handled).
- **Interaction with compile**: capturing the compiled callable may double-graph
  (inductor cudagraphs + manual). The prototype captures with the existing
  serve compile config but with a STATIC batch dim; if inductor refuses, fall
  back to capturing the eager forward (still removes launch overhead, just
  without the fusion win — the eager-vs-replay bench covers this).
- **Memory**: each graph holds static IO + intermediate buffers. ~8 Npad
  buckets x (up to 4 B tiers) at 12 GB should fit (the largest captured here is
  `64x512`), but late buckets (Npad up to 3000+) are deliberately NOT captured
  (compute-bound, gated out, and memory-prohibitive).

Overall: the prototype is minimal (one standalone script, zero packages/
changes), directly measurable (replay ms vs compiled-direct ms at the exact
launch-bound shapes), and parity-gated (in-script fp16 maxabsdiff + the
authoritative compile/overlap test). A measurable win is plausible because the
bottleneck is precisely the cost graphs remove. The honest open question is
whether B-padding compute waste eats the launch-overhead savings at small g —
which is exactly what the bench answers, making this worth prototyping.
