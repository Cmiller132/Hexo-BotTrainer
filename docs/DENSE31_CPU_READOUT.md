# Dense 31-tap convolution: CPU investigation read-out

Date: 2026-07-10. Branch: `raytap-31tap-cpu`. Status: CPU reference
investigation complete.

## Bottom line

The dense31 reference operator, recompute-in-backward Function, mode plumbing,
and all-distance ray-tap checkpoint surgery are implemented and pass the
combined dense31/ray-tap CPU suite. The surgery test includes the required
negative control: folding only distance 1 does not reproduce a trained-alpha
ray-tap network.

The expressivity probe did **not** produce the hypothesized single-block
advantage for dense31. P1 ray-tap and P2 dense31 converged to essentially the
same validation result, with ray-tap marginally ahead in this run. This is a
material negative finding for Design A's motivating claim on this synthetic
task; no post-hoc tuning or rerun was used to manufacture a gap.

On CPU, dense31 took 1.51--2.15 times the wall time of ray-tap in the matched
benchmark rows. These ratios do not predict a future GPU kernel.

## What was built

- A shell-wise D6 gather index for center plus six directions at distances
  1--5, and tap-count-generic tied-weight materialization.
- `dense31` architecture plumbing through trunk convolution blocks,
  checkpoint metadata, and state-dict shape fallback. The stem and heads stay
  baseline 7-tap operators.
- Fresh initialization that samples exactly the baseline seven-tap parameter
  stream and zeroes shells 2--5 without consuming RNG.
- A shared visibility-masked shell-major gather based on the existing ray-tap
  masking helpers, plus a custom autograd Function that recomputes the gather
  for `grad_weight` and scatter-adds `grad_x`.
- All-distance warm-start surgery: every direction block at distance `k` is
  the source direction block column-scaled by trained `alpha[k]`; `alpha` is
  removed, metadata becomes `dense31`, and optimizer state is cleared.
- CPU correctness/oracle tests, an expressivity probe, and a fixed-shape
  micro-benchmark. Full raw measurements are in
  [DENSE31_EXPRESSIVITY_CURVES.csv](DENSE31_EXPRESSIVITY_CURVES.csv) and
  [DENSE31_CPU_BENCH.csv](DENSE31_CPU_BENCH.csv).

## Correctness result

Environment:

- Windows `C:/Python314/python.exe`
- PyTorch `2.10.0+cu126`
- `CUDA_VISIBLE_DEVICES=-1`; `torch.cuda.is_available() == False`
- `PYTHONPATH=packages/hexfield_eq/python;C:\Users\epicm\AppData\Roaming\Python\Python314\site-packages`

Command:

```text
C:/Python314/python.exe -m pytest \
  tests/test_hexfield_eq_dense31.py tests/test_hexfield_eq_raytap.py -q -ra
```

Result: **37 passed, 6 skipped, 0 failed** in 12.95 seconds. Three skips were
for the deliberately unbuilt Rust extension. The remaining three were
CUDA/Triton serve tests, all of which skipped as required; no CUDA-marked test
ran.

The dense31 coverage includes shell-wise gather-index sanity, all 12 D6 group
elements at full-network level, fresh-init equivalence and far-shell liveness,
all-distance surgery equivalence plus the distance-1-only negative control,
Function-vs-naive outputs and gradients, float64 gradcheck, saved-tensor
discipline, mode/meta/state-dict behavior, environment validation, and
optimizer/gradient-group classification. The combined run also retains the
entire existing ray-tap regression suite.

## Expressivity probe

Method: 64 training and 24 validation positions, 8--24 stones per side sampled
inside a radius-6 disk, 200 AdamW steps, batch size 4, cosine LR decay from
`2e-3`, and eight CPU threads. All arms used the same examples, batch schedule,
step budget, and depth-matched initialization seed. To keep the four-arm CPU
probe comfortably within its time budget, it used the same regular D6
structure at `C=48`, `C_ORBIT=4`; parameter counts below therefore describe
the probe models, not the production-width network.

The generated dataset contained 49,401 legal-cell labels and 57 positives
(0.1154%). The rarity of positives makes the absolute BCE small and makes AUC
more sensitive to a small number of validation positives.

| Arm | Operator / depth | Parameters | Steps | Val BCE | Val AUC | Seconds |
|---|---|---:|---:|---:|---:|---:|
| P1 | ray-tap / 1 block | 11,217 | 200 | 0.01814 | 0.99331 | 21.5 |
| P2 | dense31 / 1 block | 20,393 | 200 | 0.01815 | 0.99294 | 23.3 |
| P3 | ray-tap / 2 blocks | 13,973 | 200 | 0.00925 | 0.98284 | 41.1 |
| P4 | dense31 / 2 blocks | 32,325 | 200 | 0.00870 | 0.98347 | 45.3 |

P1 and P2 had identical step-0 metrics and stayed nearly coincident throughout
the curve. At step 200, dense31 was worse by 0.0000066 BCE and 0.00037 AUC,
which is operationally a tie rather than evidence of a dense31 advantage.
Depth roughly halved BCE for both operators; P4 modestly beat P3 on BCE and
AUC, but this does not rescue the missing P1-vs-P2 separation. The complete
20-step curves, including training BCE and elapsed time, are preserved in the
CSV rather than summarized away.

## CPU micro-benchmark

Method: one equipped convolution, `Npad=448`, `C=192`, `torch.set_num_threads(8)`,
one warm-up and three measured repetitions, with the median reported. The
reference rows use ordinary autograd; Function rows use the recompute path
(`_RayTapTaps` for ray-tap and `_Dense31ConvFn` for dense31).

| B | Phase | Path | Ray-tap median (ms) | Dense31 median (ms) | Dense31 / ray-tap |
|---:|---|---|---:|---:|---:|
| 2 | forward | reference | 4.398 | 8.250 | 1.876x |
| 2 | forward | Function | 5.152 | 10.750 | 2.086x |
| 2 | forward + backward | reference | 15.665 | 23.650 | 1.510x |
| 2 | forward + backward | Function | 16.455 | 34.552 | 2.100x |
| 8 | forward | reference | 26.813 | 57.674 | 2.151x |
| 8 | forward | Function | 26.672 | 45.497 | 1.706x |
| 8 | forward + backward | reference | 65.556 | 106.475 | 1.624x |
| 8 | forward + backward | Function | 79.152 | 126.561 | 1.599x |

Recomputation was slower than the naive reference in both dense31
forward+backward cases, as expected for a memory-saving strategy. The B=8
forward-only dense31 Function median was faster than its reference median;
with only three timed samples and visible max-time outliers in several rows,
that should be treated as allocator/cache noise rather than a kernel result.

## Deviations and limitations

- The probe used `C=48`, `C_ORBIT=4` instead of production `C=192`,
  `C_ORBIT=16` so all four arms would complete quickly on CPU. The D6 tying,
  tap geometry, hard visibility, and relative model definitions were
  unchanged.
- The plan says "12 signed directions" for the synthetic label. Code reality
  has six entries in `DIRECTIONS` (the positive and negative directions of
  three axes); the 12 ray-length slots are two visibility sides times those
  six directions. The independent label walker therefore checks all six
  geometric signed directions under the own-side pattern.
- `_Dense31ConvFn` retains the small `(B,N)` output mask in addition to `x`,
  tap indices, reach, and generated weight because backward must reproduce the
  masked output epilogue. It does not retain either `(B,N,31*C)` or
  `(B,N,31,C)` gathered storage.
- The synthetic positive rate was only 0.1154%. This is part of the measured
  negative result and was not corrected by resampling or class weighting
  after seeing the outcome.
- The benchmark is a short CPU micro-benchmark, not a throughput prediction
  for a fused GPU implementation.

## GPU-side work deliberately not done

No CUDA allocation, GPU timing, Triton/K1/split-serve dense31 kernel, fp16
serve folding, CUDA graph integration, or `inference.py` fast-profile work was
attempted. No production configuration, soak script, `_triton_*.py` internal,
real training corpus, WSL environment, or other worktree was touched. The Rust
extension was not built. Designs B/C and learned gates remain out of scope.
