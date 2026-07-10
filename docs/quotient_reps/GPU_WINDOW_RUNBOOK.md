# Phase-B idle-GPU window runbook

Run this only in an owner-approved idle window. The benchmark scripts refuse
GPU execution unless `--allow-gpu` is explicit; their CPU smoke modes do not
touch CUDA.

## 1. Safety preconditions

1. Confirm the live soak is paused and will not auto-resume during the window.
2. Confirm the Phase-R/prefit job and all of its data workers have finished.
3. Run `nvidia-smi` and require zero compute processes, near-zero utilization,
   and no unexplained allocation. If any condition is unclear, stop.
4. Use one Torch process at a time. Do not overlap either benchmark, pytest, or
   the later WSL aggregate.
5. Start from the `quotient-phase-b` worktree and preserve all output in the
   operator log before editing the results document.

## 2. WSL setup

The production Triton and Rust-pack stack lives in the WSL build environment,
so use it for the GPU measurements:

```bash
cd /mnt/e/Hexo-BotTrainer-hexgt-quotient-phase-b
source /root/.venvs/hexgt-build/bin/activate
unset $(env | awk -F= '/^HEXFIELD/{print $1}')
unset CUDA_VISIBLE_DEVICES
export PYTHONPATH=packages/hexfield_eq/python:packages/hexo_engine/python:packages/hexo_utils/python:packages/hexo_train/python:packages/hexo_runner/python
```

Re-run the idle check immediately before each of the next three commands.

## 3. Tile benchmark

```bash
python -B scripts/bench_quotient_tile.py --allow-gpu --warmup 10 --iterations 30
```

Paste the dense, fused-conv, and attention markdown tables into
`RESULTS_PHASE_B.md` section 5.1. Compare dense-width efficiency to C=192. If
C=160 drops by more than 10%, re-nominate C=176 with
`reg:8,mirror:8,axis:8,triv:8` as required by Phase-B section 7.1.

## 4. Full-network serve benchmark

```bash
python -B scripts/bench_quotient_serve.py --allow-gpu --warmup 5 --iterations 20
```

Paste its markdown table into `RESULTS_PHASE_B.md` section 5.2. Judge each arm
against the alpha=4/7 projection: B1 1.615x, B2 1.466x, B3 2.068x. PASS means
the measured speedup is within +/-20% of that value; otherwise record the
measured miss and explanation rather than changing the criterion.

## 5. CUDA typed-serve gate

```bash
python -B -m pytest -p no:cacheprovider tests/test_hexfield_eq_typed_serve.py::test_mixed_half_kernels_and_cuda_graphs -q -ra
```

Record the pass and tail beside the section 5.2 benchmark evidence.

## 6. Deferred WSL aggregate

Run this only after WSL is free, still one Torch process at a time. From WSL:

```bash
cd /mnt/e/Hexo-BotTrainer-hexgt-quotient-phase-b
source /root/.venvs/hexgt-build/bin/activate
unset $(env | awk -F= '/^HEXFIELD/{print $1}')
export CUDA_VISIBLE_DEVICES=-1
export PYTHONPATH=packages/hexfield_eq/python:packages/hexo_engine/python:packages/hexo_utils/python:packages/hexo_train/python:packages/hexo_runner/python
python -B -m pytest -p no:cacheprovider \
  tests/test_hexfield_eq_typed_model.py \
  tests/test_hexfield_eq_typed_regression.py \
  tests/test_hexfield_eq_typed_serve.py \
  tests/test_hexfield_eq_typed_checkpoint_meta.py \
  tests/test_hexfield_eq_equivariance.py \
  tests/test_hexfield_eq_perm_fold.py \
  tests/test_hexfield_eq_serve.py \
  tests/test_hexfield_eq_triton_ray.py \
  tests/test_hexfield_eq_ray_block.py \
  tests/test_hexfield_eq_register_lane.py \
  tests/test_hexfield_eq_checkpoint_meta.py \
  tests/test_hexfield_eq_rust_parity.py \
  tests/test_hexfield_eq_derivation.py \
  tests/test_hexfield_eq_reps_group.py \
  tests/test_hexfield_eq_reps_homdims.py \
  tests/test_hexfield_eq_reps_parity.py \
  tests/test_hexfield_eq_reps_typed_layers.py \
  tests/test_hexfield_eq_reps_toynet.py -q -ra --tb=short
```

Append the aggregate count and tail to `RESULTS_PHASE_B.md` section 4. Do not
start a second Torch command until this one exits.
