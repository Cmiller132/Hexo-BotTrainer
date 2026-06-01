"""Inference-backend benchmark scaffolding for the dense_cnn 96x6 + P7 model.

`bench_harness` is the shared, reusable harness (model load, representative
inputs, timing with full warmup + mean/stdev/p50/p95, and FP32-reference
correctness comparison). Variants are pluggable; FP32 / FP16-AMP / BF16 are
provided. See `bench_harness.__doc__` for invocation.
"""
