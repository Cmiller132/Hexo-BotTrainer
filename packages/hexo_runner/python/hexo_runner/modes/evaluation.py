"""Evaluation runner mode (never implemented).

The original plan: build a fixed series of matches between configured
opponents on top of batch mode, mark sessions as evaluation runs so players
reduce exploration/noise, and own score keeping/strength estimates. That
shared layer was never built — each model package wrote its own SealBot eval
harness instead (dense_cnn_restnet/evaluation.py, hexo_models/dense_cnn/
evaluation.py, hexo_models/hexgt/evaluation.py, hexgnn/evaluation.py).
"""

from __future__ import annotations


# UNUSED(2026-06-12): never-implemented stub; only references are this
# definition and the re-export in modes/__init__.py (grep packages/tests/
# scripts excl. archive). Superseded by the per-model eval harnesses listed
# in the module docstring.
def run_evaluation(config: object) -> object:
    """Run an evaluation suite by preparing eval match configs for batch mode."""

    raise NotImplementedError("evaluation mode will be built on runner sessions.")
