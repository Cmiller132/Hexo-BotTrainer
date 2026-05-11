"""Shared model-facing utilities for Hexo model packages."""

from .inference import (
    InferenceServer,
    NetworkOutput,
    collate_encoded_states,
    encoded_state_tensor,
    legal_mask_from_state,
)
from .model_api import ModelPlugin, TensorBatch, load_model_plugin
from .replay import (
    ReplayBuffer,
    ReplaySample,
    batch_from_samples,
    inspect_replay,
    iter_jsonl,
    list_replay_files,
    write_jsonl,
)
from .rust_bridge import import_rust_module, run_uniform_selfplay, rust_available

__all__ = [
    "InferenceServer",
    "ModelPlugin",
    "NetworkOutput",
    "ReplayBuffer",
    "ReplaySample",
    "TensorBatch",
    "batch_from_samples",
    "collate_encoded_states",
    "encoded_state_tensor",
    "import_rust_module",
    "inspect_replay",
    "iter_jsonl",
    "legal_mask_from_state",
    "list_replay_files",
    "load_model_plugin",
    "run_uniform_selfplay",
    "rust_available",
    "write_jsonl",
]

__version__ = "0.1.0"
