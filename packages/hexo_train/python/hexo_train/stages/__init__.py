"""Default stage implementations for the shared training pipeline.

Each module owns boring orchestration for one area. Model plugins can override
any stage through `ComponentOverrides.stage_handlers` or by exposing a method
with the same stage name.
"""

from .artifacts import write_diagnostics
from .checkpoint import (
    load_or_initialize_checkpoint,
    save_checkpoint,
    update_selfplay_checkpoint_pointer,
)
from .samples import (
    build_sample_window,
    finalize_pending_samples,
    prepare_sample_store,
    refresh_sample_index,
)
from .selfplay import maybe_generate_selfplay
from .training import train_steps

__all__ = [
    "build_sample_window",
    "finalize_pending_samples",
    "load_or_initialize_checkpoint",
    "maybe_generate_selfplay",
    "prepare_sample_store",
    "refresh_sample_index",
    "save_checkpoint",
    "train_steps",
    "update_selfplay_checkpoint_pointer",
    "write_diagnostics",
]
