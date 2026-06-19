"""Persisted trainer state for the KataGo-style replay buffer (the train-bucket
reuse governor + window bookkeeping).

hexfield shuffles the window IN RAM and never materializes on-disk shuffle dirs,
so this carries no shuffle-output bookkeeping. It is serialized into the
checkpoint ``meta`` by the saver and restored by the loader on the RESUME branch
only — never on an ``initialize_from`` warm start, which must begin with a fresh
governor.

Versioned with a missing-key/version-mismatch -> fresh-state fallback so old
checkpoints (which carry no ``train_state``) resume cleanly instead of raising.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

# Bump when the persisted schema changes incompatibly; a mismatch loads fresh.
TRAIN_STATE_VERSION = 1


@dataclass
class HexfieldTrainState:
    # Monotone count of self-play rows ever seen by the governor. NEVER
    # decremented: window selection uses the live manifest total, but the bucket
    # accrual is driven by this cumulative counter so a pruned / regenerated
    # window can't spuriously trip the reload branch.
    total_num_data_rows: int = 0
    # Cumulative gradient samples consumed (diagnostics / reuse accounting).
    global_step_samples: int = 0
    # First global row index still inside the current window.
    window_start_data_row_idx: int = 0
    # Train-bucket reuse governor. ``level`` is credited by each new self-play row
    # * max_train_bucket_per_new_data and debited by effective_rows at selection
    # time; ``level_at_row`` is the cumulative-row watermark the last accrual was
    # computed against.
    train_bucket_level: float = 0.0
    train_bucket_level_at_row: int = 0
    train_steps_since_last_reload: int = 0
    # Optional no-repeat-files set (defaults OFF for hexfield's single-game shards,
    # so this normally stays empty; kept so a run that opts in survives resume).
    data_files_used: set[str] = field(default_factory=set)
    version: int = TRAIN_STATE_VERSION

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "HexfieldTrainState":
        """Tolerant load. ``None``/non-mapping/old-format/version-mismatch all
        return a FRESH state (old checkpoints carry no ``train_state``)."""
        if not isinstance(raw, Mapping):
            return cls()
        if int(raw.get("version", 0)) != TRAIN_STATE_VERSION:
            # Incompatible persisted schema -> start the governor fresh rather
            # than misinterpreting fields. Safe: the bucket simply re-accrues.
            return cls()
        return cls(
            total_num_data_rows=int(raw.get("total_num_data_rows", 0)),
            global_step_samples=int(raw.get("global_step_samples", 0)),
            window_start_data_row_idx=int(raw.get("window_start_data_row_idx", 0)),
            train_bucket_level=float(raw.get("train_bucket_level", 0.0)),
            train_bucket_level_at_row=int(raw.get("train_bucket_level_at_row", 0)),
            train_steps_since_last_reload=int(raw.get("train_steps_since_last_reload", 0)),
            data_files_used=set(str(item) for item in raw.get("data_files_used", ()) or ()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "total_num_data_rows": int(self.total_num_data_rows),
            "global_step_samples": int(self.global_step_samples),
            "window_start_data_row_idx": int(self.window_start_data_row_idx),
            "train_bucket_level": float(self.train_bucket_level),
            "train_bucket_level_at_row": int(self.train_bucket_level_at_row),
            "train_steps_since_last_reload": int(self.train_steps_since_last_reload),
            "data_files_used": sorted(self.data_files_used),
        }
