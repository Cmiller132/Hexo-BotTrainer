"""Sample finalizer for Hexformer AR self-play."""

from __future__ import annotations

from typing import Any


class HexformerSampleFinalizer:
    """Self-play writes finalized Hexformer samples directly into the buffer."""

    def finalize(self, *, ctx: Any, components: Any, epoch: int) -> dict[str, Any]:
        trainer = components.model.trainer
        return {
            "status": "completed",
            "epoch": epoch,
            "sample_count": trainer.buffer.sample_count,
            "note": "hexformer_ar self-play finalizes sparse samples as games finish",
        }
