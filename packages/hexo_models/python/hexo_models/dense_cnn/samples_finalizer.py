"""Sample finalization boundary for dense CNN self-play."""

from __future__ import annotations

from typing import Any, Mapping

from .samples import SampleBuffer


class DenseCNNSampleFinalizer:
    """Reports finalized samples written during self-play.

    Dense CNN self-play appends already-finalized compressed samples after each
    completed game because the terminal value is known in the same process.
    This object keeps the generic epoch lifecycle explicit.
    """

    def __init__(self, buffer: SampleBuffer) -> None:
        self.buffer = buffer

    def finalize(self, *, ctx: Any, components: Any, epoch: int) -> Mapping[str, Any]:
        _ = (ctx, components)
        return {
            "status": "completed",
            "epoch": epoch,
            "buffer_count": self.buffer.sample_count,
            "compressed_bytes": self.buffer.compressed_bytes,
            "note": "dense_cnn self-play finalizes samples immediately after each game",
        }

