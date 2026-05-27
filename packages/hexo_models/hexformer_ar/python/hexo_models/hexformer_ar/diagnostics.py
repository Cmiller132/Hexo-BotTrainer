"""Diagnostics helpers for Hexformer AR."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .samples import CompressedHexformerSample


def sparse_sample_preview(records: Sequence[CompressedHexformerSample], *, limit: int = 8) -> list[Mapping[str, Any]]:
    preview = []
    for index, record in enumerate(tuple(records)[:limit]):
        sample = record.decode()
        payload = sample.input_payload
        preview.append(
            {
                "index": index,
                "game_id": sample.game_id,
                "turn_index": sample.turn_index,
                "candidate_count": len(payload.get("candidate_action_ids", ())),
                "metadata": dict(sample.metadata),
            }
        )
    return preview
