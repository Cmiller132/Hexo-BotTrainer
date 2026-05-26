"""Python exposure for the Rust-backed `.hxr` record codec."""

from __future__ import annotations

from ._rust import (
    AbortRecord,
    HEXO_RECORD_MAGIC,
    HEXO_RECORD_SCHEMA_VERSION,
    HexoRecord,
    HexoRecordFile,
    HexoRecordGameWriter,
    HexoRecordPlayer,
)

__all__ = [
    "AbortRecord",
    "HEXO_RECORD_MAGIC",
    "HEXO_RECORD_SCHEMA_VERSION",
    "HexoRecord",
    "HexoRecordFile",
    "HexoRecordGameWriter",
    "HexoRecordPlayer",
]
