"""Runner record and result contracts.

The most-imported path of this package: all four model lineages'
selfplay/evaluation modules import AbortRecord/HexoRecordFile/HexoRecordPlayer
from here (e.g. packages/dense_cnn_restnet/python/dense_cnn_restnet/
selfplay.py), and hexo_frontend reads .hxr files through the same types.
"""

from .record import (
    AbortRecord,
    HEXO_RECORD_MAGIC,
    HEXO_RECORD_SCHEMA_VERSION,
    HexoRecord,
    HexoRecordFile,
    HexoRecordGameWriter,
    HexoRecordPlayer,
)
from .results import BatchResult, GameResult, GameStatus

__all__ = [
    "AbortRecord",
    "BatchResult",
    "GameResult",
    "GameStatus",
    "HEXO_RECORD_MAGIC",
    "HEXO_RECORD_SCHEMA_VERSION",
    "HexoRecord",
    "HexoRecordFile",
    "HexoRecordGameWriter",
    "HexoRecordPlayer",
]
