"""Shared training sample helpers.

Sample utilities describe training-facing records, sample buffers, and sampling
contracts. Core game records stay in `hexo_runner.records`; models write
sample records during self-play.
"""

from .records import (
    ModelSamplePayload,
    PolicyOutputRecord,
    TrainingSampleRecord,
)
from .index import SampleIndex, refresh_sample_index
from .sampling import SampleBatch, SampleRequest, sample_training_samples
from .schema import SAMPLE_SCHEMA_VERSION, SampleSchema
from .store import SampleStore, open_sample_store
from .targets import LegalPolicyValueTarget, build_legal_policy_value_target
from .window import SampleWindow, build_sample_window
from .writer import SampleWriteResult, append_samples

__all__ = [
    "LegalPolicyValueTarget",
    "ModelSamplePayload",
    "PolicyOutputRecord",
    "SAMPLE_SCHEMA_VERSION",
    "SampleBatch",
    "SampleIndex",
    "SampleRequest",
    "SampleSchema",
    "SampleStore",
    "SampleWindow",
    "SampleWriteResult",
    "TrainingSampleRecord",
    "append_samples",
    "build_sample_window",
    "build_legal_policy_value_target",
    "open_sample_store",
    "refresh_sample_index",
    "sample_training_samples",
]
