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
from .sampling import SampleBatch, SampleRequest, sample_training_samples
from .schema import SAMPLE_SCHEMA_VERSION, SampleSchema
from .targets import LegalPolicyValueTarget, build_legal_policy_value_target

__all__ = [
    "LegalPolicyValueTarget",
    "ModelSamplePayload",
    "PolicyOutputRecord",
    "SAMPLE_SCHEMA_VERSION",
    "SampleBatch",
    "SampleRequest",
    "SampleSchema",
    "TrainingSampleRecord",
    "build_legal_policy_value_target",
    "sample_training_samples",
]
