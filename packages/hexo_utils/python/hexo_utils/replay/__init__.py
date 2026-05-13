"""Shared replay helpers.

Replay utilities describe training-facing records and sampling contracts. Core
game records stay in `hexo_runner.records`; replay records attach policy data
and model-owned extensions for training.
"""

from .records import (
    ModelExtensionRecord,
    PolicyLogitRecord,
    ReplayDecisionRecord,
)
from .sampling import ReplayBatch, ReplaySampleRequest, sample_replay_records
from .schema import REPLAY_SCHEMA_VERSION, ReplaySchema
from .targets import LegalPolicyValueTarget, build_legal_policy_value_target

__all__ = [
    "LegalPolicyValueTarget",
    "ModelExtensionRecord",
    "PolicyLogitRecord",
    "REPLAY_SCHEMA_VERSION",
    "ReplayBatch",
    "ReplayDecisionRecord",
    "ReplaySampleRequest",
    "ReplaySchema",
    "build_legal_policy_value_target",
    "sample_replay_records",
]
