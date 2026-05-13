"""Shared training sample helpers.

The package is intentionally small:

- `buffer.py` owns storage, indexing, window selection, and sample requests;
- `records.py` owns schema identifiers and neutral training record shapes;
- `targets.py` owns common legal-action policy/value target helpers.

Core game records stay in `hexo_runner.records`; models write sample records
during self-play and decide how to turn them into tensors.
"""

from .buffer import (
    SampleBatch,
    SampleIndex,
    SampleRequest,
    SampleStore,
    SampleWindow,
    SampleWriteResult,
    append_samples,
    build_sample_window,
    open_sample_store,
    refresh_sample_index,
    sample_training_samples,
)
from .records import (
    ModelSamplePayload,
    PolicyOutputRecord,
    SAMPLE_SCHEMA_VERSION,
    SampleSchema,
    TrainingSampleRecord,
)
from .targets import (
    LegalPolicyTargetHelper,
    LegalPolicyValueTarget,
    ScalarValueTargetHelper,
    build_legal_policy_value_target,
)

__all__ = [
    "LegalPolicyValueTarget",
    "LegalPolicyTargetHelper",
    "ModelSamplePayload",
    "PolicyOutputRecord",
    "SAMPLE_SCHEMA_VERSION",
    "ScalarValueTargetHelper",
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
