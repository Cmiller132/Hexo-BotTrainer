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
    SampleChunkInfo,
    SampleIndex,
    SampleIndexEntry,
    SampleManifest,
    SampleRequest,
    SampleStore,
    SampleWindow,
    SampleWriteResult,
    append_samples,
    build_sample_window,
    iter_sample_records,
    load_sample_manifest,
    open_sample_store,
    read_sample_records,
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
    "SampleChunkInfo",
    "SampleIndex",
    "SampleIndexEntry",
    "SampleManifest",
    "SampleRequest",
    "SampleSchema",
    "SampleStore",
    "SampleWindow",
    "SampleWriteResult",
    "TrainingSampleRecord",
    "append_samples",
    "build_sample_window",
    "build_legal_policy_value_target",
    "iter_sample_records",
    "load_sample_manifest",
    "open_sample_store",
    "read_sample_records",
    "refresh_sample_index",
    "sample_training_samples",
]
