"""hexfield — the variable-geometry hex-lattice lineage.

Spec: docs/specs/hexfield_model_spec.md (all §12 decision points resolved
2026-06-12). The model's domain is the engine-true support set
(stones ∪ full legal set ∪ 1-ring halo); every engine-legal cell carries a
policy logit. Greenfield code: dense_cnn / dense_cnn_restnet are imported in
tests only, as executable oracles, never at runtime.
"""

from . import constants, geometry, support, features

__all__ = ["constants", "geometry", "support", "features"]
