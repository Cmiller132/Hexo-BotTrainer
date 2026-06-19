"""hexfield — the variable-geometry hex-lattice lineage.

The model's domain is the engine-true support set
(stones ∪ full legal set ∪ 1-ring halo); every engine-legal cell carries a
policy logit. The sibling dense_cnn_restnet lineage is a porting reference
(its constructions are the test oracles) but is never imported by this
package — at runtime or in tests.
"""

from . import constants, geometry, support, features

__all__ = ["constants", "geometry", "support", "features"]
