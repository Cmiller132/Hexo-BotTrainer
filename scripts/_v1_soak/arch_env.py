"""Import-time arch env for the main_3 ep90 checkpoint (best-eval, owner-designated).

hexfield_eq binds several architecture constants (NUM_FEATURES / stem lift, ray
geometry, group order) at MODULE IMPORT from HEXFIELD_EQ_* env vars. The ep90 net
is the A5 ray7lut2 arm (feature_version=2 -> 46 planes, c=192, heads=3,
c_orbit=16, group_order=12, raytap=both + additive LUT, trunk CCACCACA, reg lane
on). These values come straight from the checkpoint's arch_meta. This module sets
them (without clobbering an explicit user value) and MUST be imported BEFORE the
first ``import hexfield_eq`` in any V1-soak script.
"""

from __future__ import annotations

import os

_ARCH = {
    "HEXFIELD_EQ_FEATURE_VERSION": "2",
    "HEXFIELD_EQ_CHANNELS": "192",
    "HEXFIELD_EQ_ATTENTION_HEADS": "3",
    "HEXFIELD_EQ_C_ORBIT": "16",
    "HEXFIELD_EQ_GROUP_ORDER": "12",
    "HEXFIELD_EQ_TRUNK": "CCACCACA",
    "HEXFIELD_EQ_RAYTAP": "both",
    "HEXFIELD_EQ_RAYTAP_LUT": "additive",
    "HEXFIELD_EQ_REG_LANE": "1",
    "HEXFIELD_EQ_RAY_BLOCKERS": "1",
}

for _k, _v in _ARCH.items():
    os.environ.setdefault(_k, _v)
