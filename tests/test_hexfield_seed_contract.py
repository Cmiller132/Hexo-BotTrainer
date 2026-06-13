"""M6 gate: the mix_seed hash + stream ids are a written contract — golden
vectors copied from dense mcts.rs's own unit tests."""

import pytest

try:
    from hexfield import _rust as hexfield_rust
except ImportError:  # pragma: no cover
    hexfield_rust = None

needs_native = pytest.mark.skipif(hexfield_rust is None, reason="native module not built")


@needs_native
def test_mix_seed_golden_vectors() -> None:
    assert hexfield_rust.mix_seed(0, 0, 0, 0) == 0x7DE5_3DE7_72EA_694C
    assert hexfield_rust.mix_seed(1, 2, 3, 4) == 0xA6A9_B091_CFF9_D67A
    a = hexfield_rust.mix_seed(123_456_789, 987_654_321, 42, 0)
    b = hexfield_rust.mix_seed(123_456_789, 987_654_321, 42, 1)
    assert a != b  # streams are independent
