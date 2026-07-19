#!/usr/bin/env python3
"""Cargo-free exact D6 census of Hexo's first two reply placements.

This script enumerates positions, not move-order histories: Player 1's two
stones are unordered once its turn completes. It performs no solving and
makes no verdict claims.
"""

from collections import Counter


RADIUS = 8


def distance(left: tuple[int, int], right: tuple[int, int] = (0, 0)) -> int:
    q = left[0] - right[0]
    r = left[1] - right[1]
    return max(abs(q), abs(r), abs(q + r))


def transform(coord: tuple[int, int], symmetry: int) -> tuple[int, int]:
    if not 0 <= symmetry < 12:
        raise ValueError("D6 symmetry must be in [0, 12)")
    q, r = coord
    if symmetry >= 6:
        r = -q - r
    for _ in range(symmetry % 6):
        q, r = -r, q + r
    return q, r


def canonical(coords: tuple[tuple[int, int], ...], unordered: bool) -> tuple[tuple[int, int], ...]:
    images = []
    for symmetry in range(12):
        image = tuple(transform(coord, symmetry) for coord in coords)
        images.append(tuple(sorted(image)) if unordered else image)
    return min(images)


def main() -> None:
    disk = [
        (q, r)
        for q in range(-RADIUS, RADIUS + 1)
        for r in range(-RADIUS, RADIUS + 1)
        if (q, r) != (0, 0) and distance((q, r)) <= RADIUS
    ]
    first_orbits: dict[tuple[tuple[int, int], ...], int] = {}
    for first in disk:
        key = canonical((first,), unordered=False)
        first_orbits.setdefault(key, len({transform(first, symmetry) for symmetry in range(12)}))

    raw_pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for first in disk:
        for q in range(-16, 17):
            for r in range(-16, 17):
                second = (q, r)
                if second in ((0, 0), first):
                    continue
                if distance(second) <= RADIUS or distance(second, first) <= RADIUS:
                    raw_pairs.add(tuple(sorted((first, second))))
    pair_orbits = Counter(canonical(pair, unordered=True) for pair in raw_pairs)

    first_sizes = Counter(first_orbits.values())
    pair_sizes = Counter(pair_orbits.values())
    assert len(disk) == 216
    assert len(first_orbits) == 24
    assert sum(size * count for size, count in first_sizes.items()) == 216
    assert len(raw_pairs) == 42_768
    assert len(pair_orbits) == 3_684
    assert sum(size * count for size, count in pair_sizes.items()) == 42_768

    print("OPENING_CENSUS schema=1 radius=8")
    print(f"PLY2 raw={len(disk)} d6={len(first_orbits)} orbit_hist={dict(sorted(first_sizes.items()))}")
    print(
        f"PLY3 raw_unordered={len(raw_pairs)} d6={len(pair_orbits)} "
        f"orbit_hist={dict(sorted(pair_sizes.items()))}"
    )
    for index, (representative, orbit) in enumerate(sorted(first_orbits.items()), 1):
        q, r = representative[0]
        print(f"PLY2_REP index={index:02d} q={q} r={r} orbit={orbit}")


if __name__ == "__main__":
    main()
