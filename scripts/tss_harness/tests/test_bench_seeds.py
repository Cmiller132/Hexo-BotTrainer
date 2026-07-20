from __future__ import annotations

import json
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.tss_harness.bench_seeds import (
    BANDS,
    PER_BAND,
    SeedSetError,
    build,
    load_and_verify,
    select_seed_rows,
)


def source_rows(per_band: int = 72):
    rows = []
    for band in BANDS:
        placements = band * 10 + 3
        for candidate in range(per_band):
            # Legality is the production driver's responsibility; this fixture
            # only exercises deterministic JSON stratification/pinning.
            moves = [[candidate, index - placements // 2] for index in range(placements)]
            rows.append(
                {
                    "id": f"b{band}-{candidate}",
                    "source": "selfplay",
                    "placements": placements,
                    "moves": moves,
                }
            )
    return rows


class SeedSelectionTests(unittest.TestCase):
    def test_stratification_is_hash_ranked_and_input_order_independent(self):
        rows = source_rows()
        first = select_seed_rows(rows)
        shuffled = list(rows)
        random.Random(999).shuffle(shuffled)
        second = select_seed_rows(shuffled)
        self.assertEqual(first, second)
        self.assertEqual(
            Counter(row["band"] for row in first),
            Counter({band: PER_BAND for band in BANDS}),
        )
        self.assertTrue(all(len(row["stable_hash"]) == 64 for row in first))

    def test_build_verify_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source.jsonl"
            frozen = tmp_path / "bench.jsonl"
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in source_rows()),
                encoding="utf-8",
            )
            manifest = build(source, frozen)
            checked, positions = load_and_verify(frozen)
            self.assertEqual(checked, manifest)
            self.assertEqual(len(positions), 4 * PER_BAND)

            lines = frozen.read_text(encoding="utf-8").splitlines()
            tampered = json.loads(lines[-1])
            tampered["moves"][0][0] += 1
            lines[-1] = json.dumps(tampered)
            frozen.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(SeedSetError):
                load_and_verify(frozen)


if __name__ == "__main__":
    unittest.main()
