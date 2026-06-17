"""CPU-only self-check for the pure-Python helpers in hexfield/eval_arena.py.

Importing the module proper pulls in torch + hexfield._rust (CUDA), which is
forbidden while the live GPU run is training. So we extract ONLY the numpy/
torch-free helper functions from the source text by AST and exec them in an
isolated namespace, then assert their numeric behavior. This validates the
actual code text without importing the CUDA-touching package.
"""
import ast
import math
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / (
    "packages/hexfield/python/hexfield/eval_arena.py"
)

WANT = {
    "_percentile",
    "_length_stats",
    "_opening_dedup",
    "_pentanomial_block",
    "_wilson_ci",
}


def _load_pure_helpers():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    picked = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in WANT
    ]
    missing = WANT - {n.name for n in picked}
    assert not missing, f"helpers not found in source: {sorted(missing)}"
    module = ast.Module(body=picked, type_ignores=[])
    ns: dict = {"Any": object, "round": round, "sorted": sorted, "sum": sum,
                "min": min, "max": max, "len": len, "int": int, "float": float}
    exec(compile(module, str(SRC), "exec"), ns)
    return ns


def main() -> None:
    ns = _load_pure_helpers()
    wilson = ns["_wilson_ci"]
    length_stats = ns["_length_stats"]
    opening_dedup = ns["_opening_dedup"]
    pentanomial = ns["_pentanomial_block"]
    percentile = ns["_percentile"]

    # --- Wilson CI -------------------------------------------------------- #
    assert wilson(0, 0) == (0.0, 1.0)
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi and 0.39 < lo < 0.41 and 0.59 < hi < 0.61, (lo, hi)
    lo, hi = wilson(100, 100)
    assert hi == 1.0 and 0.95 < lo < 0.97, (lo, hi)  # all wins -> upper pinned
    # Symmetry: CI for w wins of n mirrors CI for (n-w) wins.
    lo1, hi1 = wilson(30, 100)
    lo2, hi2 = wilson(70, 100)
    assert abs(lo1 - (1 - hi2)) < 1e-9 and abs(hi1 - (1 - lo2)) < 1e-9
    print("wilson_ci OK")

    # --- percentile / length_stats --------------------------------------- #
    assert percentile([], 0.5) is None
    assert length_stats([]) is None
    st = length_stats([10, 20, 30, 40, 100])
    assert st["n"] == 5 and st["min"] == 10 and st["max"] == 100
    assert st["median"] == 30 and st["mean"] == 40.0, st
    print("length_stats OK")

    # --- opening_dedup ---------------------------------------------------- #
    dd = opening_dedup([(1, 2), (1, 2), (3, 4), (5, 6), (3, 4)])
    assert dd["n_games"] == 5 and dd["distinct_openings"] == 3, dd
    # two duplicate groups, keyed by first member index (as strings)
    assert set(dd["duplicate_groups"].keys()) == {"0", "2"}, dd
    assert dd["duplicate_groups"]["0"] == [0, 1]
    assert dd["duplicate_groups"]["2"] == [2, 4]
    print("opening_dedup OK")

    # --- pentanomial block ------------------------------------------------ #
    # 4 full pairs: scores 2,1,0,1 ; plus one singleton (1 game, A won).
    pairs = [
        {"pair_index": 0, "n_games": 2, "n_decided": 2, "a_wins": 2, "b_wins": 0,
         "pentanomial_a_score": 2},
        {"pair_index": 1, "n_games": 2, "n_decided": 2, "a_wins": 1, "b_wins": 1,
         "pentanomial_a_score": 1},
        {"pair_index": 2, "n_games": 2, "n_decided": 2, "a_wins": 0, "b_wins": 2,
         "pentanomial_a_score": 0},
        {"pair_index": 3, "n_games": 2, "n_decided": 2, "a_wins": 1, "b_wins": 1,
         "pentanomial_a_score": 1},
        {"pair_index": 4, "n_games": 1, "n_decided": 1, "a_wins": 1, "b_wins": 0,
         "pentanomial_a_score": 1},
    ]
    blk = pentanomial(pairs)
    assert blk["n_pairs"] == 5 and blk["n_full_pairs"] == 4, blk
    assert blk["histogram_a_wins"] == {"0": 1, "1": 2, "2": 1}, blk
    # pair stats over decided games: [1.0, 0.5, 0.0, 0.5, 1.0] -> mean 0.6
    assert blk["n_informative_pairs"] == 5, blk
    assert abs(blk["pair_winrate_mean"] - 0.6) < 1e-9, blk
    stats = [1.0, 0.5, 0.0, 0.5, 1.0]
    mean = sum(stats) / len(stats)
    var = sum((x - mean) ** 2 for x in stats) / (len(stats) - 1)
    se = math.sqrt(var / len(stats))
    assert abs(blk["pair_winrate_se"] - round(se, 4)) < 1e-9, (blk, se)
    # A perfectly even sweep (all 0.5) -> zero SE.
    even = [{"pair_index": i, "n_games": 2, "n_decided": 2, "a_wins": 1,
             "b_wins": 1, "pentanomial_a_score": 1} for i in range(3)]
    blk2 = pentanomial(even)
    assert blk2["pair_winrate_se"] == 0.0 and blk2["pair_winrate_mean"] == 0.5, blk2
    # Both-truncated pairs (n_decided==0) are dropped from mean/SE.
    blk3 = pentanomial([
        {"pair_index": 0, "n_games": 2, "n_decided": 0, "a_wins": 0, "b_wins": 0,
         "pentanomial_a_score": 0},
    ])
    assert blk3["n_informative_pairs"] == 0 and blk3["pair_winrate_mean"] is None, blk3
    print("pentanomial_block OK")

    print("ALL SELF-CHECKS PASSED")


if __name__ == "__main__":
    main()
