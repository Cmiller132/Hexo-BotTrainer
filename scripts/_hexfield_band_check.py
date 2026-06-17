"""M9 exploration band check: compare hexfield soak self-play exploration
numbers against the dense reference run. Extracts the comparable fields from
both lineages' selfplay diagnostics and flags out-of-band metrics.

Usage: python scripts/_hexfield_band_check.py [hexfield_run_dir] [dense_run_dir]
"""

import json
import sys
from pathlib import Path

HEX = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/e/Hexo-BotTrainer/runs/hexfield_soak")
DENSE = Path(sys.argv[2] if len(sys.argv) > 2
             else "/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_sanity_1")


def load(run, lineage):
    diag = run / "diagnostics"
    files = sorted(diag.glob(f"{lineage}.selfplay.epoch_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text()), files[-1].name


def dense_ref():
    got = load(DENSE, "dense_cnn")
    if not got:
        return None
    d, name = got
    return {
        "source": name,
        "mean_game_length": d.get("temperature_control", {}).get("expected_game_length"),
        "search_pos_per_s": d.get("search_positions_per_second"),
        "truncated_games": d.get("truncated_games"),
        "completed_games": d.get("completed_games") or d.get("games_finished"),
        "policy_init_avg_plies": d.get("policy_init", {}).get("avg_plies"),
    }


def hex_latest():
    got = load(HEX, "hexfield")
    if not got:
        return None
    h, name = got
    return {
        "source": name,
        "mean_game_length": h.get("mean_game_length"),
        "p90_game_length": h.get("p90_game_length"),
        "root_policy_entropy_mean": h.get("root_policy_entropy_mean"),
        "root_value_mean": h.get("root_value_mean"),
        "truncated_games": h.get("truncated_games"),
        "games_finished": h.get("games_finished"),
        "full_fast_init": (h.get("full_decisions"), h.get("total_decisions")),
        "rows_written": h.get("rows_written"),
        "scheduler": h.get("scheduler", {}),
    }


print("=== DENSE reference (sanity_1) ===")
ref = dense_ref()
print(json.dumps(ref, indent=2) if ref else "  (none)")
print("\n=== HEXFIELD soak (latest epoch) ===")
hx = hex_latest()
print(json.dumps(hx, indent=2, default=str) if hx else "  (no selfplay epoch yet)")

if ref and hx:
    print("\n=== BAND CHECK ===")
    gl_h = hx["mean_game_length"]
    gl_d = ref["mean_game_length"]
    if gl_h and gl_d:
        ratio = gl_h / gl_d
        print(f"game length: hex {gl_h:.1f} vs dense {gl_d:.1f}  (ratio {ratio:.2f}; "
              f"{'IN BAND' if 0.5 <= ratio <= 2.0 else 'OUT OF BAND'})")
    ent = hx["root_policy_entropy_mean"]
    if ent is not None:
        print(f"root policy entropy: {ent:.3f}  "
              f"({'healthy (>0.5, not collapsed)' if ent > 0.5 else 'LOW — possible collapse'})")
    tg = hx["truncated_games"]
    fg = hx["games_finished"]
    if fg:
        print(f"truncation rate: {tg}/{fg}  "
              f"({'OK' if tg / max(fg,1) < 0.5 else 'HIGH — games not finishing'})")
    print(f"rows written: {hx['rows_written']}  ({'OK' if hx['rows_written'] else 'ZERO — no training data!'})")
