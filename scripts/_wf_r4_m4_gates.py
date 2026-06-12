"""main_4 launch-gate + per-epoch extraction (G6/G7/G2 counters + eval)."""
import json
from pathlib import Path

RUN = Path("/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main_4")

for ep in range(6, 20):
    f = RUN / "diagnostics" / f"epoch_{ep:06d}.json"
    if not f.exists():
        break
    d = json.load(open(f))["metadata"]["result"]
    sp = d["selfplay"]
    tr = d.get("training") or {}
    comp = tr.get("loss_components") or {}
    tc = sp.get("temperature_control", {})
    line = (f"ep{ep}: dec/g={sp['total_decisions']/max(sp.get('games_finished',1),1):.1f} "
            f"rows={sp.get('raw_samples')} EMA={tc.get('expected_game_length', float('nan')):.2f} "
            f"trunc={sp.get('truncated_games')} "
            f"fwo={sp.get('frozen_win_overrides', 'MISSING')}/{sp.get('frozen_win_override_failures', 'MISSING')} "
            f"train.samples={tr.get('samples')} polCE={comp.get('policy', float('nan')):.3f} "
            f"valCE={comp.get('value', float('nan')):.3f} "
            f"pos/s={sp.get('search_positions_per_second', 0):.1f}")
    print(line)
    ev = d.get("evaluation")
    if ev and ev.get("status") == "completed":
        print(f"   EVAL: wins={ev.get('wins')}/{ev.get('games')} mean_turns={ev.get('mean_turns')}")
print("---G6: ep6 EMA must be 115.0 +- 0.5 (150=prior unapplied, 160.86=contaminated copy)")
print("---G7: ep6 train.samples must be >= 18000 (expected ~22400; unseeded ~6400)")
