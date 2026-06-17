import json, glob, os
d = "/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main1/diagnostics"
print(f"{'ep':>3} {'pos':>7} {'games':>5} {'dec/game':>8} {'pos/s':>6} {'exp_len':>7} {'self_min':>8} {'flush_mean':>10}")
for f in sorted(glob.glob(f"{d}/dense_cnn.selfplay.epoch_*.json")):
    j = json.load(open(f))
    ep = j.get("epoch")
    pos = j.get("searched_positions", 0)
    games = j.get("games_finished", 0) or 1
    pps = j.get("positions_per_second", 0)
    elapsed = j.get("elapsed_seconds", 0)
    tc = j.get("temperature_control", {})
    exp = tc.get("expected_game_length", 0)
    sd = j.get("scheduler_diagnostics", {})
    fm = sd.get("mean_flush_states", 0)
    print(f"{ep:>3} {pos:>7} {games:>5} {pos/games:>8.1f} {pps:>6.2f} {exp:>7.1f} {elapsed/60:>8.1f} {fm:>10.1f}")
print("=== evaluations ===")
for f in sorted(glob.glob(f"{d}/dense_cnn.evaluation.epoch_*.json")):
    j = json.load(open(f))
    print(j.get("epoch"), "wins", j.get("wins"), "/", j.get("games"), "mean_turns", j.get("mean_turns"))
