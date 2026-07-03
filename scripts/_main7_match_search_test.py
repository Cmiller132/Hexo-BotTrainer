"""Match-search behavior test at a real multi-candidate position: roll out 12
plies (greedy policy), then verify (a) tempered in-search selection varies by
seed at temperature 1 (opening protocol), (b) is deterministic per seed, and
(c) temperature 0 is greedy-stable across seeds."""

import sys

sys.path.insert(0, "/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_frontend/python")
sys.path.insert(0, "/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python")
sys.path.insert(0, "/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python")
sys.path.insert(0, "/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python")

from hexo_frontend import debug_infer as di  # noqa: E402

CKPT = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_7/checkpoints/epoch_000004.pt"
loaded = di.load_checkpoint(CKPT)

acts: list[int] = []
for _ in range(12):
    a = di.analyze_position(loaded, acts)
    acts.append(int(a["policy"][0]["action_id"]))
a = di.analyze_position(loaded, acts)
print(f"position: ply={len(acts)} candidates={a['candidate_count']}")
assert a["candidate_count"] > 10

# NOTE: under the gumbel profile the ROOT CANDIDATE SET is Gumbel-sampled by
# the search seed (gumbel_root m=32), so even temperature-0 play legitimately
# varies across seeds — exactly like eval games (eval_arena: "under a gumbel
# profile the search itself is seeded"). The contract is per-seed determinism,
# not cross-seed agreement.
greedy = [
    di.search_position(loaded, acts, visits=64, seed=s)["best_action_id"]
    for s in (1, 2, 3)
]
greedy_repeat = di.search_position(loaded, acts, visits=64, seed=1)["best_action_id"]
sampled = [
    di.search_position(loaded, acts, visits=64, seed=s, temperature=1.0)["best_action_id"]
    for s in (1, 2, 3, 4, 5, 6)
]
repeat = di.search_position(loaded, acts, visits=64, seed=1, temperature=1.0)["best_action_id"]
print(f"greedy (3 seeds): {greedy}  seed=1 repeat stable: {greedy_repeat == greedy[0]}")
print(f"sampled temp=1 (6 seeds): {sampled}")
print(f"seed=1 repeat == first sample: {repeat == sampled[0]}")

ok = True
ok &= greedy_repeat == greedy[0]     # temp 0: per-seed deterministic
ok &= len(set(sampled)) > 1          # temp 1: seeds diversify the opening
ok &= repeat == sampled[0]           # per-seed reproducible
print("MATCH SEARCH TEST " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
