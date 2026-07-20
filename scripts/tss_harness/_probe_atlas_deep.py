"""The 5 atlas rows certified WIN/LOSS that Lane C could not decide at the
20k labeling cap: probe at 100k with dual pass. Known truth = the atlas
certificate; any verified verdict must AGREE (disagreement = soundness
alarm somewhere).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "_v1_soak"))

import arch_env  # noqa: F401
import corpus_lib
from hexfield_eq import _rust

V1_RAWS = Path("/mnt/e/Hexo-BotTrainer-hexgt/.claude/worktrees/v1-soak/raws")
CAP = 100_000

deep = []
for line in open(V1_RAWS / "lanec_labels.jsonl"):
    r = json.loads(line)
    if r["source"] == "atlas" and r["status"] == "unknown":
        deep.append(r)
print(f"atlas-deep positions: {len(deep)}", flush=True)

# moves live in the puzzle set (source atlas_deep rows)
moves_by_id = {}
for line in open(ROOT / "scripts" / "tss_harness" / "sets" / "puzzle_v3.jsonl"):
    row = json.loads(line)
    moves_by_id[row["pos_id"]] = row["moves"]

for r in deep:
    moves = moves_by_id.get(r["pos_id"])
    if moves is None:
        print(f"{r['pos_id']}: not in puzzle set (holdout-only or filtered), skip")
        continue
    state = corpus_lib.build_state(list(moves))
    out = _rust.hexfield_eq_deep_solve_batch(
        [state], CAP, "both", 0, False, False, True, True)[0]
    truth = r["prior"]["atlas_status"].lower()
    verdict = out["status"]
    agree = ("AGREE" if verdict == truth else
             ("undecided" if verdict == "unknown" else "!!! DISAGREE !!!"))
    print(f"{r['pos_id']}: truth={truth} solved={verdict} "
          f"nodes={out['deep_nodes']} vf={out['deep_verify_failed']} -> {agree}",
          flush=True)
