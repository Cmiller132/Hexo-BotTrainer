"""READ-ONLY: test the 'soft-Z asymmetrically weakens losing labels' hypothesis.

Reads ACTUAL stored soft-Z targets from every per-game compact shard and joins to
the game winner (from the .hxr record by game_id). Recovers root_value = 2*value - z.
Splits mean target / mean root_value by eventual WINNER vs LOSER side-to-move and by
game phase, across epochs. No model, no GPU.
"""
import glob, re
import numpy as np
from hexo_runner.records import HexoRecordFile

RUN = "/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main3/selfplay"
PLAYERS = ("player0", "player1")
PHASES = [("open(0-10)", 0, 10), ("mid(11-30)", 11, 30), ("late(31+)", 31, 10**9)]


def winners_for_epoch(ep):
    """game_index -> winner ('player0'/'player1'/None) from the .hxr record."""
    hxr = f"{RUN}/epoch_{ep:06d}.hxr"
    if not glob.glob(hxr):
        return {}
    out = {}
    for r in HexoRecordFile.open(hxr).iter_records():
        m = re.search(r"selfplay-(\d+)$", r.game_id)
        if not m:
            continue
        w = r.winner
        out[int(m.group(1))] = (str(getattr(w, "value", w)) if w is not None else None)
    return out


def phase_of(t):
    for name, lo, hi in PHASES:
        if lo <= t <= hi:
            return name
    return "late(31+)"


print(f"{'ep':>2} {'phase':>11} | {'WIN: n   tgt    rv':>22} | {'LOSE: n   tgt    rv':>22} | softening(|win|-|lose|)")
print("-" * 92)
epoch_summary = []
for ep in range(0, 7):
    win = winners_for_epoch(ep)
    if not win:
        continue
    # accumulators[phase][role] = [sum_tgt, sum_rv, n]
    acc = {ph: {"W": [0.0, 0.0, 0], "L": [0.0, 0.0, 0]} for ph, _, _ in PHASES}
    for sh in sorted(glob.glob(f"{RUN}/epoch_{ep:06d}_game_*.npz")):
        gi = int(re.search(r"game_(\d+)\.npz$", sh).group(1))
        w = win.get(gi)
        if w not in ("player0", "player1"):
            continue  # skip truncated/draw
        d = np.load(sh)
        val = d["value"].astype(np.float64)
        cp = d["current_player"].astype(int)
        ti = d["turn_index"].astype(int)
        for i in range(len(val)):
            player = PLAYERS[cp[i]]
            z = 1.0 if player == w else -1.0
            rv = 2.0 * val[i] - z  # recover root_value from soft-Z blend (lambda=0.5)
            role = "W" if z > 0 else "L"
            a = acc[phase_of(ti[i])][role]
            a[0] += val[i]; a[1] += rv; a[2] += 1
    tot = {"W": [0.0, 0.0, 0], "L": [0.0, 0.0, 0]}
    for ph, _, _ in PHASES:
        for role in ("W", "L"):
            a = acc[ph][role]
            for k in range(3):
                tot[role][k] += a[k]
        W, L = acc[ph]["W"], acc[ph]["L"]
        if W[2] and L[2]:
            wt, wr = W[0]/W[2], W[1]/W[2]
            lt, lr = L[0]/L[2], L[1]/L[2]
            asym = abs(wt) - abs(lt)
            print(f"{ep:>2} {ph:>11} | {W[2]:>5} {wt:+.3f} {wr:+.3f} | {L[2]:>5} {lt:+.3f} {lr:+.3f} | {asym:+.3f}")
    # epoch totals
    wt, wr = tot["W"][0]/max(1,tot["W"][2]), tot["W"][1]/max(1,tot["W"][2])
    lt, lr = tot["L"][0]/max(1,tot["L"][2]), tot["L"][1]/max(1,tot["L"][2])
    epoch_summary.append((ep, tot["W"][2], wt, wr, tot["L"][2], lt, lr))
    print(f"{ep:>2} {'ALL':>11} | {tot['W'][2]:>5} {wt:+.3f} {wr:+.3f} | {tot['L'][2]:>5} {lt:+.3f} {lr:+.3f} | {abs(wt)-abs(lt):+.3f}")
    print("-" * 92)

print("\n=== FEEDBACK-LOOP CHECK: does loser-label softening grow across epochs? ===")
print(f"{'ep':>2} | mean tgt WIN | mean tgt LOSE | mean rv WIN | mean rv LOSE | |win|-|lose| (asym) | rv_win+rv_lose")
for ep, wn, wt, wr, ln, lt, lr in epoch_summary:
    print(f"{ep:>2} | {wt:+.3f}       | {lt:+.3f}        | {wr:+.3f}      | {lr:+.3f}       | {abs(wt)-abs(lt):+.3f}              | {wr+lr:+.3f}")
print("\nInterpretation:")
print("  symmetric softening => |tgt WIN| ~= |tgt LOSE| (asym ~0) and rv_win+rv_lose ~0.")
print("  hypothesis (asym weakening) => |tgt LOSE| < |tgt WIN| (asym>0) AND rv_LOSE > 0 / rv_win+rv_lose>0, growing over epochs.")
