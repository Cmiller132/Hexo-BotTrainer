"""READ-ONLY CPU value-head diagnostics for hexgt_rl_main3.

Distinguishes BUG vs BIAS for the value-head optimism:
  (A) In-distribution calibration: replay real self-play games, evaluate at the
      TRUE side-to-move each ply. mean(v_pred), corr(v_pred, z), sign-accuracy,
      calibration buckets, and opening (empty-board) value.
  (B) Optimism probe (owner-swap): vA + vB on the IDENTICAL board -> reproduces
      the calib metric, AND records whether the swap creates an off-distribution
      stone-count parity (probe-artifact check).
No GPU, no training mutation. Run: CUDA_VISIBLE_DEVICES= python _value_diag_readonly.py
"""
import glob, statistics as st
import torch

import hexo_engine.api as eng
from hexo_engine.types import AxialCoord, PlacementAction, unpack_coord_id
from hexo_runner.records import HexoRecordFile
from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.features import build_graph_tensors
from hexo_models.hexgt.collate import collate_graphs
from hexo_models.hexgt import rust_bridge
from hexo_models.hexgt.losses import decode_binned_value

RADIUS = 3
RUN = "/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main3"
CKPTS = [
    ("pretrain", f"{RUN}/pretrain/hexgt_model3_pretrain.pt"),
    ("epoch0", f"{RUN}/checkpoints/hexgt_rl_epoch000000.pt"),
    ("epoch3", f"{RUN}/checkpoints/hexgt_rl_epoch000003.pt"),
    ("epoch6", f"{RUN}/checkpoints/hexgt_rl_epoch000006.pt"),
]
torch.set_grad_enabled(False)
m = None


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["arch"]
    net = HexgtNetwork(
        token_dim=a["token_dim"], gnn_layers=a["gnn_layers"], ctx_layers=a["ctx_layers"],
        ffn_dim=a["ffn_dim"], attention_heads=a["attention_heads"],
        short_term_value_horizons=tuple(a.get("short_term_value_horizons", ())),
        value_pma_seeds=int(a.get("value_pma_seeds", 2)),
        value_head_use_side=bool(a.get("value_head_use_side", True)),
    ).eval()
    sd = {k: v for k, v in ck["model"].items() if not k.startswith("short_term_value_heads")}
    net.load_state_dict(sd, strict=False)
    return net


def val_from_facts(facts):
    g = build_graph_tensors(facts)
    b = collate_graphs([g])
    return float(decode_binned_value(m.forward_policy_value(b)["value"].float())[0])


def swap_owner(facts):
    own = list(facts["nodes"]["node_owner"])
    sw = [1 if o == 0 else (0 if o == 1 else o) for o in own]
    return {**facts, "nodes": {**facts["nodes"], "node_owner": sw}}


def side_counts(facts):
    """(own_stones, opp_stones) the SIDE node will encode = node_owner 0/1 counts."""
    own = list(facts["nodes"]["node_owner"])
    return own.count(0), own.count(1)


def games(hxr):
    rf = HexoRecordFile.open(hxr)
    out = []
    for r in rf.iter_records():
        out.append([unpack_coord_id(int(c)) for c in r.action_ids])
    return out


def analyze(hxr_files, maxgames=80):
    # In-distribution calibration containers
    vpred, zout = [], []          # v_pred at true side-to-move vs final outcome z
    open_vals = []                # empty/near-opening value (ply 0)
    # Probe containers
    sums, both_pos = [], 0
    parity_real, parity_swap = [], []  # own-opp at evaluated perspective
    ng = 0
    for f in hxr_files:
        for coords in games(f):
            if ng >= maxgames:
                break
            s = eng.new_game()
            # determine terminal winner by replaying ALL actions on a clone
            sc = eng.new_game()
            for c in coords:
                eng.apply_action(sc, PlacementAction(AxialCoord(q=int(c.q), r=int(c.r))))
            term = eng.terminal(sc)
            winner = None
            if term is not None and getattr(term, "winner", None) is not None:
                winner = str(getattr(term.winner, "value", term.winner))
            # walk plies
            for i, c in enumerate(coords):
                tm = eng.terminal(s)
                if tm is None:
                    facts = rust_bridge.graph_facts(s, RADIUS)
                    _cp = eng.current_player(s)
                    cp = str(getattr(_cp, "value", _cp))
                    if i == 0:
                        open_vals.append(val_from_facts(facts))
                    # in-distribution: only score where we have a decisive winner
                    if winner in ("player0", "player1") and 6 <= i <= 60:
                        v = val_from_facts(facts)
                        z = 1.0 if winner == cp else -1.0
                        vpred.append(v); zout.append(z)
                    # probe (owner-swap) on FirstStone-resolved midgame, plies 10-50
                    if 10 <= i <= 50 and facts.get("meta", {}).get("first_stone") is None:
                        ow, op = side_counts(facts)
                        vA = val_from_facts(facts)
                        sf = swap_owner(facts)
                        vB = val_from_facts(sf)
                        sums.append(vA + vB)
                        if vA > 0.05 and vB > 0.05:
                            both_pos += 1
                        parity_real.append(ow - op)
                        parity_swap.append(op - ow)
                eng.apply_action(s, PlacementAction(AxialCoord(q=int(c.q), r=int(c.r))))
            ng += 1
        if ng >= maxgames:
            break
    return dict(vpred=vpred, zout=zout, open_vals=open_vals, sums=sums,
                both_pos=both_pos, parity_real=parity_real, parity_swap=parity_swap, ng=ng)


def corr(a, b):
    if len(a) < 3:
        return float("nan")
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da > 0 and db > 0 else float("nan")


hxr = sorted(glob.glob(f"{RUN}/selfplay/epoch_000006.hxr")) + sorted(glob.glob(f"{RUN}/selfplay/epoch_000005.hxr"))
print(f"eval games file(s): {hxr}\n")
for label, path in CKPTS:
    m = load_model(path)
    r = analyze(hxr, maxgames=80)
    vp, zo = r["vpred"], r["zout"]
    n = len(vp)
    mean_vp = st.mean(vp) if vp else float("nan")
    mean_z = st.mean(zo) if zo else float("nan")
    c = corr(vp, zo)
    signacc = (sum(1 for v, z in zip(vp, zo) if (v > 0) == (z > 0)) / n) if n else float("nan")
    # calibration buckets
    buckets = {}
    for v, z in zip(vp, zo):
        k = round(max(-0.9, min(0.9, v)) * 2) / 2  # nearest 0.5
        buckets.setdefault(k, []).append(z)
    cal = " ".join(f"v~{k:+.1f}:z={st.mean(b):+.2f}(n{len(b)})" for k, b in sorted(buckets.items()))
    v_win = [v for v, z in zip(vp, zo) if z > 0]
    v_lose = [v for v, z in zip(vp, zo) if z < 0]
    rev = f"mean v|WON={st.mean(v_win):+.3f}(n{len(v_win)})  mean v|LOST={st.mean(v_lose):+.3f}(n{len(v_lose)})  frac(v<0 | LOST)={sum(1 for v in v_lose if v<0)/max(1,len(v_lose)):.0%}"
    sums = r["sums"]
    opt = st.mean(sums) if sums else float("nan")
    print(f"===== {label} =====")
    print(f"  IN-DISTRIBUTION (true side-to-move, {n} pos, decisive games):")
    print(f"    mean v_pred = {mean_vp:+.3f}   mean z(outcome) = {mean_z:+.3f}   bias(v-z) = {mean_vp-mean_z:+.3f}")
    print(f"    corr(v_pred, z) = {c:+.3f}   sign-accuracy = {signacc:.1%}")
    print(f"    opening(ply0) mean v = {st.mean(r['open_vals']):+.3f} (n{len(r['open_vals'])})  [calibrated~0]")
    print(f"    calib: {cal}")
    print(f"    defense: {rev}")
    print(f"  PROBE (owner-swap, {len(sums)} pos):  mean(vA+vB) = {opt:+.3f}  both>0.05 = {r['both_pos']}/{len(sums)} = {r['both_pos']/max(1,len(sums)):.0%}")
    print(f"    parity own-opp  real: mean={st.mean(r['parity_real']):+.2f} swapped: mean={st.mean(r['parity_swap']):+.2f}  (swap flips sign of stone-count diff)")
    print()
