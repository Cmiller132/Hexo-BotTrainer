"""Soft-Z causal experiment (owner-approved, behavior-changing but SIDE checkpoints only).

Run is HALTED (breaker tripped on the STV-graft optimizer bug). GPU is free.

Tests: does disabling soft-Z (lambda=0, pure hard-z value targets) reduce the
value-head opening optimism? lambda is BAKED into shards at finalize, so we cannot
just flip a config flag on a re-train of existing shards -- we recompute the value
target to hard-z from the stored records (winner via .hxr join + current_player).

Two controlled passes from the SAME epoch-6 checkpoint (last stable; epoch 7 never
completed), identical group sampling (Random(7)), fresh optimizer (identical across
arms; also sidesteps the graft optimizer-shape crash), 1024 steps x batch 64 = 65,536
samples, STV graft applied as the driver would:
  - lambda05: original shards (soft-Z 0.5, == what epoch 7 would have trained)
  - lambda0 : hard-z mirror shards (soft-Z disabled)
Saves hexgt_rl_epoch000007_lambda0.pt / _lambda05.pt. Does NOT touch hexgt_rl_latest.pt.
Eval (CPU) compares epoch6 / lambda0 / lambda05: opening value, optimism probe,
calibration, corr(v,z), won/lost gap.
"""
import glob, re, time, random, statistics as st
from pathlib import Path
import numpy as np
import torch

import hexo_engine.api as eng
from hexo_engine.types import AxialCoord, PlacementAction, unpack_coord_id
from hexo_runner.records import HexoRecordFile
from hexo_models.hexgt.architecture import HexgtNetwork, expand_stv_readout_columns
from hexo_models.hexgt.config import parse_hexgt_config
from hexo_models.hexgt.trainer import HexgtTrainer
from hexo_models.hexgt.features import build_graph_tensors
from hexo_models.hexgt.collate import collate_graphs
from hexo_models.hexgt import rust_bridge
from hexo_models.hexgt.losses import decode_binned_value

RUN = "/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main3"
SP = f"{RUN}/selfplay"
SP0 = f"{RUN}/selfplay_lambda0"          # hard-z mirror
CK = f"{RUN}/checkpoints"
EP6 = f"{CK}/hexgt_rl_epoch000006.pt"
PLAYERS = ("player0", "player1")
STV_HORIZONS = (4, 12, 24)
WINDOW_EPOCHS = list(range(0, 8))        # current_epoch=7, base_window=8 -> epochs 0..7
RADIUS = 3


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- mirror shards
def winners_for_epoch(ep):
    hxr = f"{SP}/epoch_{ep:06d}.hxr"
    if not glob.glob(hxr):
        return {}
    out = {}
    for r in HexoRecordFile.open(hxr).iter_records():
        m = re.search(r"selfplay-(\d+)$", r.game_id)
        if m:
            w = r.winner
            out[int(m.group(1))] = (str(getattr(w, "value", w)) if w is not None else None)
    return out


def build_mirror():
    Path(SP0).mkdir(parents=True, exist_ok=True)
    total = changed = 0
    for ep in WINDOW_EPOCHS:
        win = winners_for_epoch(ep)
        for sh in sorted(glob.glob(f"{SP}/epoch_{ep:06d}_game_*.npz")):
            gi = int(re.search(r"game_(\d+)\.npz$", sh).group(1))
            d = dict(np.load(sh, allow_pickle=True))
            cp = d["current_player"].astype(int)
            w = win.get(gi)
            val = d["value"].astype(np.float32).copy()
            if w in ("player0", "player1"):
                z = np.where(np.array([PLAYERS[c] for c in cp]) == w, 1.0, -1.0).astype(np.float32)
            else:
                z = np.zeros_like(val)  # truncated/draw -> hard z = 0
            d["value"] = z
            np.savez_compressed(f"{SP0}/{Path(sh).name}", **d)
            total += 1; changed += int(np.count_nonzero(np.abs(z - val) > 1e-6))
    log(f"mirror built: {total} shards -> {SP0} ({changed} rows changed value)")


# ---------------------------------------------------------------- training pass
def load_grafted(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck["arch"]
    net = HexgtNetwork(
        token_dim=a["token_dim"], gnn_layers=a["gnn_layers"], ctx_layers=a["ctx_layers"],
        ffn_dim=a["ffn_dim"], attention_heads=a["attention_heads"],
        short_term_value_horizons=tuple(a.get("short_term_value_horizons", ())),
        value_pma_seeds=int(a.get("value_pma_seeds", 2)),
        value_head_use_side=bool(a.get("value_head_use_side", True)),
    ).to(device)
    grafted = expand_stv_readout_columns(net, ck["model"])  # STV 168->504 zero-init (as driver)
    miss, unexp = net.load_state_dict(ck["model"], strict=False)
    bad = [k for k in miss if not k.startswith("_")]
    assert not bad and not unexp, f"load issue miss={bad} unexp={unexp}"
    return net, ck, a, grafted


def make_cfg(device):
    return parse_hexgt_config({
        "device": str(device),
        "architecture": {"candidate_radius": RADIUS, "short_term_value_horizons": list(STV_HORIZONS)},
        "training": {"learning_rate": 2e-4, "warmup_steps": 200, "batch_size": 64,
                     "short_term_value_weight": 0.1},
    })


def epoch_of(path):
    return int(re.search(r"epoch_(\d+)_game", path).group(1))


def run_pass(shard_root, tag, device, steps=1024):
    net, ck, arch, grafted = load_grafted(EP6, device)
    cfg = make_cfg(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)  # FRESH (identical both arms)
    trainer = HexgtTrainer(model=net, config=cfg, optimizer=opt)
    trainer.load_train_state(ck.get("train_state"))
    start_step = trainer.train_state.global_step
    shards = sorted(p for e in WINDOW_EPOCHS for p in glob.glob(f"{shard_root}/epoch_{e:06d}_game_*.npz"))
    weights = [0.9 ** (7 - epoch_of(p)) for p in shards]
    rng = random.Random(7)
    target = start_step + steps
    comp_sum, comp_n, t0 = {}, 0, time.perf_counter()
    while trainer.train_state.global_step < target and shards:
        group = rng.choices(shards, weights=weights, k=30)
        hist = trainer.train_on_shards(group, batch_size=64,
                                       max_steps=target - trainer.train_state.global_step)
        for h in hist:
            for k, v in h.items():
                comp_sum[k] = comp_sum.get(k, 0.0) + v
            comp_n += 1
    avg = lambda k: comp_sum.get(k, float("nan")) / comp_n if comp_n else float("nan")
    log(f"[{tag}] grafted={len(grafted)} start_step={start_step} -> {trainer.train_state.global_step} "
        f"({comp_n} steps, {(time.perf_counter()-t0)/60:.1f} min) "
        f"total={avg('total'):.4f} policy={avg('policy'):.4f} value={avg('value'):.4f} opp={avg('opp_policy'):.4f}")
    payload = {"model": net.state_dict(), "optimizer": opt.state_dict(),
               "train_state": trainer.train_state.to_dict(), "arch": arch,
               "step": trainer.train_state.global_step, "rl_epoch": 7,
               "feature_schema_version": int(ck.get("feature_schema_version", 1)),
               "epoch_train_complete": True,
               "_experiment": f"lambda-experiment {tag}; fresh-opt; {steps}steps x b64; from epoch6"}
    out = f"{CK}/hexgt_rl_{tag}.pt"
    torch.save(payload, out)
    log(f"[{tag}] saved {out}  (latest NOT touched)")
    return out


# ---------------------------------------------------------------- eval (CPU)
def swap_owner(facts):
    own = list(facts["nodes"]["node_owner"])
    return {**facts, "nodes": {**facts["nodes"], "node_owner": [1 if o == 0 else (0 if o == 1 else o) for o in own]}}


def corr(a, b):
    if len(a) < 3: return float("nan")
    ma, mb = st.mean(a), st.mean(b)
    da = sum((x-ma)**2 for x in a) ** .5; db = sum((y-mb)**2 for y in b) ** .5
    return sum((x-ma)*(y-mb) for x, y in zip(a, b))/(da*db) if da and db else float("nan")


def eval_ckpt(path, hxr_files, device="cpu", maxgames=80):
    net, _ck, _a, _g = load_grafted(path, device)
    net.eval()
    def val(facts):
        b = collate_graphs([build_graph_tensors(facts)])
        return float(decode_binned_value(net.forward_policy_value(b)["value"].float())[0])
    vp, zo, opens, sums, both = [], [], [], [], 0
    ng = 0
    with torch.no_grad():
        for f in hxr_files:
            for r in HexoRecordFile.open(f).iter_records():
                if ng >= maxgames: break
                coords = [unpack_coord_id(int(c)) for c in r.action_ids]
                sc = eng.new_game()
                for c in coords: eng.apply_action(sc, PlacementAction(AxialCoord(q=int(c.q), r=int(c.r))))
                t = eng.terminal(sc)
                winner = str(getattr(t.winner, "value", t.winner)) if (t and getattr(t, "winner", None)) else None
                s = eng.new_game()
                for i, c in enumerate(coords):
                    if eng.terminal(s) is None:
                        facts = rust_bridge.graph_facts(s, RADIUS)
                        cp = str(getattr(eng.current_player(s), "value", ""))
                        if i == 0: opens.append(val(facts))
                        if winner in ("player0", "player1") and 6 <= i <= 60:
                            vp.append(val(facts)); zo.append(1.0 if winner == cp else -1.0)
                        if 10 <= i <= 50 and facts.get("meta", {}).get("first_stone") is None:
                            vA = val(facts); vB = val(swap_owner(facts)); sums.append(vA+vB)
                            if vA > 0.05 and vB > 0.05: both += 1
                    eng.apply_action(s, PlacementAction(AxialCoord(q=int(c.q), r=int(c.r))))
                ng += 1
            if ng >= maxgames: break
    n = len(vp)
    vw = [v for v, z in zip(vp, zo) if z > 0]; vl = [v for v, z in zip(vp, zo) if z < 0]
    return dict(
        opening=st.mean(opens) if opens else float("nan"),
        probe=st.mean(sums) if sums else float("nan"),
        both=both/max(1, len(sums)),
        mean_v=st.mean(vp) if vp else float("nan"),
        corr=corr(vp, zo), n=n,
        v_won=st.mean(vw) if vw else float("nan"), v_lost=st.mean(vl) if vl else float("nan"),
        gap=(st.mean(vw)-st.mean(vl)) if vw and vl else float("nan"),
    )


if __name__ == "__main__":
    torch.set_grad_enabled(True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device(train)={dev}")
    log("=== phase 1: build hard-z mirror shards ===")
    build_mirror()
    log("=== phase 2: train lambda=0.5 control (original shards) ===")
    p05 = run_pass(SP, "epoch000007_lambda05", dev)
    log("=== phase 3: train lambda=0 treatment (hard-z shards) ===")
    p0 = run_pass(SP0, "epoch000007_lambda0", dev)
    log("=== phase 4: eval (CPU) ===")
    torch.set_grad_enabled(False)
    hxr = sorted(glob.glob(f"{SP}/epoch_000006.hxr")) + sorted(glob.glob(f"{SP}/epoch_000005.hxr"))
    rows = [("epoch6 (baseline, lambda0.5 trained)", EP6),
            ("epoch7 lambda0.5 (control)", p05),
            ("epoch7 lambda0  (treatment)", p0)]
    print(f"\n{'checkpoint':<38} | opening | probe(both) | mean_v corr | v_won/v_lost gap | n")
    print("-" * 110)
    for name, path in rows:
        r = eval_ckpt(path, hxr)
        print(f"{name:<38} | {r['opening']:+.3f}  | {r['probe']:+.3f}({r['both']:.0%}) | "
              f"{r['mean_v']:+.3f} {r['corr']:+.3f} | {r['v_won']:+.3f}/{r['v_lost']:+.3f} {r['gap']:+.3f} | {r['n']}")
    log("DONE. latest checkpoint untouched; run still halted (flag in place).")
