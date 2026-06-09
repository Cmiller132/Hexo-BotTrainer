"""Shared read-only helpers for the hexgnn exploration analysis (CPU only)."""
from __future__ import annotations
import math, re
from pathlib import Path
import torch
torch.set_grad_enabled(False)

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id
from hexgnn.architecture import HexgnnNetwork
from hexgnn.inference import HexgnnInference

RUN = "/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1"
CKPT_DIR = f"{RUN}/checkpoints"
SP_DIR = f"{RUN}/selfplay"

# Live exploration curve (scripts/_rl_launch_hexgnn.sh).
TEMP_START, TEMP_FLOOR, TEMP_HALFLIFE = 1.0, 0.3, 33.0
def cur_temp(ply: int) -> float:
    return TEMP_FLOOR + (TEMP_START - TEMP_FLOOR) * (2.0 ** (-float(ply) / TEMP_HALFLIFE))

BANDS = [("0-20", 0, 20), ("20-50", 20, 50), ("50+", 50, 10**9)]
def band_of(ply: int) -> str:
    for nm, lo, hi in BANDS:
        if lo <= ply < hi:
            return nm
    return "50+"

def load_model(epoch: int | str = 41):
    path = f"{CKPT_DIR}/hexgnn_rl_epoch{int(epoch):06d}.pt" if isinstance(epoch, int) \
        else f"{CKPT_DIR}/{epoch}"
    p = torch.load(path, map_location="cpu")
    ms = p["model"] if "model" in p else p["model_state"]
    a = p.get("arch", {})
    net = HexgnnNetwork(
        node_feature_dim=int(a.get("node_feature_dim", 32)),
        token_dim=int(a.get("token_dim", 96)),
        gnn_layers=int(a.get("gnn_layers", 2)),
        attention_heads=int(a.get("attention_heads", 4)),
        value_pma_seeds=int(a.get("value_pma_seeds", 2)),
        value_head_use_side=bool(a.get("value_head_use_side", True)),
        steerable_channels=int(a.get("steerable_channels", 4)),
    )
    net.load_state_dict(ms, strict=True)
    net.eval()
    return net, p

class Evaluator:
    def __init__(self, net, n=4):
        self.inf = HexgnnInference(net, device="cpu", fp16=False)
        self.n = int(n)
    def value(self, state) -> float:
        """Value from the perspective of the player to move at `state`, in [-1,1]."""
        return float(self.inf.evaluate_states([state], self.n)[0]["value"])
    def priors(self, state):
        """(candidate_ids ndarray, priors ndarray, value) for `state`."""
        o = self.inf.evaluate_states([state], self.n)[0]
        return o["candidate_ids"], o["priors"], float(o["value"])

def apply(state, action_id: int):
    engine.apply_action(state, engine.PlacementAction(unpack_coord_id(int(action_id))))

def clone(state):
    return engine.clone_state(state)

def new_game(seed: int = 0):
    return engine.new_game(seed=seed)

def shard_index_from_game_id(game_id: str) -> int:
    return int(re.search(r"selfplay-(\d+)$", game_id).group(1))

def entropy(weights) -> float:
    ws = [float(w) for w in weights if float(w) > 0.0]
    tot = sum(ws)
    if tot <= 0:
        return 0.0
    return -sum((w / tot) * math.log(w / tot) for w in ws)

def top1_fraction(weights) -> float:
    ws = [float(w) for w in weights if float(w) > 0.0]
    tot = sum(ws)
    return (max(ws) / tot) if tot > 0 and ws else 0.0
