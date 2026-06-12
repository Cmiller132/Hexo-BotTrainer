"""Head-to-head arena: dense_cnn_restnet_main_3 ckpt epoch 10 vs epoch 5.

Plays N eval-style games between two checkpoints of the same run, both sides
using the production rust MCTS at full strength (512 visits, eval knobs: no
Dirichlet noise, no root temperature). Mirrors the production eval game
protocol (dense_cnn_restnet/evaluation.py):
  - alternate seats: even game index -> ckpt10 is P0, odd -> ckpt5 is P0
  - opening diversity: plies 0-7 sample the move from the visit distribution
    at temperature 0.6 (per-root RNG => per-game divergence), then argmax
  - truncate at max_actions=1024
All games run CONCURRENTLY (cross-game leaf batching, same orchestration as
the production eval loop), one persistent MCTS session + one DenseCNNInference
per checkpoint.

READ-ONLY w.r.t. the run dir.

Usage:
  python _wf_r4_h2h_arena.py N_GAMES OUT_JSON [DEVICE] [VISITS] [MAX_WALL_MIN]
"""
from __future__ import annotations

import json
import math
import sys
import time
import types
from pathlib import Path

PKG = "dense_cnn_restnet"
PKG_DIR = Path("/mnt/e/Hexo-BotTrainer-hexgt/packages/dense_cnn_restnet/python/dense_cnn_restnet")
if PKG not in sys.modules:
    stub = types.ModuleType(PKG)
    stub.__path__ = [str(PKG_DIR)]
    sys.modules[PKG] = stub

import torch  # noqa: E402

torch.set_grad_enabled(False)

import hexo_engine as engine  # noqa: E402
from hexo_engine.types import unpack_coord_id  # noqa: E402

from dense_cnn_restnet.architecture import RestnetNetwork  # noqa: E402
from dense_cnn_restnet.inference import DenseCNNInference  # noqa: E402
from dense_cnn_restnet.mcts import new_mcts_session  # noqa: E402

RUN_DIR = Path("/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main_3")
CKPT_A_EPOCH = 10  # "ckpt10"
CKPT_B_EPOCH = 5   # "ckpt5"

# Search knobs mirrored from configs/dense_cnn_restnet_main_3.toml
# [model.config.selfplay], eval-style (NO Dirichlet noise, NO root temperature).
C_PUCT = 1.5
FPU_REDUCTION = 0.20
VIRTUAL_LOSS = 1.0
WIDEN_MASS = 0.95
WIDEN_MAX = 96
WIDEN_MIN = 2
FORCED_K = 2.0
VIRTUAL_BATCH = 4
ACTIVE_ROOT_LIMIT = 256
MAX_ACTIONS = 1024
OPENING_PLIES = 8        # total plies 0-7 sampled ...
OPENING_TEMPERATURE = 0.6  # ... at this temperature (mirrors eval opening knobs)

GAME_SEED_BASE = 777_000
SEED_BASE = {"ckpt10": 911_000_000, "ckpt5": 522_000_000}


def log(*a):
    print(*a, flush=True)


def build_net() -> RestnetNetwork:
    return RestnetNetwork(
        in_channels=13, channels=96, blocks_type="R_R_R_T_R_R_T_R",
        attention_heads=4, mlp_ratio=2, embed_kernel_size=3,
        residual_conv="hex", attention_impl="sdpa", attention_scope="disk",
        dropout=0.0, short_term_value_horizons=(2, 6, 16), moves_left_head=True,
    ).eval()


def load_ckpt(path: Path) -> RestnetNetwork:
    net = build_net()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state", payload.get("model"))
    missing, unexpected = net.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if "disk" not in k and "relative_index" not in k]
    if real_missing or unexpected:
        raise RuntimeError(
            f"checkpoint {path.name} load mismatch: missing={real_missing[:8]} "
            f"unexpected={list(unexpected)[:8]}"
        )
    return net


def wilson_ci(wins: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def length_stats(xs):
    if not xs:
        return None
    return {
        "n": len(xs),
        "mean": round(sum(xs) / len(xs), 1),
        "median": pct(xs, 0.5),
        "p90": pct(xs, 0.9),
        "min": min(xs),
        "max": max(xs),
    }


class Game:
    __slots__ = ("index", "seed", "a_is_p0", "state", "actions", "done",
                 "status", "winner", "decisions")

    def __init__(self, index: int):
        self.index = index
        self.seed = GAME_SEED_BASE + index
        self.a_is_p0 = index % 2 == 0  # even -> ckpt10 plays P0
        self.state = engine.new_game(seed=self.seed)
        self.actions: list[int] = []  # packed action ids, in order
        self.done = False
        self.status = "active"
        self.winner: str | None = None  # "ckpt10" | "ckpt5" | None
        self.decisions = {"ckpt10": 0, "ckpt5": 0}

    def role_of(self, tag: str) -> engine.Player:
        if tag == "ckpt10":
            return engine.Player.PLAYER_0 if self.a_is_p0 else engine.Player.PLAYER_1
        return engine.Player.PLAYER_1 if self.a_is_p0 else engine.Player.PLAYER_0

    def to_move(self) -> str:
        cur = engine.current_player(self.state)
        if cur == self.role_of("ckpt10"):
            return "ckpt10"
        return "ckpt5"


def main():
    n_games = int(sys.argv[1])
    out_json = Path(sys.argv[2])
    device = sys.argv[3] if len(sys.argv) > 3 else "cuda"
    visits = int(sys.argv[4]) if len(sys.argv) > 4 else 512
    max_wall_min = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0

    t0 = time.perf_counter()
    log("=" * 78)
    log(f"H2H ARENA main_3 ep{CKPT_A_EPOCH} (ckpt10) vs ep{CKPT_B_EPOCH} (ckpt5)  "
        f"games={n_games} visits={visits} device={device} "
        f"max_wall_min={max_wall_min or 'none'}")
    log(f"knobs: c_puct={C_PUCT} fpu={FPU_REDUCTION} widen={WIDEN_MASS}/{WIDEN_MAX}/{WIDEN_MIN} "
        f"forced_k={FORCED_K} vloss={VIRTUAL_LOSS} vb={VIRTUAL_BATCH} "
        f"no-noise no-root-temp opening T{OPENING_TEMPERATURE} x {OPENING_PLIES} plies "
        f"max_actions={MAX_ACTIONS}")
    log("=" * 78)

    if device.startswith("cuda") and not torch.cuda.is_available():
        log("[FATAL] CUDA requested but unavailable")
        raise SystemExit(2)

    nets = {}
    infers = {}
    sessions = {}
    for tag, epoch in (("ckpt10", CKPT_A_EPOCH), ("ckpt5", CKPT_B_EPOCH)):
        path = RUN_DIR / "checkpoints" / f"epoch_{epoch:06d}.pt"
        nets[tag] = load_ckpt(path)
        infers[tag] = DenseCNNInference(
            nets[tag], device=device, return_logits=False,
            max_batch_size=512, bucket_pad_multiple=16,
            fp16_model=device.startswith("cuda"), fp16_allow_fallback=True,
            attention_kv_gather=device.startswith("cuda"),
        )
        sessions[tag] = new_mcts_session(max_states=262_144)
        log(f"loaded {tag} <- {path}")

    games = [Game(i) for i in range(n_games)]
    rounds = 0
    roots_searched = 0
    search_elapsed = 0.0
    budget_hit = False

    def finalize(g: Game, sess_keys_owner: str | None = None):
        term = engine.terminal(g.state)
        if term is not None:
            g.status = "completed"
            if term.winner is None:
                g.winner = None  # draw
            else:
                w = str(term.winner)  # "player0" / "player1"
                role10 = "player0" if g.a_is_p0 else "player1"
                g.winner = "ckpt10" if w == role10 else "ckpt5"
        elif budget_hit:
            g.status = "aborted_budget"
        else:
            g.status = "truncated"  # max_actions reached, no terminal
        g.done = True
        for s in sessions.values():
            s.discard(g.index)

    def settle(g: Game) -> bool:
        if engine.terminal(g.state) is not None or len(g.actions) >= MAX_ACTIONS:
            finalize(g)
            return True
        return False

    def dump(partial: bool):
        completed = [g for g in games if g.status == "completed"]
        wins10 = sum(1 for g in completed if g.winner == "ckpt10")
        wins5 = sum(1 for g in completed if g.winner == "ckpt5")
        draws = sum(1 for g in completed if g.winner is None)
        decided = wins10 + wins5
        lo, hi = wilson_ci(wins10, decided)
        p0_games = [g for g in completed if g.a_is_p0]
        p1_games = [g for g in completed if not g.a_is_p0]
        openings = {}
        for g in games:
            key = tuple(g.actions[:OPENING_PLIES])
            openings.setdefault(key, []).append(g.index)
        dup_groups = {str(v[0]): v for v in openings.values() if len(v) > 1}
        lengths_all = [len(g.actions) for g in completed]
        out = {
            "meta": {
                "run": RUN_DIR.name,
                "ckpt10": f"epoch_{CKPT_A_EPOCH:06d}.pt",
                "ckpt5": f"epoch_{CKPT_B_EPOCH:06d}.pt",
                "games_requested": n_games,
                "visits": visits, "device": device,
                "knobs": {
                    "c_puct": C_PUCT, "fpu_reduction": FPU_REDUCTION,
                    "virtual_loss": VIRTUAL_LOSS,
                    "widening": [WIDEN_MASS, WIDEN_MAX, WIDEN_MIN],
                    "forced_playout_k": FORCED_K,
                    "virtual_batch": VIRTUAL_BATCH,
                    "dirichlet_noise": None, "root_policy_temperature": None,
                    "opening_temperature": OPENING_TEMPERATURE,
                    "opening_plies": OPENING_PLIES,
                    "max_actions": MAX_ACTIONS,
                },
                "partial": partial,
                "elapsed_minutes": round((time.perf_counter() - t0) / 60.0, 2),
                "rounds": rounds, "roots_searched": roots_searched,
                "search_elapsed_minutes": round(search_elapsed / 60.0, 2),
            },
            "score": {
                "completed": len(completed),
                "truncated": sum(1 for g in games if g.status == "truncated"),
                "aborted_budget": sum(1 for g in games if g.status == "aborted_budget"),
                "ckpt10_wins": wins10, "ckpt5_wins": wins5, "draws": draws,
                "ckpt10_winrate_decided": round(wins10 / decided, 4) if decided else None,
                "ckpt10_winrate_ci95": [round(lo, 4), round(hi, 4)] if decided else None,
                "by_seat": {
                    "ckpt10_as_P0": {
                        "n": len(p0_games),
                        "ckpt10_wins": sum(1 for g in p0_games if g.winner == "ckpt10"),
                        "ckpt5_wins": sum(1 for g in p0_games if g.winner == "ckpt5"),
                        "draws": sum(1 for g in p0_games if g.winner is None),
                    },
                    "ckpt10_as_P1": {
                        "n": len(p1_games),
                        "ckpt10_wins": sum(1 for g in p1_games if g.winner == "ckpt10"),
                        "ckpt5_wins": sum(1 for g in p1_games if g.winner == "ckpt5"),
                        "draws": sum(1 for g in p1_games if g.winner is None),
                    },
                },
            },
            "game_lengths": {
                "overall": length_stats(lengths_all),
                "ckpt10_as_P0": length_stats([len(g.actions) for g in p0_games]),
                "ckpt10_as_P1": length_stats([len(g.actions) for g in p1_games]),
                "ckpt10_won": length_stats([len(g.actions) for g in completed if g.winner == "ckpt10"]),
                "ckpt5_won": length_stats([len(g.actions) for g in completed if g.winner == "ckpt5"]),
            },
            "opening_dedup": {
                "n_games": len(games),
                "distinct_8ply_openings": len(openings),
                "duplicate_groups": dup_groups,
            },
            "games": [
                {
                    "index": g.index, "seed": g.seed,
                    "ckpt10_seat": "P0" if g.a_is_p0 else "P1",
                    "status": g.status, "winner": g.winner,
                    "plies": len(g.actions),
                    "opening_8": list(g.actions[:OPENING_PLIES]),
                }
                for g in games
            ],
        }
        out_json.write_text(json.dumps(out, indent=1), encoding="utf-8")
        return out

    last_dump_done = 0
    try:
        while True:
            active = [g for g in games if not g.done]
            if not active:
                break
            if max_wall_min and (time.perf_counter() - t0) / 60.0 > max_wall_min:
                budget_hit = True
                log(f"[BUDGET] wall {max_wall_min} min exceeded; aborting "
                    f"{len(active)} unfinished games")
                for g in active:
                    finalize(g)
                break
            rounds += 1
            plies_this_round = 0
            for tag in ("ckpt10", "ckpt5"):
                batch = [g for g in games if not g.done and g.to_move() == tag]
                if not batch:
                    continue
                move_temperatures = [
                    OPENING_TEMPERATURE if len(g.actions) < OPENING_PLIES else 0.0
                    for g in batch
                ]
                started = time.perf_counter()
                searches = sessions[tag].run(
                    [g.index for g in batch],
                    [g.state for g in batch],
                    infers[tag],
                    visits=visits,
                    c_puct=C_PUCT,
                    temperature=0.0,
                    seed=SEED_BASE[tag] + rounds * 1_000_003,
                    virtual_batch_size=VIRTUAL_BATCH,
                    active_root_limit=ACTIVE_ROOT_LIMIT,
                    fpu_reduction=FPU_REDUCTION,
                    virtual_loss=VIRTUAL_LOSS,
                    widening_policy_mass=WIDEN_MASS,
                    widening_max_children=WIDEN_MAX,
                    widening_min_children=WIDEN_MIN,
                    forced_playout_k=FORCED_K,
                    move_temperatures=move_temperatures,
                )
                search_elapsed += time.perf_counter() - started
                roots_searched += len(batch)
                if len(searches) != len(batch):
                    raise RuntimeError(
                        f"MCTS returned {len(searches)} results for {len(batch)} games")
                for g, s in zip(batch, searches):
                    aid = int(s.action_id)
                    engine.apply_action(g.state, engine.PlacementAction(unpack_coord_id(aid)))
                    g.actions.append(aid)
                    g.decisions[tag] += 1
                    plies_this_round += 1
                    if settle(g):
                        log(f"  game {g.index:3d} done: status={g.status} "
                            f"winner={g.winner} plies={len(g.actions)} "
                            f"ckpt10_seat={'P0' if g.a_is_p0 else 'P1'} "
                            f"opening={g.actions[:OPENING_PLIES]}")
            if plies_this_round == 0:
                raise RuntimeError("no progress in a round; aborting to avoid a hang")
            done_now = sum(1 for g in games if g.done)
            if rounds % 10 == 0 or done_now != last_dump_done:
                el = time.perf_counter() - t0
                sims = roots_searched * visits
                log(f"[round {rounds}] done {done_now}/{n_games} active {n_games - done_now} "
                    f"plies {sum(len(g.actions) for g in games)} "
                    f"elapsed {el/60:.1f}m (~{sims/max(el,1e-9):.0f} sims/s upper-bound)")
            if done_now != last_dump_done:
                last_dump_done = done_now
                dump(partial=True)
    finally:
        out = dump(partial=any(not g.done for g in games))
        s = out["score"]
        log("-" * 78)
        log(f"FINAL: ckpt10 {s['ckpt10_wins']} - {s['ckpt5_wins']} ckpt5 "
            f"(draws {s['draws']}, truncated {s['truncated']}, "
            f"aborted_budget {s['aborted_budget']}) "
            f"winrate={s['ckpt10_winrate_decided']} CI95={s['ckpt10_winrate_ci95']}")
        log(f"lengths overall: {out['game_lengths']['overall']}")
        log(f"openings distinct: {out['opening_dedup']['distinct_8ply_openings']}"
            f"/{out['opening_dedup']['n_games']}")
        log(f"wrote {out_json} ({(time.perf_counter()-t0)/60:.1f} min total)")


if __name__ == "__main__":
    main()
