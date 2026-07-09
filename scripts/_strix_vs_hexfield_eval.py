"""Head-to-head eval through hexo_runner: hexo-strix vs a hexfield checkpoint,
each running ITS OWN Gumbel MCTS at a matched simulation budget.

- hexo-strix side: `StrixMctsPlayer` — the real Rust `hexo_rs` Gumbel MCTS with
  the ported model as eval_fn (m_actions=16, c_visit=50, c_scale=1.0).
- hexfield side: `HexfieldMain6Player` (defined here) — hexfield's native
  `HexfieldMctsSession` gumbel search, loaded from a checkpoint, using the
  model's own selfplay divergences (gumbel_root/SH/nonroot on) with gumbel_m
  overridden to match.

Both run through `hexo_runner`'s match loop as `RunnerPlayer`s, one placement
per `decide` (a HeXO turn is two placements; each searched separately). Gumbel
root noise is ON on both sides (per-move seeds derived from a per-game seed) so
the games are diverse yet reproducible. Colors alternate across games.

MUST RUN under WSL Python 3.12 (the hexfield `_rust` extension is linux/cp312),
in a venv where hexo_engine, hexfield._rust, torch+CUDA, hexo_runner AND the
strix `hexo_rs` wheel all import (e.g. /root/.venvs/hexgt-build).

Usage (from WSL):
  python scripts/_strix_vs_hexfield_eval.py \
      --strix-ckpt /mnt/c/Users/epicm/Downloads/checkpoint_00237000.pt \
      --hexfield-ckpt /path/to/hexfield/checkpoints/epoch_000075.pt \
      --hexfield-config /mnt/e/Hexo-BotTrainer-hexgt/configs/hexfield_main_9.toml \
      --games 40 --sims 256 --m 16
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import tomllib
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for _pkg in ("hexo_runner", "hexfield", "hexo_strix"):
    _p = _REPO / "packages" / _pkg / "python"
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import hexo_engine as engine  # noqa: E402
from hexo_engine import api  # noqa: E402
from hexo_engine.types import AxialCoord, PlacementAction  # noqa: E402

from hexo_runner.player import DecisionResult, PlayerIdentity  # noqa: E402

# The threaded/batched runner + batch server are imported inside main().


# --------------------------------------------------------------------------
# hexfield checkpoint -> RunnerPlayer running its native gumbel search
# --------------------------------------------------------------------------
class HexfieldPlayer:
    """RunnerPlayer wrapping a hexfield checkpoint's native Gumbel MCTS.

    A fresh native session is created per player (i.e. per game) so tree reuse
    across a game's placements is correct and games don't share cache. The net +
    evaluator are shared across games (passed in) to avoid reloading the model.

    Thread-safety: :class:`hexfield.inference.HexfieldEvaluator` holds per-call
    mutable state (a CUDA-graph cache, copy streams, an optional serve-half model
    copy) and is NOT safe under concurrent calls, so a SHARED lock serialises the
    search across all hexfield players. The Strix side runs lock-free (its shared
    model lives only in the batch server's single batcher thread), so concurrent
    games still overlap Strix search + graph builds and coalesce Strix leaves.
    """

    def __init__(self, *, evaluator, sp, overrides, session, sims, seed, lock,
                 label, model_tag, identity_id="hexfield", device="cuda"):
        self._ev = evaluator
        self._sp = sp
        self._ov = overrides
        self._session = session
        self._sims = sims
        self._seed = seed
        self._lock = lock
        self._tag = model_tag
        self._n = 0
        self.identity = PlayerIdentity(
            player_id=identity_id,
            label=label,
            metadata={"model": model_tag, "search": "gumbel", "sims": sims, "device": device},
        )

    def setup_worker(self, ctx) -> None: ...
    def start_game(self, ctx) -> None:
        self._n = 0
    def observe_transition(self, t) -> None: ...
    def finish_game(self, s) -> None: ...
    def close(self) -> None: ...

    def decide(self, state) -> DecisionResult:
        from hexfield.geometry import unpack_action_id

        move_seed = self._seed * 1_000_003 + self._n
        self._n += 1
        sp = self._sp
        # Serialise the (non-thread-safe) evaluator across concurrent games.
        with self._lock:
            res = self._session.search(
                [0], (state,),
                visits=self._sims,
                c_puct=sp.c_puct,
                temperature=0.0,
                seed=move_seed,
                evaluator=self._ev,
                virtual_batch_size=sp.virtual_batch_size,
                active_root_limit=1,
                widening_policy_mass=sp.widening_policy_mass,
                widening_max_children=sp.widening_max_children,
                widening_min_children=sp.widening_min_children,
                fpu_reduction=sp.fpu_reduction,
                tss_enabled=sp.tss_enabled,
                search_parity_mode=sp.search_parity_mode,
                divergence_overrides=self._ov,
            )[0]
        q, r = unpack_action_id(int(res["action_id"]))
        action = PlacementAction(AxialCoord(q=int(q), r=int(r)))
        if not api.is_legal_action(state, action):
            legal = api.legal_actions(state)
            return DecisionResult(action=legal[0],
                                  diagnostics={"model": self._tag, "fallback": True})
        return DecisionResult(action=action, diagnostics={
            "model": self._tag, "search": "gumbel",
            "root_value": res.get("root_value"), "visits": res.get("visits")})


def make_hexfield_factory(ckpt, config_path, *, sims, m_actions, device):
    """Load the hexfield net + config once; return a per-game player factory.

    The returned factory builds a fresh per-game session but shares ONE evaluator
    and ONE lock across games (see HexfieldPlayer thread-safety note). The model
    tag / label are derived from the checkpoint path so the prints are accurate
    for any ``--hexfield-ckpt`` (not hardcoded to a specific epoch).
    """
    import threading

    from hexfield.config import parse_hexfield_config, build_divergence_overrides
    from hexfield.eval_arena import _load_hexfield_net
    from hexfield.inference import HexfieldEvaluator
    from hexfield import _rust

    with open(config_path, "rb") as fh:
        toml = tomllib.load(fh)
    cfg = parse_hexfield_config(toml["model"]["config"])
    sp = dataclasses.replace(cfg.selfplay, gumbel_m=int(m_actions))
    overrides = build_divergence_overrides(sp)
    net = _load_hexfield_net(ckpt)
    evaluator = HexfieldEvaluator(net, device=device)
    lock = threading.Lock()
    # Tag from the run + epoch in the checkpoint path, e.g.
    # ".../hexfield_main_6/checkpoints/epoch_000073.pt" -> "hexfield_main_6.ep73".
    ckpt_path = Path(ckpt)
    run_name = ckpt_path.parent.parent.name or "hexfield"
    ep = "".join(c for c in ckpt_path.stem if c.isdigit()).lstrip("0") or "0"
    model_tag = f"{run_name}.ep{ep}"
    label = f"{model_tag} (gumbel {sims})"
    n_params = sum(p.numel() for p in net.parameters())
    print(f"hexfield {model_tag}: {n_params:,} params | gumbel m={sp.gumbel_m} "
          f"c_visit={sp.gumbel_c_visit} c_scale={sp.gumbel_c_scale} sims={sims} device={device}",
          flush=True)

    def factory(game_seed: int) -> HexfieldPlayer:
        session = _rust.HexfieldMctsSession(max_states=sp.cache_max_states)
        return HexfieldPlayer(
            evaluator=evaluator, sp=sp, overrides=overrides, session=session,
            sims=sims, seed=game_seed, lock=lock, label=label, model_tag=model_tag,
            device=device,
        )

    return factory


def main() -> None:
    import time

    from hexo_strix.batch_server import StrixBatchServer
    from hexo_strix.loader import load_strix_checkpoint
    from hexo_strix.match_runner import run_threaded_matches
    from hexo_strix.mcts_player import StrixMctsPlayer

    ap = argparse.ArgumentParser()
    ap.add_argument("--strix-ckpt", default="/mnt/e/Hexo-BotTrainer/anchors/strix/checkpoint_00237000.pt")
    ap.add_argument("--hexfield-ckpt", default="/mnt/e/Hexo-BotTrainer/runs/hexfield_main_9/checkpoints/epoch_000030.pt")
    ap.add_argument("--hexfield-config", default=str(_REPO / "configs" / "hexfield_main_9.toml"))
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--sims", type=int, default=256)
    ap.add_argument("--m", type=int, default=16, help="gumbel m_actions for BOTH sides")
    ap.add_argument("--strix-device", default="cuda",
                    help="device for the Strix GNN batch server (default cuda)")
    ap.add_argument("--hexfield-device", default="cuda")
    ap.add_argument("--threads", type=int, default=8,
                    help="concurrent games in flight (Strix leaves coalesce across them)")
    ap.add_argument("--linger-ms", type=float, default=0.8,
                    help="batch-server linger window (ms) to gather cross-game leaves")
    ap.add_argument("--max-actions", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--out-dir", default=str(_REPO / "runs" / "strix_vs_hexfield"))
    args = ap.parse_args()

    hexf_tag = Path(args.hexfield_ckpt).parent.parent.name
    print(f"=== hexo-strix vs {hexf_tag} | {args.games} games | {args.sims} sims/move | "
          f"m={args.m} | threads={args.threads} | strix-dev={args.strix_device} | "
          f"gumbel noise ON both sides | alternating colors ===", flush=True)

    # --- Strix side: one shared model in a cross-game batch server. ---
    strix_ck = load_strix_checkpoint(args.strix_ckpt, device="cpu")
    n_strix = sum(p.numel() for p in strix_ck.model.parameters())
    server = StrixBatchServer(
        strix_ck.model, device=args.strix_device, linger_s=args.linger_ms / 1000.0
    )
    print(f"hexo-strix: {n_strix:,} params | server device={args.strix_device} "
          f"linger={args.linger_ms}ms", flush=True)

    def strix_factory(game_seed: int) -> StrixMctsPlayer:
        return StrixMctsPlayer(
            checkpoint=strix_ck, device=args.strix_device, sims=args.sims,
            m_actions=args.m, disable_gumbel_noise=False, seed=game_seed,
            identity_id="hexo-strix", server=server,
        )

    make_hexfield = make_hexfield_factory(
        args.hexfield_ckpt, args.hexfield_config, sims=args.sims, m_actions=args.m,
        device=args.hexfield_device,
    )

    def on_game(row: dict) -> None:
        print(f"  game {row['game']:3d}: strix={row['strix_seat']} "
              f"winner={row['winner']} -> {row['outcome']:11s} turns={row['turns']}",
              flush=True)

    t0 = time.time()
    try:
        report = run_threaded_matches(
            strix_factory, make_hexfield,
            games=args.games, seed=args.seed, max_actions=args.max_actions,
            out_dir=args.out_dir, n_threads=args.threads, on_game=on_game,
        )
    finally:
        server.close()
    elapsed = time.time() - t0

    sc = report["score"]
    wr = sc["strix_winrate_decided"]
    print("\n================ RESULT (hexo-strix perspective) ================", flush=True)
    print(f"  completed={sc['completed']}/{args.games}  "
          f"W-L-D = {sc['strix_wins']}-{sc['opp_wins']}-{sc['draws']}", flush=True)
    print(f"  strix win rate (decided): {wr:.1%}" if wr is not None else "  no decided games", flush=True)
    print(f"  strix score (draws=0.5):  {sc['strix_score']:.3f}" if sc['strix_score'] is not None else "", flush=True)
    print(f"  wall time: {elapsed:.1f}s  ({elapsed/max(args.games,1):.1f}s/game)", flush=True)
    print(f"  batch server: {server.n_forwards} forwards, {server.n_graphs} leaves, "
          f"max batch={server.max_seen_batch}, "
          f"mean batch={server.n_graphs/max(server.n_forwards,1):.1f}", flush=True)


if __name__ == "__main__":
    main()
