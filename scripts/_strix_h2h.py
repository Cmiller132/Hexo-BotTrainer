"""Head-to-head / smoke driver for the ported hexo-strix bot.

Plays the hexo-strix HeXONet checkpoint (raw-policy greedy) against an
opponent through THIS repo's engine + runner, alternating colors, and reports
the win rate from hexo-strix's perspective. This is the entry point for adding
hexo-strix to the eval routine as a fixed-strength anchor.

Opponents:
  --vs random    a uniform-random legal-move baseline (default; self-contained)
  --vs self      hexo-strix vs itself (color-symmetry / sanity)
  --vs sealbot   SealBot minimax (needs a SealBot checkout; SEALBOT_PATH)

Usage:
  python scripts/_strix_h2h.py --ckpt ~/Downloads/checkpoint_00237000.pt \
      --vs random --games 10

Environment note: this repo's pure-Python packages (hexo_runner) may not be on
the default import path in every shell; this script prepends the local
`packages/*/python` dirs so it resolves the current worktree's runner while
still using the compiled hexo_engine from site-packages.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _bootstrap_path() -> None:
    # Prepend only hexo_runner's source so it resolves to this worktree (its
    # installed editable entry is broken on this machine). hexo_engine,
    # hexo_utils, hexo_models keep their installed COMPILED builds — their local
    # python/ dirs lack the built _rust extension, so shadowing them breaks.
    p = _REPO / "packages" / "hexo_runner" / "python"
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


_bootstrap_path()

import hexo_engine as engine  # noqa: E402

from hexo_runner.modes.match import run_match  # noqa: E402
from hexo_runner.player import DecisionResult, PlayerIdentity  # noqa: E402
from hexo_runner.records import GameStatus  # noqa: E402
from hexo_runner.session import GameSpec  # noqa: E402

from hexo_strix import make_strix_factory, make_strix_mcts_factory  # noqa: E402


class RandomPlayer:
    """Uniform-random legal-move RunnerPlayer baseline."""

    def __init__(self, seed: int = 0, player_id: str = "random") -> None:
        self.identity = PlayerIdentity(player_id=player_id, label="random")
        self._rng = random.Random(seed)

    def setup_worker(self, context) -> None: ...
    def start_game(self, context) -> None: ...
    def observe_transition(self, transition) -> None: ...
    def finish_game(self, final_summary) -> None: ...
    def close(self) -> None: ...

    def decide(self, state) -> DecisionResult:
        legal = engine.legal_actions(state)
        return DecisionResult(action=legal[self._rng.randrange(len(legal))])


def _make_strix(kind: str, device: str, ckpt: str, sims: int, noise: bool, ident: str):
    """Build the hexo-strix side: real Gumbel MCTS or raw-policy greedy."""
    if kind == "mcts":
        return make_strix_mcts_factory(
            ckpt, device=device, sims=sims,
            disable_gumbel_noise=not noise, identity_id=ident,
        )
    return make_strix_factory(ckpt, device=device, identity_id=ident)


def _make_opponent(kind: str, device: str, ckpt: str, sims: int, noise: bool):
    if kind == "random":
        return lambda seed: RandomPlayer(seed=seed, player_id="random")
    if kind == "self":
        return _make_strix("mcts", device, ckpt, sims, noise, "hexo-strix-b")
    if kind == "sealbot":
        from hexo_runner.adapters.sealbot import SealBotConfig, SealBotPlayer

        cfg = SealBotConfig(variant="best", time_limit=0.05)
        cfg.validate()
        return lambda seed: SealBotPlayer(cfg, player_id="sealbot-best-50ms")
    raise ValueError(f"unknown opponent {kind!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/Downloads/checkpoint_00237000.pt"))
    ap.add_argument("--vs", default="random", choices=["random", "self", "sealbot"])
    ap.add_argument("--search", default="mcts", choices=["mcts", "policy"],
                    help="hexo-strix side: real Gumbel MCTS (default) or raw-policy greedy")
    ap.add_argument("--sims", type=int, default=256, help="MCTS simulations per placement (eval default 256)")
    ap.add_argument("--noise", action="store_true", help="enable Gumbel root noise (stochastic, matches hexo-strix default)")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-actions", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out-dir", default=str(_REPO / "runs" / "strix_h2h"))
    args = ap.parse_args()

    make_strix = _make_strix(args.search, args.device, args.ckpt, args.sims, args.noise, "hexo-strix")
    make_opp = _make_opponent(args.vs, args.device, args.ckpt, args.sims, args.noise)
    print(f"hexo-strix side: search={args.search} sims={args.sims} noise={args.noise}", flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    wins = losses = draws = completed = 0
    total_turns = 0
    for g in range(args.games):
        seed = args.seed + g
        strix = make_strix(seed)
        opp = make_opp(seed)
        # Alternate colors: even games strix=player0 (moves first).
        strix_is_p0 = (g % 2 == 0)
        players = (strix, opp) if strix_is_p0 else (opp, strix)
        spec = GameSpec(game_id=f"strix_vs_{args.vs}_{g:03d}", seed=seed,
                        is_evaluation=True, max_actions=args.max_actions)
        result = run_match(spec, players, output_dir=out)
        if result.status != GameStatus.COMPLETED:
            print(f"  game {g}: status={result.status} (skipped in tally)", flush=True)
            continue
        completed += 1
        total_turns += getattr(result, "action_count", 0) or 0
        winner = result.winner  # "player0"/"player1"/None
        strix_seat = "player0" if strix_is_p0 else "player1"
        if winner is None:
            draws += 1
            outcome = "draw"
        elif str(winner) == strix_seat:
            wins += 1
            outcome = "WIN"
        else:
            losses += 1
            outcome = "loss"
        print(f"  game {g}: strix={strix_seat} winner={winner} -> {outcome}", flush=True)

    decided = wins + losses
    win_rate = (wins / decided) if decided else float("nan")
    print(f"\n=== hexo-strix vs {args.vs} ({args.games} games) ===", flush=True)
    print(f"  completed={completed} wins={wins} losses={losses} draws={draws}", flush=True)
    print(f"  WIN RATE (decided): {win_rate:.1%}", flush=True)


if __name__ == "__main__":
    main()
