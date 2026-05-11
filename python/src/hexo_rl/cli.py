"""Command-line entry points for Hexo RL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_config
from .replay import inspect_replay, update_replay_from_cycle
from .selfplay import RustSelfplayBridge, run_selfplay_cycle
from .train import train_one_cycle


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def command_test_engine(_: argparse.Namespace) -> int:
    config = load_config("configs/dev.yaml")
    _print_json(RustSelfplayBridge(config).test_engine())
    return 0


def command_random_game(_: argparse.Namespace) -> int:
    config = load_config("configs/dev.yaml")
    _print_json(RustSelfplayBridge(config).random_game())
    return 0


def command_selfplay(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    result = run_selfplay_cycle(config, args.cycle)
    _print_json(result.__dict__)
    return 0


def command_train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    result = train_one_cycle(config)
    _print_json(result.__dict__)
    return 0


def command_loop(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    cycles = args.cycles if args.cycles is not None else config.loop.cycles
    for cycle in range(cycles):
        selfplay = run_selfplay_cycle(config, cycle)
        replay_path = update_replay_from_cycle(config, selfplay.output_dir)
        training = train_one_cycle(config)
        _print_json(
            {
                "cycle": cycle,
                "selfplay": selfplay.__dict__,
                "replay_path": replay_path,
                "training": training.__dict__,
            }
        )
    return 0


def command_inspect_replay(args: argparse.Namespace) -> int:
    _print_json(inspect_replay(Path(args.path)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hexo-rl")
    subcommands = parser.add_subparsers(dest="command", required=True)

    test_engine = subcommands.add_parser("test-engine")
    test_engine.set_defaults(func=command_test_engine)

    random_game = subcommands.add_parser("random-game")
    random_game.set_defaults(func=command_random_game)

    selfplay = subcommands.add_parser("selfplay")
    selfplay.add_argument("config")
    selfplay.add_argument("--cycle", type=int, default=1)
    selfplay.set_defaults(func=command_selfplay)

    train = subcommands.add_parser("train")
    train.add_argument("config")
    train.set_defaults(func=command_train)

    loop = subcommands.add_parser("loop")
    loop.add_argument("config")
    loop.add_argument("--cycles", type=int)
    loop.set_defaults(func=command_loop)

    inspect = subcommands.add_parser("inspect-replay")
    inspect.add_argument("path")
    inspect.set_defaults(func=command_inspect_replay)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

