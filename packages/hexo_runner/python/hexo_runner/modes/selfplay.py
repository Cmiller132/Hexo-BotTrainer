"""Pseudo-code self-play boundary for the future Python runner.

Target responsibility split:

Python:
- load config
- load model/checkpoint
- create an evaluator object
- call the redesigned Rust self-play entry point
- validate the returned manifest
- hand validated replay to training

Rust:
- own game rules
- own MCTS/search
- own state encoding
- own replay sample semantics once the contract is defined
- write replay files and the self-play manifest once the contract is defined
"""

from __future__ import annotations


def run_selfplay_cycle() -> None:
    """Design placeholder.

    Pseudo-code:

    config = load_config()
    model = load_checkpoint(config.model)
    evaluator = make_evaluator(model, config.inference)
    manifest = rust.selfplay(config.rust_contract, evaluator)
    validate_manifest(manifest)
    return manifest
    """

    raise NotImplementedError("Python self-play orchestration is being redesigned")
