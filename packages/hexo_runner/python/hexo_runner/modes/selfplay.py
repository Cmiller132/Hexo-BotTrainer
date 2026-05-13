"""Self-play runner mode.

Self-play generates games from model-backed players and writes detached core
game records. It should not build model tensors itself; model packages write
trainable samples into their own buffers while self-play runs.
"""

from __future__ import annotations


def run_selfplay_cycle(config: object) -> object:
    """Generate self-play records once player, batch, and storage wiring exist.

    Intended flow:

    1. Receive configured `RunnerPlayer` instances from the caller.
    2. Build match configs and call batch mode.
    3. Write detached core game records with position history and runner metadata.
    4. Let model-owned players observe transitions and maintain sample writers.
    5. Return game record manifests and runner result summaries.
    """

    raise NotImplementedError("self-play will be built on batch mode and records.")
