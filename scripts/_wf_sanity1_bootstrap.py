"""Short prefit for dense_cnn_restnet_sanity_1 (owner directive 2026-06-12):
~5k rows from main_3 EPOCH 5 selfplay games SHORTER than 80 decisions (the
healthy, decisive slice of the ckpt5-era corpus), trained for a few passes on
the sanity architecture (policy/value/opp_policy/moves_left — no stvalue).

Reuses the HF bootstrap's prefit half (production shuffle + trainer); only the
data selection differs: instead of replaying human games, copy the selected
main_3 game shards (they already carry 512-visit search policy targets, hard z,
and real moves_left targets — strictly better prefit data than one-hot BC).

Game length = turn_index[0] + moves_left[0] + 1 (decisions; moves_left is the
raw decisions-remaining in schema v4). Truncated games (moves_left < 0) are
excluded by construction (also all >= 80).
"""

from __future__ import annotations

import importlib.util
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _pkg in ("hexo_engine", "hexo_utils", "hexo_runner", "hexo_train", "hexo_models"):
    _p = ROOT / "packages" / _pkg / "python"
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
_rest = ROOT / "packages" / "dense_cnn_restnet" / "python"
if str(_rest) not in sys.path:
    sys.path.insert(0, str(_rest))

MAIN3_SELFPLAY = Path("/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main_3/selfplay")
OUT_DIR = Path("/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_sanity_1_prefit")
OUT_CKPT = OUT_DIR / "restnet_sanity1_prefit.pt"
CONFIG_PATH = ROOT / "configs" / "dense_cnn_restnet_sanity_1.toml"
MAX_DECISIONS = 80
TARGET_ROWS = 5000
PREFIT_EPOCHS = 4

spec = importlib.util.spec_from_file_location(
    "bootstrap_hf", ROOT / "scripts" / "bootstrap_dense_cnn_restnet_hf.py"
)
bootstrap_hf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap_hf)


def select_games() -> list[tuple[Path, int, int]]:
    """Return (npz_path, decisions, rows) for ep5 games shorter than the cap."""
    selected = []
    shards = sorted(MAIN3_SELFPLAY.glob("epoch_000005_game_*.npz"))
    if not shards:
        raise SystemExit(f"no main_3 epoch-5 shards under {MAIN3_SELFPLAY}")
    for shard in shards:
        with np.load(shard) as data:
            ml = float(data["moves_left"][0])
            if ml < 0:  # truncated game: no real length, and poison anyway
                continue
            decisions = int(data["turn_index"][0]) + int(ml) + 1
            rows = int(data["num_rows"])
        if decisions < MAX_DECISIONS:
            selected.append((shard, decisions, rows))
    return selected


def main() -> int:
    games = select_games()
    total_available = sum(rows for _, _, rows in games)
    print(f"[select] {len(games)} games < {MAX_DECISIONS} decisions "
          f"(of 384 ep5 games), {total_available} rows available", flush=True)
    if not games:
        raise SystemExit("no qualifying games")

    rng = random.Random(0)
    rng.shuffle(games)
    chosen, rows_total = [], 0
    for game in games:
        if rows_total >= TARGET_ROWS:
            break
        chosen.append(game)
        rows_total += game[2]
    lengths = sorted(d for _, d, _ in chosen)
    print(f"[select] using {len(chosen)} games / {rows_total} rows "
          f"(target {TARGET_ROWS}); length min/med/max = "
          f"{lengths[0]}/{lengths[len(lengths)//2]}/{lengths[-1]}", flush=True)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    selfplay_dir = OUT_DIR / "selfplay"
    selfplay_dir.mkdir(parents=True)
    for index, (shard, _, _) in enumerate(chosen):
        # Names must match the prefit glob epoch_*_w*_g*.npz.
        shutil.copy2(shard, selfplay_dir / f"epoch_000000_w00_g{index:05d}.npz")

    config = bootstrap_hf._build_config(str(CONFIG_PATH), prefit_batch=0)
    a = config.architecture
    print(f"[prefit] arch: {a.channels}ch {a.blocks_type} horizons={a.short_term_value_horizons} "
          f"moves_left={a.moves_left_head} batch={config.training.batch_size} "
          f"epochs={PREFIT_EPOCHS}", flush=True)
    model, optimizer, pf = bootstrap_hf.prefit(config, OUT_DIR, prefit_epochs=PREFIT_EPOCHS)

    import torch
    from dense_cnn_restnet.architecture import RestnetNetwork

    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": 0,
        "metadata": {
            "bootstrap": "main3_ep5_short_games",
            "source": str(MAIN3_SELFPLAY),
            "max_decisions": MAX_DECISIONS,
            "games": len(chosen),
            "rows": rows_total,
            "prefit_epochs": PREFIT_EPOCHS,
            "prefit": pf,
            "arch": {"channels": a.channels, "blocks_type": a.blocks_type,
                     "short_term_value_horizons": list(a.short_term_value_horizons),
                     "moves_left_head": bool(a.moves_left_head)},
        },
    }
    torch.save(payload, OUT_CKPT)
    print(f"[bootstrap] wrote {OUT_CKPT} ({OUT_CKPT.stat().st_size/1e6:.1f} MB)", flush=True)

    fresh = RestnetNetwork(
        in_channels=a.input_channels, channels=a.channels, blocks_type=a.blocks_type,
        attention_heads=a.attention_heads, mlp_ratio=a.mlp_ratio,
        embed_kernel_size=a.embed_kernel_size, residual_conv=a.residual_conv,
        attention_impl=a.attention_impl, dropout=a.dropout,
        short_term_value_horizons=a.short_term_value_horizons)
    fresh.load_state_dict(payload["model_state"], strict=True)
    print(f"[bootstrap] VERIFIED strict load "
          f"({sum(p.numel() for p in fresh.parameters()):,} params)", flush=True)
    (OUT_DIR / "restnet_sanity1_prefit.bootstrap.json").write_text(
        json.dumps(payload["metadata"], indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
