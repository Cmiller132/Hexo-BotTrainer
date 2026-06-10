"""Pretrain dense_cnn_restnet on the external HuggingFace human corpus.

Dataset: timmyburn/hexo-bootstrap-corpus ("Hexo Human Corpus") — 6902 decisive
human Hex Tac Toe games as JSONL move-lists (NO planes/policy/value targets).
This is NOT training-ready data; it needs conversion via replay + behavioral
cloning.

Pipeline (reuses the dense_cnn_restnet / dense_cnn sample + shuffle + trainer
machinery; the only new code is the human-game -> samples replay):

  1. CONVERT: replay each human game move-by-move through the engine. At each
     position record a compact sample with a ONE-HOT policy target on the human's
     played move (root prior = same one-hot). Per-position value comes from the
     game outcome via finalize_game_samples (winner = the last mover, since a
     six-in-a-row is completed by the player who just moved; cross-checked against
     the engine terminal and the dataset `winner` field). Samples are written as
     canonical dense NPZ shards with write_selfplay_npz (the SAME writer self-play
     uses -> D6/schema handling is correct, no poisoning risk).
  2. PREFIT: build the RestnetNetwork from --config and train it on all shards with
     the production DenseCNNTrainer step for a few passes.
  3. Save {model_state, optimizer_state, epoch=0} and verify a strict load into a
     fresh RestnetNetwork.

Usage:
  python scripts/bootstrap_dense_cnn_restnet_hf.py \
    --config configs/dense_cnn_restnet_main1.toml \
    --jsonl runs/dense_cnn_restnet_main1_prefit/hf_corpus/hexo_human_corpus.jsonl \
    --out  runs/dense_cnn_restnet_main1_prefit/restnet_hf_prefit.pt \
    --threads 24 --prefit-epochs 4 --prefit-batch 128

  # validate-only (replay a few games, check legality/terminal/winner, no write):
  python scripts/bootstrap_dense_cnn_restnet_hf.py --config ... --jsonl ... --validate 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _pkg in ("hexo_engine", "hexo_utils", "hexo_runner", "hexo_train", "hexo_models"):
    _p = ROOT / "packages" / _pkg / "python"
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
# dense_cnn_restnet is a standalone additive package (reuses _rust.dense_cnn).
_rest = ROOT / "packages" / "dense_cnn_restnet" / "python"
if str(_rest) not in sys.path:
    sys.path.insert(0, str(_rest))

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def _load_games(jsonl_path: Path) -> list[dict]:
    games = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                games.append(json.loads(line))
    return games


def _replay_game(game: dict, *, horizons: tuple[int, ...], validate: bool = False):
    """Replay one human game into finalized dense samples.

    Returns (finalized_samples, info_dict) or (None, reason) on a faithfulness
    failure (illegal move / non-terminal / non-decisive).
    """
    import hexo_engine as engine
    from hexo_engine.types import AxialCoord, PlacementAction
    from dense_cnn_restnet.samples import sample_from_state, finalize_game_samples

    moves = game["moves"]
    game_id = str(game.get("game_hash", "hf"))
    state = engine.new_game(seed=0)
    pending = []
    players_seq: list[str] = []
    for ti, mv in enumerate(moves):
        x, y = int(mv[0]), int(mv[1])
        action = PlacementAction(AxialCoord(q=x, r=y))
        if not engine.is_legal_action(state, action):
            return None, f"illegal move idx {ti} {(x, y)} (game {game_id})"
        aid = int(engine.action_id(action))
        sample = sample_from_state(
            state, game_id=game_id, turn_index=ti,
            policy={aid: 1.0}, root_prior_policy={aid: 1.0},
            metadata={"bootstrap": "hf_human", "sample_source": "hf_human",
                      "search_visits": 0, "mcts_sims_exact": False},
        )
        players_seq.append(sample.current_player)
        pending.append((sample.current_player, sample, 0.0))
        engine.apply_action(state, action)

    term = engine.terminal(state)
    if term is None or term.winner is None:
        return None, f"non-terminal/non-decisive after {len(moves)} moves (game {game_id})"
    if not pending:
        return None, f"no samples (game {game_id})"

    engine_winner = str(term.winner)
    lastmover_winner = pending[-1][0]
    # Use the engine terminal winner (authoritative); it matches the sample
    # current_player string space (Player StrEnum: 'player0'/'player1').
    #
    # horizons=() — NOT the architecture horizons: short-term-value targets are
    # EMAs of future MCTS root values, and human replay has no search, so every
    # root_value above is 0.0. Passing real horizons here would emit stvalue
    # targets of exactly 0 with mask=1, training the stvalue heads (and, through
    # them, the trunk) toward a constant during the whole prefit. With no
    # short_term_value on the samples the shard writer (which still receives the
    # architecture horizons for schema parity) writes stvalue_mask=0, so the
    # masked stvalue loss contributes exactly nothing.
    finalized = finalize_game_samples(pending, winner=engine_winner, horizons=(), truncated=False)
    info = {
        "moves": len(moves),
        "samples": len(finalized),
        "engine_winner": engine_winner,
        "lastmover_winner": lastmover_winner,
        "lastmover_agrees": engine_winner == lastmover_winner,
        "dataset_winner": int(game.get("winner", 0)),
        "first_player": players_seq[0] if players_seq else None,
    }
    return finalized, info


def _worker_convert(spec: dict) -> dict:
    """Replay a slice of games and write one NPZ shard per game."""
    from dense_cnn_restnet.replay import write_selfplay_npz

    wid = int(spec["worker_id"])
    horizons = tuple(int(h) for h in spec["horizons"])
    selfplay_dir = Path(spec["selfplay_dir"])
    games = spec["games"]

    written = 0
    n_games = 0
    skipped = 0
    skip_reasons: list[str] = []
    # winner mapping cross-check: dataset_winner(1/-1) -> first_player(winner?)
    map_counts: dict[str, int] = {}
    lastmover_disagree = 0
    for gi, game in enumerate(games):
        finalized, info = _replay_game(game, horizons=horizons, validate=True)
        if finalized is None:
            skipped += 1
            if len(skip_reasons) < 5:
                skip_reasons.append(info)
            continue
        # cross-check: does dataset winner=1 correspond to engine_winner==player0?
        dsw = info["dataset_winner"]
        key = f"ds{dsw:+d}->{info['engine_winner']}"
        map_counts[key] = map_counts.get(key, 0) + 1
        if not info["lastmover_agrees"]:
            lastmover_disagree += 1
        shard = selfplay_dir / f"epoch_000000_w{wid:02d}_g{gi:05d}.npz"
        write_selfplay_npz(shard, finalized, raw_rows=len(finalized), epoch=0,
                           game_id=str(game.get("game_hash", f"w{wid}g{gi}")),
                           short_term_value_horizons=horizons)
        written += len(finalized)
        n_games += 1
    return {"worker_id": wid, "games": n_games, "samples": written, "skipped": skipped,
            "skip_reasons": skip_reasons, "map_counts": map_counts,
            "lastmover_disagree": lastmover_disagree}


def convert_parallel(out_dir: Path, games: list[dict], *, threads: int, horizons: tuple[int, ...]) -> dict:
    selfplay_dir = out_dir / "selfplay"
    selfplay_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[list[dict]] = [games[i::threads] for i in range(threads)]
    specs = [{"worker_id": w, "games": chunks[w], "horizons": list(horizons),
              "selfplay_dir": str(selfplay_dir)} for w in range(threads) if chunks[w]]
    print(f"[convert] {len(specs)} workers over {len(games)} games", flush=True)
    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=len(specs)) as ex:
        futs = {ex.submit(_worker_convert, s): s["worker_id"] for s in specs}
        done = 0
        for fut in as_completed(futs):
            r = fut.result(); results.append(r); done += 1
            print(f"  [convert] worker {r['worker_id']:>2} done: {r['samples']} samples / {r['games']} games "
                  f"({r['skipped']} skipped) [{done}/{len(specs)}]", flush=True)
    elapsed = time.perf_counter() - t0
    total_samples = sum(r["samples"] for r in results)
    total_games = sum(r["games"] for r in results)
    total_skipped = sum(r["skipped"] for r in results)
    lastmover_disagree = sum(r["lastmover_disagree"] for r in results)
    map_counts: dict[str, int] = {}
    for r in results:
        for k, v in r["map_counts"].items():
            map_counts[k] = map_counts.get(k, 0) + v
    sample_reasons = [s for r in results for s in r["skip_reasons"]][:10]
    summary = {"games_in": len(games), "games_converted": total_games, "samples": total_samples,
               "skipped": total_skipped, "lastmover_disagree": lastmover_disagree,
               "winner_map_counts": map_counts, "skip_reasons_sample": sample_reasons,
               "elapsed_s": round(elapsed, 1)}
    print(f"[convert] DONE: {json.dumps(summary, indent=2)}", flush=True)
    return summary


def prefit(config, out_dir: Path, *, prefit_epochs: int):
    import torch
    from types import SimpleNamespace
    from dense_cnn_restnet.architecture import RestnetNetwork
    from dense_cnn_restnet.replay import (
        build_katago_shuffle, shuffle_train_files, latest_shuffle_dir,
        DenseNpzSampleWindow, npz_row_count)
    from dense_cnn_restnet.trainer import DenseCNNTrainer

    a = config.architecture
    model = RestnetNetwork(
        in_channels=a.input_channels, channels=a.channels, blocks_type=a.blocks_type,
        attention_heads=a.attention_heads, mlp_ratio=a.mlp_ratio,
        embed_kernel_size=a.embed_kernel_size, residual_conv=a.residual_conv,
        attention_impl=a.attention_impl, dropout=a.dropout,
        short_term_value_horizons=a.short_term_value_horizons)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate,
                                 weight_decay=config.training.weight_decay)
    trainer = DenseCNNTrainer(model=model, config=config, optimizer=optimizer)

    selfplay_dir = out_dir / "selfplay"
    raw_shards = sorted(selfplay_dir.glob("epoch_*_w*_g*.npz"))
    if not raw_shards:
        raise SystemExit("no converted shards to prefit on")
    raw_rows = sum(npz_row_count(s) for s in raw_shards)
    shuffle_result = build_katago_shuffle(
        selfplay_dir=selfplay_dir, shuffled_root=out_dir / "shuffleddata",
        scratch_dir=out_dir / "shufflescratch", epoch=0, seed=0,
        min_rows=1, keep_target_rows=raw_rows * 4,
        taper_window_exponent=1.0, expand_window_per_row=1.0, taper_window_scale=None,
        approx_rows_per_out_file=4000, batch_size=config.training.batch_size,
        worker_group_size=4000, validation_fraction=0.0)
    shuffle_dir = shuffle_result.shuffle_dir or latest_shuffle_dir(out_dir / "shuffleddata")
    if shuffle_dir is None:
        raise SystemExit("shuffle produced no output")
    shards = list(shuffle_train_files(shuffle_dir))
    total_rows = sum(npz_row_count(s) for s in shards)
    print(f"  [prefit] shuffled {raw_rows} raw rows ({len(raw_shards)} game shards) -> "
          f"{total_rows} rows in {len(shards)} shuffled shards", flush=True)
    window = DenseNpzSampleWindow(
        files=tuple(shards), seed=0, epoch=0,
        index=SimpleNamespace(sample_count=total_rows, store=shuffle_dir),
        window_size=total_rows, target_rows=total_rows, shuffle_dir=shuffle_dir,
        metadata={"bootstrap": True})
    ctx = SimpleNamespace(output_dir=out_dir, config=SimpleNamespace(run=SimpleNamespace(seed=0)))
    components = SimpleNamespace(shared=SimpleNamespace())
    losses = []
    try:
        for ep in range(prefit_epochs):
            res = trainer.train_passes(passes=1, sample_window=window, sample_symmetries=None,
                                       ctx=ctx, components=components, epoch=ep)
            losses.append(res.get("loss"))
            print(f"  [prefit] epoch {ep}: status={res.get('status')} steps={res.get('steps')} "
                  f"loss={res.get('loss')}", flush=True)
    finally:
        trainer.close()
    return model, optimizer, {"total_rows": total_rows, "shards": len(shards), "losses": losses}


def _build_config(config_path: str, prefit_batch: int | None):
    from dataclasses import replace
    from dense_cnn_restnet.config import parse_model1_config
    raw = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))["model"]["config"]
    config = parse_model1_config(raw)
    if prefit_batch and prefit_batch > 0:
        config = replace(config, training=replace(config.training, batch_size=int(prefit_batch)))
    return config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pretrain dense_cnn_restnet on the HF human corpus.")
    ap.add_argument("--config", required=True, help="restnet run config (arch read from here)")
    ap.add_argument("--jsonl", required=True, help="path to hexo_human_corpus.jsonl")
    ap.add_argument("--out", help="output prefit checkpoint path (required unless --validate)")
    ap.add_argument("--threads", type=int, default=min(24, os.cpu_count() or 16))
    ap.add_argument("--prefit-epochs", type=int, default=4)
    ap.add_argument("--prefit-batch", type=int, default=0, help="override training batch for prefit (0=config)")
    ap.add_argument("--max-games", type=int, default=0, help="cap games (0=all)")
    ap.add_argument("--work-dir", default=None, help="scratch dir for shards (default: <out>.bootstrap_work)")
    ap.add_argument("--skip-generation", action="store_true", help="prefit on existing shards in --work-dir")
    ap.add_argument("--validate", type=int, default=0, help="replay N games, check, and exit (no write/prefit)")
    args = ap.parse_args(argv)

    config = _build_config(args.config, args.prefit_batch)
    horizons = tuple(int(h) for h in config.architecture.short_term_value_horizons)
    jsonl = Path(args.jsonl)

    if args.validate:
        games = _load_games(jsonl)[: args.validate]
        print(f"[validate] replaying {len(games)} games...", flush=True)
        ok = 0; agree = 0; mapc: dict[str, int] = {}; total_samples = 0
        for g in games:
            fin, info = _replay_game(g, horizons=horizons, validate=True)
            if fin is None:
                print(f"  FAIL: {info}", flush=True); continue
            ok += 1; total_samples += info["samples"]
            agree += int(info["lastmover_agrees"])
            mapc[f"ds{info['dataset_winner']:+d}->{info['engine_winner']}"] = \
                mapc.get(f"ds{info['dataset_winner']:+d}->{info['engine_winner']}", 0) + 1
        print(f"[validate] ok={ok}/{len(games)} samples={total_samples} "
              f"lastmover==engine_winner: {agree}/{ok}", flush=True)
        print(f"[validate] winner mapping (dataset -> engine): {json.dumps(mapc, indent=2)}", flush=True)
        return 0

    if not args.out:
        ap.error("--out is required unless --validate")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir) if args.work_dir else out_path.with_suffix(".bootstrap_work")
    work_dir.mkdir(parents=True, exist_ok=True)

    a = config.architecture
    print(f"[bootstrap] restnet arch: {a.channels}ch blocks_type={a.blocks_type} "
          f"heads={a.attention_heads} impl={a.attention_impl} conv={a.residual_conv} "
          f"prefit_epochs={args.prefit_epochs} prefit_batch={config.training.batch_size}", flush=True)

    if args.skip_generation:
        conv = {"skipped": True}
        print("[convert] skipped (reusing existing shards)", flush=True)
    else:
        games = _load_games(jsonl)
        if args.max_games:
            games = games[: args.max_games]
        conv = convert_parallel(work_dir, games, threads=args.threads, horizons=horizons)
        # Faithfulness gate: bail if too many games failed to replay or winner
        # cross-check is inconsistent (would silently poison the value targets).
        if conv["games_converted"] == 0:
            print("[bootstrap] FATAL: zero games converted", flush=True); return 1
        fail_frac = conv["skipped"] / max(1, conv["games_in"])
        if fail_frac > 0.10:
            print(f"[bootstrap] FATAL: {fail_frac:.1%} of games failed to replay "
                  f"(reasons: {conv['skip_reasons_sample']}) — coordinate/engine mismatch?", flush=True)
            return 1
        if conv["lastmover_disagree"] > 0:
            print(f"[bootstrap] WARN: {conv['lastmover_disagree']} games where last-mover != engine winner "
                  "(unexpected for six-in-a-row; value targets still use engine winner).", flush=True)

    model, optimizer, pf = prefit(config, work_dir, prefit_epochs=args.prefit_epochs)

    import torch
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": 0,
        "metadata": {"bootstrap": "hf_human_corpus", "dataset": "timmyburn/hexo-bootstrap-corpus",
                     "prefit_epochs": int(args.prefit_epochs), "conversion": conv, "prefit": pf,
                     "arch": {"channels": a.channels, "blocks_type": a.blocks_type,
                              "attention_heads": a.attention_heads, "residual_conv": a.residual_conv}},
    }
    torch.save(payload, out_path)
    print(f"[bootstrap] wrote checkpoint {out_path} ({out_path.stat().st_size/1e6:.1f} MB)", flush=True)

    # Verify strict load into a fresh RestnetNetwork.
    from dense_cnn_restnet.architecture import RestnetNetwork
    fresh = RestnetNetwork(
        in_channels=a.input_channels, channels=a.channels, blocks_type=a.blocks_type,
        attention_heads=a.attention_heads, mlp_ratio=a.mlp_ratio,
        embed_kernel_size=a.embed_kernel_size, residual_conv=a.residual_conv,
        attention_impl=a.attention_impl, dropout=a.dropout,
        short_term_value_horizons=a.short_term_value_horizons)
    missing, unexpected = fresh.load_state_dict(payload["model_state"], strict=False)
    if missing or unexpected:
        print(f"[bootstrap] FATAL: state_dict mismatch missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}")
        return 1
    fresh.load_state_dict(payload["model_state"], strict=True)
    print(f"[bootstrap] VERIFIED: loads strict into a fresh restnet "
          f"({sum(p.numel() for p in fresh.parameters()):,} params).", flush=True)
    (out_path.parent / (out_path.stem + ".bootstrap.json")).write_text(
        json.dumps(payload["metadata"], indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
