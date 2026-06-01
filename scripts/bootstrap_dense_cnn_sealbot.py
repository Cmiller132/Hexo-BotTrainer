"""Build a SealBot-bootstrapped dense_cnn prefit checkpoint (current NPZ pipeline).

Reusable, parameterized entry point — re-run this whenever the architecture changes
(it reads the arch from the --config, so 96x6+P7 today, something else tomorrow).

Pipeline (reuses existing components; the only new machinery is parallel generation):
  1. GENERATE in parallel across N worker processes (saturate the 7950X). Each worker
     plays SealBot(best)-vs-SealBot games at a configurable per-turn time budget, and
     records a one-hot policy target on SealBot's chosen move (root prior = same one-hot);
     values come from the game outcome via finalize_game_samples. Samples are written as
     canonical dense self-play NPZ shards with write_selfplay_npz (SAME writer self-play
     uses -> D6/schema handling is correct, no poisoning risk).
  2. PREFIT: train the target model on ALL generated shards with the production
     DenseCNNTrainer step (FP16, real loss) for a few passes.
  3. Save {model_state, optimizer_state, epoch} and verify it loads strict into a fresh
     target model.

Usage:
  python scripts/bootstrap_dense_cnn_sealbot.py \
    --config configs/dense_cnn_model1_target_96x6.toml \
    --out runs/dense_cnn_model1_target_96x6/checkpoints/bootstrap_sealbot_prefit.pt \
    --target-samples 16000 --threads 32 --turn-time-ms 100 --prefit-epochs 8

Note: torch and the trainer are imported lazily (only in the prefit stage) so the
spawned generation workers stay light and start fast.
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
for _pkg in ("hexo_models", "hexo_train", "hexo_engine", "hexo_runner", "hexo_utils"):
    _p = ROOT / "packages" / _pkg / "python"
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


# --- generation worker (top-level + picklable for Windows spawn; no torch) -----------

def _worker_generate(spec: dict) -> dict:
    """Play SealBot games until this worker's sample quota; write NPZ shards."""
    import hexo_engine as engine
    from hexo_models.dense_cnn.samples import sample_from_state, finalize_game_samples
    from hexo_models.dense_cnn.replay import write_selfplay_npz
    from hexo_runner.adapters.sealbot import SealBotConfig, SealBotPlayer
    from hexo_runner.loop import run_match_loop
    from hexo_runner.records import HexoRecordFile, HexoRecordPlayer
    from hexo_runner.player import WorkerContext
    from hexo_runner.session import GameSpec

    wid = int(spec["worker_id"])
    quota = int(spec["quota_samples"])
    horizons = tuple(int(h) for h in spec["horizons"])
    selfplay_dir = Path(spec["selfplay_dir"])
    variant = spec["variant"]
    time_limit = float(spec["time_limit"])
    seed = int(spec["seed"])
    max_actions = int(spec["max_actions"])

    cfg = SealBotConfig(path=spec["sealbot_path"], variant=variant, time_limit=time_limit,
                        response_timeout=max(30.0, time_limit * 20.0 + 5.0))
    cfg.validate()
    sink: list = []
    counter = [0]
    source = f"sealbot_{variant}_bootstrap"

    class _Recorder:
        def __init__(self, inner):
            self.inner = inner
            self.identity = inner.identity
            self.game_id = ""

        def setup_worker(self, ctx):
            self.inner.setup_worker(ctx)

        def start_game(self, ctx):
            self.game_id = ctx.game_id
            self.inner.start_game(ctx)

        def decide(self, state):
            decision = self.inner.decide(state)
            action_id = int(engine.action_id(decision.action))
            ti = int(counter[0])
            sample = sample_from_state(
                state, game_id=self.game_id, turn_index=ti,
                policy={action_id: 1.0}, root_prior_policy={action_id: 1.0},
                metadata={"bootstrap": source, "sample_source": source, "search_visits": 0,
                          "mcts_sims_exact": False},
            )
            sink.append((sample.current_player, sample, 0.0))
            counter[0] = ti + 1
            return decision

        def observe_transition(self, t):
            self.inner.observe_transition(t)

        def finish_game(self, s):
            self.inner.finish_game(s)

        def close(self):
            self.inner.close()

    p0 = _Recorder(SealBotPlayer(cfg, player_id=f"sb-{variant}-w{wid}-p0"))
    p1 = _Recorder(SealBotPlayer(cfg, player_id=f"sb-{variant}-w{wid}-p1"))
    worker = WorkerContext(worker_id=wid, engine_metadata=engine.engine_metadata())
    p0.setup_worker(worker)
    p1.setup_worker(worker)
    players_payload = [
        HexoRecordPlayer(f"sb-w{wid}-p0", "player0", f"SealBot {variant} W{wid} P0"),
        HexoRecordPlayer(f"sb-w{wid}-p1", "player1", f"SealBot {variant} W{wid} P1"),
    ]
    record_path = selfplay_dir.parent / "records" / f"worker_{wid:02d}.hxr"
    record_path.parent.mkdir(parents=True, exist_ok=True)

    games = 0
    written = 0
    winners: dict[str, int] = {}
    try:
        with HexoRecordFile.create(record_path, engine.engine_metadata(), players_payload) as record_file:
            while written < quota:
                sink.clear()
                counter[0] = 0
                game_id = f"sb-w{wid:02d}-g{games:05d}"
                result = run_match_loop(
                    GameSpec(game_id=game_id, seed=seed + games, max_actions=max_actions,
                             mode="bootstrap", is_evaluation=False),
                    (p0, p1), record_file, worker_context=worker,
                    setup_players=False, close_players=False,
                )
                truncated = str(result.status) != "completed" or result.winner is None
                finalized = finalize_game_samples(
                    list(sink),
                    winner=str(result.winner) if result.winner is not None else None,
                    horizons=horizons, truncated=truncated,
                )
                if finalized:
                    shard = selfplay_dir / f"epoch_000000_w{wid:02d}_g{games:05d}.npz"
                    write_selfplay_npz(shard, finalized, raw_rows=len(finalized), epoch=0,
                                       game_id=game_id, short_term_value_horizons=horizons)
                    written += len(finalized)
                games += 1
                if result.winner is not None and not truncated:
                    winners[str(result.winner)] = winners.get(str(result.winner), 0) + 1
    finally:
        p0.close()
        p1.close()
    return {"worker_id": wid, "games": games, "samples": written, "winners": winners}


def generate_parallel(out_dir: Path, *, target_samples: int, threads: int, horizons: tuple[int, ...],
                      sealbot_path: str, variant: str, time_limit: float, seed: int,
                      max_actions: int) -> dict:
    selfplay_dir = out_dir / "selfplay"
    selfplay_dir.mkdir(parents=True, exist_ok=True)
    per_worker = math.ceil(target_samples / threads)
    specs = [
        {"worker_id": w, "quota_samples": per_worker, "horizons": list(horizons),
         "selfplay_dir": str(selfplay_dir), "variant": variant, "time_limit": time_limit,
         "seed": seed + w * 100_000, "max_actions": max_actions, "sealbot_path": sealbot_path}
        for w in range(threads)
    ]
    print(f"[gen] {threads} workers x {per_worker} samples each (target ~{target_samples}) "
          f"@ {int(time_limit*1000)} ms/turn variant={variant}", flush=True)
    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=threads) as ex:
        futures = {ex.submit(_worker_generate, s): s["worker_id"] for s in specs}
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            print(f"  [gen] worker {r['worker_id']:>2} done: {r['samples']} samples / {r['games']} games "
                  f"({done}/{threads} workers)", flush=True)
    elapsed = time.perf_counter() - t0
    total_samples = sum(r["samples"] for r in results)
    total_games = sum(r["games"] for r in results)
    winners: dict[str, int] = {}
    for r in results:
        for k, v in r["winners"].items():
            winners[k] = winners.get(k, 0) + v
    summary = {"threads": threads, "samples": total_samples, "games": total_games,
               "winners": winners, "elapsed_s": round(elapsed, 1),
               "samples_per_s": round(total_samples / max(elapsed, 1e-9), 1)}
    print(f"[gen] DONE: {summary}", flush=True)
    return summary


# --- prefit (torch + trainer imported lazily here) -----------------------------------

def prefit(config, out_dir: Path, *, prefit_epochs: int):
    import torch
    from types import SimpleNamespace
    from hexo_models.dense_cnn.architecture import Model1Network
    from hexo_models.dense_cnn.replay import (
        build_katago_shuffle, shuffle_train_files, latest_shuffle_dir,
        DenseNpzSampleWindow, npz_row_count)
    from hexo_models.dense_cnn.trainer import DenseCNNTrainer

    a = config.architecture
    model = Model1Network(in_channels=a.input_channels, channels=a.channels, blocks=a.residual_blocks,
                          dropout=a.dropout, short_term_value_horizons=a.short_term_value_horizons)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate,
                                 weight_decay=config.training.weight_decay)
    trainer = DenseCNNTrainer(model=model, config=config, optimizer=optimizer)

    selfplay_dir = out_dir / "selfplay"
    raw_shards = sorted(selfplay_dir.glob("epoch_*_w*_g*.npz"))
    if not raw_shards:
        raise SystemExit("no SealBot shards to prefit on")
    raw_rows = sum(npz_row_count(s) for s in raw_shards)
    # Shuffle ACROSS games into full-size shards with the canonical shuffler, so
    # training batches are not single-game-correlated and steps are full (256),
    # not one tiny batch per game. For a bootstrap we want to keep EVERY sample
    # (not a power-law replay window): with taper_exponent=1, expand_per_row=1,
    # min_rows=1 the window formula reduces to exactly usable_rows, and a large
    # keep_target_rows means no cap -> all rows shuffled in once.
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
        trainer.close()  # release the spawn expansion pool (torch workers) at prefit end
    return model, optimizer, {"total_rows": total_rows, "shards": len(shards), "losses": losses}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SealBot-bootstrap a dense_cnn prefit checkpoint.")
    ap.add_argument("--config", required=True, help="target run config (arch is read from here)")
    ap.add_argument("--out", required=True, help="output prefit checkpoint path")
    ap.add_argument("--target-samples", type=int, default=16000)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 16)
    ap.add_argument("--turn-time-ms", type=int, default=100)
    ap.add_argument("--prefit-epochs", type=int, default=8)
    ap.add_argument("--sealbot-path", default=os.environ.get("SEALBOT_PATH", "E:/SealBot"))
    ap.add_argument("--sealbot-variant", default="best")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-actions", type=int, default=1024)
    ap.add_argument("--work-dir", default=None, help="scratch dir for shards (default: <out>.bootstrap_work)")
    ap.add_argument("--skip-generation", action="store_true", help="prefit on existing shards in --work-dir")
    args = ap.parse_args(argv)

    from hexo_models.dense_cnn.config import parse_model1_config
    config = parse_model1_config(tomllib.loads(Path(args.config).read_text(encoding="utf-8"))["model"]["config"])
    horizons = tuple(int(h) for h in config.architecture.short_term_value_horizons)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir) if args.work_dir else out_path.with_suffix(".bootstrap_work")
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bootstrap] config={args.config} arch={config.architecture.channels}x"
          f"{config.architecture.residual_blocks} P7 target_samples={args.target_samples} "
          f"threads={args.threads} turn={args.turn_time_ms}ms prefit_epochs={args.prefit_epochs}", flush=True)

    if args.skip_generation:
        gen = {"skipped": True}
        print("[gen] skipped (reusing existing shards)", flush=True)
    else:
        gen = generate_parallel(work_dir, target_samples=args.target_samples, threads=args.threads,
                                horizons=horizons, sealbot_path=args.sealbot_path,
                                variant=args.sealbot_variant, time_limit=args.turn_time_ms / 1000.0,
                                seed=args.seed, max_actions=args.max_actions)

    model, optimizer, pf = prefit(config, work_dir, prefit_epochs=args.prefit_epochs)

    import torch
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        # epoch=0: this is an INITIALIZE_FROM prefit, not a resume. The pipeline sets
        # start_epoch = checkpoint_epoch + 1, so 0 makes the fresh run begin at epoch 1
        # and run all loop.epochs (the actual prefit-epoch count is kept in metadata).
        "epoch": 0,
        "metadata": {"bootstrap": "sealbot", "prefit_epochs": int(args.prefit_epochs),
                     "generation": gen, "prefit": pf,
                     "arch": {"channels": config.architecture.channels,
                              "blocks": config.architecture.residual_blocks,
                              "policy_head": "fully_conv_P7"}},
    }
    torch.save(payload, out_path)
    print(f"[bootstrap] wrote checkpoint {out_path} ({out_path.stat().st_size/1e6:.1f} MB)", flush=True)

    # Verify: a fresh target model loads this state_dict strict.
    from hexo_models.dense_cnn.architecture import Model1Network
    a = config.architecture
    fresh = Model1Network(in_channels=a.input_channels, channels=a.channels, blocks=a.residual_blocks,
                          dropout=a.dropout, short_term_value_horizons=a.short_term_value_horizons)
    missing, unexpected = fresh.load_state_dict(payload["model_state"], strict=False)
    if missing or unexpected:
        print(f"[bootstrap] FATAL: state_dict mismatch missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}")
        return 1
    fresh.load_state_dict(payload["model_state"], strict=True)
    print(f"[bootstrap] VERIFIED: loads strict into a fresh {a.channels}x{a.residual_blocks} + P7 model "
          f"({sum(p.numel() for p in fresh.parameters()):,} params).")
    (out_path.parent / (out_path.stem + ".bootstrap.json")).write_text(
        json.dumps(payload["metadata"], indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
