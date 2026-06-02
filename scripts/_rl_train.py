"""BC-seeded RL self-play training for hexgt (Model 2).

Closes the loop: self-play (game-driven, dynamic-GNN MCTS) -> compact shards ->
train (HexgtTrainer over a sliding replay window) -> checkpoint -> periodic
head-to-head eval vs dense_cnn epoch-24 (and optional SealBot). Seeded from the
converged BC checkpoint (step-6009). RAM-disciplined (stream shard groups, free
between groups) and crash-safe + RESUMABLE (every epoch writes
`hexgt_rl_latest.pt` carrying model + optimizer + train-state + rl_epoch; a
restart picks up where it left off) so a supervisor can run it unattended.

Usage (under the supervisor, see _rl_supervise.sh):
  python scripts/_rl_train.py --bc-seed runs/hexgt_bc/hexgt_bc_step006009.pt \
      --out-dir runs/hexgt_rl --epochs 40 --games-per-epoch 64
"""

from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from hexo_models.hexgt.architecture import HexgtNetwork, zero_init_expanded_feature_columns
from hexo_models.hexgt.config import parse_hexgt_config
from hexo_models.hexgt.constants import FEATURE_SCHEMA_VERSION
from hexo_models.hexgt.selfplay import run_selfplay_games
from hexo_models.hexgt.trainer import HexgtTrainer


def log(msg, fh=None):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n"); fh.flush()


def build_model(arch_meta, device):
    return HexgtNetwork(
        token_dim=arch_meta["token_dim"], gnn_layers=arch_meta["gnn_layers"],
        ctx_layers=arch_meta["ctx_layers"], ffn_dim=arch_meta["ffn_dim"],
        attention_heads=arch_meta["attention_heads"],
        short_term_value_horizons=tuple(arch_meta["short_term_value_horizons"]),
    ).to(device)


def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def eval_due(rl_epoch, eval_every, total_epochs):
    """Is an eval scheduled at this ABSOLUTE epoch? Keyed off the absolute epoch
    (not a resume-relative counter) so a restart can never shift the schedule."""
    return (rl_epoch % eval_every == 0) or (rl_epoch == total_epochs - 1)


def eval_result_path(eval_dir, rl_epoch):
    return Path(eval_dir) / f"epoch_{rl_epoch:06d}_eval.json"


def eval_missing(eval_dir, rl_epoch):
    """An eval is 'missing' if its result JSON was never written (e.g. a restart
    killed the process mid-eval). The JSON is written only after the eval fully
    completes, so its absence is the authoritative 'not done' marker."""
    return not eval_result_path(eval_dir, rl_epoch).exists()


def select_window_epochs(current_epoch, epoch_positions, *, base_window, pool_cap):
    """Recent epochs (newest->oldest) whose cumulative positions fit under
    ``pool_cap``, always including at least ``base_window`` newest epochs. The
    window GROWS over training up to the cap, then drops the oldest. The callable
    ``epoch_positions(e)`` returns the position count of epoch ``e``. Returns
    ``(sorted_epochs, total_positions)``.

    RAM-safety by construction: this returns only an epoch LIST (the driver then
    lists shard PATHS for them) — no shard CONTENTS are loaded here. The pool stays
    disk-resident; the driver reads ONE ~group-sized batch of shards at a time at
    train-read (expand -> train -> free), so peak RAM is bounded by one group +
    model + optimizer, NOT by the (up to ``pool_cap``) pool. ``epoch_positions`` is
    queried lazily once per candidate epoch and is expected to read only cheap
    metadata (a one-line self-play summary, or shard ``num_rows`` headers)."""

    epochs: list[int] = []
    total = 0
    for e in range(int(current_epoch), -1, -1):
        pos = max(0, int(epoch_positions(e)))
        within_floor = (current_epoch - e) < base_window
        if epochs and not within_floor and total + pos > pool_cap:
            break
        epochs.append(e)
        total += pos
    return sorted(epochs), total


def epoch_recency_weight(epoch, current_epoch, decay):
    """Geometric recency weight ``decay^(current_epoch - epoch)``: the newest epoch
    weighs 1.0, one epoch old ``decay``, two ``decay^2`` … so sampling shards
    proportional to it over-represents recent self-play and decays old games out."""
    return float(decay) ** max(0, int(current_epoch) - int(epoch))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-seed", default="/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_bc/hexgt_bc_step006009.pt")
    ap.add_argument("--out-dir", default="/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--games-per-epoch", type=int, default=64)
    ap.add_argument("--active", type=int, default=48)
    ap.add_argument("--vbatch", type=int, default=64)
    ap.add_argument("--visits", type=int, default=128)
    # Self-play move cap. 512 (was 240) lets long-tail games run to their natural
    # terminal — the 240 cap truncated real games (max length pegged at 240) and a
    # truncated game is not a true terminal, distorting the length distribution +
    # value targets. Eval uses its own higher cap (--eval-max-actions=1024).
    ap.add_argument("--max-actions", type=int, default=512)
    ap.add_argument("--train-steps-per-epoch", type=int, default=400)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--replay-window-epochs", type=int, default=8)  # MINIMUM window (floor)
    # Recency-weighted, cap-bounded replay (KataGo-style). The window grows beyond
    # the floor to include as many recent epochs as fit under --replay-pool-cap
    # positions (dropping the oldest); shards are sampled PROPORTIONAL to
    # decay^(age_in_epochs), so recent epochs are over-represented and old ones
    # decay out. RAM stays bounded by ONE shard group (pool is disk-resident).
    ap.add_argument("--replay-recency-decay", type=float, default=0.9)
    ap.add_argument("--replay-pool-cap", type=int, default=500_000)
    ap.add_argument("--group-size", type=int, default=30)
    ap.add_argument("--n", type=int, default=3)
    # Self-play exploration config (the ablation-chosen knobs).
    ap.add_argument("--total-alpha", type=float, default=6.6)        # Dirichlet sum (derived)
    ap.add_argument("--eps", type=float, default=0.25)               # root noise fraction
    ap.add_argument("--root-policy-temperature", type=float, default=1.0)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--final-temperature", type=float, default=0.2)
    ap.add_argument("--temperature-decay-moves", type=int, default=30)
    ap.add_argument("--temperature-floor", type=float, default=0.1)
    ap.add_argument("--forced-playout-k", type=float, default=2.0)
    # MCTS nucleus widening. Defaults match configs/hexgt_model2.toml (documented
    # intent) + dense_cnn's 96x8 run. The driver previously passed NONE of these,
    # so they silently fell to the parse defaults (max_children=32), narrowing
    # search vs the intended 96.
    ap.add_argument("--widening-max-children", type=int, default=96)
    ap.add_argument("--widening-min-children", type=int, default=2)
    ap.add_argument("--widening-policy-mass", type=float, default=0.95)
    # Eval
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--eval-games", type=int, default=40)
    ap.add_argument("--eval-visits", type=int, default=200)
    ap.add_argument("--eval-opening-moves", type=int, default=10)
    ap.add_argument("--eval-opening-temperature", type=float, default=0.6)
    # Eval games run to completion (a long Hexo game truncated to a draw would
    # deflate the win rate); keep this well above the self-play max_actions.
    ap.add_argument("--eval-max-actions", type=int, default=1024)
    ap.add_argument("--dense-ckpt", default="/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8/checkpoints/epoch_000024.pt")
    ap.add_argument("--dense-config", default="/mnt/e/Hexo-BotTrainer-hexgt/configs/dense_cnn_model1_target_96x8.toml")
    ap.add_argument("--sealbot", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    selfplay_dir = out_dir / "selfplay"; selfplay_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = out_dir / "eval"; eval_dir.mkdir(parents=True, exist_ok=True)
    fh = open(out_dir / "rl_train.log", "a")
    rng = random.Random(args.seed)

    cfg = parse_hexgt_config({
        "device": str(device),
        "architecture": {"candidate_radius": args.n},
        "training": {"learning_rate": args.lr, "warmup_steps": args.warmup, "batch_size": args.batch},
        "selfplay": {
            "search_visits": args.visits, "max_actions": args.max_actions,
            "c_puct": args.c_puct, "root_policy_temperature": args.root_policy_temperature,
            "root_dirichlet_noise_enabled": True, "root_dirichlet_total_alpha": args.total_alpha,
            "root_dirichlet_noise_fraction": args.eps,
            "temperature": args.temperature, "final_temperature": args.final_temperature,
            "temperature_decay_moves": args.temperature_decay_moves,
            "temperature_floor": args.temperature_floor,
            "forced_playout_k": args.forced_playout_k,
            "widening_max_children": args.widening_max_children,
            "widening_min_children": args.widening_min_children,
            "widening_policy_mass": args.widening_policy_mass,
        },
    })

    # --- model: resume from RL latest if present, else seed from BC -----------
    latest = ckpt_dir / "hexgt_rl_latest.pt"
    start_epoch = 0
    if latest.exists():
        ck = torch.load(latest, map_location=device, weights_only=False)
        model = build_model(ck["arch"], device)
        model.load_state_dict(ck["model"])
        arch_meta = ck["arch"]
        start_epoch = int(ck.get("rl_epoch", 0)) + 1
        seed_desc = f"RESUME from {latest.name} (rl_epoch={ck.get('rl_epoch')}, step={ck.get('step')})"
    else:
        ck = torch.load(args.bc_seed, map_location=device, weights_only=False)
        arch_meta = ck["arch"]
        model = build_model(arch_meta, device)
        model.load_state_dict(ck["model"])
        seed_desc = (f"SEED from {Path(args.bc_seed).name} (step={ck.get('step')}"
                     f"{', rl_epoch=' + str(ck['rl_epoch']) if 'rl_epoch' in ck else ''})")

    # ZERO-INIT LAYER-EXPANSION (one-time): a checkpoint predating the current
    # feature schema (no/lower feature_schema_version) carries random, never-
    # trained node_in columns for the newly-activated slots. Zero them so the
    # FIRST forward after resume is byte-identical to that checkpoint; the next
    # save stamps the current version, so later resumes skip this. Gated on the
    # ABSOLUTE version in the checkpoint, so the supervisor's fixed relaunch
    # command can't re-zero already-learned columns.
    loaded_fsv = int(ck.get("feature_schema_version", 1))
    if loaded_fsv < FEATURE_SCHEMA_VERSION:
        zeroed = zero_init_expanded_feature_columns(model)
        seed_desc += (f" + ZERO-INIT feature-expansion v{loaded_fsv}->v{FEATURE_SCHEMA_VERSION} "
                      f"(zeroed {len(zeroed)} node_in cols {zeroed})")

    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    trainer = HexgtTrainer(model=model, config=cfg, optimizer=opt)
    # Load optimizer momentum from the seed/resume checkpoint when present (the BC
    # seed has none -> fresh momentum; an RL-epoch checkpoint does -> warm-start).
    if "optimizer" in ck:
        try:
            opt.load_state_dict(ck["optimizer"])
            log(f"  (optimizer state restored from {Path(args.bc_seed).name if not latest.exists() else latest.name})", fh)
        except Exception as exc:  # optimizer state shape drift -> start fresh momentum
            log(f"  (optimizer state not restored: {exc})", fh)
    if latest.exists() and "train_state" in ck:
        trainer.load_train_state(ck["train_state"])

    compiled = (not args.no_compile) and device.type == "cuda"
    if compiled:
        model.forward_policy_value = torch.compile(model.forward_policy_value, dynamic=True)

    log(f"=== hexgt RL start: {seed_desc} | params={nparams:,} n={args.n} device={device} "
        f"compile={compiled} ===", fh)
    log(f"    epochs={args.epochs} games/epoch={args.games_per_epoch} visits={args.visits} "
        f"active={args.active} vbatch={args.vbatch} | train_steps/epoch={args.train_steps_per_epoch} "
        f"batch={args.batch} lr={args.lr} replay_window={args.replay_window_epochs} ep", fh)

    def save(tag, rl_epoch):
        payload = {
            "model": model.state_dict(), "optimizer": opt.state_dict(),
            "train_state": trainer.train_state.to_dict(), "arch": arch_meta,
            "step": trainer.train_state.global_step, "rl_epoch": rl_epoch,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
        }
        torch.save(payload, ckpt_dir / f"hexgt_rl_{tag}.pt")
        torch.save(payload, ckpt_dir / "hexgt_rl_latest.pt")

    _epoch_pos_cache: dict[int, int] = {}

    def epoch_positions(e):
        """Position count for epoch ``e`` (memoized). Prefers the one-file self-play
        summary (``searched_positions``); falls back to summing shard ``num_rows``
        headers. Only COMPLETED epochs are queried (self-play finishes before the
        train step), so the count is stable and safe to cache. Reads only metadata
        — never shard contents — so the pool stays disk-resident."""
        if e in _epoch_pos_cache:
            return _epoch_pos_cache[e]
        n = 0
        spj = eval_dir / f"epoch_{e:06d}_selfplay.json"
        if spj.exists():
            try:
                with open(spj) as f:
                    n = int(json.load(f).get("searched_positions") or 0)
            except Exception:
                n = 0
        if not n:
            for sh in selfplay_dir.glob(f"epoch_{e:06d}_game_*.npz"):
                try:
                    with np.load(sh) as z:
                        n += int(z["num_rows"])
                except Exception:
                    pass
        _epoch_pos_cache[e] = n
        return n

    def build_replay_window(current_epoch):
        """Recency-weighted, cap-bounded replay window for ``current_epoch``.
        Returns ``(shards, weights, epochs, total_pos)`` where ``weights[i]`` is the
        recency weight of ``shards[i]``'s epoch (the driver samples groups
        proportional to these). Holds only PATHS + floats — no shard data."""
        epochs, total = select_window_epochs(
            current_epoch, epoch_positions,
            base_window=args.replay_window_epochs, pool_cap=args.replay_pool_cap,
        )
        shards: list = []
        weights: list = []
        for e in epochs:
            w = epoch_recency_weight(e, current_epoch, args.replay_recency_decay)
            for sh in sorted(selfplay_dir.glob(f"epoch_{e:06d}_game_*.npz")):
                shards.append(sh)
                weights.append(w)
        return shards, weights, epochs, total

    def run_eval(rl_epoch):
        # Lazy import keeps the dense_cnn eval deps off the hot path.
        import tomllib
        from hexo_models.hexgt.evaluation import make_hexgt_factory, run_head_to_head

        cfg_eval = parse_hexgt_config({"device": str(device), "architecture": {"candidate_radius": args.n},
                                       "selfplay": {"search_visits": args.eval_visits}})
        model.eval()
        # Slight opening variety decorrelates the eval games -> smoother win rate
        # (still deterministic/repeatable via the per-(game,move) seed).
        make_hexgt = make_hexgt_factory(model, cfg_eval, device=str(device),
                                        fp16=(device.type == "cuda"), identity_id="hexgt-rl",
                                        opening_moves=args.eval_opening_moves,
                                        opening_temperature=args.eval_opening_temperature)
        results = {}
        # vs dense_cnn e24
        try:
            from hexo_models.dense_cnn.config import parse_model1_config
            from hexo_models.dense_cnn.architecture import Model1Network
            from hexo_models.dense_cnn.trainer import DenseCNNTrainer
            from hexo_models.dense_cnn.player import DenseCNNPlayer
            with open(args.dense_config, "rb") as dfh:
                toml = tomllib.load(dfh)
            mc = dict(toml["model"]["config"]); mc["device"] = str(device)
            mc["selfplay"] = {**mc.get("selfplay", {}), "search_visits": int(args.eval_visits)}
            dcfg = parse_model1_config(mc); darch = dcfg.architecture
            dmodel = Model1Network(in_channels=darch.input_channels, channels=darch.channels,
                                   blocks=darch.residual_blocks, dropout=darch.dropout,
                                   short_term_value_horizons=darch.short_term_value_horizons)
            dck = torch.load(args.dense_ckpt, map_location="cpu", weights_only=False)
            dmodel.load_state_dict(dck["model_state"])
            dopt = torch.optim.AdamW(dmodel.parameters(), lr=1e-4)
            dtrainer = DenseCNNTrainer(model=dmodel, config=dcfg, optimizer=dopt)

            def make_dense(seed):
                return DenseCNNPlayer(identity_id="dense-cnn-e24", model=dmodel, trainer=dtrainer,
                                      record_samples=False, eval_seed=seed)
            # FIXED eval seed across epochs -> the SAME games each time, so a
            # win-rate change is the model improving, not seed variance (paired).
            r = run_head_to_head(make_hexgt, make_dense, games=args.eval_games,
                                 output_dir=eval_dir / f"epoch_{rl_epoch:06d}_vs_dense",
                                 base_seed=4242, max_actions=args.eval_max_actions,
                                 game_id_prefix=f"e{rl_epoch}vsdense")
            results["vs_dense_cnn_e24"] = r.as_dict()
        except Exception:
            results["vs_dense_cnn_e24_error"] = traceback.format_exc()
        # vs SealBot (optional)
        if args.sealbot:
            try:
                from hexo_runner.adapters.sealbot import SealBotConfig, SealBotPlayer
                sb = SealBotConfig(variant="best", time_limit=0.05); sb.validate()

                def make_sb(_seed):
                    return SealBotPlayer(sb, player_id="sealbot-best-50ms")
                rs = run_head_to_head(make_hexgt, make_sb, games=args.eval_games,
                                      output_dir=eval_dir / f"epoch_{rl_epoch:06d}_vs_sealbot",
                                      base_seed=4242, max_actions=args.eval_max_actions,
                                      game_id_prefix=f"e{rl_epoch}vssb")
                results["vs_sealbot"] = rs.as_dict()
            except Exception as exc:
                results["vs_sealbot_error"] = str(exc)
        model.train()
        return results

    def do_eval(rl_epoch):
        """Run the periodic eval for `rl_epoch`, write its result JSON, and log it.
        Shared by the in-loop scheduler and the resume backfill so a restart never
        skips a due eval."""
        ev = run_eval(rl_epoch)
        with open(eval_result_path(eval_dir, rl_epoch), "w") as ev_fh:
            json.dump(ev, ev_fh, indent=2)
        vd = ev.get("vs_dense_cnn_e24", {})
        msg = (f"epoch {rl_epoch} EVAL vs dense_cnn e24: "
               f"{vd.get('wins')}W/{vd.get('losses')}L/{vd.get('draws')}D "
               f"= {vd.get('win_rate', float('nan')):.1%} (visits={args.eval_visits}, games={args.eval_games})")
        if "vs_sealbot" in ev:
            vs = ev["vs_sealbot"]
            msg += (f" | vs SealBot: {vs.get('wins')}W/{vs.get('losses')}L/{vs.get('draws')}D "
                    f"= {vs.get('win_rate', float('nan')):.1%}")
        log(f">>> {msg}", fh)

    last_save = None
    try:
        # Pre-training BC-seed baseline at the exact eval settings -> the anchor
        # the RL trend is measured against (the documented BC h2h was 55% @ 40
        # games/visits=200; this re-confirms it at our 24-game/visits-200 setting).
        if start_epoch == 0:
            base_ev = run_eval(-1)
            with open(eval_dir / "baseline_bc_eval.json", "w") as bfh:
                json.dump(base_ev, bfh, indent=2)
            bd = base_ev.get("vs_dense_cnn_e24", {})
            bmsg = (f">>> BC-SEED BASELINE (pre-RL) vs dense_cnn e24: "
                    f"{bd.get('wins')}W/{bd.get('losses')}L/{bd.get('draws')}D "
                    f"= {bd.get('win_rate', float('nan')):.1%} (visits={args.eval_visits}, games={args.eval_games})")
            if "vs_sealbot" in base_ev:
                bs = base_ev["vs_sealbot"]
                bmsg += (f" | vs SealBot: {bs.get('wins')}W/{bs.get('losses')}L/{bs.get('draws')}D "
                         f"= {bs.get('win_rate', float('nan')):.1%}")
            log(bmsg, fh)

        # Resume eval-safety: if we resumed PAST an epoch whose eval was due but
        # never completed (e.g. a restart killed the process mid-eval), run it now
        # rather than skipping ahead — so a restart never drops a scheduled eval.
        prev = start_epoch - 1
        if prev >= 0 and eval_due(prev, args.eval_every, args.epochs) and eval_missing(eval_dir, prev):
            log(f"resume: backfilling missing eval for epoch {prev} (was due, no result on disk)", fh)
            do_eval(prev)

        for rl_epoch in range(start_epoch, args.epochs):
            ep_t0 = time.perf_counter()
            # --- 1) self-play ----------------------------------------------------
            model.eval()
            sp = run_selfplay_games(
                model, cfg, num_games=args.games_per_epoch, output_dir=selfplay_dir,
                epoch=rl_epoch, device=str(device), fp16=(device.type == "cuda"),
                base_seed=1000 + rl_epoch * 7919, active_games=args.active,
                virtual_batch_size=args.vbatch, collect_examples=3,
            )
            log(f"epoch {rl_epoch} selfplay: {sp.completed_games}C/{sp.truncated_games}T games, "
                f"{sp.searched_positions} pos, {sp.positions_per_second:.1f} pos/s | "
                f"cand={sp.mean_candidate_count:.0f} | "
                f"Q1 decisive={sp.decisive_fraction:.1%} len_med={sp.game_length_median:.0f} | "
                f"Q2 uniq_open={sp.opening_unique_fraction:.1%} m2H={sp.move2_entropy:.2f} | "
                f"Q3 visitH={sp.mean_visit_entropy:.2f} priorH={sp.mean_prior_entropy:.2f} | "
                f"Q4 |val|={sp.mean_abs_value:.2f} draw={sp.draw_fraction:.1%} | "
                f"Q5 forced={sp.forced_move_fraction:.1%}", fh)
            # Also persist the full self-play metric dict per epoch for later analysis.
            with open(eval_dir / f"epoch_{rl_epoch:06d}_selfplay.json", "w") as sp_fh:
                json.dump(sp.as_dict(), sp_fh, indent=2)
            # Dump example traces for play-style analysis.
            with open(eval_dir / f"epoch_{rl_epoch:06d}_examples.json", "w") as ex_fh:
                json.dump(sp.example_games, ex_fh)

            # --- 2) train over the recency-weighted, cap-bounded replay window ---
            model.train()
            shards, weights, win_epochs, pool_pos = build_replay_window(rl_epoch)
            target = trainer.train_state.global_step + args.train_steps_per_epoch
            comp_sum = {}; comp_n = 0; tr_t0 = time.perf_counter()
            while trainer.train_state.global_step < target and shards:
                # Sample a group of shards PROPORTIONAL to recency weight (with
                # replacement). Each group is loaded, expanded, trained, and freed
                # before the next, so peak RAM is bounded by ONE group (not the
                # disk-resident pool).
                group = rng.choices(shards, weights=weights, k=args.group_size)
                hist = trainer.train_on_shards(group, batch_size=args.batch,
                                               max_steps=target - trainer.train_state.global_step)
                for h in hist:
                    for k, v in h.items():
                        comp_sum[k] = comp_sum.get(k, 0.0) + v
                    comp_n += 1

            def avg(k):
                return comp_sum.get(k, float("nan")) / comp_n if comp_n else float("nan")
            ms_per = (time.perf_counter() - tr_t0) / max(1, comp_n) * 1000
            span = f"{win_epochs[0]}-{win_epochs[-1]}" if win_epochs else "-"
            log(f"epoch {rl_epoch} train: {comp_n} steps (step={trainer.train_state.global_step}) "
                f"total={avg('total'):.4f} policy={avg('policy'):.4f} value={avg('value'):.4f} "
                f"opp={avg('opp_policy'):.4f} prune={trainer.prune_rate:.1%} {ms_per:.0f}ms/step "
                f"window={len(win_epochs)}ep[{span}] {len(shards)}sh pool~{pool_pos // 1000}k "
                f"decay={args.replay_recency_decay}", fh)

            # --- 3) checkpoint ---------------------------------------------------
            save(f"epoch{rl_epoch:06d}", rl_epoch); last_save = rl_epoch

            # --- 4) periodic eval (absolute-epoch schedule) ----------------------
            if eval_due(rl_epoch, args.eval_every, args.epochs):
                do_eval(rl_epoch)

            log(f"epoch {rl_epoch} done in {(time.perf_counter()-ep_t0)/60:.1f} min", fh)
        log(f"=== hexgt RL DONE (through epoch {last_save}) ===", fh)
    except Exception:
        log("EXCEPTION — saving crash checkpoint:\n" + traceback.format_exc(), fh)
        if last_save is not None or start_epoch < args.epochs:
            save("crash", last_save if last_save is not None else start_epoch)
        raise
    finally:
        fh.close()


if __name__ == "__main__":
    main()
