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

import torch

from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.config import parse_hexgt_config
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
    ap.add_argument("--replay-window-epochs", type=int, default=8)
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
        }
        torch.save(payload, ckpt_dir / f"hexgt_rl_{tag}.pt")
        torch.save(payload, ckpt_dir / "hexgt_rl_latest.pt")

    def replay_window(current_epoch):
        lo = max(0, current_epoch - args.replay_window_epochs + 1)
        shards = []
        for e in range(lo, current_epoch + 1):
            shards += sorted(selfplay_dir.glob(f"epoch_{e:06d}_game_*.npz"))
        return shards

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

            # --- 2) train over the replay window --------------------------------
            model.train()
            window = replay_window(rl_epoch)
            rng.shuffle(window)
            target = trainer.train_state.global_step + args.train_steps_per_epoch
            comp_sum = {}; comp_n = 0; tr_t0 = time.perf_counter()
            wi = 0
            while trainer.train_state.global_step < target and window:
                group = window[wi:wi + args.group_size]
                if not group:
                    rng.shuffle(window); wi = 0; continue
                wi += args.group_size
                hist = trainer.train_on_shards(group, batch_size=args.batch,
                                               max_steps=target - trainer.train_state.global_step)
                for h in hist:
                    for k, v in h.items():
                        comp_sum[k] = comp_sum.get(k, 0.0) + v
                    comp_n += 1

            def avg(k):
                return comp_sum.get(k, float("nan")) / comp_n if comp_n else float("nan")
            ms_per = (time.perf_counter() - tr_t0) / max(1, comp_n) * 1000
            log(f"epoch {rl_epoch} train: {comp_n} steps (step={trainer.train_state.global_step}) "
                f"total={avg('total'):.4f} policy={avg('policy'):.4f} value={avg('value'):.4f} "
                f"opp={avg('opp_policy'):.4f} prune={trainer.prune_rate:.1%} {ms_per:.0f}ms/step "
                f"window={len(window)} shards", fh)

            # --- 3) checkpoint ---------------------------------------------------
            save(f"epoch{rl_epoch:06d}", rl_epoch); last_save = rl_epoch

            # --- 4) periodic eval ------------------------------------------------
            if (rl_epoch % args.eval_every == 0) or (rl_epoch == args.epochs - 1):
                ev = run_eval(rl_epoch)
                with open(eval_dir / f"epoch_{rl_epoch:06d}_eval.json", "w") as ev_fh:
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
