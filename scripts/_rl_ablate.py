"""One short BC-seeded RL ablation run under a given EXPLORATION config.

Measures how the exploration constants shape LEARNING + self-play DATA QUALITY
(not static strength). Per epoch: self-play (with Q1-Q5 metrics) -> train over a
small replay window -> L1 (parallel deterministic head-to-head vs the FROZEN BC
seed) -> L2 (fixed-holdout policy/value loss). Writes a per-config JSON the
comparison report reads.

Methodology: Dirichlet alpha/eps + the temperature schedule are SELF-PLAY-ONLY
(L1 eval is greedy + noiseless) -> judge them by Q1-Q5 + the learning deltas, NOT
by L1 win rate. c_puct / root_policy_temperature DO affect L1.

Usage: python scripts/_rl_ablate.py --name C1 --total-alpha 6.6 --eps 0.25 \
    --root-policy-temperature 1.0 --c-puct 1.5 --temp-initial 1.0 --temp-final 0.2 \
    --temp-decay 30 --temp-floor 0.1
"""
from __future__ import annotations

import argparse, json, random, time, traceback
from pathlib import Path

import torch

from hexo_models.dense_cnn.compact_io import read_compact_shard
from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.config import parse_hexgt_config
from hexo_models.hexgt.evaluation import HexgtBatchedSearcher, run_head_to_head_parallel
from hexo_models.hexgt.expand import build_training_batch
from hexo_models.hexgt.losses import decode_binned_value, segment_softmax_cross_entropy
from hexo_models.hexgt.selfplay import run_selfplay_games
from hexo_models.hexgt.trainer import HexgtTrainer


def log(msg, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n"); fh.flush()


def seg_argmax(values, seg, num_seg, device):
    best = torch.full((num_seg,), float("-inf"), device=device).scatter_reduce(0, seg, values, reduce="amax", include_self=True)
    order = torch.arange(values.shape[0], device=device)
    cand = torch.where(values >= (best[seg] - 1e-6), order, torch.full_like(order, 1 << 30))
    return torch.full((num_seg,), 1 << 30, dtype=torch.long, device=device).scatter_reduce(0, seg, cand, reduce="amin", include_self=True)


@torch.no_grad()
def heldout_eval(model, rows, n, device, batch=128):
    """L2: fixed-holdout policy CE + top-1 agreement + value MAE."""
    model.eval()
    fp16 = device.type == "cuda"
    tot_ce = 0.0; tot = 0; agree = 0; val_abs = 0.0
    for start in range(0, len(rows), batch):
        b, t = build_training_batch(rows[start:start + batch], n=n, horizons=(), prune_max_dropped_mass=0.15)
        if b is None:
            continue
        b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        t = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()}
        ng = int(b["num_graphs"]); seg = b["candidate_graph"]
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
            out = model.forward_policy_value(b)
        pol = out["policy"].float()
        tot_ce += float(segment_softmax_cross_entropy(pol, t["policy"].float(), seg, ng)) * ng
        tot += ng
        pred = seg_argmax(pol, seg, ng, device); tgt = seg_argmax(t["policy"].float(), seg, ng, device)
        valid = (pred < (1 << 30)) & (tgt < (1 << 30))
        agree += int((valid & (pred == tgt)).sum())
        vt = t["value"]; tgt_s = decode_binned_value(vt.float()) if vt.ndim == 2 else vt.float()
        val_abs += float((decode_binned_value(out["value"].float()) - tgt_s).abs().sum())
    model.train()
    return dict(policy_ce=tot_ce / max(1, tot), top1=agree / max(1, tot), value_mae=val_abs / max(1, tot), graphs=tot)


def build_model(arch, device):
    return HexgtNetwork(
        token_dim=arch["token_dim"], gnn_layers=arch["gnn_layers"], ctx_layers=arch["ctx_layers"],
        ffn_dim=arch["ffn_dim"], attention_heads=arch["attention_heads"],
        short_term_value_horizons=tuple(arch["short_term_value_horizons"]),
    ).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    # exploration knobs (the ablation variables)
    ap.add_argument("--total-alpha", type=float, default=6.6)
    ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--root-policy-temperature", type=float, default=1.0)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--temp-initial", type=float, default=1.0)
    ap.add_argument("--temp-final", type=float, default=0.2)
    ap.add_argument("--temp-decay", type=int, default=30)
    ap.add_argument("--temp-floor", type=float, default=0.1)
    ap.add_argument("--forced-playout-k", type=float, default=2.0)
    # run budget
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--games-per-epoch", type=int, default=48)
    ap.add_argument("--active", type=int, default=64)
    ap.add_argument("--vbatch", type=int, default=64)
    ap.add_argument("--visits", type=int, default=96)
    ap.add_argument("--max-actions", type=int, default=240)
    ap.add_argument("--train-steps-per-epoch", type=int, default=200)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--replay-window-epochs", type=int, default=3)
    ap.add_argument("--group-size", type=int, default=24)
    ap.add_argument("--n", type=int, default=3)
    # L1 (vs frozen BC seed) + L2 (holdout)
    ap.add_argument("--l1-games", type=int, default=60)
    ap.add_argument("--l1-visits", type=int, default=96)
    # Slight opening variety decorrelates the L1 games -> smoother win rate (the
    # games are otherwise near-identical under greedy + fixed seeds). Per-game
    # seeded, so still repeatable across epochs (valid paired comparison).
    ap.add_argument("--l1-opening-moves", type=int, default=10)
    ap.add_argument("--l1-opening-temperature", type=float, default=0.6)
    ap.add_argument("--bc-seed", default="/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_bc/hexgt_bc_step006009.pt")
    ap.add_argument("--holdout-glob", default="/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8/selfplay/epoch_000022_game_*.npz")
    ap.add_argument("--holdout-shards", type=int, default=60)
    ap.add_argument("--out-dir", default="/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_ablation")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sp_dir = out_dir / args.name / "selfplay"; sp_dir.mkdir(parents=True, exist_ok=True)
    fh = open(out_dir / f"ablate_{args.name}.log", "a")
    rng = random.Random(args.seed)

    sched = ((0, args.temp_initial), (args.temp_decay, args.temp_final)) if args.temp_decay > 0 else ()
    cfg = parse_hexgt_config({
        "device": str(device), "architecture": {"candidate_radius": args.n},
        "training": {"learning_rate": args.lr, "warmup_steps": args.warmup, "batch_size": args.batch},
        "selfplay": {
            "search_visits": args.visits, "max_actions": args.max_actions,
            "c_puct": args.c_puct, "root_policy_temperature": args.root_policy_temperature,
            "root_dirichlet_noise_enabled": True, "root_dirichlet_total_alpha": args.total_alpha,
            "root_dirichlet_noise_fraction": args.eps,
            "temperature": args.temp_initial, "final_temperature": args.temp_final,
            "temperature_decay_moves": args.temp_decay, "temperature_floor": args.temp_floor,
            "forced_playout_k": args.forced_playout_k,
        },
    })
    cfg_l1 = parse_hexgt_config({"device": str(device), "architecture": {"candidate_radius": args.n},
                                 "selfplay": {"search_visits": args.l1_visits, "c_puct": args.c_puct,
                                              "root_policy_temperature": args.root_policy_temperature}})

    ck = torch.load(args.bc_seed, map_location=device, weights_only=False)
    arch = ck["arch"]
    model = build_model(arch, device); model.load_state_dict(ck["model"])
    frozen = build_model(arch, device); frozen.load_state_dict(ck["model"]); frozen.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    trainer = HexgtTrainer(model=model, config=cfg, optimizer=opt)
    if (not args.no_compile) and device.type == "cuda":
        model.forward_policy_value = torch.compile(model.forward_policy_value, dynamic=True)
        # frozen reference stays EAGER: a 2nd inductor workspace would duplicate
        # VRAM for a model only used in the periodic L1 eval.
    frozen_searcher = HexgtBatchedSearcher(frozen, cfg_l1, device=str(device), fp16=(device.type == "cuda"),
                                           visits=args.l1_visits, opening_moves=args.l1_opening_moves,
                                           opening_temperature=args.l1_opening_temperature)

    def _free():
        # Release each phase's cached CUDA pool so self-play / train / L1 peaks do
        # not STACK (each phase reuses freed memory instead of allocating on top).
        if device.type == "cuda":
            torch.cuda.empty_cache()

    holdout_rows = []
    import glob as _glob
    for p in sorted(_glob.glob(args.holdout_glob))[:args.holdout_shards]:
        holdout_rows.extend(read_compact_shard(Path(p)))

    log(f"=== ABLATE {args.name}: total_alpha={args.total_alpha} eps={args.eps} rpt={args.root_policy_temperature} "
        f"c_puct={args.c_puct} temp={args.temp_initial}->{args.temp_final}@{args.temp_decay} | "
        f"epochs={args.epochs} games/ep={args.games_per_epoch} visits={args.visits} active={args.active} "
        f"compile={not args.no_compile} | L1 vs frozen BC seed ({args.l1_games}g@{args.l1_visits}) "
        f"holdout_rows={len(holdout_rows)} ===", fh)

    results = {"name": args.name, "config": {
        "total_alpha": args.total_alpha, "eps": args.eps, "root_policy_temperature": args.root_policy_temperature,
        "c_puct": args.c_puct, "temp_schedule": [args.temp_initial, args.temp_final, args.temp_decay, args.temp_floor],
        "forced_playout_k": args.forced_playout_k}, "epochs": []}
    try:
        for ep in range(args.epochs):
            t0 = time.perf_counter()
            model.eval()
            sp = run_selfplay_games(model, cfg, num_games=args.games_per_epoch, output_dir=sp_dir, epoch=ep,
                                    device=str(device), fp16=(device.type == "cuda"), base_seed=2000 + ep * 6361,
                                    active_games=args.active, virtual_batch_size=args.vbatch, collect_examples=2)
            _free()
            # train over the recent replay window (this run's own shards)
            model.train()
            lo = max(0, ep - args.replay_window_epochs + 1)
            window = []
            for e in range(lo, ep + 1):
                window += sorted(sp_dir.glob(f"epoch_{e:06d}_game_*.npz"))
            rng.shuffle(window)
            target = trainer.train_state.global_step + args.train_steps_per_epoch
            csum = {}; cn = 0; wi = 0
            while trainer.train_state.global_step < target and window:
                grp = window[wi:wi + args.group_size]
                if not grp:
                    rng.shuffle(window); wi = 0; continue
                wi += args.group_size
                for h in trainer.train_on_shards(grp, batch_size=args.batch, max_steps=target - trainer.train_state.global_step):
                    for k, v in h.items():
                        csum[k] = csum.get(k, 0.0) + v
                    cn += 1
            tr = {k: csum[k] / cn for k in csum} if cn else {}
            # L1: current (compiled) vs frozen BC seed
            model.eval()
            cur_searcher = HexgtBatchedSearcher(model, cfg_l1, device=str(device), fp16=(device.type == "cuda"),
                                                visits=args.l1_visits, opening_moves=args.l1_opening_moves,
                                                opening_temperature=args.l1_opening_temperature)
            l1 = run_head_to_head_parallel(cur_searcher, frozen_searcher, games=args.l1_games,
                                           base_seed=9000, max_actions=1024, active_limit=args.active)
            model.train()
            # L2: fixed holdout
            l2 = heldout_eval(model, holdout_rows, args.n, device, batch=args.batch) if holdout_rows else {}
            rec = {"epoch": ep, "selfplay": sp.as_dict(), "train": tr,
                   "L1_vs_frozen_bc": l1.as_dict(), "L2_holdout": l2, "minutes": (time.perf_counter() - t0) / 60}
            results["epochs"].append(rec)
            log(f"{args.name} ep{ep}: L1_vs_frozen={l1.win_rate:.1%} ({l1.wins}-{l1.losses}-{l1.draws}) | "
                f"L2 CE={l2.get('policy_ce', float('nan')):.3f} top1={l2.get('top1', float('nan')):.1%} "
                f"valMAE={l2.get('value_mae', float('nan')):.3f} | "
                f"Q1 decisive={sp.decisive_fraction:.1%} len_med={sp.game_length_median:.0f} | "
                f"Q2 uniq_open={sp.opening_unique_fraction:.1%} m2H={sp.move2_entropy:.2f} | "
                f"Q3 visitH={sp.mean_visit_entropy:.2f} priorH={sp.mean_prior_entropy:.2f} | "
                f"Q4 |val|={sp.mean_abs_value:.2f} draw={sp.draw_fraction:.1%} | "
                f"Q5 forced={sp.forced_move_fraction:.1%} | {sp.positions_per_second:.1f}pos/s {rec['minutes']:.1f}min", fh)
            with open(out_dir / f"ablate_{args.name}.json", "w") as jf:
                json.dump(results, jf, indent=2)
        log(f"=== ABLATE {args.name} DONE ===", fh)
    except Exception:
        log("EXCEPTION:\n" + traceback.format_exc(), fh)
        with open(out_dir / f"ablate_{args.name}.json", "w") as jf:
            json.dump(results, jf, indent=2)
        raise
    finally:
        fh.close()


if __name__ == "__main__":
    main()
