"""BC-seeded RL self-play training for hexgnn (transformer-free hexgt lineage).

A stripped parallel of scripts/_rl_train.py: same resumable, crash-safe, RAM-
disciplined self-play -> compact-shards -> train -> checkpoint -> periodic eval
loop and the SAME exploration/PCR/soft-Z/policy-surprise/LR-decay knobs, but
driving the hexgnn model (GNN trunk + policy/value/opp heads, NO context
transformer, NO STV lookahead heads). Because hexgnn has no STV heads there is no
STV graft on resume — the trunk + policy/value/opp load EXACTLY.

It is fully ADDITIVE: it imports the top-level `hexgnn` package (which reuses the
already-built native `hexo_models._rust.hexgt` accelerator read-only) and writes
to its own run dir, so it never touches the live hexgt_rl_main3 run.

Usage (CPU dev smoke; the owner picks the real launch settings — see
scripts/_rl_launch_hexgnn.sh):
  python scripts/_rl_train_hexgnn.py --bc-seed runs/hexgnn_rl/pretrain/hexgnn_pretrain.pt \
      --out-dir runs/hexgnn_rl --epochs 40 --games-per-epoch 64 --device cpu
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

# Self-bootstrap the local-worktree package paths (mirrors the production
# launcher's PYTHONPATH) so a long run never imports a stale installed wheel and
# the additive top-level `hexgnn` package resolves regardless of how it is invoked.
_ROOT = Path(__file__).resolve().parents[1]
for _pkg in ("hexo_engine", "hexo_utils", "hexo_runner", "hexo_train", "hexo_models"):
    _p = _ROOT / "packages" / _pkg / "python"
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
_hexgnn_path = _ROOT / "packages" / "hexgnn" / "python"
if str(_hexgnn_path) not in sys.path:
    sys.path.insert(0, str(_hexgnn_path))

from hexgnn.architecture import (
    HexgnnNetwork,
    expand_value_readout_columns,
    zero_init_expanded_feature_columns,
)
from hexgnn.config import parse_hexgnn_config
from hexgnn.constants import FEATURE_SCHEMA_VERSION, feature_slots_after
from hexgnn.selfplay import run_selfplay_games
from hexgnn.trainer import HexgnnTrainer


def log(msg, fh=None):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n"); fh.flush()


def log_gpu_mem(tag, fh=None, reset=True):
    if not torch.cuda.is_available():
        return
    st = torch.cuda.memory_stats()
    res_peak = st.get("reserved_bytes.all.peak", 0) / 1e9
    res_now = st.get("reserved_bytes.all.current", 0) / 1e9
    alloc_peak = torch.cuda.max_memory_allocated() / 1e9
    log(f"  GPU mem [{tag}]: reserved_peak={res_peak:.2f}GB reserved_now={res_now:.2f}GB "
        f"alloc_peak={alloc_peak:.2f}GB", fh)
    if reset:
        torch.cuda.reset_peak_memory_stats()


def build_model(arch_meta, device):
    return HexgnnNetwork(
        token_dim=arch_meta["token_dim"], gnn_layers=arch_meta["gnn_layers"],
        attention_heads=arch_meta.get("attention_heads", 4),
        value_pma_seeds=int(arch_meta.get("value_pma_seeds", 2)),
        value_head_use_side=bool(arch_meta.get("value_head_use_side", True)),
    ).to(device)


def _validate_strict_resume_load(load_info) -> None:
    """hexgnn has no aux-head graft, so a resume must load EXACTLY: nothing missing,
    nothing unexpected (after the optional value-readout zero-init expansion)."""
    missing = list(load_info.missing_keys)
    unexpected = list(load_info.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(f"hexgnn resume load mismatch: missing={missing} unexpected={unexpected}")


def eval_due(rl_epoch, eval_every, total_epochs):
    return (rl_epoch % eval_every == 0) or (rl_epoch == total_epochs - 1)


def eval_result_path(eval_dir, rl_epoch):
    return Path(eval_dir) / f"epoch_{rl_epoch:06d}_eval.json"


def eval_missing(eval_dir, rl_epoch):
    return not eval_result_path(eval_dir, rl_epoch).exists()


def select_window_epochs(current_epoch, epoch_positions, *, base_window, pool_cap):
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
    return float(decay) ** max(0, int(current_epoch) - int(epoch))


def resume_plan(loaded_epoch, epoch_train_complete):
    if epoch_train_complete:
        return loaded_epoch + 1, False
    return loaded_epoch, True


def should_skip_selfplay(rl_epoch, start_epoch, resume_incomplete_train):
    return bool(resume_incomplete_train) and rl_epoch == start_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-seed", default=str(_ROOT / "runs/hexgnn_rl/pretrain/hexgnn_pretrain.pt"))
    # FROM-SCRATCH (no pretraining): build a random-init model from these arch args
    # instead of loading a seed. The arch is stamped into every checkpoint so resume
    # rebuilds the same shape. Ignored once a hexgnn_rl_latest.pt exists (resume).
    ap.add_argument("--from-scratch", action="store_true",
                    help="random-init the model from --token-dim/--gnn-layers/... (no --bc-seed)")
    ap.add_argument("--token-dim", type=int, default=128)
    ap.add_argument("--gnn-layers", type=int, default=3)
    ap.add_argument("--attention-heads", type=int, default=4)
    ap.add_argument("--value-pma-seeds", type=int, default=2)
    ap.add_argument("--value-head-use-side", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out-dir", default=str(_ROOT / "runs/hexgnn_rl"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--games-per-epoch", type=int, default=64)
    ap.add_argument("--active", type=int, default=64)
    ap.add_argument("--vbatch", type=int, default=16)
    ap.add_argument("--visits", type=int, default=512)
    ap.add_argument("--max-actions", type=int, default=512)
    ap.add_argument("--train-steps-per-epoch", type=int, default=512)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--lr-decay", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--lr-decay-start-step", type=int, default=0)
    ap.add_argument("--lr-decay-halflife-steps", type=float, default=0.0)
    ap.add_argument("--lr-min", type=float, default=0.0)
    ap.add_argument("--replay-window-epochs", type=int, default=8)
    ap.add_argument("--replay-recency-decay", type=float, default=0.9)
    ap.add_argument("--replay-pool-cap", type=int, default=500_000)
    ap.add_argument("--group-size", type=int, default=30)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--total-alpha", type=float, default=6.6)
    ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--root-policy-temperature", type=float, default=1.0)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--final-temperature", type=float, default=0.2)
    ap.add_argument("--temperature-decay-moves", type=int, default=30)
    ap.add_argument("--temperature-floor", type=float, default=0.1)
    ap.add_argument("--temperature-halflife", type=float, default=0.0)
    ap.add_argument("--forced-playout-k", type=float, default=2.0)
    ap.add_argument("--pcr", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--pcr-full-proportion", type=float, default=0.5)
    ap.add_argument("--pcr-fast-visits", type=int, default=170)
    ap.add_argument("--widening-max-children", type=int, default=96)
    ap.add_argument("--widening-min-children", type=int, default=2)
    ap.add_argument("--widening-policy-mass", type=float, default=0.95)
    ap.add_argument("--policy-surprise", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--policy-surprise-uniform-fraction", type=float, default=0.5)
    ap.add_argument("--policy-surprise-max-weight", type=float, default=8.0)
    ap.add_argument("--soft-z-lambda", type=float, default=0.0)
    # Eval
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--eval-games", type=int, default=40)
    ap.add_argument("--eval-visits", type=int, default=200)
    ap.add_argument("--eval-opening-moves", type=int, default=10)
    ap.add_argument("--eval-opening-temperature", type=float, default=0.6)
    ap.add_argument("--eval-max-actions", type=int, default=1024)
    ap.add_argument("--dense-ckpt", default="/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8/checkpoints/epoch_000024.pt")
    ap.add_argument("--dense-config", default=str(_ROOT / "configs/dense_cnn_model1_target_96x8.toml"))
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

    cfg = parse_hexgnn_config({
        "device": str(device),
        "architecture": {"candidate_radius": args.n},
        "training": {"learning_rate": args.lr, "warmup_steps": args.warmup, "batch_size": args.batch,
                     "lr_decay_enabled": bool(args.lr_decay),
                     "lr_decay_start_step": args.lr_decay_start_step,
                     "lr_decay_halflife_steps": args.lr_decay_halflife_steps,
                     "lr_min": args.lr_min},
        "samples": {
            "policy_surprise_enabled": bool(args.policy_surprise),
            "policy_surprise_uniform_fraction": args.policy_surprise_uniform_fraction,
            "policy_surprise_max_weight": args.policy_surprise_max_weight,
            "soft_z_lambda": args.soft_z_lambda,
        },
        "selfplay": {
            "search_visits": args.visits, "max_actions": args.max_actions,
            "c_puct": args.c_puct, "root_policy_temperature": args.root_policy_temperature,
            "root_dirichlet_noise_enabled": True, "root_dirichlet_total_alpha": args.total_alpha,
            "root_dirichlet_noise_fraction": args.eps,
            "temperature": args.temperature, "final_temperature": args.final_temperature,
            "temperature_decay_moves": args.temperature_decay_moves,
            "temperature_floor": args.temperature_floor,
            "temperature_halflife": args.temperature_halflife,
            "forced_playout_k": args.forced_playout_k,
            "widening_max_children": args.widening_max_children,
            "widening_min_children": args.widening_min_children,
            "widening_policy_mass": args.widening_policy_mass,
            "pcr_enabled": bool(args.pcr),
            "pcr_full_proportion": args.pcr_full_proportion,
            "pcr_fast_visits": args.pcr_fast_visits,
        },
    })

    latest = ckpt_dir / "hexgnn_rl_latest.pt"
    start_epoch = 0
    resume_incomplete_train = False
    vr_expanded = False
    if latest.exists():
        ck = torch.load(latest, map_location=device, weights_only=False)
        arch_meta = dict(ck["arch"])
        model = build_model(arch_meta, device)
        vr_expanded = expand_value_readout_columns(model, ck["model"])
        _validate_strict_resume_load(model.load_state_dict(ck["model"], strict=False))
        loaded_epoch = int(ck.get("rl_epoch", 0))
        epoch_train_complete = bool(ck.get("epoch_train_complete", True))
        start_epoch, resume_incomplete_train = resume_plan(loaded_epoch, epoch_train_complete)
        seed_desc = (f"RESUME from {latest.name} (rl_epoch={ck.get('rl_epoch')}, step={ck.get('step')})"
                     + ("" if epoch_train_complete else f" + RE-TRAIN incomplete epoch {loaded_epoch}"))
    elif args.from_scratch:
        # FRESH random init — no pretraining. Arch comes straight from CLI args and
        # is stamped at the CURRENT feature-schema version, so no seed load, no
        # value-readout expansion, and no feature zero-init are needed.
        ck = None
        arch_meta = {
            "token_dim": args.token_dim, "gnn_layers": args.gnn_layers,
            "attention_heads": args.attention_heads, "value_pma_seeds": args.value_pma_seeds,
            "value_head_use_side": bool(args.value_head_use_side),
        }
        model = build_model(arch_meta, device)
        seed_desc = (f"FROM SCRATCH (random init, no pretraining): token_dim={args.token_dim} "
                     f"gnn_layers={args.gnn_layers} heads={args.attention_heads} "
                     f"pma_k={args.value_pma_seeds} use_side={bool(args.value_head_use_side)}")
    else:
        ck = torch.load(args.bc_seed, map_location=device, weights_only=False)
        arch_meta = dict(ck["arch"])
        model = build_model(arch_meta, device)
        vr_expanded = expand_value_readout_columns(model, ck["model"])
        _validate_strict_resume_load(model.load_state_dict(ck["model"], strict=False))
        seed_desc = (f"SEED from {Path(args.bc_seed).name} (step={ck.get('step')}"
                     f"{', rl_epoch=' + str(ck['rl_epoch']) if 'rl_epoch' in ck else ''})")

    loaded_fsv = int(ck.get("feature_schema_version", 1)) if ck is not None else FEATURE_SCHEMA_VERSION
    if loaded_fsv < FEATURE_SCHEMA_VERSION:
        target_slots = feature_slots_after(loaded_fsv)
        zeroed = zero_init_expanded_feature_columns(model, target_slots)
        seed_desc += (f" + ZERO-INIT feature-expansion v{loaded_fsv}->v{FEATURE_SCHEMA_VERSION} "
                      f"(zeroed {len(zeroed)} node_in cols {zeroed})")
    if vr_expanded:
        seed_desc += " + EXPAND value-readout to [SIDE|PMA] (zero-init, identical first step)"

    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    trainer = HexgnnTrainer(model=model, config=cfg, optimizer=opt)
    trainer.cuda_retry_log = lambda m: log(m, fh)
    if ck is not None and "optimizer" in ck:
        try:
            opt.load_state_dict(ck["optimizer"])
            log(f"  (optimizer state restored from {Path(args.bc_seed).name if not latest.exists() else latest.name})", fh)
        except Exception as exc:
            log(f"  (optimizer state not restored: {exc})", fh)
        drifted = 0
        for group in opt.param_groups:
            for p in group["params"]:
                st = opt.state.get(p)
                if not st:
                    continue
                ea = st.get("exp_avg")
                if ea is not None and tuple(ea.shape) != tuple(p.shape):
                    opt.state.pop(p, None)
                    drifted += 1
        if drifted:
            log(f"  (reset fresh momentum for {drifted} shape-drifted param(s) after column-expansion)", fh)
    if latest.exists() and "train_state" in ck:
        trainer.load_train_state(ck["train_state"])

    compiled = (not args.no_compile) and device.type == "cuda"
    if compiled:
        model.forward_policy_value = torch.compile(model.forward_policy_value, dynamic=True)

    if device.type == "cuda":
        alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        expandable = "expandable_segments:true" in alloc_conf.lower()
        total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
        warn = "" if (compiled and expandable) else "  <-- WARNING: expected compile+expandable for the production mem profile"
        log(f"    GPU mem config: compile={compiled} expandable_segments={'ON' if expandable else 'OFF'} "
            f"(PYTORCH_CUDA_ALLOC_CONF={alloc_conf or 'unset'}) total={total_gb:.1f}GB{warn}", fh)

    log(f"=== hexgnn RL start: {seed_desc} | params={nparams:,} n={args.n} device={device} "
        f"compile={compiled} ===", fh)
    log(f"    epochs={args.epochs} games/epoch={args.games_per_epoch} visits={args.visits} "
        f"active={args.active} vbatch={args.vbatch} | train_steps/epoch={args.train_steps_per_epoch} "
        f"batch={args.batch} lr={args.lr} replay_window={args.replay_window_epochs} ep | "
        f"loss_w: policy={cfg.training.policy_weight} value={cfg.training.value_weight} "
        f"opp={cfg.training.opp_policy_weight} "
        f"soft_z_lambda={cfg.samples.soft_z_lambda} policy_surprise={cfg.samples.policy_surprise_enabled}", fh)

    def save(tag, rl_epoch, epoch_train_complete=True):
        payload = {
            "model": model.state_dict(), "optimizer": opt.state_dict(),
            "train_state": trainer.train_state.to_dict(), "arch": arch_meta,
            "step": trainer.train_state.global_step, "rl_epoch": rl_epoch,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "epoch_train_complete": epoch_train_complete,
        }
        torch.save(payload, ckpt_dir / f"hexgnn_rl_{tag}.pt")
        torch.save(payload, ckpt_dir / "hexgnn_rl_latest.pt")

    _epoch_pos_cache: dict[int, int] = {}

    def epoch_positions(e):
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
        import tomllib
        from hexgnn.evaluation import make_hexgnn_factory, run_head_to_head

        cfg_eval = parse_hexgnn_config({"device": str(device), "architecture": {"candidate_radius": args.n},
                                        "selfplay": {"search_visits": args.eval_visits}})
        model.eval()
        make_hexgnn = make_hexgnn_factory(model, cfg_eval, device=str(device),
                                          fp16=(device.type == "cuda"), identity_id="hexgnn-rl",
                                          opening_moves=args.eval_opening_moves,
                                          opening_temperature=args.eval_opening_temperature)
        results = {}
        try:
            if not Path(args.dense_ckpt).exists():
                raise FileNotFoundError(f"dense_cnn e24 checkpoint missing: {args.dense_ckpt}")
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
            r = run_head_to_head(make_hexgnn, make_dense, games=args.eval_games,
                                 output_dir=eval_dir / f"epoch_{rl_epoch:06d}_vs_dense",
                                 base_seed=4242, max_actions=args.eval_max_actions,
                                 game_id_prefix=f"e{rl_epoch}vsdense")
            results["vs_dense_cnn_e24"] = r.as_dict()
        except FileNotFoundError as exc:
            log(f"  eval: vs-dense_cnn SKIPPED — {exc}", fh)
            results["vs_dense_cnn_e24_skipped"] = str(args.dense_ckpt)
        except Exception:
            results["vs_dense_cnn_e24_error"] = traceback.format_exc()
        if args.sealbot:
            try:
                from hexo_runner.adapters.sealbot import SealBotConfig, SealBotPlayer
                sb = SealBotConfig(variant="best", time_limit=0.05); sb.validate()

                def make_sb(_seed):
                    return SealBotPlayer(sb, player_id="sealbot-best-50ms")
                rs = run_head_to_head(make_hexgnn, make_sb, games=args.eval_games,
                                      output_dir=eval_dir / f"epoch_{rl_epoch:06d}_vs_sealbot",
                                      base_seed=4242, max_actions=args.eval_max_actions,
                                      game_id_prefix=f"e{rl_epoch}vssb")
                results["vs_sealbot"] = rs.as_dict()
            except Exception as exc:
                results["vs_sealbot_error"] = str(exc)
        model.train()
        return results

    def do_eval(rl_epoch):
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
    cur_epoch = None
    cur_epoch_shards_ready = False
    run_sanitized_logits = 0
    run_sanitized_excluded = 0
    try:
        if start_epoch == 0:
            base_ev = run_eval(-1)
            with open(eval_dir / "baseline_bc_eval.json", "w") as bfh:
                json.dump(base_ev, bfh, indent=2)
            bd = base_ev.get("vs_dense_cnn_e24", {})
            log(f">>> BC-SEED BASELINE (pre-RL) vs dense_cnn e24: "
                f"{bd.get('wins')}W/{bd.get('losses')}L/{bd.get('draws')}D "
                f"= {bd.get('win_rate', float('nan')):.1%}", fh)

        prev = start_epoch - 1
        if prev >= 0 and eval_due(prev, args.eval_every, args.epochs) and eval_missing(eval_dir, prev):
            log(f"resume: backfilling missing eval for epoch {prev}", fh)
            do_eval(prev)

        for rl_epoch in range(start_epoch, args.epochs):
            ep_t0 = time.perf_counter()
            cur_epoch = rl_epoch
            skip_selfplay = should_skip_selfplay(rl_epoch, start_epoch, resume_incomplete_train)
            cur_epoch_shards_ready = skip_selfplay
            if skip_selfplay:
                log(f"epoch {rl_epoch}: RE-TRAIN after mid-training crash — reusing shards", fh)
            else:
                model.eval()
                sp = run_selfplay_games(
                    model, cfg, num_games=args.games_per_epoch, output_dir=selfplay_dir,
                    epoch=rl_epoch, device=str(device), fp16=(device.type == "cuda"),
                    base_seed=1000 + rl_epoch * 7919, active_games=args.active,
                    virtual_batch_size=args.vbatch, collect_examples=3,
                )
                cur_epoch_shards_ready = True
                run_sanitized_logits += int(sp.sanitized_logit_events)
                run_sanitized_excluded += int(sp.sanitized_samples_excluded)
                saniti_str = (
                    f" | SANITIZED {sp.sanitized_logit_events} logits/"
                    f"{sp.sanitized_search_rounds} rounds excl={sp.sanitized_samples_excluded} "
                    f"(run {run_sanitized_logits} logits/{run_sanitized_excluded} excl)"
                    if sp.sanitized_logit_events else ""
                )
                pcr_str = (
                    f" | PCR full={sp.full_search_count}/{sp.full_search_count + sp.fast_search_count} "
                    f"rec={sp.recorded_positions} (~{sp.mean_search_visits:.0f}v/move)"
                    if sp.pcr_enabled else ""
                )
                log(f"epoch {rl_epoch} selfplay: {sp.completed_games}C/{sp.truncated_games}T games, "
                    f"{sp.searched_positions} pos, {sp.positions_per_second:.1f} pos/s | "
                    f"cand={sp.mean_candidate_count:.0f} | "
                    f"Q1 decisive={sp.decisive_fraction:.1%} len_med={sp.game_length_median:.0f} | "
                    f"Q3 visitH={sp.mean_visit_entropy:.2f} priorH={sp.mean_prior_entropy:.2f} | "
                    f"Q4 |val|={sp.mean_abs_value:.2f} draw={sp.draw_fraction:.1%}{saniti_str}{pcr_str}", fh)
                log_gpu_mem(f"epoch {rl_epoch} selfplay", fh)
                with open(eval_dir / f"epoch_{rl_epoch:06d}_selfplay.json", "w") as sp_fh:
                    json.dump(sp.as_dict(), sp_fh, indent=2)
                with open(eval_dir / f"epoch_{rl_epoch:06d}_examples.json", "w") as ex_fh:
                    json.dump(sp.example_games, ex_fh)

            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
            model.train()
            shards, weights, win_epochs, pool_pos = build_replay_window(rl_epoch)
            target = trainer.train_state.global_step + args.train_steps_per_epoch
            comp_sum = {}; comp_n = 0; tr_t0 = time.perf_counter()
            while trainer.train_state.global_step < target and shards:
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
                f"opp={avg('opp_policy'):.4f} prune={trainer.prune_rate:.1%} lr={avg('lr'):.2e} {ms_per:.0f}ms/step "
                f"window={len(win_epochs)}ep[{span}] {len(shards)}sh "
                f"pool~{pool_pos // 1000}k/{args.replay_pool_cap // 1000}k decay={args.replay_recency_decay}", fh)

            save(f"epoch{rl_epoch:06d}", rl_epoch); last_save = rl_epoch
            if eval_due(rl_epoch, args.eval_every, args.epochs):
                do_eval(rl_epoch)
            log(f"epoch {rl_epoch} done in {(time.perf_counter()-ep_t0)/60:.1f} min", fh)
        log(f"=== hexgnn RL DONE (through epoch {last_save}) ===", fh)
    except Exception:
        log("EXCEPTION — saving crash checkpoint:\n" + traceback.format_exc(), fh)
        if cur_epoch is not None and cur_epoch_shards_ready:
            save("crash", cur_epoch, epoch_train_complete=False)
            log(f"  (crash during epoch {cur_epoch} training -> saved INCOMPLETE; resume will re-train it)", fh)
        else:
            log("  (crash before training shards were ready; leaving last checkpoint as the resume point)", fh)
        raise
    finally:
        fh.close()


if __name__ == "__main__":
    main()
