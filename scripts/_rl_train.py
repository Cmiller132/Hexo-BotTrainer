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
import gc
import json
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from hexo_models.hexgt.architecture import (
    HexgtNetwork,
    expand_stv_readout_columns,
    expand_value_readout_columns,
    zero_init_expanded_feature_columns,
)
from hexo_models.hexgt.config import parse_hexgt_config
from hexo_models.hexgt.constants import FEATURE_SCHEMA_VERSION, feature_slots_after
from hexo_models.hexgt.selfplay import run_selfplay_games
from hexo_models.hexgt.trainer import HexgtTrainer


def log(msg, fh=None):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n"); fh.flush()


def measure_value_optimism(model, hxr_path, device, n_radius, max_pos=64):
    """Cheap per-epoch value-head CALIBRATION probe (seconds, diagnostic-only).

    At FirstStone midgame positions, evaluate the IDENTICAL board from BOTH
    perspectives via a facts-level owner-swap (0<->1) — `build_graph_tensors`
    re-derives every feature (including the SIDE-node own/opp counts the value
    head reads) from the swapped owners, so the swap is an exact perspective flip.
    Returns mean(v_side + v_oppside): 0 = zero-sum-consistent / calibrated, >0 =
    optimism (both sides predict winning), <0 = pessimism. Tracks calibration
    drift over the run (pretrain seed ~ -0.06; the old dense_cnn head was +0.82).

    Wrapped end-to-end in try/except: a probe failure must NEVER crash training.
    """
    try:
        import statistics as _st

        import hexo_engine.api as _eng
        from hexo_engine.types import AxialCoord, PlacementAction, unpack_coord_id
        from hexo_runner.records import HexoRecordFile
        from hexo_models.hexgt.features import build_graph_tensors
        from hexo_models.hexgt.collate import collate_graphs
        from hexo_models.hexgt import rust_bridge
        from hexo_models.hexgt.losses import decode_binned_value

        if not Path(hxr_path).exists():
            return {"optimism_sum_mean": float("nan"), "optimism_positions": 0}

        def _val(facts):
            b = collate_graphs([build_graph_tensors(facts)])
            b = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in b.items()}
            return float(decode_binned_value(model.forward_policy_value(b)["value"].float().cpu())[0])

        def _swap(facts):
            own = list(facts["nodes"]["node_owner"])
            sw = [1 if o == 0 else (0 if o == 1 else o) for o in own]
            return {**facts, "nodes": {**facts["nodes"], "node_owner": sw}}

        was_training = model.training
        model.eval()
        sums, both_pos = [], 0
        with torch.no_grad():
            rf = HexoRecordFile.open(str(hxr_path))
            for rec in rf.iter_records():
                if len(sums) >= max_pos:
                    break
                coords = [unpack_coord_id(int(c)) for c in rec.action_ids]
                s = _eng.new_game()
                for i, c in enumerate(coords):
                    if 10 <= i <= 50:
                        facts = rust_bridge.graph_facts(s, n_radius)
                        if facts.get("meta", {}).get("first_stone") is None:
                            va, vb = _val(facts), _val(_swap(facts))
                            sums.append(va + vb)
                            if va > 0.05 and vb > 0.05:
                                both_pos += 1
                            if len(sums) >= max_pos:
                                break
                    _eng.apply_action(s, PlacementAction(AxialCoord(q=int(c.q), r=int(c.r))))
        if was_training:
            model.train()
        if not sums:
            return {"optimism_sum_mean": float("nan"), "optimism_positions": 0}
        return {
            "optimism_sum_mean": _st.mean(sums),
            "optimism_sum_median": _st.median(sums),
            "optimism_both_positive_frac": both_pos / len(sums),
            "optimism_positions": len(sums),
        }
    except Exception as exc:  # diagnostic only — never abort the epoch
        return {"optimism_sum_mean": float("nan"), "optimism_positions": 0,
                "optimism_error": repr(exc)}


def log_gpu_mem(tag, fh=None, reset=True):
    """Log the CUDA caching-allocator RESERVED peak (what governs OOM / verifies the
    run is on the ~2.4GB compiled+expandable path, not the ~11.8GB eager path) plus
    the live reserved and the true allocated peak, then reset the peak so the next
    window is measured fresh. No-op without CUDA."""
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


# KataGo-style short-term-value (EMA) auxiliary heads. Horizon h uses the EMA of
# future ROOT values with decay lambda = h/(h+1) (effective look-ahead h moves,
# see dense_cnn.samples._short_term_value_targets — the shared, proven target).
# Training-only aux heads (with_aux); they never touch the search/play forward, so
# enabling them leaves policy/value output identical. NEW-SAMPLES-ONLY: the root-
# value trajectory isn't persisted, so targets ramp in over post-deploy epochs.
STV_HORIZONS = (4, 12, 24)


def _validate_stv_resume_load(load_info) -> None:
    """strict=False load is used so the NEW short-term-value heads can be fresh-
    init while the trunk + policy/value/opp_policy load EXACTLY. Guard that the
    ONLY thing missing from the checkpoint is stv-head params, and nothing in the
    checkpoint went unused — so a real trunk/key mismatch can't slip through."""
    unexpected = list(load_info.unexpected_keys)
    nonstv_missing = [k for k in load_info.missing_keys if not k.startswith("short_term_value_heads")]
    if unexpected or nonstv_missing:
        raise RuntimeError(
            f"STV resume load mismatch: unexpected={unexpected} non-stv-missing={nonstv_missing}")


def build_model(arch_meta, device):
    return HexgtNetwork(
        token_dim=arch_meta["token_dim"], gnn_layers=arch_meta["gnn_layers"],
        ctx_layers=arch_meta["ctx_layers"], ffn_dim=arch_meta["ffn_dim"],
        attention_heads=arch_meta["attention_heads"],
        short_term_value_horizons=tuple(arch_meta["short_term_value_horizons"]),
        # PMA value-head seed count (default k=2 if a pre-PMA checkpoint omits it).
        value_pma_seeds=int(arch_meta.get("value_pma_seeds", 2)),
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


def resume_plan(loaded_epoch, epoch_train_complete):
    """Decide the next epoch to run from a checkpoint's (rl_epoch, completion).

    Returns ``(start_epoch, resume_incomplete_train)``. A COMPLETE epoch advances
    to the next one (normal). An INCOMPLETE one (mid-training crash) re-runs THAT
    epoch's training — ``resume_incomplete_train=True`` makes the first loop
    iteration skip self-play and re-train from the existing on-disk shards."""
    if epoch_train_complete:
        return loaded_epoch + 1, False
    return loaded_epoch, True


def should_skip_selfplay(rl_epoch, start_epoch, resume_incomplete_train):
    """Self-play is skipped ONLY for the crashing epoch being re-trained on resume
    (its shards already exist) — and only on the first loop iteration."""
    return bool(resume_incomplete_train) and rl_epoch == start_epoch


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
    # Aux loss weight shared by the 3 short-term-value (lookahead) heads. Default
    # 0.25 (the config default) is a no-op; the run passes 0.10 so the 3 stv heads
    # collectively (~0.10x7.8=0.78 effective) ~= the value head instead of ~3x it,
    # restoring policy dominance of the shared trunk (~main 62 / aux 38).
    ap.add_argument("--short-term-value-weight", type=float, default=0.25)
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
    # KataGo-style smooth exponential-halflife move temperature. >0 takes precedence
    # over the linear decay: temp(ply)=floor+(temperature-floor)*2**(-ply/halflife).
    # MUST be threaded into the cfg dict below (the driver does not read TOML [selfplay]).
    ap.add_argument("--temperature-halflife", type=float, default=0.0)
    ap.add_argument("--forced-playout-k", type=float, default=2.0)
    # KataGo Playout Cap Randomization (Wu 2020). Per MOVE, with prob
    # --pcr-full-proportion do a FULL search (--visits, recorded, with noise + forced
    # playouts + the temperature schedule); else a FAST search (--pcr-fast-visits, NOT
    # recorded, no noise, greedy). Only full positions become policy/value training
    # rows (~halves rows/epoch); fast moves still advance the game + feed the dense STV/
    # opp aux chain. MUST be threaded into the cfg dict (the driver does not read TOML
    # [selfplay]) — same trap as --soft-z-lambda / --temperature-halflife. Default OFF
    # so the relaunch command is byte-identical until the launcher opts in. See
    # docs/analysis/HEXGT_PCR_KATAGO_MAPPING.md. n=170 == KataGo's 1/6 ratio at N=1024.
    ap.add_argument("--pcr", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--pcr-full-proportion", type=float, default=0.5)
    ap.add_argument("--pcr-fast-visits", type=int, default=170)
    # MCTS nucleus widening. Defaults match configs/hexgt_model2.toml (documented
    # intent) + dense_cnn's 96x8 run. The driver previously passed NONE of these,
    # so they silently fell to the parse defaults (max_children=32), narrowing
    # search vs the intended 96.
    ap.add_argument("--widening-max-children", type=int, default=96)
    ap.add_argument("--widening-min-children", type=int, default=2)
    ap.add_argument("--widening-policy-mass", type=float, default=0.95)
    # KataGo policy-surprise sample weighting (row duplication by KL(visits||prior)).
    # The driver builds its config from ARGS only and never read the [samples]
    # section of any TOML, so configs/hexgt_model3.toml's `policy_surprise_enabled
    # = true` NEVER reached the self-play config -> epochs 0-4 trained WITHOUT it
    # (effective_samples == raw, policy_surprise_mean == 0.0). Default ON here so it
    # engages at the next supervisor relaunch without changing the fixed relaunch
    # command; --no-policy-surprise opts out. Values match the intended TOML.
    ap.add_argument("--policy-surprise", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--policy-surprise-uniform-fraction", type=float, default=0.5)
    ap.add_argument("--policy-surprise-max-weight", type=float, default=8.0)
    # Soft-Z value-target blend: value = (1-lambda)*z + lambda*root_value at finalize.
    # MUST be threaded into the cfg dict below (the driver does NOT read any TOML
    # [samples] section, so a TOML soft_z_lambda is silently ignored -- same trap as
    # policy_surprise). Default 0.5 (the prior behavior); the launcher sets 0 to
    # disable soft-Z (pure hard outcome targets) per the 2026-06-04 owner decision.
    ap.add_argument("--soft-z-lambda", type=float, default=0.5)
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
        "architecture": {"candidate_radius": args.n,
                         "short_term_value_horizons": list(STV_HORIZONS)},
        "training": {"learning_rate": args.lr, "warmup_steps": args.warmup, "batch_size": args.batch,
                     "short_term_value_weight": args.short_term_value_weight},
        # Self-play row finalization. policy_surprise_* MUST be threaded here: the
        # driver does not read any TOML [samples] section, so omitting this leaves
        # the dataclass default policy_surprise_enabled=False (the bug that kept it
        # off through epoch 4). selfplay.py:run_selfplay_games reads
        # config.samples.policy_surprise_enabled to gate materialize_policy_surprise_rows.
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
            # KataGo Playout Cap Randomization (threaded here; TOML [selfplay] unread).
            "pcr_enabled": bool(args.pcr),
            "pcr_full_proportion": args.pcr_full_proportion,
            "pcr_fast_visits": args.pcr_fast_visits,
        },
    })

    # --- model: resume from RL latest if present, else seed from BC -----------
    latest = ckpt_dir / "hexgt_rl_latest.pt"
    start_epoch = 0
    # STV GRAFT: force the short-term-value horizons on (overriding a checkpoint
    # built with []), build the model WITH the 3 aux heads, and load strict=False
    # so the trunk + policy/value/opp_policy load EXACTLY while the new stv heads
    # keep fresh init. _validate_stv_resume_load guards that ONLY stv-head params
    # were missing. The stv heads are aux (with_aux), so policy/value/play output
    # is identical at resume; only the training loss gains the stvalue_* terms.
    resume_incomplete_train = False
    vr_expanded = False
    if latest.exists():
        ck = torch.load(latest, map_location=device, weights_only=False)
        arch_meta = dict(ck["arch"])
        arch_meta["short_term_value_horizons"] = list(STV_HORIZONS)
        model = build_model(arch_meta, device)
        vr_expanded = expand_value_readout_columns(model, ck["model"])
        stv_expanded = expand_stv_readout_columns(model, ck["model"])
        _validate_stv_resume_load(model.load_state_dict(ck["model"], strict=False))
        loaded_epoch = int(ck.get("rl_epoch", 0))
        # RE-TRAIN-ON-CRASH: a checkpoint whose epoch training did NOT complete (a
        # mid-training crash) RE-RUNS that epoch's training from the existing
        # self-play shards instead of advancing past it (which under-trained/skipped
        # the epoch). Old checkpoints lack the flag -> treated as complete (advance),
        # preserving prior behavior.
        epoch_train_complete = bool(ck.get("epoch_train_complete", True))
        start_epoch, resume_incomplete_train = resume_plan(loaded_epoch, epoch_train_complete)
        had_stv = bool(tuple(ck["arch"].get("short_term_value_horizons", ())))
        seed_desc = (f"RESUME from {latest.name} (rl_epoch={ck.get('rl_epoch')}, step={ck.get('step')})"
                     + ("" if had_stv else f" + GRAFT STV-heads {STV_HORIZONS} (fresh-init, aux-only)")
                     + (f" + EXPAND STV readout to [SIDE|PMA] (zero-init PMA cols, identical first step): {len(stv_expanded)} heads" if stv_expanded else "")
                     + ("" if epoch_train_complete else f" + RE-TRAIN incomplete epoch {loaded_epoch} (reuse shards, no re-self-play)"))
    else:
        ck = torch.load(args.bc_seed, map_location=device, weights_only=False)
        arch_meta = dict(ck["arch"])
        arch_meta["short_term_value_horizons"] = list(STV_HORIZONS)
        model = build_model(arch_meta, device)
        vr_expanded = expand_value_readout_columns(model, ck["model"])
        stv_expanded = expand_stv_readout_columns(model, ck["model"])
        _validate_stv_resume_load(model.load_state_dict(ck["model"], strict=False))
        seed_desc = (f"SEED from {Path(args.bc_seed).name} (step={ck.get('step')}"
                     f"{', rl_epoch=' + str(ck['rl_epoch']) if 'rl_epoch' in ck else ''})"
                     f" + STV-heads {STV_HORIZONS}")

    # ZERO-INIT LAYER-EXPANSION (one-time): a checkpoint predating the current
    # feature schema (no/lower feature_schema_version) carries random, never-
    # trained node_in columns for the newly-activated slots. Zero them so the
    # FIRST forward after resume is byte-identical to that checkpoint; the next
    # save stamps the current version, so later resumes skip this. Gated on the
    # ABSOLUTE version in the checkpoint, so the supervisor's fixed relaunch
    # command can't re-zero already-learned columns.
    loaded_fsv = int(ck.get("feature_schema_version", 1))
    if loaded_fsv < FEATURE_SCHEMA_VERSION:
        # Zero only the columns introduced AFTER the checkpoint's version (a v2
        # checkpoint keeps its trained v2 columns; only the new v3 slots are
        # zeroed), so the first forward is identical and trained weights survive.
        target_slots = feature_slots_after(loaded_fsv)
        zeroed = zero_init_expanded_feature_columns(model, target_slots)
        seed_desc += (f" + ZERO-INIT feature-expansion v{loaded_fsv}->v{FEATURE_SCHEMA_VERSION} "
                      f"(zeroed {len(zeroed)} node_in cols {zeroed})")
    if vr_expanded:
        seed_desc += " + EXPAND value-readout to global-pool [side|mean|max] (zero-init, identical first step)"

    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    trainer = HexgtTrainer(model=model, config=cfg, optimizer=opt)
    trainer.cuda_retry_log = lambda m: log(m, fh)  # surface transient-CUDA retries in the run log
    # Load optimizer momentum from the seed/resume checkpoint when present (the BC
    # seed has none -> fresh momentum; an RL-epoch checkpoint does -> warm-start).
    if "optimizer" in ck:
        try:
            opt.load_state_dict(ck["optimizer"])
            log(f"  (optimizer state restored from {Path(args.bc_seed).name if not latest.exists() else latest.name})", fh)
        except Exception as exc:  # optimizer state shape drift -> start fresh momentum
            log(f"  (optimizer state not restored: {exc})", fh)
        # COLUMN-EXPANSION OPTIMIZER RECONCILIATION: Optimizer.load_state_dict maps
        # saved moments to params POSITIONALLY and does NOT check per-tensor shapes,
        # so a zero-init column-expansion (STV / value readout, e.g. 168->504) silently
        # installs OLD-width exp_avg/exp_avg_sq onto a now-wider param; the mismatch
        # only surfaces later inside adam.step()'s _foreach_lerp_ (RuntimeError: size
        # of tensor a (168) must match b (504)) -- past the try/except above. Drop the
        # stale per-param state for any param whose moment shape != the param shape so
        # Adam reinitializes fresh ZERO moments for exactly those params (correct for
        # the zero-init'd new columns), while every unchanged param keeps warm-start.
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

    # One-time confirmation that the production memory/compile config is engaged:
    # torch.compile active + expandable_segments (so self-play runs on the ~2.4GB
    # compiled+expandable path, not the ~11.8GB eager cell). Loud if either is OFF.
    if device.type == "cuda":
        alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        expandable = "expandable_segments:true" in alloc_conf.lower()
        total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
        warn = "" if (compiled and expandable) else "  <-- WARNING: expected compile+expandable for the production mem profile"
        log(f"    GPU mem config: compile={compiled} expandable_segments={'ON' if expandable else 'OFF'} "
            f"(PYTORCH_CUDA_ALLOC_CONF={alloc_conf or 'unset'}) total={total_gb:.1f}GB{warn}", fh)

    log(f"=== hexgt RL start: {seed_desc} | params={nparams:,} n={args.n} device={device} "
        f"compile={compiled} ===", fh)
    log(f"    epochs={args.epochs} games/epoch={args.games_per_epoch} visits={args.visits} "
        f"active={args.active} vbatch={args.vbatch} | train_steps/epoch={args.train_steps_per_epoch} "
        f"batch={args.batch} lr={args.lr} replay_window={args.replay_window_epochs} ep | "
        f"loss_w: policy={cfg.training.policy_weight} value={cfg.training.value_weight} "
        f"opp={cfg.training.opp_policy_weight} stv={cfg.training.short_term_value_weight} "
        f"soft_z_lambda={cfg.samples.soft_z_lambda} policy_surprise={cfg.samples.policy_surprise_enabled}", fh)
    _sp = cfg.selfplay
    if _sp.temperature_halflife > 0:
        _temp_desc = (f"halflife start={_sp.temperature} floor={_sp.temperature_floor} "
                      f"halflife={_sp.temperature_halflife} "
                      f"[t(0)={_sp.temperature_floor + (_sp.temperature - _sp.temperature_floor):.3f} "
                      f"t(80)={_sp.temperature_floor + (_sp.temperature - _sp.temperature_floor) * 2 ** (-80 / _sp.temperature_halflife):.3f} "
                      f"t(200)={_sp.temperature_floor + (_sp.temperature - _sp.temperature_floor) * 2 ** (-200 / _sp.temperature_halflife):.3f}]")
    else:
        _temp_desc = (f"linear start={_sp.temperature} final={_sp.final_temperature} "
                      f"decay_moves={_sp.temperature_decay_moves} floor={_sp.temperature_floor}")
    log(f"    move-temperature: {_temp_desc}", fh)
    # PCR confirmation: a loud startup line so the effective full/fast caps + proportion
    # are verifiable in the log (the driver builds cfg from CLI args, not TOML). When OFF
    # this prints the plain full-search visit count so the line is always present.
    if _sp.pcr_enabled:
        _exp_visits = (_sp.pcr_full_proportion * _sp.search_visits
                       + (1.0 - _sp.pcr_full_proportion) * _sp.pcr_fast_visits)
        log(f"    PCR (KataGo playout-cap-randomization): ENABLED p_full={_sp.pcr_full_proportion} "
            f"full_visits={_sp.search_visits} fast_visits={_sp.pcr_fast_visits} "
            f"-> ~{_exp_visits:.0f} visits/move avg; full searches recorded (noise+forced-playouts"
            f"+temperature), fast NOT recorded (no-noise, greedy). recorded rows/epoch ~halve.", fh)
    else:
        log(f"    PCR (KataGo playout-cap-randomization): DISABLED (every move full search "
            f"@ {_sp.search_visits} visits, all recorded)", fh)

    def save(tag, rl_epoch, epoch_train_complete=True):
        payload = {
            "model": model.state_dict(), "optimizer": opt.state_dict(),
            "train_state": trainer.train_state.to_dict(), "arch": arch_meta,
            "step": trainer.train_state.global_step, "rl_epoch": rl_epoch,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            # RE-TRAIN-ON-CRASH marker: True only when this epoch's training reached
            # its step target. A crash mid-training writes False so resume re-runs
            # the epoch's training (from existing shards) instead of skipping it.
            "epoch_train_complete": epoch_train_complete,
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
        # vs dense_cnn e24 — SKIPPED cleanly when the e24 checkpoint is unavailable
        # (it was lost in the E: integrity event). SealBot (below) is the strength
        # benchmark; a missing dense ckpt must not nan/spam or break the run.
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
            # FIXED eval seed across epochs -> the SAME games each time, so a
            # win-rate change is the model improving, not seed variance (paired).
            r = run_head_to_head(make_hexgt, make_dense, games=args.eval_games,
                                 output_dir=eval_dir / f"epoch_{rl_epoch:06d}_vs_dense",
                                 base_seed=4242, max_actions=args.eval_max_actions,
                                 game_id_prefix=f"e{rl_epoch}vsdense")
            results["vs_dense_cnn_e24"] = r.as_dict()
        except FileNotFoundError as exc:
            log(f"  eval: vs-dense_cnn SKIPPED — {exc}", fh)
            results["vs_dense_cnn_e24_skipped"] = str(args.dense_ckpt)
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
    cur_epoch = None            # epoch currently being processed (for crash bookkeeping)
    cur_epoch_shards_ready = False  # cur_epoch's self-play shards are on disk (re-trainable)
    # Per-RUN non-finite-logit sanitization tally (per-epoch comes from the self-play
    # result). A rising count is a red flag (fp16 overflow / model instability), so it
    # is surfaced in the run log alongside the per-epoch breakdown.
    run_sanitized_logits = 0
    run_sanitized_excluded = 0
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
            cur_epoch = rl_epoch
            # On a RE-TRAIN resume (mid-training crash), the crashing epoch's
            # self-play shards are already on disk — skip self-play and go straight
            # to re-training over them (only for the first epoch after such a resume).
            skip_selfplay = should_skip_selfplay(rl_epoch, start_epoch, resume_incomplete_train)
            cur_epoch_shards_ready = skip_selfplay  # shards already exist in this case
            # --- 1) self-play ----------------------------------------------------
            if skip_selfplay:
                log(f"epoch {rl_epoch}: RE-TRAIN after mid-training crash — reusing "
                    f"existing self-play shards, skipping self-play", fh)
            else:
                model.eval()
                sp = run_selfplay_games(
                    model, cfg, num_games=args.games_per_epoch, output_dir=selfplay_dir,
                    epoch=rl_epoch, device=str(device), fp16=(device.type == "cuda"),
                    base_seed=1000 + rl_epoch * 7919, active_games=args.active,
                    virtual_batch_size=args.vbatch, collect_examples=3,
                )
                cur_epoch_shards_ready = True  # self-play finished -> shards on disk
                run_sanitized_logits += int(sp.sanitized_logit_events)
                run_sanitized_excluded += int(sp.sanitized_samples_excluded)
                # Sanitization segment: appended only when the guard fired this epoch
                # (clean runs stay quiet; a non-zero count is loud). Placed LAST so the
                # dashboard bridge's existing selfplay-line regexes are unaffected.
                saniti_str = (
                    f" | SANITIZED {sp.sanitized_logit_events} logits/"
                    f"{sp.sanitized_search_rounds} rounds excl={sp.sanitized_samples_excluded} "
                    f"(run {run_sanitized_logits} logits/{run_sanitized_excluded} excl)"
                    if sp.sanitized_logit_events else ""
                )
                # PCR segment: full/fast mix, recorded rows, avg visits/move. Appended
                # after the Q-fields (and the sanitization segment) so the existing
                # dashboard selfplay-line parsing is unaffected. Empty when PCR is off.
                pcr_str = (
                    f" | PCR full={sp.full_search_count}/{sp.full_search_count + sp.fast_search_count} "
                    f"rec={sp.recorded_positions} (~{sp.mean_search_visits:.0f}v/move "
                    f"N={sp.pcr_full_visits}/n={sp.pcr_fast_visits} p={sp.pcr_full_proportion:.2f})"
                    if sp.pcr_enabled else ""
                )
                log(f"epoch {rl_epoch} selfplay: {sp.completed_games}C/{sp.truncated_games}T games, "
                    f"{sp.searched_positions} pos, {sp.positions_per_second:.1f} pos/s | "
                    f"cand={sp.mean_candidate_count:.0f} | "
                    f"Q1 decisive={sp.decisive_fraction:.1%} len_med={sp.game_length_median:.0f} | "
                    f"Q2 uniq_open={sp.opening_unique_fraction:.1%} m2H={sp.move2_entropy:.2f} | "
                    f"Q3 visitH={sp.mean_visit_entropy:.2f} priorH={sp.mean_prior_entropy:.2f} | "
                    f"Q4 |val|={sp.mean_abs_value:.2f} draw={sp.draw_fraction:.1%} | "
                    f"Q5 forced={sp.forced_move_fraction:.1%}{saniti_str}{pcr_str}", fh)
                # Per-epoch GPU reserved-memory peak for the self-play phase (the
                # VRAM-binding phase): verifies the run holds the compiled+expandable
                # envelope and flags any creep toward the card limit.
                log_gpu_mem(f"epoch {rl_epoch} selfplay", fh)
                # Also persist the full self-play metric dict per epoch for later analysis.
                with open(eval_dir / f"epoch_{rl_epoch:06d}_selfplay.json", "w") as sp_fh:
                    json.dump(sp.as_dict(), sp_fh, indent=2)
                # Dump example traces for play-style analysis.
                with open(eval_dir / f"epoch_{rl_epoch:06d}_examples.json", "w") as ex_fh:
                    json.dump(sp.example_games, ex_fh)

            # --- 2) train over the recency-weighted, cap-bounded replay window ---
            # Release the self-play inference reservation BEFORE training allocates.
            # run_selfplay_games builds a local HexgtInference (MCTS session cache +
            # vbatch=128 eval buffers) that goes out of scope on return but leaves
            # ~5GB cached in the torch allocator; on this 12GB card the training
            # batch + autograd graph + grads then crest the VRAM ceiling at the
            # self-play->train transition (telemetry-confirmed OOM at epoch-0 train
            # step ~6, 2026-06-04: nvidia-smi 7661->11978 MiB / 97.5%). Returning the
            # cached pool first restores the headroom. Cost: one sync + gc per epoch.
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
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
            # Per-head loss components: log the short-term-value heads (stvalue_<h>)
            # alongside total/policy/value/opp so the dashboard can surface EVERY
            # head. Built dynamically from whatever stvalue_* keys the loss emits
            # (graceful if the horizons change), sorted by horizon for stable
            # columns; inserted between opp= and prune= (additive — the bridge
            # regexes skip over it via .*?, so old log readers stay valid).
            stv_keys = sorted((k for k in comp_sum if k.startswith("stvalue_")),
                              key=lambda k: int(k.rsplit("_", 1)[1]))
            stv_str = "".join(f" {k.replace('stvalue_', 'stv')}={avg(k):.4f}" for k in stv_keys)
            log(f"epoch {rl_epoch} train: {comp_n} steps (step={trainer.train_state.global_step}) "
                f"total={avg('total'):.4f} policy={avg('policy'):.4f} value={avg('value'):.4f} "
                f"opp={avg('opp_policy'):.4f}{stv_str} prune={trainer.prune_rate:.1%} {ms_per:.0f}ms/step "
                f"window={len(win_epochs)}ep[{span}] {len(shards)}sh "
                f"pool~{pool_pos // 1000}k/{args.replay_pool_cap // 1000}k "
                f"decay={args.replay_recency_decay}", fh)

            # --- 2b) value-head CALIBRATION probe (cheap, diagnostic) ------------
            # Same-board both-perspectives optimism on this epoch's self-play boards,
            # so the dashboard/logs surface value-head calibration drift over the run
            # (the run's |val| was rising + games shortening — this quantifies whether
            # the head is staying zero-sum-consistent or drifting optimistic). Failsafe.
            calib = measure_value_optimism(
                model, selfplay_dir / f"epoch_{rl_epoch:06d}.hxr", device, args.n, max_pos=64)
            log(f"epoch {rl_epoch} calib: optimism_sum_mean={calib.get('optimism_sum_mean', float('nan')):+.4f} "
                f"both_pos={calib.get('optimism_both_positive_frac', float('nan')):.0%} "
                f"n={calib.get('optimism_positions', 0)} "
                f"(0=calibrated >0=optimistic)", fh)
            with open(eval_dir / f"epoch_{rl_epoch:06d}_calibration.json", "w") as cfh:
                json.dump(calib, cfh, indent=2)

            # --- 3) checkpoint ---------------------------------------------------
            save(f"epoch{rl_epoch:06d}", rl_epoch); last_save = rl_epoch

            # --- 4) periodic eval (absolute-epoch schedule) ----------------------
            if eval_due(rl_epoch, args.eval_every, args.epochs):
                do_eval(rl_epoch)

            log(f"epoch {rl_epoch} done in {(time.perf_counter()-ep_t0)/60:.1f} min", fh)
        log(f"=== hexgt RL DONE (through epoch {last_save}) ===", fh)
    except Exception:
        log("EXCEPTION — saving crash checkpoint:\n" + traceback.format_exc(), fh)
        if cur_epoch is not None and cur_epoch_shards_ready:
            # Crash during/after TRAINING: the crashing epoch's self-play shards are
            # on disk, so keep the partial weights but mark this epoch's training
            # INCOMPLETE — resume RE-TRAINS it from those shards (no re-self-play,
            # no skip). This is the common case (the CUDA crashes hit the training
            # forward/backward).
            save("crash", cur_epoch, epoch_train_complete=False)
            log(f"  (crash during epoch {cur_epoch} training -> saved INCOMPLETE; resume will re-train it)", fh)
        else:
            # Crash during self-play (weights unchanged since the last checkpoint):
            # leave the existing latest checkpoint as the resume point so resume
            # cleanly redoes this epoch (self-play + train) from the last completed
            # epoch — overwriting would risk skipping the epoch.
            log("  (crash before training shards were ready; leaving last checkpoint as the resume point)", fh)
        raise
    finally:
        fh.close()


if __name__ == "__main__":
    main()
