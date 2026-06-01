"""Head-to-head: hexgt (a trained checkpoint) vs dense_cnn epoch-24 and vs
SealBot, deterministic + matched search budget (the "is the GNN competitive?"
answer). Uses the hexgt run_head_to_head match driver (alternates colors, fixed
per-game seeds, greedy/no-noise eval). dense_cnn e24 is loaded READ-ONLY from
the live tree; nothing there is mutated.

Usage:
  python scripts/_head_to_head.py --hexgt-ckpt <ckpt.pt> --dense-ckpt <e24.pt> \
      --dense-config configs/dense_cnn_model1_target_96x8.toml --games 40 --visits 200
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import torch

from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.config import parse_hexgt_config
from hexo_models.hexgt.evaluation import make_hexgt_factory, run_head_to_head


def build_hexgt(ckpt_path, visits, device, compile_model):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck["arch"]
    model = HexgtNetwork(
        token_dim=a["token_dim"], gnn_layers=a["gnn_layers"], ctx_layers=a["ctx_layers"],
        ffn_dim=a["ffn_dim"], attention_heads=a["attention_heads"],
        short_term_value_horizons=tuple(a["short_term_value_horizons"]),
    ).to(device).eval()
    model.load_state_dict(ck["model"])
    if compile_model and device.type == "cuda":
        model.forward_policy_value = torch.compile(model.forward_policy_value, dynamic=True)
    cfg = parse_hexgt_config({"device": str(device), "selfplay": {"search_visits": visits}})
    step = ck.get("step", ck.get("train_state", {}).get("global_step"))
    print(f"hexgt: {sum(p.numel() for p in model.parameters()):,} params, step={step}", flush=True)
    return make_hexgt_factory(model, cfg, device=str(device), fp16=(device.type == "cuda"),
                              identity_id="hexgt", virtual_batch_size=0)


def build_densecnn(ckpt_path, config_path, visits, device):
    from hexo_models.dense_cnn.config import parse_model1_config
    from hexo_models.dense_cnn.architecture import Model1Network
    from hexo_models.dense_cnn.trainer import DenseCNNTrainer
    from hexo_models.dense_cnn.player import DenseCNNPlayer

    with open(config_path, "rb") as fh:
        toml = tomllib.load(fh)
    mc = dict(toml["model"]["config"])
    mc["device"] = str(device)
    mc.setdefault("selfplay", {})
    mc["selfplay"] = {**mc.get("selfplay", {}), "search_visits": int(visits)}
    cfg = parse_model1_config(mc)
    arch = cfg.architecture
    model = Model1Network(
        in_channels=arch.input_channels, channels=arch.channels, blocks=arch.residual_blocks,
        dropout=arch.dropout, short_term_value_horizons=arch.short_term_value_horizons,
    )
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = DenseCNNTrainer(model=model, config=cfg, optimizer=opt)
    print(f"dense_cnn e24: {sum(p.numel() for p in model.parameters()):,} params, "
          f"{arch.channels}ch x {arch.residual_blocks}blk, visits={visits}", flush=True)

    def factory(seed: int) -> DenseCNNPlayer:
        return DenseCNNPlayer(identity_id="dense-cnn-e24", model=model, trainer=trainer,
                              record_samples=False, eval_seed=seed)
    return factory


def report(name, r):
    print(f"\n=== {name} (from hexgt's perspective) ===", flush=True)
    print(f"  games={r.games} completed={r.completed} wins={r.wins} losses={r.losses} draws={r.draws}", flush=True)
    print(f"  WIN RATE (decided): {r.win_rate:.1%}   mean_turns={r.mean_turns:.1f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hexgt-ckpt", required=True)
    ap.add_argument("--dense-ckpt", default="/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8/checkpoints/epoch_000024.pt")
    ap.add_argument("--dense-config", default="configs/dense_cnn_model1_target_96x8.toml")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--visits", type=int, default=200)
    ap.add_argument("--max-actions", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_h2h")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--sealbot", action="store_true", help="also play vs SealBot best-50ms if available")
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"matched search budget: {args.visits} visits/move both sides; {args.games} games, "
          f"alternating colors, deterministic (greedy, no noise)", flush=True)

    make_hexgt = build_hexgt(args.hexgt_ckpt, args.visits, device, not args.no_compile)

    # --- hexgt vs dense_cnn e24 (the primary, self-contained answer) ----------
    make_dense = build_densecnn(args.dense_ckpt, args.dense_config, args.visits, device)
    r = run_head_to_head(make_hexgt, make_dense, games=args.games, output_dir=out / "vs_dense",
                         base_seed=args.seed, max_actions=args.max_actions, game_id_prefix="vsdense")
    report("hexgt vs dense_cnn epoch-24", r)

    # --- hexgt vs SealBot best-50ms (optional; needs a SealBot checkout) -------
    if args.sealbot:
        from hexo_runner.adapters.sealbot import SealBotConfig, SealBotPlayer
        sb_cfg = SealBotConfig(variant="best", time_limit=0.05)
        try:
            sb_cfg.validate()
            ok = True
        except Exception as exc:
            ok = False
            print(f"\n[SealBot unavailable: {exc}] — skipping hexgt-vs-SealBot.", flush=True)
        if ok:
            def make_sb(_seed):
                return SealBotPlayer(sb_cfg, player_id="sealbot-best-50ms")
            rs = run_head_to_head(make_hexgt, make_sb, games=args.games, output_dir=out / "vs_sealbot",
                                  base_seed=args.seed, max_actions=args.max_actions, game_id_prefix="vssb")
            report("hexgt vs SealBot best-50ms", rs)


if __name__ == "__main__":
    main()
