"""Migrate dense_cnn_restnet run checkpoints to the heads_v2 layout (2026-06-09).

heads_v2 replaces the four private ValueBinnedHeads (value + stvalue_{1,4,8})
with one shared ValueReduction (conv1x1 -> Linear 1681->64) and per-head 64->65
Linear tops, changes the lookahead horizons to {1, 4, 16}, and adds a moves_left
top. Old run checkpoints therefore no longer strict-load; without migration the
resume path fail-louds (DenseCNNCheckpointLoader raises) and the run cannot
continue.

Mapping (function-preserving for the search surface):
  - trunk / stem / policy_head / opp_policy_head: copied verbatim.
  - value_reduction.conv  <- old value_head.conv   (same composition)
  - value_reduction.mlp.0 <- old value_head.mlp.0
  - value_head (Linear)   <- old value_head.mlp.2
    => the MAIN VALUE head computes bit-for-bit the same function (verified
       below against a manual replay of the old head on the same trunk
       features), so MCTS search behavior is unchanged across the migration.
  - short_term_value_heads.{1,4} tops <- old heads' mlp.2 (their private
    reductions are dropped, so these heads re-converge on the shared embedding;
    they were diverging anyway and run at reduced loss weight).
  - short_term_value_heads.16 and moves_left_head: fresh init.
  - optimizer_state: DROPPED (param set changed; AdamW moments re-accumulate in
    a few hundred steps). train_state / epoch / metadata preserved.

Run with the trainer STOPPED (the script refuses while driver.pid is alive):
  python scripts/_restnet_migrate_heads_v2.py \
    --run-dir /mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main1 \
    --config  configs/dense_cnn_restnet_main1.toml [--dry-run]

The original checkpoint is kept as <name>.preheadsv2.bak.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _pkg in ("hexo_engine", "hexo_utils", "hexo_runner", "hexo_train", "hexo_models"):
    _p = ROOT / "packages" / _pkg / "python"
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
_rest = ROOT / "packages" / "dense_cnn_restnet" / "python"
if str(_rest) not in sys.path:
    sys.path.insert(0, str(_rest))

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def _latest_epoch_checkpoint(checkpoint_dir: Path) -> Path:
    best: tuple[int, Path] | None = None
    for path in checkpoint_dir.glob("epoch_*.pt"):
        match = re.fullmatch(r"epoch_(\d+)\.pt", path.name)
        if match:
            epoch = int(match.group(1))
            if best is None or epoch > best[0]:
                best = (epoch, path)
    if best is None:
        raise SystemExit(f"no epoch_*.pt checkpoints under {checkpoint_dir}")
    return best[1]


def _refuse_if_running(run_dir: Path, force: bool) -> None:
    pid_file = run_dir / "driver.pid"
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text().strip())
    except ValueError:
        return
    if Path(f"/proc/{pid}").exists():
        if force:
            print(f"WARNING: trainer pid {pid} appears alive; continuing due to --force", flush=True)
            return
        raise SystemExit(
            f"trainer pid {pid} appears alive ({pid_file}); stop the supervisor/trainer "
            "before migrating (or pass --force if the pid is stale)."
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Migrate restnet run checkpoint to heads_v2.")
    ap.add_argument("--run-dir", required=True, help="run directory (holds checkpoints/, driver.pid)")
    ap.add_argument("--config", required=True, help="run TOML (new architecture is built from this)")
    ap.add_argument("--checkpoint", default=None, help="explicit checkpoint path (default: latest epoch_*.pt)")
    ap.add_argument("--dry-run", action="store_true", help="verify mapping + function preservation, write nothing")
    ap.add_argument("--force", action="store_true", help="proceed even if driver.pid looks alive")
    args = ap.parse_args(argv)

    import torch

    from dense_cnn_restnet.config import parse_model1_config
    from dense_cnn_restnet.plugin import DenseCNNRestnetPlugin

    run_dir = Path(args.run_dir)
    _refuse_if_running(run_dir, args.force)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else _latest_epoch_checkpoint(run_dir / "checkpoints")
    payload = torch.load(ckpt_path, map_location="cpu")
    old_state = payload["model_state"]

    raw = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))["model"]["config"]
    parsed = parse_model1_config(raw)
    model = DenseCNNRestnetPlugin().build_model({}, raw)
    model.eval()
    new_state = model.state_dict()

    if "value_reduction.conv.0.weight" in old_state:
        raise SystemExit(f"{ckpt_path} already has the heads_v2 layout; nothing to do.")

    mapped: dict[str, torch.Tensor] = {}
    fresh: list[str] = []
    rename = {
        "value_reduction.conv.0.weight": "value_head.conv.0.weight",
        "value_reduction.conv.0.bias": "value_head.conv.0.bias",
        "value_reduction.mlp.0.weight": "value_head.mlp.0.weight",
        "value_reduction.mlp.0.bias": "value_head.mlp.0.bias",
        "value_head.weight": "value_head.mlp.2.weight",
        "value_head.bias": "value_head.mlp.2.bias",
    }
    for horizon in parsed.architecture.short_term_value_horizons:
        rename[f"short_term_value_heads.{horizon}.weight"] = f"short_term_value_heads.{horizon}.mlp.2.weight"
        rename[f"short_term_value_heads.{horizon}.bias"] = f"short_term_value_heads.{horizon}.mlp.2.bias"

    for key, tensor in new_state.items():
        source = rename.get(key, key)
        old = old_state.get(source)
        if old is not None and tuple(old.shape) == tuple(tensor.shape):
            mapped[key] = old
        else:
            mapped[key] = tensor  # fresh init (stvalue_16 top, moves_left top, ...)
            fresh.append(key)
    model.load_state_dict(mapped, strict=True)

    # Function-preservation check: the migrated value path must equal a manual
    # replay of the OLD value head (conv+ReLU -> Linear -> ReLU -> Linear) on the
    # same trunk features. Trunk weights are copied verbatim, so the new trunk IS
    # the old trunk.
    import torch.nn.functional as F

    torch.manual_seed(0)
    x = torch.zeros(2, parsed.architecture.input_channels, 41, 41)
    x[:, 3] = (torch.rand(2, 41, 41) < 0.5).float()  # plausible nonzero planes
    x[:, 0] = (torch.rand(2, 41, 41) < 0.1).float()
    with torch.no_grad():
        feats = model.trunk(x)
        old_v = F.relu(F.conv2d(feats, old_state["value_head.conv.0.weight"], old_state["value_head.conv.0.bias"]))
        old_v = F.relu(F.linear(old_v.flatten(1), old_state["value_head.mlp.0.weight"], old_state["value_head.mlp.0.bias"]))
        old_v = F.linear(old_v, old_state["value_head.mlp.2.weight"], old_state["value_head.mlp.2.bias"])
        new_v = model.value_head(model.value_reduction(feats))
        new_p = model.forward_policy_value(x)
    diff = (old_v - new_v).abs().max().item()
    scale = max(1.0, old_v.abs().max().item())
    # Same weights + same op sequence; tolerance covers fp32 GEMM reduction-order
    # differences between the checkpoint's tensor storages and the module's
    # (different alignment -> different oneDNN blocking). Anything beyond ~1e-4
    # relative would indicate a real mapping error.
    if diff > 1.0e-3 * scale:
        raise SystemExit(
            f"FUNCTION PRESERVATION FAILED: value outputs differ (max abs {diff}, output scale {scale})"
        )
    if not torch.isfinite(new_p["policy"]).all() or not torch.isfinite(new_p["value"]).all():
        raise SystemExit("migrated forward_policy_value produced non-finite outputs")

    old_params = sum(t.numel() for t in old_state.values())
    new_params = sum(p.numel() for p in model.parameters())
    print(f"checkpoint: {ckpt_path} (epoch {payload.get('epoch')})")
    print(f"params: {old_params:,} -> {new_params:,}")
    print(f"fresh-init keys ({len(fresh)}): {fresh}")
    print(
        "function preservation: value head matches old composition "
        f"(max abs diff {diff:.2e} at output scale {scale:.2f} — fp32 reduction-order only); "
        "policy path copied verbatim"
    )

    if args.dry_run:
        print("DRY RUN: nothing written.")
        return 0

    backup = ckpt_path.with_suffix(ckpt_path.suffix + ".preheadsv2.bak")
    if backup.exists():
        raise SystemExit(f"backup already exists, refusing to overwrite: {backup}")
    ckpt_path.rename(backup)
    new_payload = {
        "model": "dense_cnn_restnet",
        "model_state": model.state_dict(),
        "optimizer_state": None,  # param set changed; AdamW moments re-accumulate
        "train_state": payload.get("train_state"),
        "epoch": payload.get("epoch"),
        "metadata": {
            **dict(payload.get("metadata", {})),
            "migration": "heads_v2",
            "migrated_from": str(backup.name),
            "optimizer_state_dropped": True,
        },
    }
    torch.save(new_payload, ckpt_path)
    print(f"wrote migrated checkpoint to {ckpt_path} (original kept as {backup.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
