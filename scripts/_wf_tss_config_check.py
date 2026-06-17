"""Scratch: parse configs/dense_cnn_restnet_sanity_1.toml through the real
restnet config parser (strict unknown-key rejection) and print the resolved
sanity-relevant knobs."""

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "dense_cnn_restnet" / "python"))

from dense_cnn_restnet.config import parse_model1_config  # noqa: E402

raw = tomllib.loads((ROOT / "configs" / "dense_cnn_restnet_sanity_1.toml").read_text())
cfg = parse_model1_config(raw["model"]["config"])
sp = cfg.selfplay
print("tss_enabled:", sp.tss_enabled)
print("root_fpu_zero_under_noise:", sp.root_fpu_zero_under_noise)
print("scheduler:", sp.scheduler, "| visits:", sp.search_visits, "| pcr:", sp.pcr_enabled)
print("policy_init_fraction:", sp.policy_init_fraction)
print("temperature_schedule:", sp.temperature_schedule, "| halflife_fraction:", sp.temperature_halflife_fraction)
print("root_policy_temperature:", sp.root_policy_temperature, "| fpu:", sp.fpu_reduction)
print("frozen_win_override:", sp.frozen_win_override)
print("stv horizons:", cfg.architecture.short_term_value_horizons, "| moves_left_head:", cfg.architecture.moves_left_head)
print("drop_truncated_rows:", cfg.samples.drop_truncated_rows, "| C1 knee:", cfg.samples.length_decay_moves_left_knee)
print("train_samples:", cfg.training.train_samples_per_epoch)
print("eval:", cfg.evaluation.games_per_epoch, "every", cfg.evaluation.eval_every, "| refs:", cfg.evaluation.reference_checkpoints)
print("compile:", cfg.performance.inference_use_torch_compile, "| kv_gather:", cfg.performance.attention_kv_gather, "| fp16:", cfg.performance.inference_fp16_model)
print("PARSE OK")
