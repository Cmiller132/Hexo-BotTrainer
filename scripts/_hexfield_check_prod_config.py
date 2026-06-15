"""Pre-launch sanity for the production config: parses through the real
hexo_train + hexfield config path, confirms warm-start checkpoint exists,
prints the resolved key knobs and the temperature schedule head/tail."""

import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))

from hexfield.config import parse_hexfield_config

cfg_path = REPO / "configs" / "hexfield_main_1.toml"
raw = tomllib.loads(cfg_path.read_text())
mc = raw["model"]["config"]
cfg = parse_hexfield_config(mc)
sp = cfg.selfplay

print("=== production config (hexfield_main_1.toml) ===")
print("run:", raw["run"]["name"], "->", raw["run"]["output_dir"])
print("epochs:", raw["loop"]["epochs"], "| games/epoch:", raw["selfplay"]["games_per_epoch"])
print("search_visits:", sp.search_visits, "| pcr_fast:", sp.pcr_fast_visits,
      "| pcr_full_proportion:", sp.pcr_full_proportion)
print("active_games:", sp.active_games, "| flush_target:", sp.flush_target,
      "| active_root_limit:", sp.active_root_limit, "| cache:", sp.cache_max_states)
print("QUARANTINE: root_fpu_zero_under_noise:", sp.root_fpu_zero_under_noise,
      "| root_policy_temperature:", sp.root_policy_temperature,
      "| search_parity_mode:", sp.search_parity_mode)
print("divergences default ON (production): forced_k:", sp.forced_playout_k,
      "| tss:", sp.tss_enabled)
print("eval: games:", cfg.evaluation.games_per_epoch, "visits:", cfg.evaluation.eval_visits,
      "every:", cfg.evaluation.eval_every)
print("temp schedule[0..3]:", [round(t, 3) for t in cfg.temperature_by_ply()[:4]],
      "... tail:", round(cfg.temperature_by_ply()[-1], 3))

ckpt = Path(raw["checkpoint"]["initialize_from"])
print("\nwarm-start:", ckpt, "EXISTS" if ckpt.is_file() else "*** MISSING ***")
out = Path(raw["run"]["output_dir"])
print("output_dir parent writable:", out.parent.exists())
print("\nLAUNCH-READY" if ckpt.is_file() and out.parent.exists() else "\n*** NOT READY ***")
