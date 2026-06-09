import tomllib, pathlib
from hexgnn.config import parse_hexgnn_config
from hexgnn.plugin import get_plugin
raw = tomllib.loads(pathlib.Path("/mnt/e/Hexo-BotTrainer-hexgt/configs/hexgnn_model.toml").read_text())["model"]["config"]
cfg = parse_hexgnn_config(raw)
print("parsed arch:", cfg.architecture)
m = get_plugin().build_model({}, raw)
print("plugin built model params:", sum(p.numel() for p in m.parameters()))
# removed keys must be rejected
for bad in ("ctx_layers", "ffn_dim", "short_term_value_horizons"):
    try:
        parse_hexgnn_config({"architecture": {bad: 3}})
        print("FAIL: did not reject", bad)
    except ValueError as e:
        print("OK rejected arch.", bad)
try:
    parse_hexgnn_config({"training": {"short_term_value_weight": 0.1}})
    print("FAIL: did not reject short_term_value_weight")
except ValueError:
    print("OK rejected training.short_term_value_weight")
print("CONFIG/PLUGIN OK")
