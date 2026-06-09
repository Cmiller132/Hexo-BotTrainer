export VIRTUAL_ENV=/root/.venvs/hexgt-build
export PATH="$VIRTUAL_ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES=
R=/mnt/e/Hexo-BotTrainer-hexgt
export PYTHONPATH="$R/packages/hexo_engine/python:$R/packages/hexo_utils/python:$R/packages/hexo_runner/python:$R/packages/hexo_train/python:$R/packages/hexo_models/python:$R/packages/hexo_frontend/python"
$VIRTUAL_ENV/bin/python - <<'PY'
import hexo_models._rust as r
print("native submodules:", [m for m in dir(r) if not m.startswith("_")], "| dense_cnn:", hasattr(r,"dense_cnn"))
# load the dense_cnn plugin
from hexo_train.registry import load_model_plugin
try:
    pl = load_model_plugin("dense_cnn")
    print("plugin loaded:", type(pl).__name__)
except Exception as e:
    print("plugin load via name failed:", repr(e))
    pl = None
# parse the 96x8 config
import tomllib, pathlib
cfg = tomllib.loads(pathlib.Path("/mnt/e/Hexo-BotTrainer-hexgt/configs/dense_cnn_model1_target_96x8.toml").read_text())
mc = cfg.get("model",{}).get("config",{})
print("config model.config keys:", list(mc.keys())[:8])
print("search_visits:", mc.get("selfplay",{}).get("search_visits"), "channels:", mc.get("architecture",{}).get("channels") if "architecture" in mc else mc.get("channels"))
PY
echo "exit=$?"
