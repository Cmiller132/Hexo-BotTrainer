export VIRTUAL_ENV=/root/.venvs/hexgt-build; export PATH="$VIRTUAL_ENV/bin:$PATH"; export CUDA_VISIBLE_DEVICES=
R=/mnt/e/Hexo-BotTrainer-hexgt
export PYTHONPATH="$R/packages/hexo_engine/python:$R/packages/hexo_utils/python:$R/packages/hexo_runner/python:$R/packages/hexo_train/python:$R/packages/hexo_models/python"
$VIRTUAL_ENV/bin/python - <<'PY'
import tomllib, pathlib, torch
from hexo_models.dense_cnn.plugin import DenseCNNPlugin
cfg = tomllib.loads(pathlib.Path("/mnt/e/Hexo-BotTrainer-hexgt/configs/dense_cnn_rl_main1.toml").read_text())
mc = cfg["model"]["config"]
arch = mc["architecture"]
print("config arch -> channels:", arch["channels"], "residual_blocks:", arch["residual_blocks"], "input_channels:", arch["input_channels"])
net = DenseCNNPlugin().build_model({}, mc)
n = sum(p.numel() for p in net.parameters())
print(f"BUILD OK (64x10). dense_cnn 64x10 params: {n:,}")
PY
echo "exit=$?"
