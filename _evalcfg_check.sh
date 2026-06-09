export VIRTUAL_ENV=/root/.venvs/hexgt-build; export PATH="$VIRTUAL_ENV/bin:$PATH"; export CUDA_VISIBLE_DEVICES=
R=/mnt/e/Hexo-BotTrainer-hexgt
export PYTHONPATH="$R/packages/hexo_engine/python:$R/packages/hexo_utils/python:$R/packages/hexo_runner/python:$R/packages/hexo_train/python:$R/packages/hexo_models/python"
$VIRTUAL_ENV/bin/python - <<'PY'
import tomllib, pathlib
from hexo_models.dense_cnn.config import parse_model1_config
cfg = tomllib.loads(pathlib.Path("/mnt/e/Hexo-BotTrainer-hexgt/configs/dense_cnn_rl_main1.toml").read_text())
mc = cfg["model"]["config"]
parsed = parse_model1_config(mc)   # this raises if 'eval_every' is an unknown key
print("PARSED OK. eval_every =", parsed.evaluation.eval_every, "| games_per_epoch =", parsed.evaluation.games_per_epoch)
# gate logic check (epoch % eval_every == 0 evals; else skips), epochs are 1-based in the loop
ee = parsed.evaluation.eval_every
evals = [e for e in range(1, 31) if not (ee > 1 and e % ee != 0)]
skips = [e for e in range(1, 31) if (ee > 1 and e % ee != 0)]
print("with eval_every=%d -> EVAL epochs:" % ee, evals)
print("                     SKIP epochs:", skips[:12], "...")
PY
echo "exit=$?"
