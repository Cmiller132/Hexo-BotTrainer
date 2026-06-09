export VIRTUAL_ENV=/root/.venvs/hexgt-build
export PATH="$VIRTUAL_ENV/bin:$HOME/.cargo/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/mnt/e/Hexo-BotTrainer-hexgt
export PYTHONPATH="$R/packages/hexgnn/python:$R/packages/hexo_engine/python:$R/packages/hexo_utils/python:$R/packages/hexo_runner/python:$R/packages/hexo_train/python:$R/packages/hexo_models/python"
echo "########## A) new .so import + RESUME load of latest.pt ##########"
$VIRTUAL_ENV/bin/python - <<'PY'
import torch
import hexo_models._rust as r
print("native submodules:", [m for m in dir(r) if not m.startswith("_")])
assert hasattr(r, "hexgnn"), "hexgnn native submodule missing!"
from hexgnn.architecture import HexgnnNetwork
ck = torch.load("/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1/checkpoints/hexgnn_rl_latest.pt", map_location="cpu", weights_only=False)
a = dict(ck["arch"]); print("arch:", a, "| rl_epoch:", ck.get("rl_epoch"), "step:", ck.get("step"))
m = HexgnnNetwork(token_dim=a["token_dim"], gnn_layers=a["gnn_layers"], attention_heads=a.get("attention_heads",4),
                  value_pma_seeds=a.get("value_pma_seeds",2), value_head_use_side=a.get("value_head_use_side",True),
                  steerable_channels=a.get("steerable_channels",0))
miss, unexp = m.load_state_dict(ck["model"], strict=False)
print(f"RESUME strict load -> missing={list(miss)} unexpected={list(unexp)}")
assert not miss and not unexp, "RESUME LOAD MISMATCH"
print("RESUME LOAD: OK (exact)")
PY
echo "exitA=$?"
echo ""
echo "########## B) self-play smoke on new build (int32 path), 8 games ##########"
timeout 90 $VIRTUAL_ENV/bin/python -u $R/_hexgnn_posps.py 96 2 0 512 256 8 512 16 2 4 2>&1 | grep -iE "CONFIG|pos/s|Error|Traceback|panic" | grep -v "mcts-trace" | tail -12
echo "exitB=$?"
