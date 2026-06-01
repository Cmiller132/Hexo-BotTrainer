"""Fast standalone check of the TRT build + correctness gate on representative inputs."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
for p in ("hexo_engine", "hexo_utils", "hexo_runner", "hexo_train", "hexo_models", "hexo_frontend"):
    sys.path.insert(0, str(REPO / "packages" / p / "python"))
import tomllib, torch
from hexo_models.dense_cnn.architecture import optimized_model1_for_inference
from hexo_models.dense_cnn.plugin import DenseCNNPlugin
from hexo_models.dense_cnn import trt_backend

section = tomllib.load(open(REPO / "configs" / "dense_cnn_model1_target_96x6.toml", "rb"))["model"]["config"]
m = DenseCNNPlugin().build_model(game_spec={}, config=section)
m.load_state_dict(torch.load(REPO / "runs/dense_cnn_model1_target_96x6/checkpoints/bootstrap_sealbot_prefit.pt", map_location="cpu")["model_state"], strict=True)
opt = optimized_model1_for_inference(m).to("cuda", memory_format=torch.channels_last).eval()
g = torch.Generator().manual_seed(3)
sample = (torch.rand((128, 13, 41, 41), generator=g) > 0.7).float()
fwd, info = trt_backend.build_trt_forward(opt, max_batch=1024, device="cuda", sample_inputs=sample)
import json
print("GATE:", json.dumps({k: info.get(k) for k in ("adopted","reason","policy_argmax_match","policy_max_abs_err","value_max_abs_err","build_seconds")}))
