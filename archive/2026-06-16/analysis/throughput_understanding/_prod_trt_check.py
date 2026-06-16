import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
for p in ("hexo_engine", "hexo_utils", "hexo_runner", "hexo_train", "hexo_models", "hexo_frontend"):
    sys.path.insert(0, str(REPO / "packages" / p / "python"))
import tomllib, torch
from hexo_models.dense_cnn.plugin import DenseCNNPlugin
from hexo_models.dense_cnn.inference import DenseCNNInference
s = tomllib.load(open(REPO / "configs/dense_cnn_model1_target_96x6.toml", "rb"))["model"]["config"]
m = DenseCNNPlugin().build_model(game_spec={}, config=s)
m.load_state_dict(torch.load(REPO / "runs/dense_cnn_model1_target_96x6/checkpoints/bootstrap_sealbot_prefit.pt", map_location="cpu")["model_state"], strict=True)
inf = DenseCNNInference(m, device="cuda", amp=True, max_batch_size=1024, use_trt=True)
print("PROD-PATH trt_adopted=", inf.trt_info.get("adopted"), "argmax=", inf.trt_info.get("policy_argmax_match"),
      "val_err=", inf.trt_info.get("value_max_abs_err"), "using_trt_forward=", inf._trt_forward is not None)
