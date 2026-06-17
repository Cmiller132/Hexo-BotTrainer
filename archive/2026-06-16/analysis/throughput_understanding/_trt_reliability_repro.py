"""Reliability test: build the TRT engine N times in one process via the
subprocess-isolated builder. The bug was ~33% in-process build failures; confirm
100% adopt now."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
for p in ("hexo_engine", "hexo_utils", "hexo_runner", "hexo_train", "hexo_models", "hexo_frontend"):
    sys.path.insert(0, str(REPO / "packages" / p / "python"))
import tomllib, torch
from hexo_models.dense_cnn.architecture import optimized_model1_for_inference
from hexo_models.dense_cnn.plugin import DenseCNNPlugin
from hexo_models.dense_cnn import trt_backend

s = tomllib.load(open(REPO / "configs/dense_cnn_model1_target_96x6.toml", "rb"))["model"]["config"]
m = DenseCNNPlugin().build_model(game_spec={}, config=s)
m.load_state_dict(torch.load(REPO / "runs/dense_cnn_model1_target_96x6/checkpoints/bootstrap_sealbot_prefit.pt", map_location="cpu")["model_state"], strict=True)
opt = optimized_model1_for_inference(m).to("cuda", memory_format=torch.channels_last).eval()

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
adopted = 0
for i in range(N):
    fwd, info = trt_backend.build_trt_forward(opt, max_batch=1024, device="cuda")
    ok = fwd is not None
    adopted += int(ok)
    print(f"  build {i+1}/{N}: adopted={ok} reason={info.get('reason')} argmax={info.get('policy_argmax_match')} build_s={info.get('build_seconds')}", flush=True)
print(f"RELIABILITY: {adopted}/{N} adopted ({100*adopted/N:.0f}%)", flush=True)
