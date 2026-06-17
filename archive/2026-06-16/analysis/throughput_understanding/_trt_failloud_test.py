import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
for p in ("hexo_engine","hexo_utils","hexo_runner","hexo_train","hexo_models","hexo_frontend"):
    sys.path.insert(0, str(REPO/"packages"/p/"python"))
import tomllib, torch
from hexo_models.dense_cnn.architecture import optimized_model1_for_inference
from hexo_models.dense_cnn.plugin import DenseCNNPlugin
from hexo_models.dense_cnn import trt_backend
s=tomllib.load(open(REPO/"configs/dense_cnn_model1_target_96x6.toml","rb"))["model"]["config"]
m=DenseCNNPlugin().build_model(game_spec={},config=s)
m.load_state_dict(torch.load(REPO/"runs/dense_cnn_model1_target_96x6/checkpoints/bootstrap_sealbot_prefit.pt",map_location="cpu")["model_state"],strict=True)
opt=optimized_model1_for_inference(m).to("cuda",memory_format=torch.channels_last).eval()
# (1) normal build -> adopt
fwd,info=trt_backend.build_trt_forward(opt,max_batch=256,device="cuda")
print("TEST1 normal: adopted=",info.get("adopted"))
# (2) deliberately-broken gate (argmax_match_min=2.0 impossible), fallback OFF -> must RAISE loudly
try:
    trt_backend.build_trt_forward(opt,max_batch=256,device="cuda",argmax_match_min=2.0,allow_torch_fallback=False)
    print("TEST2 FAIL: returned instead of raising!")
except trt_backend.TRTAdoptError as e:
    print("TEST2 PASS: raised TRTAdoptError loudly (no silent fallback)")
# (3) same but fallback ON -> returns (None,info), no raise
fwd3,info3=trt_backend.build_trt_forward(opt,max_batch=256,device="cuda",argmax_match_min=2.0,allow_torch_fallback=True)
print("TEST3 fallback-opt-in: fwd is None=",fwd3 is None,"(logged torch fallback)")
