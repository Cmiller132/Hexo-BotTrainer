import time, torch
import hexo_engine as engine
from hexgnn.architecture import HexgnnNetwork
from hexgnn.inference import HexgnnInference

ckpt = "/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1/checkpoints/hexgnn_rl_epoch000041.pt"
p = torch.load(ckpt, map_location="cpu")
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
print("ckpt keys", list(p.keys()), "rl_epoch", p.get("rl_epoch"), "step", p.get("step"))
print("arch", p.get("arch"))
ms = p["model"] if "model" in p else p["model_state"]
arch = p.get("arch", {})
net = HexgnnNetwork(node_feature_dim=int(arch.get("node_feature_dim", 32)),
                    token_dim=int(arch.get("token_dim", 96)),
                    gnn_layers=int(arch.get("gnn_layers", 2)),
                    attention_heads=int(arch.get("attention_heads", 4)),
                    value_pma_seeds=int(arch.get("value_pma_seeds", 2)),
                    value_head_use_side=bool(arch.get("value_head_use_side", True)),
                    steerable_channels=int(arch.get("steerable_channels", 4)))
missing, unexpected = net.load_state_dict(ms, strict=False)
print("missing", list(missing)[:6], "unexpected", list(unexpected)[:6],
      "nparams", sum(x.numel() for x in net.parameters()))
net.eval()
inf = HexgnnInference(net, device="cpu", fp16=False)
s = engine.new_game(seed=1)
t = time.time()
out = inf.evaluate_states([s], 4)
print("eval_states ok; ncand", len(out[0]["candidate_ids"]), "value", round(out[0]["value"], 3),
      "dt", round(time.time() - t, 3))
