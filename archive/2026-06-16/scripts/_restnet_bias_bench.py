"""Measure the B-speedup: per-forward relative-bias gather (uncached) vs the
version-keyed inference cache, on the real 41x41 model, eval/no-grad (the self-play
inference path). CPU so it does not contend with the GPU prefit."""
import sys, time
from pathlib import Path
ROOT = Path("/mnt/e/Hexo-BotTrainer-hexgt")
for p in ("hexo_engine", "hexo_utils", "hexo_runner", "hexo_train", "hexo_models"):
    sys.path.insert(0, str(ROOT / "packages" / p / "python"))
sys.path.insert(0, str(ROOT / "packages" / "dense_cnn_restnet" / "python"))
import torch
from dense_cnn_restnet.architecture import RestnetNetwork, RelPosMHSA
from dense_cnn_restnet.constants import INPUT_CHANNELS, BOARD_SIZE

torch.manual_seed(0)
net = RestnetNetwork(channels=96, blocks_type="R_R_R_T_R_R_T_R").eval()
for m in net.modules():
    if isinstance(m, RelPosMHSA):
        with torch.no_grad():
            m.relative_bias_table.normal_(std=0.05)
x = torch.randn(8, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
K = 12

def run(uncached):
    with torch.no_grad():
        net(x)  # warm
        t0 = time.perf_counter()
        for _ in range(K):
            if uncached:
                for m in net.modules():
                    if isinstance(m, RelPosMHSA):
                        m._cached_bias_key = None  # force re-gather every forward
            net.forward_policy_value(x)
        return (time.perf_counter() - t0) / K

t_uncached = run(True)
t_cached = run(False)
print(f"41x41 forward_policy_value batch={x.shape[0]} (CPU, eval): "
      f"uncached {t_uncached*1000:.1f} ms/fwd, cached {t_cached*1000:.1f} ms/fwd, "
      f"speedup {t_uncached/t_cached:.3f}x ({100*(t_uncached-t_cached)/t_uncached:.1f}% off the gather)")
