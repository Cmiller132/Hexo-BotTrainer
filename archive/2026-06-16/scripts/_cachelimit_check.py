import torch
import torch._dynamo as d

d.config.cache_size_limit = 10 * 2 + 8
d.config.accumulated_cache_size_limit = 10 * 4 + 16
m = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)).eval()
c = torch.compile(m, fullgraph=True, dynamic=False)
with torch.inference_mode():
    for bs in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        c(torch.zeros(bs, 8))
print("CACHE-LIMIT OK: 10 static shapes compiled, no FailOnRecompileLimitHit")
