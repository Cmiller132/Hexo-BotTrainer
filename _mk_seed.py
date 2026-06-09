"""Make a tiny fresh hexgnn seed checkpoint for the driver CPU smoke."""
import sys
from pathlib import Path
import torch
from hexgnn.architecture import HexgnnNetwork
from hexgnn.constants import FEATURE_SCHEMA_VERSION

out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
arch = dict(token_dim=32, gnn_layers=2, attention_heads=4, value_pma_seeds=2, value_head_use_side=True)
m = HexgnnNetwork(token_dim=arch["token_dim"], gnn_layers=arch["gnn_layers"],
                  attention_heads=arch["attention_heads"], value_pma_seeds=arch["value_pma_seeds"],
                  value_head_use_side=arch["value_head_use_side"])
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
torch.save({"model": m.state_dict(), "optimizer": opt.state_dict(), "arch": arch,
            "feature_schema_version": FEATURE_SCHEMA_VERSION, "rl_epoch": 0, "step": 0}, out)
print("wrote seed", out, "params", sum(p.numel() for p in m.parameters()))
