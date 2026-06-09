"""Part 3 graft validation: STV heads given the [SIDE|PMA] readout must be
bit-identical at init (zero-init PMA columns), and shapes/D6 must hold.

OLD architecture (git HEAD) vs NEW (working tree, STV reads value_emb + expander):
load the SAME checkpoint into both (NEW via expand_value/expand_stv) and compare
every head output on a real batch. Run on CPU (no GPU disturbance)."""
import importlib.util
import sys
import types

import torch

CKPT = "/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main3/checkpoints/hexgt_rl_epoch000005.pt"
PKG = "/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_models/hexgt/python/hexo_models/hexgt"
torch.manual_seed(0)
torch.set_grad_enabled(False)


def load_module(path, name):
    # Load as a submodule of the installed hexo_models.hexgt package so the OLD
    # architecture's relative imports (`from .constants import ...`) resolve.
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    m.__package__ = "hexo_models.hexgt"
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# NEW architecture (working tree)
import hexo_engine.api as eng
from hexo_engine.types import AxialCoord, PlacementAction, unpack_coord_id
from hexo_runner.records import HexoRecordFile
from hexo_models.hexgt.features import build_graph_tensors
from hexo_models.hexgt.collate import collate_graphs
from hexo_models.hexgt import rust_bridge
from hexo_models.hexgt import architecture as arch_new

# OLD architecture from git HEAD -> temp file -> module
import subprocess
old_src = subprocess.check_output(
    ["git", "-C", "/mnt/e/Hexo-BotTrainer-hexgt", "show",
     "HEAD:packages/hexo_models/hexgt/python/hexo_models/hexgt/architecture.py"]).decode()
open("/tmp/arch_old.py", "w").write(old_src)
# arch_old imports `from .constants import ...` style? Check: it uses absolute imports
# from hexo_models.hexgt.* so it loads fine as a standalone module.
arch_old = load_module("/tmp/arch_old.py", "arch_old")

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
a = ck["arch"]
print("ckpt arch:", a)
common = dict(token_dim=a["token_dim"], gnn_layers=a["gnn_layers"], ctx_layers=a["ctx_layers"],
              ffn_dim=a["ffn_dim"], attention_heads=a["attention_heads"],
              short_term_value_horizons=tuple(a["short_term_value_horizons"]),
              value_pma_seeds=int(a.get("value_pma_seeds", 2)))

# OLD model: loads the checkpoint directly (matching shapes)
old = arch_old.HexgtNetwork(**common).eval()
mi, un = old.load_state_dict(ck["model"], strict=False)
assert not [k for k in mi if not k.startswith("_")], f"old missing {mi}"
assert not un, f"old unexpected {un}"

# NEW model: expand value + STV readout columns in a COPY of the state dict, then load
new = arch_new.HexgtNetwork(**common).eval()
sd = {k: v.clone() for k, v in ck["model"].items()}
vr = arch_new.expand_value_readout_columns(new, sd)
stv = arch_new.expand_stv_readout_columns(new, sd)
print("expanded value:", vr, "| stv heads expanded:", stv)
mi2, un2 = new.load_state_dict(sd, strict=False)
assert not [k for k in mi2 if not k.startswith("_")], f"new missing {mi2}"
assert not un2, f"new unexpected {un2}"

# shape check: STV first Linear widened
for h in common["short_term_value_horizons"]:
    w = new.short_term_value_heads[str(h)][0].weight
    exp = (1 + common["value_pma_seeds"]) * common["token_dim"]
    assert tuple(w.shape) == (common["token_dim"], exp), (h, w.shape, exp)
print("STV first-linear shapes OK:", {h: tuple(new.short_term_value_heads[str(h)][0].weight.shape) for h in common["short_term_value_horizons"]})

# Build a few real batches from epoch-5 self-play games
hxr = "/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main3/selfplay/epoch_000005.hxr"
batches = []
rf = HexoRecordFile.open(hxr)
for rec in rf.iter_records():
    coords = [unpack_coord_id(int(c)) for c in rec.action_ids]
    s = eng.new_game()
    for i, c in enumerate(coords):
        if i in (12, 20, 28, 36):
            batches.append(collate_graphs([build_graph_tensors(rust_bridge.graph_facts(s, 3))]))
        eng.apply_action(s, PlacementAction(AxialCoord(q=int(c.q), r=int(c.r))))
        if len(batches) >= 6:
            break
    if len(batches) >= 6:
        break
print("built", len(batches), "real batches")

# Compare ALL head outputs (both with_aux paths)
max_diff = {}
for b in batches:
    o_old = old.forward(b)            # with_aux=True
    o_new = new.forward(b)
    for k in o_old:
        d = (o_old[k] - o_new[k]).abs().max().item()
        max_diff[k] = max(max_diff.get(k, 0.0), d)
    # also the inference path (with_aux=False): policy+value
    pv_old = old.forward_policy_value(b)
    pv_new = new.forward_policy_value(b)
    for k in pv_old:
        d = (pv_old[k] - pv_new[k]).abs().max().item()
        max_diff[f"pv_{k}"] = max(max_diff.get(f"pv_{k}", 0.0), d)

print("\nMAX ABS DIFF (OLD vs NEW) across heads:")
for k, v in sorted(max_diff.items()):
    print(f"  {k:16s} {v:.3e}")
worst = max(max_diff.values())
print(f"\nWORST: {worst:.3e}  -> {'BIT-IDENTICAL (PASS)' if worst == 0.0 else ('OK<1e-6' if worst < 1e-6 else 'FAIL')}")

print("\n(D6 invariance checked via the test suite)")
