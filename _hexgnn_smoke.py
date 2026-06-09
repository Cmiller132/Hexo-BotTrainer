import random
import torch
import hexgnn
from hexgnn.architecture import HexgnnNetwork
from hexgnn.graph_build import batch_from_states
import hexo_engine.api as eng

rng = random.Random(0)
states = []
while len(states) < 5:
    s = eng.new_game(seed=rng.randint(0, 10**6))
    for _ in range(rng.randint(4, 30)):
        if eng.terminal(s) is not None:
            break
        acts = list(eng.legal_actions(s))
        if not acts:
            break
        eng.apply_action(s, rng.choice(acts))
    if eng.terminal(s) is None:
        states.append(s)

b = batch_from_states(states, n=4)
m = HexgnnNetwork(token_dim=48, gnn_layers=2)
m.eval()
with torch.no_grad():
    out = m(b)
ctot = int(b["candidate_index"].shape[0])
g = int(b["num_graphs"])
print("keys:", sorted(out.keys()))
print("policy", tuple(out["policy"].shape), "expect", (ctot,))
print("opp_policy", tuple(out["opp_policy"].shape))
print("value", tuple(out["value"].shape), "g=", g)
assert out["policy"].shape == (ctot,)
assert out["opp_policy"].shape == (ctot,)
assert out["value"].shape == (g, 65)
assert not any(k.startswith("stvalue") for k in out)
assert torch.isfinite(out["policy"]).all() and torch.isfinite(out["value"]).all()

mp = HexgnnNetwork()  # defaults: token_dim 168, gnn_layers 4, PMA k=2, +side
n_default = sum(p.numel() for p in mp.parameters())
print("DEFAULT hexgnn params:", f"{n_default:,}")
print("SMOKE OK")
