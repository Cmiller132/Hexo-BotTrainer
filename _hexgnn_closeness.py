"""Behavioral-closeness + edge metric for the CONTEXT-removal phase: run the SAME
fixed-seed weights over the CURRENT graph (hexgt, WITH context) vs the NEW graph
(hexgnn, NO context) on a corpus, report per-position policy KL + value delta, and
mean edges/leaf for each. Closeness is a measured TARGET."""
import random, math
import torch
import numpy as np
import hexo_engine.api as eng
from hexgnn.architecture import HexgnnNetwork
import hexgnn.graph_build as gb_new           # hexgnn: CONTEXT removed
import hexo_models.hexgt.graph_build as gb_old  # hexgt: WITH context
from hexgnn.losses import decode_binned_value

torch.manual_seed(0)
m = HexgnnNetwork(token_dim=128, gnn_layers=3).eval()

def corpus(k, seed=5):
    rng = random.Random(seed); out=[]
    while len(out) < k:
        s = eng.new_game(seed=rng.randint(0,10**7))
        for _ in range(rng.randint(4, 70)):
            if eng.terminal(s) is not None: break
            a=list(eng.legal_actions(s))
            if not a: break
            eng.apply_action(s, rng.choice(a))
        if eng.terminal(s) is None: out.append(s)
    return out

def seg_softmax_kl(p_old, p_new):
    # both are 1-D logits over the SAME candidate set (same n, same order)
    po = torch.softmax(p_old.float(), 0); pn = torch.softmax(p_new.float(), 0)
    return float((po * (po.clamp_min(1e-9).log() - pn.clamp_min(1e-9).log())).sum())

states = corpus(96, seed=5)
kls=[]; dvs=[]; eo=[]; en=[]
with torch.no_grad():
    for s in states:
        bo = gb_old.batch_from_states([s], n=2)   # WITH context
        bn = gb_new.batch_from_states([s], n=2)   # NO context (hexgnn)
        # candidate sets identical (same builder minus context edges) -> aligned
        if int(bo["candidate_index"].shape[0]) != int(bn["candidate_index"].shape[0]):
            continue
        oo = m(bo); on = m(bn)
        kls.append(seg_softmax_kl(oo["policy"], on["policy"]))
        dvs.append(abs(float(decode_binned_value(oo["value"])[0]) - float(decode_binned_value(on["value"])[0])))
        eo.append(int(bo["edge_index"].shape[1])); en.append(int(bn["edge_index"].shape[1]))

def stats(x):
    x=sorted(x); n=len(x);
    return (sum(x)/n, x[int(0.95*n)] if n else 0)
mk,pk = stats(kls); mv,pv = stats(dvs)
print(f"positions={len(kls)}")
print(f"policy KL (with-ctx || no-ctx): mean={mk:.4f} p95={pk:.4f}   [gate mean<0.10 p95<0.25]")
print(f"value |dv|:                      mean={mv:.4f} p95={pv:.4f}   [gate mean<0.04 p95<0.08]")
print(f"edges/leaf: WITH-ctx mean={np.mean(eo):.0f}  NO-ctx mean={np.mean(en):.0f}  "
      f"cut={100*(1-np.mean(en)/np.mean(eo)):.1f}%")
