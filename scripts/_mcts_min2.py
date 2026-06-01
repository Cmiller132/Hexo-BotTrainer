"""Count evaluator calls during a single tiny search to localize the hang."""
from __future__ import annotations
import sys, time, threading
def log(m): print(m, flush=True)
import torch
import hexo_engine.api as eng
from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.inference import HexgtInference
from hexo_models.hexgt.mcts import new_mcts_session

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HexgtNetwork(short_term_value_horizons=()).to(dev).eval()
inf = HexgtInference(model, device=dev, fp16=(dev.type == "cuda"))

calls = {"n": 0}
orig = inf.evaluate_graph_facts
def wrapped(payload):
    calls["n"] += 1
    gf = payload["graph_facts"]
    out = orig(payload)
    log(f"   [eval call #{calls['n']}] n_graphs={len(gf)} offsets_last={out['candidate_row_offsets'][-1]}")
    return out
inf.evaluate_graph_facts = wrapped

s = eng.new_game(seed=1)
for _ in range(12):
    if eng.terminal(s) is not None: break
    acts = list(eng.legal_actions(s)); eng.apply_action(s, acts[len(acts)//2])
log(f"state ready legal={len(list(eng.legal_actions(s)))}")

sess = new_mcts_session(max_states=4096, n=3)
log("calling search visits=4 vbatch=16 (MAIN THREAD) ...")
res = sess.run([0], [s], inf, visits=4, c_puct=1.5, seed=1, virtual_batch_size=16,
               widening_policy_mass=0.95, widening_max_children=8, widening_min_children=2,
               root_policy_temperature=1.1)
log(f"DONE action={res[0].action_id} visits={res[0].visits} calls={calls['n']}")
