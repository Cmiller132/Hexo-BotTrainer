"""Minimal staged hexgt MCTS diagnostic — find where it hangs/fails."""
from __future__ import annotations
import sys, time
def log(m): print(m, flush=True); sys.stderr.flush()

log("1: importing")
import torch
import hexo_engine.api as eng
from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.inference import HexgtInference
from hexo_models.hexgt.mcts import new_mcts_session
from hexo_models.hexgt.graph_build import batch_from_states

log("2: build model")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HexgtNetwork(short_term_value_horizons=()).to(dev).eval()
inf = HexgtInference(model, device=dev, fp16=(dev.type == "cuda"))

log("3: two fresh states (no shard reconstruct)")
states = []
for sd in (1, 2):
    s = eng.new_game(seed=sd)
    for _ in range(12):
        if eng.terminal(s) is not None:
            break
        acts = list(eng.legal_actions(s))
        eng.apply_action(s, acts[len(acts) // 2])
    states.append(s)
log(f"   states ready, legal0={len(list(eng.legal_actions(states[0])))}")

log("4: direct evaluate_graph_facts (bypass MCTS) to test the evaluator path")
from hexo_models.hexgt import rust_bridge
facts = [rust_bridge.graph_facts(s, 3) for s in states]
t = time.perf_counter()
out = inf.evaluate_graph_facts({"graph_facts": facts, "n": 3})
log(f"   evaluator OK in {time.perf_counter()-t:.2f}s: values={len(out['values_bytes'])//4} "
    f"offsets={out['candidate_row_offsets']} prior_bytes={len(out['priors_bytes'])}")

log("5a: session.run visits=4 vbatch=16 (>=visits -> NO prefetch thread / no scope)")
sess = new_mcts_session(max_states=4096, n=3)
t = time.perf_counter()
res = sess.run(list(range(len(states))), states, inf, visits=4, c_puct=1.5, seed=1,
               virtual_batch_size=16, widening_policy_mass=0.95, widening_max_children=8,
               widening_min_children=2, root_policy_temperature=1.1)
log(f"   5a OK in {time.perf_counter()-t:.2f}s: action={res[0].action_id} visits={res[0].visits} "
    f"eval={dict(res[0].diagnostics).get('evaluation', {})}")

log("5b: session.run visits=8 vbatch=2 (the scoped select<->eval pipeline path)")
sess2 = new_mcts_session(max_states=4096, n=3)
t = time.perf_counter()
res = sess2.run(list(range(len(states))), states, inf, visits=8, c_puct=1.5, seed=1,
                virtual_batch_size=2, widening_policy_mass=0.95, widening_max_children=8,
                widening_min_children=2, root_policy_temperature=1.1)
log(f"   5b OK in {time.perf_counter()-t:.2f}s: action={res[0].action_id} visits={res[0].visits} "
    f"eval={dict(res[0].diagnostics).get('evaluation', {})}")
log("DONE")
