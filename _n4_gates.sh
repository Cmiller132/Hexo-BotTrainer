#!/bin/bash
set -euo pipefail
export VIRTUAL_ENV=/root/.venvs/hexgt-build
export PATH=$VIRTUAL_ENV/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/mnt/e/Hexo-BotTrainer-hexgt
export PYTHONPATH=$R/packages/hexgnn/python:$R/packages/hexo_engine/python:$R/packages/hexo_utils/python:$R/packages/hexo_runner/python:$R/packages/hexo_train/python:$R/packages/hexo_models/python
cd $R

echo "===== GATE 2: D6 equivariance @ n=4 (both variants via pytest) ====="
python -m pytest tests/test_hexgnn_d6.py::test_model_is_d6_equivariant tests/test_hexgnn_d6.py::test_model_with_steerable_is_d6_equivariant -q 2>&1 | tail -n 20

echo ""
echo "===== GATES 1,3,4: parity + monotonicity + counts ====="
python - <<'PY'
import importlib, random, sys
import numpy as np

eng = importlib.import_module("hexo_engine.api")

def mk_states(k, seed, lo, hi):
    rng = random.Random(seed)
    out = []
    tries = 0
    while len(out) < k and tries < 100000:
        tries += 1
        s = eng.new_game(seed=rng.randint(0, 10**7))
        nmoves = rng.randint(lo, hi)
        for _ in range(nmoves):
            if eng.terminal(s) is not None:
                break
            acts = list(eng.legal_actions(s))
            if not acts:
                break
            eng.apply_action(s, rng.choice(acts))
        if eng.terminal(s) is None:
            out.append(s)
    return out

from hexgnn import rust_bridge
mod = rust_bridge._hexgnn_module()

def rust_batch(states, n):
    d = mod.hexgnn_featurize_states(tuple(states), n)
    tn, te, tc = int(d["total_nodes"]), int(d["total_edges"]), int(d["total_candidates"])
    fd, ad = int(d["feat_dim"]), int(d["attr_dim"])
    ei = np.frombuffer(d["edge_index"], dtype=np.int32).reshape(2, te)
    return dict(
        node_feat=np.frombuffer(d["node_feat"], dtype=np.float32).reshape(tn, fd),
        node_type=np.frombuffer(d["node_type"], dtype=np.int32),
        node_graph=np.frombuffer(d["node_graph"], dtype=np.int32),
        edge_index=ei,
        edge_type=np.frombuffer(d["edge_type"], dtype=np.int32),
        edge_attr=np.frombuffer(d["edge_attr"], dtype=np.float32).reshape(te, ad),
        candidate_index=np.frombuffer(d["candidate_index"], dtype=np.int32),
        candidate_graph=np.frombuffer(d["candidate_graph"], dtype=np.int32),
        candidate_ids=np.frombuffer(d["candidate_ids"], dtype=np.int64),
        edge_dir=np.frombuffer(d["edge_dir"], dtype=np.int32),
        num_graphs=int(d["num_graphs"]),
        total_nodes=tn, total_edges=te, total_candidates=tc,
    )

# ---------- GATE 1: parity at n=4 ----------
print("\n----- GATE 1: Featurizer parity @ n=4 -----")
import torch  # noqa
from hexgnn.graph_build import batch_from_states
parity_states = mk_states(24, seed=3, lo=20, hi=90)
print(f"parity corpus states: {len(parity_states)}")
n = 4
rb = rust_batch(parity_states, n)
pb = batch_from_states(parity_states, n)
py = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in pb.items()}
parity_ok = True
detail = {}
assert rb["num_graphs"] == py["num_graphs"] == len(parity_states), "num_graphs mismatch"
for key in ("node_type","node_graph","edge_type","edge_dir","candidate_index","candidate_graph","candidate_ids"):
    eq = np.array_equal(rb[key], py[key])
    detail[key] = eq
    parity_ok &= eq
eq_ei = np.array_equal(rb["edge_index"], py["edge_index"])
detail["edge_index"] = eq_ei
parity_ok &= eq_ei
nf_maxdiff = float(np.abs(rb["node_feat"] - py["node_feat"]).max()) if rb["node_feat"].shape == py["node_feat"].shape else float("inf")
ea_maxdiff = float(np.abs(rb["edge_attr"] - py["edge_attr"]).max()) if rb["edge_attr"].shape == py["edge_attr"].shape else float("inf")
shape_ok = (rb["node_feat"].shape == py["node_feat"].shape) and (rb["edge_attr"].shape == py["edge_attr"].shape)
parity_ok &= shape_ok and (nf_maxdiff == 0.0) and (ea_maxdiff == 0.0)
print("per-buffer array_equal:", detail)
print(f"node_feat shape rust={rb['node_feat'].shape} py={py['node_feat'].shape}")
print(f"node_feat maxdiff={nf_maxdiff}  edge_attr maxdiff={ea_maxdiff}")
print(f"GATE1 parity@n4: {'PASS' if parity_ok else 'FAIL'}")

# ---------- GATE 3 + 4: candidate sets per state at n=2,4,5 ----------
print("\n----- GATE 3+4: candidate sets / monotonicity / counts -----")
corpus = mk_states(60, seed=7, lo=40, hi=80)
print(f"counts/monotonicity corpus states: {len(corpus)} (40-80 moves)")

def per_state_sets(states, n):
    """Return list over states: (set(candidate_ids), num_candidates, num_edges)."""
    b = rust_batch(states, n)
    cg = b["candidate_graph"]
    cids = b["candidate_ids"]
    ng = b["node_graph"]
    # edges per graph: use edge_index src node -> node_graph
    ei = b["edge_index"]
    src = ei[0]
    eg = ng[src]  # graph id per edge
    G = b["num_graphs"]
    sets = []
    ncand = []
    nedge = []
    for g in range(G):
        ids = cids[cg == g]
        sets.append(set(int(x) for x in ids))
        ncand.append(int((cg == g).sum()))
        nedge.append(int((eg == g).sum()))
    return sets, np.array(ncand), np.array(nedge)

s2, c2, e2 = per_state_sets(corpus, 2)
s4, c4, e4 = per_state_sets(corpus, 4)
s5, c5, e5 = per_state_sets(corpus, 5)

# GATE 3: monotonicity n2 subset n4 subset n5
viol_24 = 0
viol_45 = 0
for a, b, c in zip(s2, s4, s5):
    if not a.issubset(b):
        viol_24 += 1
    if not b.issubset(c):
        viol_45 += 1
mono_ok = (viol_24 == 0 and viol_45 == 0)
print(f"n2 subset n4 violations: {viol_24}")
print(f"n4 subset n5 violations: {viol_45}")
print(f"GATE3 monotonicity n2<=n4<=n5: {'PASS' if mono_ok else 'FAIL'}")

def stat(arr):
    return float(np.mean(arr)), float(np.percentile(arr, 90))

m2c, p2c = stat(c2); m4c, p4c = stat(c4); m5c, p5c = stat(c5)
m2e, p2e = stat(e2); m4e, p4e = stat(e4); m5e, p5e = stat(e5)
print("\n--- GATE4 candidate counts (per position) ---")
print(f"candidates  n2: mean={m2c:.2f} p90={p2c:.1f}")
print(f"candidates  n4: mean={m4c:.2f} p90={p4c:.1f}")
print(f"candidates  n5: mean={m5c:.2f} p90={p5c:.1f}")
print(f"candidates ratio n4/n2 mean={m4c/m2c:.3f}  n4/n5 mean={m4c/m5c:.3f}")
print(f"\nedges  n2: mean={m2e:.2f} p90={p2e:.1f}")
print(f"edges  n4: mean={m4e:.2f} p90={p4e:.1f}")
print(f"edges  n5: mean={m5e:.2f} p90={p5e:.1f}")
print(f"edges ratio n4/n2 mean={m4e/m2e:.3f}  n4/n5 mean={m4e/m5e:.3f}")
PY
echo "===== DONE ====="
