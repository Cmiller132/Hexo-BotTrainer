"""Self-play wall-time attribution at production shape (td128/gnn3, active=512,
visits=512, PCR off). Wraps the NN evaluator to split per-call time into:
  build  = Rust-payload deserialize (frombuffer -> tensors)         [CPU]
  forward= H2D transfer + GNN forward (synced)                       [GPU]
  post   = sanitize/softmax/decode + D2H + tobytes                   [D2H+CPU]
Everything NOT in the evaluator (total_wall - eval) = native Rust MCTS
(select/expand/backup) + Rust feature/graph build + payload serialize + the
Python self-play loop. Also records per-call leaf/node/candidate shapes to show
the opening->midgame growth, and separately times the Rust featurizer.
"""
import sys, time
import numpy as np
import torch

import hexo_engine.api as eng
import hexgnn.selfplay as SP
from hexgnn.inference import HexgnnInference
from hexgnn.losses import decode_binned_value, segment_log_softmax
from hexgnn.config import parse_hexgnn_config
from hexgnn.architecture import HexgnnNetwork
from hexgnn import rust_bridge

TD, GNN, ACTIVE, NUMG, MAXA, VIS, VB = 128, 3, 512, 512, 18, 512, 16


class ProfInference(HexgnnInference):
    T = {"build": 0.0, "forward": 0.0, "post": 0.0}
    CALLS = 0
    SHAPES = []  # (num_graphs, total_nodes, total_candidates)

    def evaluate_featurized_batch(self, payload):
        num_graphs = int(payload["num_graphs"]); tc = int(payload["total_candidates"])
        if num_graphs == 0 or tc == 0:
            return {"values_bytes": b"", "priors_bytes": b""}
        t0 = time.perf_counter()
        tn = int(payload["total_nodes"]); te = int(payload["total_edges"])
        fd = int(payload["feat_dim"]); ad = int(payload["attr_dim"])
        node_feat = torch.frombuffer(payload["node_feat"], dtype=torch.float32).reshape(tn, fd)
        node_type = torch.frombuffer(payload["node_type"], dtype=torch.int64)
        node_graph = torch.frombuffer(payload["node_graph"], dtype=torch.int64)
        if te:
            edge_index = torch.frombuffer(payload["edge_index"], dtype=torch.int64).reshape(2, te)
            edge_type = torch.frombuffer(payload["edge_type"], dtype=torch.int64)
            edge_attr = torch.frombuffer(payload["edge_attr"], dtype=torch.float32).reshape(te, ad)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.int64); edge_type = torch.zeros((0,), dtype=torch.int64)
            edge_attr = torch.zeros((0, ad), dtype=torch.float32)
        candidate_index = torch.frombuffer(payload["candidate_index"], dtype=torch.int64)
        candidate_graph = torch.frombuffer(payload["candidate_graph"], dtype=torch.int64)
        batch = {"node_feat": node_feat, "node_type": node_type, "node_graph": node_graph,
                 "edge_index": edge_index, "edge_type": edge_type, "edge_attr": edge_attr,
                 "candidate_index": candidate_index, "candidate_graph": candidate_graph,
                 "num_graphs": num_graphs}
        t1 = time.perf_counter()
        out, dev_batch = self.forward_batch(batch)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        cg = dev_batch["candidate_graph"]
        raw_policy, raw_value, _ = self._count_and_sanitize(out["policy"].float(), out["value"].float(), cg, num_graphs)
        log_probs = segment_log_softmax(raw_policy, cg, num_graphs)
        priors = log_probs.exp().cpu().numpy().astype(np.float32, copy=False)
        values = decode_binned_value(raw_value).cpu().numpy()
        values = np.nan_to_num(np.clip(values, -1.0, 1.0), nan=0.0).astype(np.float32, copy=False)
        res = {"values_bytes": np.ascontiguousarray(values).tobytes(),
               "priors_bytes": np.ascontiguousarray(priors).tobytes()}
        t3 = time.perf_counter()
        ProfInference.T["build"] += t1 - t0
        ProfInference.T["forward"] += t2 - t1
        ProfInference.T["post"] += t3 - t2
        ProfInference.CALLS += 1
        if ProfInference.CALLS % 5 == 0:  # sample shapes to bound memory
            ProfInference.SHAPES.append((num_graphs, tn, tc))
        return res


def featurize_bench(plies, seed, k=512):
    import random
    rng = random.Random(seed); st = []
    while len(st) < k:
        s = eng.new_game(seed=rng.randint(0, 10**7))
        for _ in range(plies):
            if eng.terminal(s) is not None: break
            a = list(eng.legal_actions(s));
            if not a: break
            eng.apply_action(s, rng.choice(a))
        if eng.terminal(s) is None: st.append(s)
    mod = rust_bridge._hexgt_module()
    for _ in range(3): mod.hexgt_featurize_states(tuple(st), 3)
    t0 = time.perf_counter(); N = 10
    for _ in range(N): d = mod.hexgt_featurize_states(tuple(st), 3)
    dt = (time.perf_counter() - t0) / N * 1000
    return dt, int(d["total_nodes"]), int(d["total_candidates"])


def main():
    SP.HexgnnInference = ProfInference  # monkeypatch so run_selfplay_games uses the timed evaluator
    m = HexgnnNetwork(token_dim=TD, gnn_layers=GNN).to("cuda").eval()
    m.forward_policy_value = torch.compile(m.forward_policy_value, dynamic=True)
    cfg = parse_hexgnn_config({"device": "cuda",
        "architecture": {"candidate_radius": 3, "token_dim": TD, "gnn_layers": GNN, "attention_heads": 4, "value_pma_seeds": 2},
        "samples": {"soft_z_lambda": 0.0},
        "selfplay": dict(search_visits=VIS, active_games=ACTIVE, mcts_active_root_limit=ACTIVE, c_puct=1.5,
                         root_dirichlet_noise_enabled=True, root_dirichlet_total_alpha=6.6, root_dirichlet_noise_fraction=0.30,
                         root_policy_temperature=1.0, widening_max_children=96, widening_min_children=2, widening_policy_mass=0.95,
                         max_actions=MAXA, temperature=1.0, temperature_floor=0.3, temperature_halflife=33, forced_playout_k=2.0,
                         pcr_enabled=False)})
    print(f"PROFILE td{TD}/gnn{GNN} params={sum(p.numel() for p in m.parameters()):,} "
          f"active={ACTIVE} visits={VIS} vbatch={VB} max_actions={MAXA}", flush=True)
    t0 = time.perf_counter()
    r = SP.run_selfplay_games(m, cfg, num_games=NUMG, output_dir="/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_smoke/prof",
                              epoch=0, device="cuda", fp16=True, base_seed=1, active_games=ACTIVE,
                              virtual_batch_size=VB, collect_examples=0)
    wall = time.perf_counter() - t0
    T = ProfInference.T; ev = T["build"] + T["forward"] + T["post"]; rust = wall - ev
    print(f"\n=== WALL-TIME ATTRIBUTION (whole run, wall={wall:.1f}s, {ProfInference.CALLS} eval calls) ===", flush=True)
    print(f"  searched={r.searched_positions} pos/s={r.positions_per_second:.1f}", flush=True)
    for k, v in [("Rust MCTS + featurize + serialize + py-loop", rust),
                 ("NN eval: build (frombuffer->tensors) [CPU]", T["build"]),
                 ("NN eval: forward (H2D + GNN compute)  [GPU]", T["forward"]),
                 ("NN eval: post (softmax/decode + D2H + tobytes)", T["post"])]:
        print(f"  {k:<48} {v:8.1f}s  {100*v/wall:5.1f}%", flush=True)
    sh = np.array(ProfInference.SHAPES) if ProfInference.SHAPES else np.zeros((1, 3))
    q = max(1, len(sh) // 5)
    a, b = sh[:q], sh[-q:]
    print(f"\n=== eval-call shapes (leaves, nodes, candidates), {len(sh)} sampled ===", flush=True)
    print(f"  leaves/call: min={sh[:,0].min():.0f} med={np.median(sh[:,0]):.0f} max={sh[:,0].max():.0f}", flush=True)
    print(f"  early-calls mean: leaves={a[:,0].mean():.0f} nodes/leaf={ (a[:,1]/a[:,0]).mean():.0f} cand/leaf={(a[:,2]/a[:,0]).mean():.0f}", flush=True)
    print(f"  late-calls  mean: leaves={b[:,0].mean():.0f} nodes/leaf={ (b[:,1]/b[:,0]).mean():.0f} cand/leaf={(b[:,2]/b[:,0]).mean():.0f}", flush=True)
    do, no, co = featurize_bench(6, 11); dm, nm, cm = featurize_bench(60, 12)
    print(f"\n=== Rust featurizer (512 states) ===", flush=True)
    print(f"  opening(~6 plies): {do:.1f} ms  nodes={no} cand={co}", flush=True)
    print(f"  midgame(~60 plies): {dm:.1f} ms  nodes={nm} cand={cm}", flush=True)


if __name__ == "__main__":
    main()
