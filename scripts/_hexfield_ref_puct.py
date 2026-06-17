"""Reference serial PUCT (vb=1, greedy, no noise/TSS/forced) implementing the
dense rules exactly, in Python. Compares: hexfield tree dump vs reference at
step 0 (fresh) and step 1 (reused), and reference exports vs dense outputs.
Whichever native side disagrees with the reference carries the bug."""

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np

from hexfield_testkit import api
from hexo_engine.types import AxialCoord, PlacementAction
from hexfield.geometry import unpack_action_id
from test_hexfield_search_parity import _stub_prior, _stub_value, HexfieldStub, DenseStub

from hexfield import _rust as hexfield_rust
from hexo_models._rust import dense_cnn as dense_rust

FPU = 0.2
CPUCT = 1.5
MASS, MINC, MAXC = 0.95, 2, 96


def stub_eval(state):
    ids = sorted(api.legal_action_ids(state))
    raw = [(aid, _stub_prior(i, len(ids))) for i, aid in enumerate(ids)]
    total = sum(p for _, p in raw)
    priors = [(aid, p / total) for aid, p in raw]
    priors.sort(key=lambda ap: (-ap[1], ap[0]))
    return _stub_value(len(ids)), priors


def nucleus(priors):
    total = len(priors)
    if total == 0:
        return 0
    lo = max(1, min(MINC, total))
    hi = max(min(MAXC, total), lo)
    if lo >= hi:
        return hi
    cum, count = 0.0, 0
    for p in sorted((p for _, p in priors), reverse=True):
        cum += p
        count += 1
        if cum >= MASS:
            break
    return min(max(count, lo), hi)


class Node:
    __slots__ = ("player", "eval_value", "visits", "value_sum", "edges", "stack", "cap", "state")

    def __init__(self, state, value, priors):
        self.player = api.current_player(state)
        self.eval_value = np.float32(value)
        self.visits = 0
        self.value_sum = np.float32(0.0)
        self.edges = []  # [id, prior, visits, value_sum, child|None]
        self.stack = list(priors)  # descending; next candidate = stack[len(edges)]
        self.cap = nucleus(priors)
        self.state = state

    def value(self):
        return self.eval_value if self.visits == 0 else np.float32(self.value_sum / np.float32(self.visits))


class Ref:
    def __init__(self, root_state):
        self.nodes = []
        self.table = {}
        self.root_state = api.clone_state(root_state)
        v, priors = stub_eval(root_state)
        self.nodes.append(Node(api.clone_state(root_state), v, priors))
        self.table[self._hash(root_state)] = 0
        self.completed = 0

    @staticmethod
    def _hash(state):
        return tuple(sorted(api.legal_action_ids(state))), api.current_player(state), str(api.to_python_state(state).phase), tuple(sorted((c.q, c.r, str(p)) for c, p in api.to_python_state(state).board.stones))

    def select_edge(self, nid, is_root):
        node = self.nodes[nid]
        scale = np.float32(CPUCT) * np.float32(math.sqrt(np.float32(max(node.visits, 1))))
        parent_value = node.value()
        fpu = np.float32(FPU)
        best = None  # (idx, score, visits, id)
        for idx, e in enumerate(node.edges):
            q = (parent_value - fpu) if e[2] == 0 else np.float32(e[3] / np.float32(e[2]))
            score = np.float32(q + np.float32(e[1]) * scale / np.float32(1.0 + e[2]))
            cand = (idx, score, e[2], e[0])
            if best is None or (score, -cand[2], -cand[3]) > (best[1], -best[2], -best[3]):
                best = cand
        if len(node.edges) < node.cap and len(node.edges) < len(node.stack) + len(node.edges):
            nxt = node.stack[0] if node.stack else None
            if nxt is not None:
                score = np.float32(np.float32(nxt[1]) * scale)
                cand = (None, score, 0, nxt[0])
                if best is None or (score, 0, -cand[3]) > (best[1], -best[2], -best[3]):
                    aid, prior = node.stack.pop(0)
                    node.edges.append([aid, prior, 0, np.float32(0.0), None])
                    return len(node.edges) - 1
        return best[0] if best else None

    def run(self, visits):
        target = self.completed + visits
        while self.completed < target:
            state = api.clone_state(self.root_state)
            nid = 0
            path = []
            while True:
                ei = self.select_edge(nid, nid == 0)
                if ei is None:
                    # existing-node stop
                    self.completed += 1
                    self._virtual(path)
                    node = self.nodes[nid]
                    self._backup(path, node.player, node.value())
                    break
                edge = self.nodes[nid].edges[ei]
                q, r = unpack_action_id(edge[0])
                res = api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
                path.append((nid, ei))
                if edge[4] is not None:
                    nid = edge[4]
                    continue
                self.completed += 1
                self._virtual(path)
                key = self._hash(state) if not res.terminal else None
                if res.terminal:
                    t = api.terminal(state)
                    player = api.current_player(state) if t is None else None
                    # terminal_value from current player's perspective
                    cur = api.current_player(state)
                    val = np.float32(1.0 if t.winner == cur else -1.0)
                    self._backup(path, cur, val)
                    break
                if key in self.table:
                    cid = self.table[key]
                    edge[4] = cid
                    node = self.nodes[cid]
                    self._backup(path, node.player, node.value())
                    break
                v, priors = stub_eval(state)
                child = Node(api.clone_state(state), v, priors)
                cid = len(self.nodes)
                self.nodes.append(child)
                self.table[key] = cid
                edge[4] = cid
                self._backup(path, child.player, child.value())
                break

    def _virtual(self, path):
        for nid, ei in path:
            n = self.nodes[nid]
            n.visits += 1
            n.value_sum = np.float32(n.value_sum - np.float32(1.0))
            e = n.edges[ei]
            e[2] += 1
            e[3] = np.float32(e[3] - np.float32(1.0))

    def _backup(self, path, leaf_player, leaf_value):
        for nid, ei in path:
            n = self.nodes[nid]
            v = leaf_value if n.player == leaf_player else np.float32(-leaf_value)
            n.value_sum = np.float32(n.value_sum + v + np.float32(1.0))
            e = n.edges[ei]
            e[3] = np.float32(e[3] + v + np.float32(1.0))

    def advance(self, action_id):
        root = self.nodes[0]
        edge = next(e for e in root.edges if e[0] == action_id)
        cid = edge[4]
        assert cid is not None
        # rebuild subtree
        old_nodes = self.nodes
        mapping = {}
        new_nodes = []

        def clone(old):
            if old in mapping:
                return mapping[old]
            new = len(new_nodes)
            mapping[old] = new
            n = old_nodes[old]
            copy = Node.__new__(Node)
            copy.player = n.player
            copy.eval_value = n.eval_value
            copy.visits = n.visits
            copy.value_sum = n.value_sum
            copy.edges = [[e[0], e[1], e[2], e[3], None] for e in n.edges]
            copy.stack = list(n.stack)
            copy.cap = n.cap
            copy.state = n.state
            new_nodes.append(copy)
            for i, e in enumerate(n.edges):
                if e[4] is not None:
                    copy.edges[i][4] = clone(e[4])
            return new

        clone(cid)
        q, r = unpack_action_id(action_id)
        api.apply_action(self.root_state, PlacementAction(AxialCoord(q=q, r=r)))
        self.nodes = new_nodes
        if edge[2] > self.nodes[0].visits:
            self.nodes[0].visits = edge[2]
            self.nodes[0].value_sum = np.float32(-edge[3])
        self.table = {}
        for i, n in enumerate(self.nodes):
            self.table[self._hash(n.state)] = i
        self.completed = self.nodes[0].visits
        for e in self.nodes[0].edges:
            self.completed = max(self.completed, e[2])

    def export(self):
        root = self.nodes[0]
        return root.value(), [(e[0], e[2]) for e in root.edges]


def main():
    base = api.new_game()
    for q, r in [(0, 0), (1, 1), (-1, 0)]:
        api.apply_action(base, PlacementAction(AxialCoord(q=q, r=r)))

    # --- native runs (vb=1, greedy, no TSS) ---
    def native(session, stub, parity):
        state = api.clone_state(base)
        out = []
        for step in range(2):
            kw = dict(visits=16 if step == 0 else 48, c_puct=CPUCT, temperature=0.0,
                      seed=99 + step, evaluator=stub, virtual_batch_size=1,
                      widening_policy_mass=MASS, widening_max_children=MAXC,
                      widening_min_children=MINC, fpu_reduction=FPU, tss_enabled=False)
            if parity is not None:
                kw["search_parity_mode"] = parity
            r = session.search([42], (state,), **kw)[0]
            ids = np.frombuffer(bytes(r["visit_policy_action_ids_bytes"]), dtype=np.uint32)
            w = np.frombuffer(bytes(r["visit_policy_weights_bytes"]), dtype=np.float32)
            out.append((int(r["action_id"]), round(float(r["root_value"]), 6),
                        list(zip(ids.tolist(), [round(float(x), 6) for x in w]))))
            q, rr = unpack_action_id(int(r["action_id"]))
            api.apply_action(state, PlacementAction(AxialCoord(q=q, r=rr)))
        return out

    hex_out = native(hexfield_rust.HexfieldMctsSession(max_states=65536), HexfieldStub(), True)
    dense_out = native(dense_rust.Model1MctsSession(65536), DenseStub(), None)

    # --- reference ---
    ref = Ref(base)
    ref.run(16)
    v0, dist0 = ref.export()
    baseline = dict(dist0)
    action0 = max(dist0, key=lambda iv: (iv[1], -iv[0]))[0]
    ref.advance(action0)
    pre = {e[0]: e[2] for e in ref.nodes[0].edges}
    ref.run(48)
    v1, dist1 = ref.export()
    deltas1 = [(aid, n - pre.get(aid, 0)) for aid, n in dist1 if n - pre.get(aid, 0) > 0]
    total1 = sum(n for _, n in deltas1)
    action1 = max(deltas1, key=lambda iv: (iv[1], -iv[0]))[0]

    print("step0: ref action", action0, "value", round(float(v0), 6))
    print("       hex", hex_out[0][0], hex_out[0][1], "| dense", dense_out[0][0], dense_out[0][1])
    print("step1: ref action", action1, "value", round(float(v1), 6),
          "deltas", sorted(deltas1)[:8])
    print("       hex", hex_out[1][0], hex_out[1][1])
    print("       dense", dense_out[1][0], dense_out[1][1])


if __name__ == "__main__":
    main()
