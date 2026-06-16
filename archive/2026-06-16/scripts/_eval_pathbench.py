"""Isolate why the Rust-featurized evaluator path got slower per-state than the
old Python-featurized path. Times each stage of evaluate_featurized_batch vs the
old build (Python collate) + forward, on identical states, on GPU."""
from __future__ import annotations
import argparse, random, time
import numpy as np
import torch

import hexo_engine.api as eng
from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.constants import (
    DEFAULT_ATTENTION_HEADS, DEFAULT_CTX_LAYERS, DEFAULT_FFN_DIM, DEFAULT_GNN_LAYERS, DEFAULT_TOKEN_DIM)
from hexo_models.hexgt.inference import HexgtInference
from hexo_models.hexgt.graph_build import batch_from_states
from hexo_models.hexgt import rust_bridge


def states(k, seed=1):
    rng = random.Random(seed); out=[]
    while len(out) < k:
        s = eng.new_game(seed=rng.randint(0,10**7))
        for _ in range(rng.randint(20,60)):
            if eng.terminal(s) is not None: break
            a=list(eng.legal_actions(s))
            if not a: break
            eng.apply_action(s, rng.choice(a))
        if eng.terminal(s) is None: out.append(s)
    return out


def sync(dev):
    if dev.type=="cuda": torch.cuda.synchronize()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=3)
    ap.add_argument("--batch",type=int,default=16); ap.add_argument("--iters",type=int,default=50)
    args=ap.parse_args()
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=HexgtNetwork(token_dim=DEFAULT_TOKEN_DIM,gnn_layers=DEFAULT_GNN_LAYERS,ctx_layers=DEFAULT_CTX_LAYERS,
                       attention_heads=DEFAULT_ATTENTION_HEADS,ffn_dim=DEFAULT_FFN_DIM).to(dev).eval()
    inf=HexgtInference(model,device=dev,fp16=(dev.type=="cuda"))
    ss=states(args.batch,seed=1)

    # Build the rust zero-copy payload once.
    payload=rust_bridge._hexgt_module().hexgt_featurize_states(tuple(ss),args.n)

    # warm
    for _ in range(5):
        inf.evaluate_featurized_batch(payload);
    sync(dev)

    # full new path
    t=time.perf_counter()
    for _ in range(args.iters): inf.evaluate_featurized_batch(payload)
    sync(dev); new_full=(time.perf_counter()-t)/args.iters*1000

    # stage breakdown of new path
    tn=int(payload["total_nodes"]); te=int(payload["total_edges"]); tc=int(payload["total_candidates"])
    fd=int(payload["feat_dim"]); ad=int(payload["attr_dim"]); ng=int(payload["num_graphs"])
    t=time.perf_counter()
    for _ in range(args.iters):
        nf=torch.frombuffer(payload["node_feat"],dtype=torch.float32).reshape(tn,fd)
        ntp=torch.frombuffer(payload["node_type"],dtype=torch.int64)
        ngr=torch.frombuffer(payload["node_graph"],dtype=torch.int64)
        ei=torch.frombuffer(payload["edge_index"],dtype=torch.int64).reshape(2,te)
        et=torch.frombuffer(payload["edge_type"],dtype=torch.int64)
        ea=torch.frombuffer(payload["edge_attr"],dtype=torch.float32).reshape(te,ad)
        ci=torch.frombuffer(payload["candidate_index"],dtype=torch.int64)
        cg=torch.frombuffer(payload["candidate_graph"],dtype=torch.int64)
    fb=(time.perf_counter()-t)/args.iters*1000

    batch_cpu={"node_feat":nf,"node_type":ntp,"node_graph":ngr,"edge_index":ei,"edge_type":et,
               "edge_attr":ea,"candidate_index":ci,"candidate_graph":cg,"num_graphs":ng}
    t=time.perf_counter()
    for _ in range(args.iters):
        b={k:(v.to(dev) if isinstance(v,torch.Tensor) else v) for k,v in batch_cpu.items()}
    sync(dev); h2d=(time.perf_counter()-t)/args.iters*1000

    dev_batch={k:(v.to(dev) if isinstance(v,torch.Tensor) else v) for k,v in batch_cpu.items()}
    sync(dev)
    t=time.perf_counter()
    for _ in range(args.iters):
        with torch.no_grad(), torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=(dev.type=="cuda")):
            out=model.forward_policy_value(dev_batch)
    sync(dev); fwd=(time.perf_counter()-t)/args.iters*1000

    # old path: python collate once, forward repeatedly
    oldbatch=batch_from_states(ss,args.n)
    t=time.perf_counter()
    for _ in range(args.iters): oldb=batch_from_states(ss,args.n)
    pycollate=(time.perf_counter()-t)/args.iters*1000
    oldb_dev={k:(v.to(dev) if isinstance(v,torch.Tensor) else v) for k,v in oldbatch.items()}
    sync(dev)
    t=time.perf_counter()
    for _ in range(args.iters):
        with torch.no_grad(), torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=(dev.type=="cuda")):
            out=model.forward_policy_value(oldb_dev)
    sync(dev); fwd_old=(time.perf_counter()-t)/args.iters*1000

    print(f"batch={args.batch} graphs, nodes={tn} edges={te} cands={tc}")
    print(f"NEW evaluate_featurized_batch full: {new_full:.2f} ms/call")
    print(f"  frombuffer (cpu views):  {fb:.2f} ms")
    print(f"  to(device) H2D:          {h2d:.2f} ms")
    print(f"  forward (rust batch):    {fwd:.2f} ms")
    print(f"OLD path forward (py batch): {fwd_old:.2f} ms   | python collate: {pycollate:.2f} ms")


if __name__=="__main__":
    main()
