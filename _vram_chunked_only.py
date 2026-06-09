import glob, torch
from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.inference import HexgtInference
from hexo_models.hexgt.graph_build import batch_from_states
from hexo_models.hexgt.expand import reconstruct_state
from hexo_models.dense_cnn.compact_io import read_compact_shard
device=torch.device("cuda"); MB=1024*1024
def gather(n):
    out=[]
    for sp in sorted(glob.glob("/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8/selfplay/epoch_0000*_game_*.npz")):
        rows=read_compact_shard(sp)
        for i in range(0,len(rows),max(1,len(rows)//6)):
            out.append(reconstruct_state(rows[i].placement_history))
            if len(out)>=n: return out
    return out
ck=torch.load("/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_bc/hexgt_bc_step006009.pt",map_location=device,weights_only=False)
a=ck["arch"]
m=HexgtNetwork(token_dim=a["token_dim"],gnn_layers=a["gnn_layers"],ctx_layers=a["ctx_layers"],ffn_dim=a["ffn_dim"],attention_heads=a["attention_heads"],short_term_value_horizons=tuple(a["short_term_value_horizons"])).to(device).eval()
m.load_state_dict(ck["model"])
infer=HexgtInference(m,device=device,fp16=True)  # default 200k chunk budget
print(f"{'graphs':>7} {'chunked_peak_MB':>16} {'chunks':>7}")
from hexo_models.hexgt.inference import _plan_chunks
for ng in (512,1024,2048,4096):
    st=gather(ng); b=batch_from_states(st,3)
    b={k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in b.items()}
    cnt=torch.bincount(b["candidate_graph"],minlength=int(b["num_graphs"]))
    nch=len(_plan_chunks(cnt,200000))
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        infer.forward_batch(b)
    torch.cuda.synchronize()
    print(f"{int(b['num_graphs']):>7} {torch.cuda.max_memory_allocated()/MB:>16.0f} {nch:>7}",flush=True)
    del b; torch.cuda.empty_cache()
