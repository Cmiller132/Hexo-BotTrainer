"""Quantify the shuffle phase (CPU zlib) and the memory model. Read-only.
Measures decompress + recompress throughput on a real shard and derives per-row
byte cost, then projects the 2-phase shuffle cost from the replay-window volume."""
from __future__ import annotations
import glob, json, os, time, io
import numpy as np
from hexo_models.dense_cnn.replay import INPUT_KEY

base=r"E:\Hexo-BotTrainer\runs\dense_cnn_model1_scratch_64"
g=sorted(d for d in glob.glob(os.path.join(base,"shuffleddata","*-epoch_*")) if not d.endswith(".tmp"))[-1]
shard=sorted(glob.glob(os.path.join(g,"train","*.npz")))[0]

# selfplay shard (per-game) for the input side
sp=sorted(glob.glob(os.path.join(base,"selfplay","epoch_0000*_game_*.npz")))
sp_shard=sp[len(sp)//2] if sp else None

def load_arrays(p):
    with np.load(p) as d:
        return {k:d[k] for k in d.files}, int(d[INPUT_KEY].shape[0])

# decompress throughput
t=time.perf_counter(); arrays,rows=load_arrays(shard); t_decomp=time.perf_counter()-t
uncompressed_bytes=sum(a.nbytes for a in arrays.values())
compressed_bytes=os.path.getsize(shard)
per_row_uncompressed=uncompressed_bytes/rows

# recompress throughput (savez_compressed to memory)
buf=io.BytesIO()
t=time.perf_counter(); np.savez_compressed(buf,**arrays); t_recomp=time.perf_counter()-t
# uncompressed scratch write (savez) for the P3 comparison
buf2=io.BytesIO()
t=time.perf_counter(); np.savez(buf2,**arrays); t_rawwrite=time.perf_counter()-t

decomp_mbps=uncompressed_bytes/1e6/t_decomp
recomp_mbps=uncompressed_bytes/1e6/t_recomp
raw_mbps=uncompressed_bytes/1e6/t_rawwrite

# replay window volume (target rows from config)
WINDOW_ROWS=300000
window_uncompressed_gb=per_row_uncompressed*WINDOW_ROWS/1e9
# 2-phase: phase1 reads window (decompress) + writes scratch (compress);
# phase2 reads scratch (decompress) + writes output (compress).
# => ~2x decompress + ~2x compress over the window volume.
est_decomp_s=2*per_row_uncompressed*WINDOW_ROWS/1e6/decomp_mbps
est_comp_s=2*per_row_uncompressed*WINDOW_ROWS/1e6/recomp_mbps
est_total_s=est_decomp_s+est_comp_s
# if scratch written uncompressed (P3): phase1 write + phase2 read become raw
est_p3_s=(per_row_uncompressed*WINDOW_ROWS/1e6/decomp_mbps   # p1 read window (still compressed selfplay)
          + per_row_uncompressed*WINDOW_ROWS/1e6/raw_mbps     # p1 write scratch raw
          + per_row_uncompressed*WINDOW_ROWS/1e6/(decomp_mbps*8)  # p2 read raw (~no zlib, ~8x faster)
          + per_row_uncompressed*WINDOW_ROWS/1e6/recomp_mbps)     # p2 write output compressed

out={
 "shard":os.path.basename(shard),"rows":rows,
 "compressed_MB":round(compressed_bytes/1e6,2),"uncompressed_MB":round(uncompressed_bytes/1e6,2),
 "compression_ratio":round(uncompressed_bytes/compressed_bytes,1),
 "per_row_uncompressed_KB":round(per_row_uncompressed/1e3,1),
 "decompress_MBps":round(decomp_mbps,0),"recompress_MBps":round(recomp_mbps,0),"rawwrite_MBps":round(raw_mbps,0),
 "window_rows":WINDOW_ROWS,"window_uncompressed_GB":round(window_uncompressed_gb,1),
 "group_bucket_rows":8000,"peak_RAM_per_group_GB":round(per_row_uncompressed*8000/1e9,2),
 "est_shuffle_decomp_s":round(est_decomp_s,0),"est_shuffle_comp_s":round(est_comp_s,0),
 "est_shuffle_total_s":round(est_total_s,0),"est_shuffle_P3_uncompressed_scratch_s":round(est_p3_s,0),
}
print(json.dumps(out,indent=2))
json.dump(out,open(os.path.join(os.path.dirname(__file__),"shuffle_mem_summary.json"),"w"),indent=2)
