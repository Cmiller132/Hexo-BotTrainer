"""Decompose selfplay/MCTS from per-epoch mcts_diagnostics. Read-only.
Computes evaluator batching efficiency, cache behavior, evals/position, and the
selfplay sub-phase split for steady epochs."""
import glob, json, os

D = r"E:\Hexo-BotTrainer\runs\dense_cnn_model1_scratch_64\diagnostics"


def f(x, p=1):
    return f"{x:.{p}f}" if isinstance(x, (int, float)) else str(x)


print("top-level result keys (epoch 21):")
top = json.load(open(os.path.join(D, "epoch_000021.json"), encoding="utf-8"))
res = top["metadata"]["result"]
print("  ", list(res.keys()))
for k in ("selfplay", "training", "shuffle", "evaluation", "checkpoint"):
    if k in res and isinstance(res[k], dict):
        print(f"  result['{k}'] keys:", list(res[k].keys()))

print()
hdr = ["ep", "sp_s", "search_s", "eval_s", "enc_s", "treeCPU_s", "orch_s",
       "chunks", "enc_states", "mean_bs", "max_bs", "req", "uniq", "cacheHit%",
       "evals/pos", "us/state"]
print(" ".join(f"{h:>9}" for h in hdr))
for p in sorted(glob.glob(os.path.join(D, "epoch_0000*.json"))):
    ep = int(os.path.basename(p).split("_")[1].split(".")[0])
    r = json.load(open(p, encoding="utf-8"))["metadata"]["result"]
    sp = r.get("selfplay", {})
    md = sp.get("mcts_diagnostics", {})
    if not md:
        continue
    sp_s = sp.get("elapsed_seconds", 0)
    search = sp.get("mcts_search_elapsed_seconds", 0)
    eval_s = md.get("eval_evaluator_seconds", 0)
    enc_s = md.get("eval_encoding_seconds", 0)
    tree = search - eval_s - enc_s
    orch = sp_s - search
    chunks = md.get("eval_evaluator_chunks", 0)
    enc_states = md.get("eval_encoded_states", 0)
    req = md.get("eval_requested_states", 0)
    uniq = md.get("eval_unique_states", 0)
    cache_hits = md.get("eval_cache_hits", 0)
    dup = md.get("eval_duplicate_hits", 0)
    searched = sp.get("searched_positions", 1)
    mean_bs = enc_states / chunks if chunks else 0
    cache_hit_pct = 100.0 * (cache_hits + dup) / req if req else 0
    evals_per_pos = req / searched if searched else 0
    us_state = eval_s * 1e6 / enc_states if enc_states else 0
    row = [ep, sp_s, search, eval_s, enc_s, tree, orch, chunks, enc_states,
           mean_bs, md.get("eval_max_chunk_states", 0), req, uniq,
           cache_hit_pct, evals_per_pos, us_state]
    print(" ".join(f"{f(v):>9}" for v in row))
