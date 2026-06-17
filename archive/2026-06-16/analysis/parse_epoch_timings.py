"""Extract per-phase wall-clock + selfplay sub-metrics from per-epoch diagnostics
JSON (metadata.result.{selfplay,training}). Read-only."""
import glob, json, os

D = r"E:\Hexo-BotTrainer\runs\dense_cnn_model1_scratch_64\diagnostics"


def g(d, k):
    v = d.get(k)
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v) if v is not None else "-"


hdr = ("ep", "epoch_s", "sp_s", "sp_p/s", "msearch", "eval_s", "enc_s",
       "uniq", "tr_s", "tr_steps", "tr_bs", "tr_p/s")
print(" ".join(f"{h:>9}" for h in hdr))
out = []
for p in sorted(glob.glob(os.path.join(D, "epoch_0000*.json"))):
    ep = int(os.path.basename(p).split("_")[1].split(".")[0])
    top = json.load(open(p, encoding="utf-8"))
    r = top.get("metadata", {}).get("result", {})
    sp = r.get("selfplay", {}) or {}
    tr = r.get("training", {}) or {}
    md = sp.get("mcts_diagnostics", {}) or {}
    row = dict(
        ep=ep, epoch_s=top.get("elapsed_seconds"),
        sp_s=sp.get("elapsed_seconds"), sp_ps=sp.get("positions_per_second"),
        msearch=sp.get("mcts_search_elapsed_seconds"),
        eval_s=md.get("eval_evaluator_seconds"), enc_s=md.get("eval_encoding_seconds"),
        uniq=md.get("eval_unique_states"),
        tr_s=tr.get("elapsed_seconds"), tr_steps=tr.get("steps"),
        tr_bs=tr.get("batch_size"), tr_ps=tr.get("samples_per_second"),
    )
    out.append(row)
    print(" ".join(f"{g(row, k):>9}" for k in
                    ("ep", "epoch_s", "sp_s", "sp_ps", "msearch", "eval_s",
                     "enc_s", "uniq", "tr_s", "tr_steps", "tr_bs", "tr_ps")))

with open(os.path.join(os.path.dirname(__file__), "epoch_timings.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
