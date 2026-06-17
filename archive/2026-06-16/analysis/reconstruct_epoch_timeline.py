"""Reconstruct the wall-clock timeline of one epoch from artifact mtimes +
events.jsonl epoch start/finish, to get shuffle and eval durations (which are not
separately instrumented). Read-only."""
import glob, json, os, datetime as dt

base = r"E:\Hexo-BotTrainer\runs\dense_cnn_model1_scratch_64"
DI = os.path.join(base, "diagnostics")
EP = 21


def mt(p):
    return os.path.getmtime(p) if os.path.exists(p) else None


def newest(pat):
    fs = glob.glob(pat)
    return max((os.path.getmtime(f) for f in fs), default=None)


# epoch start/finish from events.jsonl (two entries: in_progress + completed)
starts, finishes = [], []
with open(os.path.join(DI, "events.jsonl"), encoding="utf-8") as f:
    for line in f:
        if f'"epoch_{EP:06d}"' in line:
            o = json.loads(line)
            ts = o.get("timestamp") or o.get("time") or o.get("ts")
            finishes.append((o.get("status"), ts, o.get("elapsed_seconds")))
print("epoch event entries:", finishes)

last_selfplay = newest(os.path.join(base, "selfplay", f"epoch_{EP:06d}_game_*.npz"))
selfplay_hxr = mt(os.path.join(base, "selfplay", f"epoch_{EP:06d}.hxr"))
shuffle_dir = sorted(glob.glob(os.path.join(base, "shuffleddata", f"*-epoch_{EP:06d}")))
shuffle_out = newest(os.path.join(shuffle_dir[-1], "train", "*.npz")) if shuffle_dir else None
ckpt = mt(os.path.join(base, "checkpoints", f"epoch_{EP:06d}.pt"))
prev_ckpt = mt(os.path.join(base, "checkpoints", f"epoch_{EP-1:06d}.pt"))
eval_json = mt(os.path.join(DI, f"dense_cnn.evaluation.epoch_{EP:06d}.json"))


def f(ts):
    return dt.datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else "-"


print(f"prev_ckpt(ep{EP-1}):   {f(prev_ckpt)}")
print(f"last selfplay shard: {f(last_selfplay)}")
print(f"selfplay .hxr:       {f(selfplay_hxr)}")
print(f"shuffle out (train): {f(shuffle_out)}")
print(f"checkpoint(ep{EP}):    {f(ckpt)}")
print(f"eval json:           {f(eval_json)}")

# instrumented
r = json.load(open(os.path.join(DI, f"epoch_{EP:06d}.json"), encoding="utf-8"))["metadata"]["result"]
sp = r["selfplay"]["elapsed_seconds"]
tr = r["training"]["elapsed_seconds"]
epoch_total = json.load(open(os.path.join(DI, f"epoch_{EP:06d}.json"), encoding="utf-8"))["elapsed_seconds"]

# durations from timeline
shuffle_dur = (shuffle_out - last_selfplay) if (shuffle_out and last_selfplay) else None
eval_dur = (eval_json - ckpt) if (eval_json and ckpt) else None
print("\n--- phase durations (epoch %d) ---" % EP)
print(f"selfplay (instrumented):     {sp:.0f} s")
print(f"shuffle (mtime selfplay->out): {shuffle_dur:.0f} s" if shuffle_dur else "shuffle: n/a")
print(f"training (instrumented):     {tr:.0f} s")
print(f"eval (mtime ckpt->json):     {eval_dur:.0f} s" if eval_dur else "eval: n/a")
print(f"epoch total (instrumented):  {epoch_total:.0f} s")
acc = sp + (shuffle_dur or 0) + tr + (eval_dur or 0)
print(f"sum of 4 phases:             {acc:.0f} s  (residual {epoch_total-acc:.0f} s)")
out = {"epoch": EP, "selfplay_s": sp, "shuffle_s": shuffle_dur, "training_s": tr,
       "eval_s": eval_dur, "epoch_total_s": epoch_total,
       "pct": {"selfplay": round(100*sp/epoch_total,1),
               "shuffle": round(100*(shuffle_dur or 0)/epoch_total,1),
               "training": round(100*tr/epoch_total,1),
               "eval": round(100*(eval_dur or 0)/epoch_total,1)}}
json.dump(out, open(os.path.join(os.path.dirname(__file__), "epoch_timeline_summary.json"), "w"), indent=2)
print(json.dumps(out["pct"], indent=1))
