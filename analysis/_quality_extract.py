"""Read-only extraction of the 96x8 per-epoch learning series from recorded
diagnostics JSON (stdlib json only; no torch/numpy/model, no GPU)."""
import glob
import json
import os

D = "/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8/diagnostics"


def g(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


rows = {}
for f in sorted(glob.glob(os.path.join(D, "epoch_0*.json"))):
    try:
        p = json.load(open(f, encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print("READ FAIL", os.path.basename(f), repr(e))
        continue
    r = g(p, "metadata", "result", default={}) or {}
    ep = g(r, "epoch")
    if ep is None:
        continue
    tr = g(r, "training", default={}) or {}
    lc = g(tr, "loss_components", default={}) or {}
    rows.setdefault(ep, {})["train"] = dict(
        loss=g(tr, "loss"), policy=lc.get("policy"), value=lc.get("value"),
        opp=lc.get("opp_policy"), total=lc.get("total"), steps=g(tr, "steps"),
        sps=g(tr, "samples_per_second"), validation=g(tr, "validation"),
        lc_keys=sorted(lc.keys()),
    )

for f in sorted(glob.glob(os.path.join(D, "dense_cnn.selfplay.epoch_*.json"))):
    try:
        p = json.load(open(f, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        continue
    ep = p.get("epoch")
    if ep is None:
        continue
    games = p.get("games_finished") or p.get("completed_games")
    raw = p.get("raw_samples")
    rows.setdefault(ep, {})["sp"] = dict(
        games=games, raw=raw, eff=p.get("effective_samples"), trunc=p.get("truncated_games"),
        posps=p.get("positions_per_second"), searched=p.get("searched_positions"),
        mean_len=(raw / games if (isinstance(raw, (int, float)) and games) else None),
    )

for f in sorted(glob.glob(os.path.join(D, "dense_cnn.evaluation.epoch_*.json"))):
    try:
        p = json.load(open(f, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        continue
    ep = p.get("epoch")
    if ep is None:
        continue
    rows.setdefault(ep, {})["eval"] = dict(w=p.get("wins"), l=p.get("losses"), mt=p.get("mean_turns"))


def fm(x, n=3):
    return ("%.*f" % (n, x)) if isinstance(x, (int, float)) else "--"


print("ep | trainLoss policyCE valueCE oppCE | steps samp/s | sp_games sp_meanlen trunc | eval W-L wr% mt | validation")
for ep in sorted(rows):
    t = rows[ep].get("train", {})
    s = rows[ep].get("sp", {})
    e = rows[ep].get("eval", {})
    w, l = e.get("w"), e.get("l")
    wr = (100.0 * w / (w + l)) if isinstance(w, int) and isinstance(l, int) and (w + l) else None
    print(f"{ep:>2} | {fm(t.get('loss'))} {fm(t.get('policy'))} {fm(t.get('value'))} {fm(t.get('opp'))} | "
          f"{t.get('steps','--')} {fm(t.get('sps'),0)} | g={s.get('games','--')} ml={fm(s.get('mean_len'),1)} tr={s.get('trunc','--')} | "
          f"{w}-{l} {fm(wr,1)} mt={fm(e.get('mt'),1)} | val={t.get('validation')}")

# schema sanity: show the loss_components keys + validation for the latest epoch with training
last = max((ep for ep in rows if rows[ep].get("train")), default=None)
if last is not None:
    print("\nSCHEMA(ep", last, ") loss_components keys:", rows[last]["train"].get("lc_keys"))
    print("SCHEMA validation field:", repr(rows[last]["train"].get("validation")))
