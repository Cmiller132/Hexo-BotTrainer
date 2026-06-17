import json

p = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_2/diagnostics/hexfield.multistage_eval.epoch_000035.json"
d = json.load(open(p))


def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            np = f"{path}.{k}"
            if k in ("hxr_record", "hxr_games_written", "multi_checkpoint_error"):
                print(f"{np} = {v!r}")
            walk(v, np)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, f"{path}[{i}]")


walk(d)
print("---- top-level keys:", list(d.keys()))
