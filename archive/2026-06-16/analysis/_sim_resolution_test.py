#!/usr/bin/env python3
"""512 vs 1024 MCTS-visit resolution test on frozen epoch_000020.pt (read-only).

Does doubling the search budget change the search's DECISIONS, or has 512 already
converged? Loads the frozen checkpoint, replays recorded self-play .hxr positions
spanning game phases, and runs the real native MCTS at 512 vs 1024 visits with the
PRODUCTION config (c_puct 1.5, widening_max_children 96, fpu 0.20, virtual_loss 1.0,
forced_playout_k 2.0, root_policy_temperature 1.1) but Dirichlet OFF and a fixed seed
so the ONLY variable is the visit budget.

GPU SAFETY: caps this process to a small fraction of VRAM (any overrun kills THIS
job, never the live trainer) and aborts if free VRAM < MIN_FREE_MIB at startup.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path("/mnt/e/Hexo-BotTrainer")
for sub in (
    "packages/hexo_engine/python",
    "packages/hexo_utils/python",
    "packages/hexo_runner/python",
    "packages/hexo_train/python",
    "packages/hexo_models/python",
    "packages/hexo_models/dense_cnn/python",
):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

# Keep the torch allocator from grabbing the whole device; small, fragmentation-friendly.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

import torch  # noqa: E402

import hexo_engine as engine  # noqa: E402
from hexo_engine.types import unpack_coord_id  # noqa: E402
from hexo_runner.records import HexoRecordFile  # noqa: E402
from hexo_models.dense_cnn.architecture import Model1Network  # noqa: E402
from hexo_models.dense_cnn.inference import DenseCNNInference  # noqa: E402
from hexo_models.dense_cnn.mcts import new_mcts_session  # noqa: E402

CKPT = ROOT / "runs/dense_cnn_model1_target_96x8/checkpoints/epoch_000020.pt"
SELFPLAY = ROOT / "runs/dense_cnn_model1_target_96x8/selfplay"
HXR_EPOCHS = ["epoch_000020.hxr", "epoch_000019.hxr", "epoch_000018.hxr"]
PER_PHASE = 24           # positions per phase (opening/midgame/endgame)
MIN_FREE_MIB = 3000      # abort if less free VRAM than this
FORCE_CPU = os.environ.get("HEXO_SIM_TEST_CPU", "1") == "1"  # safe default: CPU
MEM_FRACTION = 0.16      # ~1.96 GiB hard cap for THIS process on a 12 GiB card
SEED = 20260601          # fixed; same for 512 and 1024 so only sims differ

# Production search config (Dirichlet OFF for a deterministic resolution test).
SEARCH = dict(
    c_puct=1.5,
    temperature=1.0,               # weights in visit_policy == visit fractions
    root_dirichlet_noise_fraction=0.0,   # no noise blended (alpha inert at 0.0)
    root_dirichlet_total_alpha=10.83,     # required to be passed alongside fraction
    root_policy_temperature=1.1,
    fpu_reduction=0.20,
    virtual_loss=1.0,
    widening_policy_mass=0.95,
    widening_max_children=96,
    widening_min_children=2,
    forced_playout_k=2.0,
    virtual_batch_size=4,
)


def pick_device() -> str:
    if FORCE_CPU:
        torch.set_num_threads(6)  # leave CPU headroom for the live trainer
        print("[safety] FORCE_CPU set -> running on CPU (zero GPU risk to live trainer)")
        return "cpu"
    if not torch.cuda.is_available():
        print("[safety] CUDA unavailable -> CPU")
        return "cpu"
    free_b, total_b = torch.cuda.mem_get_info()
    free_mib = free_b / (1024 * 1024)
    print(f"[safety] free VRAM {free_mib:.0f} MiB / {total_b/1024/1024:.0f} MiB")
    if free_mib < MIN_FREE_MIB:
        print(f"[safety] free < {MIN_FREE_MIB} MiB -> CPU fallback (won't risk trainer)")
        return "cpu"
    torch.cuda.set_per_process_memory_fraction(MEM_FRACTION, 0)
    print(f"[safety] capped this process to ~{MEM_FRACTION*total_b/1024/1024:.0f} MiB")
    return "cuda"


def load_model() -> Model1Network:
    payload = torch.load(CKPT, map_location="cpu")
    state = payload["model_state"]
    model = Model1Network(in_channels=13, channels=96, blocks=8,
                          short_term_value_horizons=(1, 4, 8))
    model.load_state_dict(state)
    model.eval()
    return model


def collect_positions() -> list[dict]:
    """Replay recorded games; snapshot nonterminal states at opening/mid/end."""
    buckets: dict[str, list] = {"opening": [], "midgame": [], "endgame": []}
    rng = random.Random(SEED)
    for fname in HXR_EPOCHS:
        path = SELFPLAY / fname
        if not path.is_file():
            continue
        with HexoRecordFile.open(path) as rf:
            records = list(rf.iter_records())
        rng.shuffle(records)
        for rec in records:
            aids = [int(a) for a in rec.action_ids]
            T = len(aids)
            if T < 12:
                continue
            targets = {
                "opening": 6,
                "midgame": T // 2,
                "endgame": max(T - 4, 8),
            }
            for phase, ply in targets.items():
                if len(buckets[phase]) >= PER_PHASE or ply >= T or ply < 1:
                    continue
                state = engine.new_game(seed=rec.seed)
                ok = True
                for a in aids[:ply]:
                    engine.apply_action(state, engine.PlacementAction(unpack_coord_id(a)))
                if engine.terminal(state) is not None:
                    ok = False
                if ok and engine.legal_action_count(state) >= 2:
                    buckets[phase].append({"phase": phase, "state": state,
                                            "ply": ply, "len": T,
                                            "nlegal": engine.legal_action_count(state)})
            if all(len(v) >= PER_PHASE for v in buckets.values()):
                break
        if all(len(v) >= PER_PHASE for v in buckets.values()):
            break
    out = []
    for v in buckets.values():
        out.extend(v)
    return out


def search(session, inference, state, visits, key):
    res = session.run([key], [engine.clone_state(state)], inference,
                      visits=visits, seed=SEED, **SEARCH)
    return res[0]


def dist(result) -> dict[int, float]:
    return {int(a): float(w) for a, w in result.visit_policy}


def kl(p: dict, q: dict, eps=1e-9) -> float:
    s = set(p) | set(q)
    out = 0.0
    for a in s:
        pi = p.get(a, 0.0)
        if pi <= 0:
            continue
        qi = q.get(a, 0.0) + eps
        out += pi * math.log((pi + eps) / qi)
    return out


def argmax(d: dict) -> int:
    return max(d.items(), key=lambda kv: kv[1])[0]


def main() -> None:
    device = pick_device()
    model = load_model()
    inference = DenseCNNInference(model, device=device, use_trt=False,
                                 optimize_for_inference=(device == "cuda"),
                                 max_batch_size=64)
    print(f"[info] model loaded on {device}; TRT off")
    positions = collect_positions()
    print(f"[info] {len(positions)} positions: " +
          ", ".join(f"{p}={sum(1 for x in positions if x['phase']==p)}"
                    for p in ("opening", "midgame", "endgame")))

    session = new_mcts_session()
    rows = []
    key = 0
    for i, pos in enumerate(positions):
        key += 2
        r512 = search(session, inference, pos["state"], 512, key)
        r1024 = search(session, inference, pos["state"], 1024, key + 1)
        p, q = dist(r512), dist(r1024)
        if not p or not q:
            continue
        a512, a1024 = argmax(p), argmax(q)
        f1_512 = max(p.values())
        f1_1024 = max(q.values())
        eff5 = sum(1 for w in p.values() if w >= 0.01)   # ~>=5 visits of 512
        rows.append({
            "phase": pos["phase"], "ply": pos["ply"], "len": pos["len"],
            "nlegal": pos["nlegal"],
            "agree": int(a512 == a1024),
            "kl": kl(p, q),
            "top1_512": f1_512, "top1_1024": f1_1024,
            "d_top1": abs(f1_512 - f1_1024),
            "dv": abs(r1024.root_value - r512.root_value),
            "v512": r512.root_value, "v1024": r1024.root_value,
            "eff_children_512": eff5, "breadth_512": len(p),
            "margin_512": sorted(p.values(), reverse=True)[0] -
                          (sorted(p.values(), reverse=True)[1] if len(p) > 1 else 0.0),
        })
        if (i + 1) % 15 == 0:
            print(f"  ...{i+1}/{len(positions)} done")

    def agg(rs):
        n = len(rs)
        if n == 0:
            return {}
        mean = lambda k: sum(r[k] for r in rs) / n
        med = lambda k: sorted(r[k] for r in rs)[n // 2]
        return {
            "n": n,
            "agree_pct": 100.0 * sum(r["agree"] for r in rs) / n,
            "kl_mean": mean("kl"), "kl_med": med("kl"),
            "d_top1_mean": mean("d_top1"), "d_top1_med": med("d_top1"),
            "dv_mean": mean("dv"), "dv_med": med("dv"), "dv_max": max(r["dv"] for r in rs),
            "eff_children_512_mean": mean("eff_children_512"),
            "eff_children_512_med": med("eff_children_512"),
            "breadth_512_mean": mean("breadth_512"),
            "top1_512_mean": mean("top1_512"),
        }

    summary = {"overall": agg(rows)}
    for ph in ("opening", "midgame", "endgame"):
        summary[ph] = agg([r for r in rows if r["phase"] == ph])
    disagreements = [r for r in rows if not r["agree"]]
    summary["disagreement_margins_512"] = sorted(r["margin_512"] for r in disagreements)

    print("\n===== SUMMARY (512 vs 1024 visits) =====")
    print(json.dumps(summary, indent=2, default=lambda x: round(x, 5)))
    out_path = ROOT / "analysis/_sim_resolution_results.json"
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
