"""A/B experiment: throughput GAIN vs search-QUALITY cost of raising
dense_cnn MCTS `virtual_batch_size` (leaves gathered per root before each NN eval).

Background: self-play runs at virtual_batch_size=4 (production) and is
overhead-bound (GPU ~38%). Raising vbatch fattens the NN forwards (more
throughput) but correlates leaf selection through virtual loss (a search-quality
cost). This script quantifies BOTH for vbatch in {4, 8, 16}, with vbatch=1 as
the sequential GOLD reference for quality.

Experiment 1 (QUALITY, deterministic, timing-independent): per diverse position,
run a single-game search at visits=512, temperature=0, NO Dirichlet noise,
forced_playout_k=0, for vbatch in [1,4,8,16] with a fresh session each time.
Compare each vbatch's visit-policy distribution against the vbatch=1 GOLD:
top-1 agreement, KL(gold||vbatch), total-variation, top1-mass delta, eff#moves.

Experiment 2 (THROUGHPUT, representative regime): reuse the production
search-only probe `performance._benchmark_selfplay_setting` (256 active games,
512 visits) at vbatch in [4,8,16]; record pos/s, mean leaf batch, cache-hit
fraction, callback fwd/s. Relative pos/s vs vbatch=4.

A live training run may be using the GPU concurrently — timing is therefore
approximate but quality metrics are timing-independent. This script does NOT
touch the live run, its config, or the supervisor.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in ("hexo_engine", "hexo_utils", "hexo_runner", "hexo_train", "hexo_models", "hexo_frontend"):
    sp = str(REPO / "packages" / p / "python")
    if sp not in sys.path:
        sys.path.insert(0, sp)

import torch

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id

CONFIG = REPO / "configs" / "dense_cnn_model1_target_96x6.toml"
CKPT_PRIMARY = REPO / "runs" / "dense_cnn_model1_target_96x6" / "checkpoints" / "epoch_000001.pt"
CKPT_FALLBACK = REPO / "runs" / "dense_cnn_model1_target_96x6" / "checkpoints" / "bootstrap_sealbot_prefit.pt"
RESULT = Path(__file__).resolve().parent / "_vbatch_quality_ab.json"

QUALITY_VBATCHES = [1, 4, 8, 16]
THROUGHPUT_VBATCHES = [4, 8, 16]
QUALITY_VISITS = 512
SNAPSHOT_DEPTHS = (2, 4, 6, 10, 15, 25, 40, 60)
NUM_SEEDS = 12  # 12 seeds * up to 8 snapshot depths -> ~60-96 positions
EPS = 1e-9


def build():
    import tomllib
    from hexo_models.dense_cnn.config import parse_model1_config
    from hexo_models.dense_cnn.plugin import DenseCNNPlugin

    raw = tomllib.loads(CONFIG.read_text())
    section = raw["model"]["config"]
    parsed = parse_model1_config(section)
    model = DenseCNNPlugin().build_model(game_spec={}, config=section)

    ckpt_path = CKPT_PRIMARY
    try:
        state = torch.load(CKPT_PRIMARY, map_location="cpu")["model_state"]
        model.load_state_dict(state, strict=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[ab] primary ckpt {CKPT_PRIMARY.name} failed ({exc!r}); falling back", flush=True)
        ckpt_path = CKPT_FALLBACK
        model.load_state_dict(torch.load(CKPT_FALLBACK, map_location="cpu")["model_state"], strict=True)
    model.eval()
    print(f"[ab] loaded checkpoint: {ckpt_path.name}", flush=True)
    return model, parsed, ckpt_path.name


def _maybe_empty_cache() -> None:
    # Only touch CUDA if it is already initialized — avoids force-initializing a
    # CUDA context (and risking an OOM) when the quality run is on CPU.
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def policy_to_dict(visit_policy) -> dict[int, float]:
    """Normalize a CompactVisitPolicy (action_id, weight) seq into a distribution."""
    d: dict[int, float] = {}
    for aid, w in visit_policy:
        d[int(aid)] = d.get(int(aid), 0.0) + float(w)
    total = sum(d.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in d.items()}


def argmax_action(dist: dict[int, float]) -> int | None:
    if not dist:
        return None
    return max(dist.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def eff_moves(dist: dict[int, float]) -> float:
    """exp(entropy) — effective number of moves."""
    if not dist:
        return 0.0
    h = -sum(p * math.log(max(p, EPS)) for p in dist.values())
    return math.exp(h)


def kl_gold_vs(gold: dict[int, float], other: dict[int, float]) -> float:
    """KL(gold || other), smoothing missing actions in `other` with EPS."""
    if not gold:
        return 0.0
    kl = 0.0
    for aid, pg in gold.items():
        if pg <= 0:
            continue
        po = other.get(aid, 0.0)
        kl += pg * math.log(pg / max(po, EPS))
    return kl


def total_variation(a: dict[int, float], b: dict[int, float]) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def generate_positions(inference, parsed):
    """Self-play GREEDY games (low-visit MCTS argmax) and snapshot diverse depths."""
    from hexo_models.dense_cnn.mcts import new_mcts_session

    sp = parsed.selfplay
    positions: list[dict] = []
    gen_session = new_mcts_session(max_states=1 << 16)
    gen_visits = 96  # cheap search just to drive a realistic greedy line
    for seed in range(NUM_SEEDS):
        state = engine.new_game(seed=seed)
        depth = 0
        snap_set = set(SNAPSHOT_DEPTHS)
        max_depth = max(SNAPSHOT_DEPTHS)
        gen_session.discard(seed)
        while depth <= max_depth:
            if engine.terminal(state) is not None:
                break
            if depth in snap_set:
                positions.append({
                    "seed": seed,
                    "depth": depth,
                    "state": engine.clone_state(state),
                    "n_legal": engine.legal_action_count(state),
                })
            res = gen_session.run(
                [seed], [state], inference,
                visits=gen_visits, c_puct=sp.c_puct, temperature=0.0, seed=seed,
                virtual_batch_size=4,
                root_dirichlet_total_alpha=None, root_dirichlet_noise_fraction=None,
                root_policy_temperature=sp.root_policy_temperature,
                fpu_reduction=sp.fpu_reduction, virtual_loss=sp.virtual_loss,
                widening_policy_mass=sp.widening_policy_mass,
                widening_max_children=sp.widening_max_children,
                widening_min_children=sp.widening_min_children,
                forced_playout_k=0.0,
            )
            aid = res[0].action_id
            engine.apply_action(state, engine.PlacementAction(unpack_coord_id(aid)))
            depth += 1
        gen_session.discard(seed)
    print(f"[ab] generated {len(positions)} nonterminal positions "
          f"(depths {sorted(set(p['depth'] for p in positions))})", flush=True)
    return positions


def run_quality_experiment(inference, parsed, positions, progress_cb=None):
    from hexo_models.dense_cnn.mcts import new_mcts_session

    sp = parsed.selfplay
    # Per-position policy dicts keyed by vbatch. Reuse ONE small session and
    # clear() between searches: a fresh allocation per (pos,vbatch) churned
    # native memory and (with the live run also on the GPU) appears to have
    # tripped a kill. clear() resets the tree so vbatch settings can't carry over.
    session = new_mcts_session(max_states=1 << 13)
    per_pos: list[dict[int, dict[int, float]]] = []
    for i, pos in enumerate(positions):
        by_vb: dict[int, dict[int, float]] = {}
        ok = True
        for vb in QUALITY_VBATCHES:
            session.clear()  # fresh tree per (position, vbatch)
            try:
                res = session.run(
                    [0], [pos["state"]], inference,
                    visits=QUALITY_VISITS, c_puct=sp.c_puct, temperature=0.0, seed=1234,
                    virtual_batch_size=vb,
                    root_dirichlet_total_alpha=None, root_dirichlet_noise_fraction=None,
                    root_policy_temperature=sp.root_policy_temperature,
                    fpu_reduction=sp.fpu_reduction, virtual_loss=sp.virtual_loss,
                    widening_policy_mass=sp.widening_policy_mass,
                    widening_max_children=sp.widening_max_children,
                    widening_min_children=sp.widening_min_children,
                    forced_playout_k=0.0,
                )
            except Exception as exc:  # noqa: BLE001  (skip a position on transient VRAM OOM)
                print(f"[ab] quality pos {i} vbatch={vb} skipped: {exc!r}", flush=True)
                _maybe_empty_cache()
                ok = False
                break
            by_vb[vb] = policy_to_dict(res[0].visit_policy)
        if ok:
            per_pos.append(by_vb)
        if (i + 1) % 5 == 0:
            print(f"[ab] quality: {i + 1}/{len(positions)} positions searched", flush=True)
            _maybe_empty_cache()
            if progress_cb is not None:
                progress_cb(_aggregate_quality(per_pos))

    # Aggregate vs vbatch=1 gold.
    return _aggregate_quality(per_pos), per_pos


def _aggregate_quality(per_pos):
    import numpy as np
    summary: dict[str, dict] = {}
    for vb in QUALITY_VBATCHES:
        if vb == 1:
            continue
        agree = 0
        kls, tvs, top1_deltas, eff_self, eff_gold = [], [], [], [], []
        counted = 0
        for by_vb in per_pos:
            gold = by_vb[1]
            other = by_vb[vb]
            if not gold or not other:
                continue
            counted += 1
            g_arg = argmax_action(gold)
            o_arg = argmax_action(other)
            if g_arg == o_arg:
                agree += 1
            kls.append(kl_gold_vs(gold, other))
            tvs.append(total_variation(gold, other))
            g_top1 = max(gold.values())
            o_top1 = max(other.values())
            top1_deltas.append(abs(o_top1 - g_top1))
            eff_self.append(eff_moves(other))
            eff_gold.append(eff_moves(gold))
        summary[str(vb)] = {
            "positions_counted": counted,
            "top1_agreement": agree / max(counted, 1),
            "mean_kl_gold_vs": float(np.mean(kls)) if kls else 0.0,
            "median_kl_gold_vs": float(np.median(kls)) if kls else 0.0,
            "max_kl_gold_vs": float(np.max(kls)) if kls else 0.0,
            "mean_total_variation": float(np.mean(tvs)) if tvs else 0.0,
            "mean_abs_top1_mass_delta": float(np.mean(top1_deltas)) if top1_deltas else 0.0,
            "mean_eff_moves_vbatch": float(np.mean(eff_self)) if eff_self else 0.0,
            "mean_eff_moves_gold": float(np.mean(eff_gold)) if eff_gold else 0.0,
        }
    return summary


def gpu_clock():
    import subprocess
    try:
        return int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.current.graphics", "--format=csv,noheader,nounits"],
            text=True).strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return -1


def run_throughput_experiment(inf, parsed, stats, probe_positions, selfplay_batch, checkpoint_cb=None):
    from hexo_models.dense_cnn import performance as perf

    def probe(vbatch, positions):
        stats["calls"] = 0; stats["rows"] = 0; stats["time_s"] = 0.0; stats["hist"] = []
        t0 = time.perf_counter()
        res = perf._benchmark_selfplay_setting(
            inference=inf, config=parsed,
            selfplay_batch_size=selfplay_batch,
            virtual_batch_size=vbatch, visits=parsed.selfplay.search_visits,
            probe_positions=positions,
        )
        wall = time.perf_counter() - t0
        hist = np.array(stats["hist"]) if stats["hist"] else np.array([0])
        diag = res.get("mcts_diagnostics", {})
        req = diag.get("eval_requested_states", 0); uniq = diag.get("eval_unique_states", 0)
        return {
            "vbatch": vbatch, "positions": res["positions"],
            "pos_per_s": res["positions_per_second"], "wall_s": wall,
            "mean_leaf_batch": float(hist.mean()),
            "p95_leaf_batch": float(np.percentile(hist, 95)),
            "callback_calls": stats["calls"], "callback_rows": stats["rows"],
            "callback_fwd_per_s": stats["rows"] / max(stats["time_s"], 1e-9),
            "leaf_evals_per_position": stats["rows"] / max(res["positions"], 1),
            "eval_requested_states": req, "eval_unique_states": uniq,
            "cache_hit_frac": 1.0 - uniq / max(req, 1),
            "clock_mhz": gpu_clock(),
        }

    print(f"[ab] throughput warm-up (throwaway, selfplay_batch={selfplay_batch})...", flush=True)
    try:
        probe(4, min(512, probe_positions))
    except Exception as exc:  # noqa: BLE001
        print(f"[ab] throughput warm-up failed: {exc!r}", flush=True)
    results = []
    for vb in THROUGHPUT_VBATCHES:
        try:
            r = probe(vb, probe_positions)
        except Exception as exc:  # noqa: BLE001
            print(f"[ab] throughput vbatch={vb} FAILED (likely VRAM contention): {exc!r}", flush=True)
            torch.cuda.empty_cache()
            continue
        results.append(r)
        print(f"  vbatch={vb:>2d} pos/s={r['pos_per_s']:6.1f} mean_batch={r['mean_leaf_batch']:6.1f} "
              f"p95={r['p95_leaf_batch']:6.1f} evals/pos={r['leaf_evals_per_position']:5.1f} "
              f"cache_hit={r['cache_hit_frac']*100:4.1f}% cb_fwd/s={r['callback_fwd_per_s']:7.0f} "
              f"clk={r['clock_mhz']}", flush=True)
        torch.cuda.empty_cache()
        if checkpoint_cb is not None:
            checkpoint_cb(results)  # persist incrementally so a kill never loses progress
    return results


def main():
    import os
    # cudnn.benchmark=True autotunes by allocating large scratch workspaces, which
    # tips us over the edge when a live training run already holds ~80% of VRAM.
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from hexo_models.dense_cnn.inference import DenseCNNInference

    model, parsed, ckpt_name = build()

    # The QUALITY experiment is the main, timing-independent deliverable. A live
    # training run holds ~80% of VRAM, and earlier GPU attempts were SIGKILLed
    # during CUDA init / first searches (out-of-VRAM, uncatchable in WSL2). So run
    # position-generation + quality on CPU by default: contention-free and exactly
    # reproduces the same deterministic search (vbatch only changes leaf-gather
    # correlation, not the math). Override with HEXO_AB_QUALITY_DEVICE=cuda.
    quality_device = os.environ.get("HEXO_AB_QUALITY_DEVICE", "cpu")
    print(f"[ab] quality/gen device = {quality_device}", flush=True)
    q_inf = DenseCNNInference(model, device=quality_device,
                              amp=(quality_device == "cuda"),
                              return_logits=False, max_batch_size=256)

    # ---- Position set (CPU/quality inference) ----
    print("[ab] generating diverse positions via greedy self-play...", flush=True)
    positions = generate_positions(q_inf, parsed)
    pos_meta = [{"seed": p["seed"], "depth": p["depth"], "n_legal": p["n_legal"]} for p in positions]

    # Probe size: 4096 was too slow under live-run GPU contention (one probe
    # ran ~5-6 min at ~12 pos/s and the long run got killed). 1536 still pushes
    # games past the shallow opening while keeping each probe tractable. The live
    # run holds ~80% of VRAM, so the throughput batch is reduced from 256 to 128
    # active games to fit under our per-process VRAM cap (timing is contention-
    # distorted regardless; the RATIO across vbatch is what matters here).
    probe_positions = 1536
    throughput_selfplay_batch = 128

    out: dict = {
        "checkpoint": ckpt_name,
        "quality_device": quality_device,
        "quality_visits": QUALITY_VISITS,
        "num_positions": len(positions),
        "position_meta": pos_meta,
        "probe_positions": probe_positions,
        "throughput_selfplay_batch": throughput_selfplay_batch,
        "quality": None,
        "throughput": [],
        "throughput_note": None,
    }

    def persist():
        RESULT.write_text(json.dumps(out, indent=2))

    # ---- Experiment 1: quality (deterministic) ----
    print("[ab] EXPERIMENT 1: search quality vs vbatch=1 gold...", flush=True)

    def quality_progress(partial):
        out["quality"] = partial
        out["quality_partial"] = True
        persist()

    quality, _ = run_quality_experiment(q_inf, parsed, positions, progress_cb=quality_progress)
    out["quality"] = quality
    out["quality_partial"] = False
    persist()  # quality is the main deliverable — never lose it
    print(f"[ab] wrote {RESULT.name} (quality done)", flush=True)

    # ---- Experiment 2: throughput (representative regime, GPU, best-effort) ----
    # Throughput requires the GPU and meaningful free VRAM. Under a live training
    # run that holds ~80% of VRAM this may OOM; each probe is guarded so a failure
    # is recorded, not fatal. Build a SEPARATE GPU inference here (the quality one
    # may be on CPU). Timing is contention-distorted; the RATIO across vbatch is
    # the signal of interest.
    print(f"[ab] EXPERIMENT 2: throughput (probe_positions={probe_positions})...", flush=True)
    throughput = []
    try:
        gpu_inf = DenseCNNInference(model, device="cuda", amp=True,
                                    return_logits=False, max_batch_size=512)
        stats: dict = {"calls": 0, "rows": 0, "time_s": 0.0, "hist": []}
        orig = gpu_inf.evaluate_model1_payload

        def wrapped(payload):
            rows = int(payload["shape"][0]); t0 = time.perf_counter()
            o = orig(payload)
            stats["calls"] += 1; stats["rows"] += rows; stats["time_s"] += time.perf_counter() - t0
            stats["hist"].append(rows)
            return o

        gpu_inf.evaluate_model1_payload = wrapped

        def tput_checkpoint(results):
            base = next((t for t in results if t["vbatch"] == 4), None)
            base_pps = base["pos_per_s"] if base else None
            for t in results:
                t["rel_pos_per_s_vs_vb4"] = (t["pos_per_s"] / base_pps) if base_pps else None
            out["throughput"] = results
            persist()

        throughput = run_throughput_experiment(gpu_inf, parsed, stats, probe_positions,
                                               throughput_selfplay_batch,
                                               checkpoint_cb=tput_checkpoint)
        tput_checkpoint(throughput)
        if not throughput:
            out["throughput_note"] = "all GPU probes failed (VRAM contention with live run)"
            persist()
    except Exception as exc:  # noqa: BLE001
        out["throughput_note"] = f"throughput experiment skipped: {exc!r}"
        persist()
        print(f"[ab] EXPERIMENT 2 skipped: {exc!r}", flush=True)
    print(f"[ab] wrote {RESULT.name} (full)", flush=True)

    # ---- Printed tables ----
    print("\n================ EXPERIMENT 1: SEARCH QUALITY (vs vbatch=1 gold) ================")
    print(f"positions={len(positions)}  visits={QUALITY_VISITS}  checkpoint={ckpt_name}")
    print(f"{'vbatch':>6} {'top1_agree':>11} {'mean_KL':>9} {'med_KL':>8} {'max_KL':>8} "
          f"{'mean_TV':>8} {'top1Δ':>7} {'eff#mv':>7} {'(gold eff#mv)':>13}")
    for vb in THROUGHPUT_VBATCHES:
        q = quality[str(vb)]
        print(f"{vb:>6} {q['top1_agreement']*100:>10.1f}% {q['mean_kl_gold_vs']:>9.4f} "
              f"{q['median_kl_gold_vs']:>8.4f} {q['max_kl_gold_vs']:>8.4f} "
              f"{q['mean_total_variation']:>8.4f} {q['mean_abs_top1_mass_delta']:>7.4f} "
              f"{q['mean_eff_moves_vbatch']:>7.2f} {q['mean_eff_moves_gold']:>13.2f}")

    print("\n================ EXPERIMENT 2: THROUGHPUT ================")
    print(f"probe_positions={probe_positions}  {throughput_selfplay_batch} active games  512 visits  "
          f"(live-run GPU contention; timing approximate, ratios are the signal)")
    if not throughput:
        print(f"  (no throughput results: {out.get('throughput_note')})")
    else:
        print(f"{'vbatch':>6} {'pos/s':>8} {'rel_vs_vb4':>11} {'mean_batch':>11} {'p95_batch':>10} "
              f"{'evals/pos':>10} {'cache_hit':>10} {'cb_fwd/s':>9} {'clk':>5}")
        for t in throughput:
            rel = t["rel_pos_per_s_vs_vb4"]
            print(f"{t['vbatch']:>6} {t['pos_per_s']:>8.1f} {rel if rel is None else f'{rel:>10.2f}x':>11} "
                  f"{t['mean_leaf_batch']:>11.1f} {t['p95_leaf_batch']:>10.1f} "
                  f"{t['leaf_evals_per_position']:>10.1f} {t['cache_hit_frac']*100:>9.1f}% "
                  f"{t['callback_fwd_per_s']:>9.0f} {t['clock_mhz']:>5}")


if __name__ == "__main__":
    main()
