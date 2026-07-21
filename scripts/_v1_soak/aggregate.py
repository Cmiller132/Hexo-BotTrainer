"""V1 SOAK phase 3: aggregate the raw arm sweep into report tables + the five
§10 decision inputs. Reads the raw (position,arm) JSONL (optionally several,
concatenated) and, for the internalization baseline, the positions JSONL that
carries each root's net prior. Emits a JSON summary and prints markdown tables.

Usage:
    python aggregate.py <out_summary.json> <raw1.jsonl> [raw2.jsonl ...] \
        [--positions <positions.jsonl>]
"""

from __future__ import annotations

import arch_env  # noqa: F401  MUST precede the hexfield_eq.geometry import in geometry_pack

import json
import statistics as st
import sys
from collections import defaultdict


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def load_raw(paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                rows.append(json.loads(line))
    return rows


def geometry_pack(q, r):
    # Must match hexfield_eq.geometry.pack_action_id: the trainer packs a coord
    # into a u32 action id. We compare cert_root_move (q,r) to the prior's
    # packed action ids, so replicate the packing here via the engine helper.
    from hexfield_eq.geometry import pack_action_id

    return pack_action_id(int(q), int(r))


def main():
    args = sys.argv[1:]
    positions_path = None
    if "--positions" in args:
        i = args.index("--positions")
        positions_path = args[i + 1]
        args = args[:i] + args[i + 2 :]
    out_path = args[0]
    raw_paths = args[1:]

    rows = load_raw(raw_paths)
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)

    summary = {"arms": {}, "n_raw_rows": len(rows)}

    for arm, rs in sorted(by_arm.items()):
        n = len(rs)
        wins = [r for r in rs if r["status"] == "win"]
        losses = [r for r in rs if r["status"] == "loss"]
        unknowns = [r for r in rs if r["status"] == "unknown"]
        walls_us = [r["wall_nanos"] / 1000.0 for r in rs]
        walls_hot = [r["wall_nanos"] / 1000.0 for r in rs if r.get("hot")]
        walls_quiet = [r["wall_nanos"] / 1000.0 for r in rs if not r.get("hot")]
        nodes = [r["deep_nodes"] for r in rs]
        cert_depths = [r["cert_depth"] for r in wins + losses if r.get("has_cert")]
        hz_cut = sum(r["horizon_cut"] for r in rs)
        hz_cut_tall = sum(r["horizon_cut_tall"] for r in rs)
        kb_death = sum(r["deep_kb_death"] for r in rs)
        zone_nodes = sum(r["zone_nodes"] for r in rs)
        zone_pos = sum(1 for r in rs if r["zone_nodes"] > 0)
        vfail = sum(r["deep_verify_failed"] for r in rs)
        zvfail = sum(r.get("zone_verify_failed", 0) for r in rs)

        entry = {
            "n": n,
            "win": len(wins),
            "loss": len(losses),
            "unknown": len(unknowns),
            "win_rate": len(wins) / n if n else None,
            "loss_rate": len(losses) / n if n else None,
            "verdict_rate": (len(wins) + len(losses)) / n if n else None,
            "wall_us_p50": pct(walls_us, 0.50),
            "wall_us_p90": pct(walls_us, 0.90),
            "wall_us_p99": pct(walls_us, 0.99),
            "wall_us_p50_quiet": pct(walls_quiet, 0.50),
            "wall_us_p90_quiet": pct(walls_quiet, 0.90),
            "wall_us_p50_hot": pct(walls_hot, 0.50),
            "wall_us_p90_hot": pct(walls_hot, 0.90),
            "nodes_mean": st.mean(nodes) if nodes else None,
            "nodes_p90": pct(nodes, 0.90),
            "cert_depth_mean": st.mean(cert_depths) if cert_depths else None,
            "cert_depth_max": max(cert_depths) if cert_depths else None,
            "horizon_cut": hz_cut,
            "horizon_cut_rate": hz_cut / n if n else None,
            "horizon_cut_tall": hz_cut_tall,
            "tall_conversion_note": "see paired ladder-vs-flat in decision inputs",
            "deep_kb_death": kb_death,
            "zone_nodes_total": zone_nodes,
            "zone_positions": zone_pos,
            "deep_verify_failed": vfail,
            "zone_verify_failed": zvfail,
        }
        # census gate + reuse (only present on with_stats arms)
        gate_eval = [r["stats_interior_gate_evaluations"] for r in rs if "stats_interior_gate_evaluations" in r]
        if gate_eval:
            ev = sum(r["stats_interior_gate_evaluations"] for r in rs if "stats_interior_gate_evaluations" in r)
            dis = sum(r["stats_interior_gate_dismissals"] for r in rs if "stats_interior_gate_dismissals" in r)
            fl = sum(r.get("stats_fragment_lookups", 0) for r in rs if "stats_fragment_lookups" in r)
            fh_ = sum(r.get("stats_fragment_hits", 0) for r in rs if "stats_fragment_hits" in r)
            tth = sum(r.get("stats_tt_hits", 0) for r in rs if "stats_tt_hits" in r)
            entry["census_gate_evaluations"] = ev
            entry["census_gate_dismissals"] = dis
            entry["census_dismissal_rate"] = dis / ev if ev else None
            entry["fragment_lookups"] = fl
            entry["fragment_hits"] = fh_
            entry["fragment_hit_rate"] = fh_ / fl if fl else None
            entry["tt_hits"] = tth
        # cert-depth histogram
        hist = defaultdict(int)
        for d in cert_depths:
            hist[d] += 1
        entry["cert_depth_hist"] = dict(sorted(hist.items()))
        summary["arms"][arm] = entry

    # ---- yields by stone-count band (primary WIN arms) ----
    band_yield = {}
    for arm in ("h16_flat_wide", "unbounded_wide"):
        bands = defaultdict(lambda: [0, 0])
        for r in by_arm.get(arm, []):
            b = r["band"]
            bands[b][1] += 1
            if r["status"] in ("win", "loss"):
                bands[b][0] += 1
        band_yield[arm] = {
            str(b): {"verdicts": v[0], "n": v[1], "rate": v[0] / v[1] if v[1] else None}
            for b, v in sorted(bands.items())
        }
    summary["band_yield"] = band_yield

    # ---- decision inputs ----
    di = {}

    # (3) horizon shape: paired flat vs ladder vs unbounded on identical positions
    def index_by_pos(arm):
        return {r["pos_id"]: r for r in by_arm.get(arm, [])}

    flat = index_by_pos("h16_flat_wide")
    ladder = index_by_pos("h16_ladder_wide")
    unb = index_by_pos("unbounded_wide")
    common = set(flat) & set(ladder) & set(unb)
    flat_v = sum(1 for p in common if flat[p]["status"] in ("win", "loss"))
    ladder_v = sum(1 for p in common if ladder[p]["status"] in ("win", "loss"))
    unb_v = sum(1 for p in common if unb[p]["status"] in ("win", "loss"))
    # tall-pass conversion: flat Unknown+horizon_cut>0 that ladder decided
    cut_eligible = [p for p in common if flat[p]["status"] == "unknown" and flat[p]["horizon_cut"] > 0]
    tall_converted = [p for p in cut_eligible if ladder[p]["status"] in ("win", "loss")]
    unb_converted = [p for p in cut_eligible if unb[p]["status"] in ("win", "loss")]
    di["horizon_shape"] = {
        "paired_positions": len(common),
        "flat_verdicts": flat_v,
        "ladder_verdicts": ladder_v,
        "unbounded_verdicts": unb_v,
        "ladder_gain_over_flat": ladder_v - flat_v,
        "unbounded_gain_over_flat": unb_v - flat_v,
        "flat_cut_eligible": len(cut_eligible),
        "tall_converted": len(tall_converted),
        "tall_conversion_rate_of_cut": len(tall_converted) / len(cut_eligible) if cut_eligible else None,
        "unbounded_converted_of_cut": len(unb_converted),
        "kill_criterion_1pct": "ladder OFF if tall_conversion_rate_of_cut < ~0.01",
    }

    # (2) affordability: wall + solves/s implied ceiling (from h16_flat_wide)
    fw = summary["arms"].get("h16_flat_wide", {})
    ub = summary["arms"].get("unbounded_wide", {})
    di["affordability"] = {
        "h16_flat_wall_us_p50": fw.get("wall_us_p50"),
        "h16_flat_wall_us_p90": fw.get("wall_us_p90"),
        "h16_flat_wall_us_p99": fw.get("wall_us_p99"),
        "unbounded_wall_us_p50": ub.get("wall_us_p50"),
        "unbounded_wall_us_p90": ub.get("wall_us_p90"),
        "census_dismissal_rate_h16": fw.get("census_dismissal_rate"),
    }

    # (deep_kb_death verdict)
    di["deep_kb_death"] = {
        arm: summary["arms"][arm]["deep_kb_death"] for arm in summary["arms"]
    }

    # (LOSS-side cost)
    loss_arms = {a: summary["arms"][a] for a in summary["arms"] if a.endswith("_loss")}
    di["loss_side"] = {
        a: {
            "n": e["n"], "loss": e["loss"], "loss_rate": e["loss_rate"],
            "win": e["win"], "unknown": e["unknown"],
            "wall_us_p50": e["wall_us_p50"], "wall_us_p90": e["wall_us_p90"],
            "nodes_mean": e["nodes_mean"],
        }
        for a, e in loss_arms.items()
    }
    # compare LOSS-side wall to WIN-side wall on the subsample
    di["loss_side"]["note"] = (
        "LOSS goal has no census early-out; compare wall/nodes to WIN arms"
    )

    # (zone delta) flat wide zone-off vs zone-on, paired
    zoff = index_by_pos("h16_flat_wide")
    zon = index_by_pos("h16_flat_zone")
    zc = set(zoff) & set(zon)
    zoff_walls = [zoff[p]["wall_nanos"] / 1000.0 for p in zc]
    zon_walls = [zon[p]["wall_nanos"] / 1000.0 for p in zc]
    # Paired wall delta (same positions): zone-on minus zone-off, so a positive
    # number is the added cost of running the zone tight+8 pass + AND-generation.
    wall_deltas = [zon[p]["wall_nanos"] / 1000.0 - zoff[p]["wall_nanos"] / 1000.0 for p in zc]
    di["zone_delta"] = {
        "paired_positions": len(zc),
        "zone_off_verdicts": sum(1 for p in zc if zoff[p]["status"] in ("win", "loss")),
        "zone_on_verdicts": sum(1 for p in zc if zon[p]["status"] in ("win", "loss")),
        "zone_on_zone_nodes_total": sum(zon[p]["zone_nodes"] for p in zc),
        "zone_on_positions_with_zone_nodes": sum(1 for p in zc if zon[p]["zone_nodes"] > 0),
        "flat_zone_nodes_total": sum(zoff[p]["zone_nodes"] for p in zc),
        "wall_us_p50_zone_off": pct(zoff_walls, 0.50),
        "wall_us_p90_zone_off": pct(zoff_walls, 0.90),
        "wall_us_p50_zone_on": pct(zon_walls, 0.50),
        "wall_us_p90_zone_on": pct(zon_walls, 0.90),
        "wall_delta_us_p50": pct(wall_deltas, 0.50),
        "wall_delta_us_p90": pct(wall_deltas, 0.90),
        "ladder_zone_positions": sum(
            1 for r in by_arm.get("h16_ladder_zone", []) if r["zone_nodes"] > 0
        ),
        "ladder_zone_nodes_total": sum(r["zone_nodes"] for r in by_arm.get("h16_ladder_zone", [])),
        "note": (
            "zone-on wall/verdict deltas CONFLATE zone AND-pruning with the "
            "production tight +8 half-budget fast-path inside tss_solve_verified "
            "(both fire only with zone on); zone_nodes isolates actual zone use."
        ),
    }

    # (4) narrow vs wide paired superiority (h16-flat)
    wide = index_by_pos("h16_flat_wide")
    narrow = index_by_pos("h16_flat_narrow")
    nc = set(wide) & set(narrow)
    def decided(x):
        return x["status"] in ("win", "loss")
    both = sum(1 for p in nc if decided(wide[p]) and decided(narrow[p]))
    wonly = sum(1 for p in nc if decided(wide[p]) and not decided(narrow[p]))
    nonly = sum(1 for p in nc if not decided(wide[p]) and decided(narrow[p]))
    neither = sum(1 for p in nc if not decided(wide[p]) and not decided(narrow[p]))
    di["narrow_vs_wide"] = {
        "paired_positions": len(nc),
        "both": both, "wide_only": wonly, "narrow_only": nonly, "neither": neither,
        "wide_verdicts": both + wonly, "narrow_verdicts": both + nonly,
    }
    # unbounded narrow vs wide too
    wideu = index_by_pos("unbounded_wide")
    narrowu = index_by_pos("unbounded_narrow")
    ncu = set(wideu) & set(narrowu)
    di["narrow_vs_wide_unbounded"] = {
        "paired_positions": len(ncu),
        "both": sum(1 for p in ncu if decided(wideu[p]) and decided(narrowu[p])),
        "wide_only": sum(1 for p in ncu if decided(wideu[p]) and not decided(narrowu[p])),
        "narrow_only": sum(1 for p in ncu if not decided(wideu[p]) and decided(narrowu[p])),
        "neither": sum(1 for p in ncu if not decided(wideu[p]) and not decided(narrowu[p])),
    }

    # (MCTS impact) would-it-flip + calibration on proven positions with net_value
    proven = [r for r in by_arm.get("unbounded_wide", []) if r["status"] in ("win", "loss") and r.get("net_value") is not None]
    win_vals = [r["net_value"] for r in proven if r["status"] == "win"]
    loss_vals = [r["net_value"] for r in proven if r["status"] == "loss"]
    # sign disagreement: proven WIN but net value <= 0, proven LOSS but net value >= 0
    win_sign_disagree = sum(1 for v in win_vals if v <= 0.0)
    loss_sign_disagree = sum(1 for v in loss_vals if v >= 0.0)
    di["calibration"] = {
        "proven_win_with_netval": len(win_vals),
        "proven_loss_with_netval": len(loss_vals),
        "win_netval_mean": st.mean(win_vals) if win_vals else None,
        "win_netval_p10": pct(win_vals, 0.10),
        "loss_netval_mean": st.mean(loss_vals) if loss_vals else None,
        "loss_netval_p90": pct(loss_vals, 0.90),
        "win_sign_disagreements": win_sign_disagree,
        "loss_sign_disagreements": loss_sign_disagree,
        "sign_disagreement_rate": (win_sign_disagree + loss_sign_disagree) / len(proven) if proven else None,
        "note": (
            "SIGN-DISAGREEMENT PROXY, not a consumed-move flip simulation: counts "
            "proven-WIN roots whose net root value (side-to-move perspective) is "
            "<= 0 (and proven-LOSS with value >= 0). This is the tactical-headroom "
            "measure of §9; an actual backup/root-move flip would require running "
            "consumption, which V1 does not."
        ),
    }

    # (§8 internalization baseline) prior mass + rank of cert root move at proven WIN roots
    if positions_path:
        prior_by_pos = {}
        with open(positions_path) as fh:
            for line in fh:
                p = json.loads(line)
                prior_by_pos[p["id"]] = p.get("prior") or []
        ranks = []
        masses = []
        top1 = 0
        considered = 0
        for r in by_arm.get("unbounded_wide", []):
            if r["status"] != "win" or "cert_root_move_q" not in r:
                continue
            prior = prior_by_pos.get(r["pos_id"])
            if not prior:
                continue
            packed = geometry_pack(r["cert_root_move_q"], r["cert_root_move_r"])
            considered += 1
            # prior is [[action_id, weight], ...] sorted desc
            rank = None
            mass = 0.0
            for i, (aid, w) in enumerate(prior):
                if int(aid) == packed:
                    rank = i
                    mass = w
                    break
            if rank is not None:
                ranks.append(rank)
                masses.append(mass)
                if rank == 0:
                    top1 += 1
        di["internalization_baseline"] = {
            "proven_win_roots_considered": considered,
            "cert_move_found_in_topk_prior": len(ranks),
            "cert_move_top1_count": top1,
            "cert_move_top1_rate": top1 / considered if considered else None,
            "cert_move_prior_mass_mean": st.mean(masses) if masses else None,
            "cert_move_rank_mean": st.mean(ranks) if ranks else None,
            "note": "baseline for the program internalization curve; rank/mass over epochs",
        }

    # (warmth sensitivity) cold single-shot vs warm persistent-solver, paired
    warmth = {}
    for cold_arm, warm_arm in (
        ("h16_flat_wide", "h16_flat_wide_warm"),
        ("unbounded_wide", "unbounded_wide_warm"),
    ):
        cold = index_by_pos(cold_arm)
        warm = index_by_pos(warm_arm)
        wcommon = set(cold) & set(warm)
        if not wcommon:
            continue
        cold_walls = [cold[p]["wall_nanos"] / 1000.0 for p in wcommon]
        warm_walls = [warm[p]["wall_nanos"] / 1000.0 for p in wcommon]
        warmth[cold_arm] = {
            "paired_positions": len(wcommon),
            "cold_verdicts": sum(1 for p in wcommon if cold[p]["status"] in ("win", "loss")),
            "warm_verdicts": sum(1 for p in wcommon if warm[p]["status"] in ("win", "loss")),
            "cold_wall_us_p50": pct(cold_walls, 0.50),
            "warm_wall_us_p50": pct(warm_walls, 0.50),
            "cold_wall_us_p90": pct(cold_walls, 0.90),
            "warm_wall_us_p90": pct(warm_walls, 0.90),
            "verdict_agreement": sum(1 for p in wcommon if cold[p]["status"] == warm[p]["status"]),
        }
    if warmth:
        di["warmth_sensitivity"] = warmth

    summary["decision_inputs"] = di

    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
