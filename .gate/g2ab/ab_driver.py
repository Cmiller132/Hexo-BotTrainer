"""G2-v2 FhwGate adoption A/B driver (CPU-only, Windows path).

Loads the freshly-built `hexfield_eq._rust` cdylib directly (bypassing the
torch-heavy package __init__), drives the pure-Rust deep-solve batch group2-on
vs group2-off at the production coverage config over the dev splits + the
19-position forcing corpus, and reports the adoption instrument:

  - verified W/L coverage (the adoption metric), nodes/decision
  - gates emitted/accepted + per-cohort firing rate
  - GATE (a) verdict parity (decided OFF but not ON => FAIL)
  - GATE (b) zero verifier failures
  - GATE (c) cross-verification: every gate-decided ON position must re-solve
    to the SAME verdict with group2 OFF at a 50k-node budget.

Never touches the GPU/WSL side; the solver is deterministic + load-robust so
only nodes/decision (never wall time) is quoted.
"""
from __future__ import annotations
import importlib.util, sys, os, json, statistics
from pathlib import Path

WT = Path(r"E:\Hexo-BotTrainer-hexgt\.claude\worktrees\g2-cert")
MAIN = Path(r"E:\Hexo-BotTrainer-hexgt")
sys.path.insert(0, str(WT / "scripts" / "_v1_soak"))
sys.path.insert(0, str(WT / "packages" / "hexo_engine" / "python"))

PYD = WT / ".gate" / "g2ab" / "_rust.pyd"
spec = importlib.util.spec_from_file_location("_rust", str(PYD))
_rust = importlib.util.module_from_spec(spec); spec.loader.exec_module(_rust)
import corpus_lib

# Production coverage config (matches TssBatchAdapter DEFAULTS + dual_pass).
CFG = dict(node_cap=500, goal="both", horizon=0, ladder=False, zone=False,
           wide=True, dual_pass=True, loss_reserve=0)


def solve_batch(states, group2, node_cap=None):
    return _rust.hexfield_eq_deep_solve_batch(
        states, node_cap or CFG["node_cap"], CFG["goal"], CFG["horizon"],
        CFG["ladder"], CFG["zone"], CFG["wide"], CFG["dual_pass"],
        CFG["loss_reserve"], group2)


def decided(r):
    return r["status"] in ("win", "loss") and int(r["deep_verify_failed"]) == 0


def load_set(name):
    p = WT / "scripts" / "tss_harness" / "sets" / f"{name}.jsonl"
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        rows.append({"id": d["pos_id"], "moves": d["moves"]})
    return rows


def main():
    cohorts = {}
    for name in ("selfplay_v1", "human_v1", "puzzle_v3"):
        cohorts[name] = load_set(name)
    corpus_path = WT / "packages" / "hexfield_eq" / "rust" / "corpus" / "forcing_corpus_moves.txt"
    cohorts["forcing_corpus"] = [
        {"id": p["id"], "moves": p["moves"]}
        for p in corpus_lib._parse_corpus_file(corpus_path)
    ]

    report = {"config": CFG, "cohorts": {}}
    parity_fail = []
    total_vf_on = total_vf_off = 0
    xverify = {"checked": 0, "agree": 0, "mismatch": []}
    grand = {"gate_positions": 0, "gate_nodes": 0}

    for cname, rows in cohorts.items():
        states, ids, valid = [], [], []
        for row in rows:
            try:
                states.append(corpus_lib.build_state([list(m) for m in row["moves"]]))
                ids.append(row["id"]); valid.append(row)
            except Exception:
                pass  # malformed/terminal replay -> skip (frozen sets should not hit)
        off = solve_batch(states, False)
        on = solve_batch(states, True)

        cov_off = {"win": 0, "loss": 0}
        cov_on = {"win": 0, "loss": 0}
        nodes_off_dec, nodes_on_dec = [], []
        gate_pos = 0; gate_nodes = 0; g2_pos = 0
        vf_on = vf_off = 0
        for i, pid in enumerate(ids):
            ro, rn = off[i], on[i]
            vf_off += int(ro["deep_verify_failed"]); vf_on += int(rn["deep_verify_failed"])
            do, dn = decided(ro), decided(rn)
            if do:
                cov_off[ro["status"]] += 1; nodes_off_dec.append(int(ro["deep_nodes"]))
            if dn:
                cov_on[rn["status"]] += 1; nodes_on_dec.append(int(rn["deep_nodes"]))
            # (a) parity: decided OFF but not ON.
            if do and not dn:
                parity_fail.append({"cohort": cname, "id": pid,
                                    "off": ro["status"], "on": rn["status"]})
            gn = int(rn.get("gate_nodes", 0))
            if gn > 0:
                gate_pos += 1; gate_nodes += gn
            if int(rn.get("group2_nodes", 0)) > 0:
                g2_pos += 1
            # (c) cross-verification: gate-decided ON must match group2-off @50k.
            if gn > 0 and dn:
                xr = solve_batch([states[i]], False, node_cap=50000)[0]
                xverify["checked"] += 1
                if xr["status"] == rn["status"] and int(xr["deep_verify_failed"]) == 0:
                    xverify["agree"] += 1
                else:
                    xverify["mismatch"].append({"cohort": cname, "id": pid,
                        "on": rn["status"], "off50k": xr["status"],
                        "off50k_vf": int(xr["deep_verify_failed"])})

        total_vf_on += vf_on; total_vf_off += vf_off
        grand["gate_positions"] += gate_pos; grand["gate_nodes"] += gate_nodes
        report["cohorts"][cname] = {
            "n": len(ids),
            "coverage_off": cov_off, "coverage_on": cov_on,
            "decided_off": sum(cov_off.values()), "decided_on": sum(cov_on.values()),
            "nodes_per_decision_off": round(statistics.mean(nodes_off_dec), 1) if nodes_off_dec else None,
            "nodes_per_decision_on": round(statistics.mean(nodes_on_dec), 1) if nodes_on_dec else None,
            "gate_positions": gate_pos, "gate_nodes": gate_nodes,
            "gate_firing_rate": round(gate_pos / max(1, len(ids)), 4),
            "group2_zone_positions": g2_pos,
            "verify_failed_off": vf_off, "verify_failed_on": vf_on,
        }

    report["gates"] = {
        "parity_fail_count": len(parity_fail),
        "parity_fail": parity_fail[:50],
        "verify_failed_total_off": total_vf_off,
        "verify_failed_total_on": total_vf_on,
        "cross_verification": xverify,
        "gate_positions_total": grand["gate_positions"],
        "gate_nodes_total": grand["gate_nodes"],
    }
    out = WT / ".gate" / "g2ab" / "ab_result.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("\n=== VERDICT INPUTS ===")
    print("parity failures (decided OFF not ON):", len(parity_fail))
    print("verify failures on/off:", total_vf_on, total_vf_off)
    print("gate-decided positions:", grand["gate_positions"], "gate nodes:", grand["gate_nodes"])
    print("cross-verify:", xverify["checked"], "checked,", xverify["agree"], "agree,",
          len(xverify["mismatch"]), "MISMATCH")


if __name__ == "__main__":
    main()
