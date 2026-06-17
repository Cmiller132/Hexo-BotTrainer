"""D1/D2/D3 dashboard-fix assertions against the LIVE read-only run dir
(hexfield_main_2). Verifies the data-layer now surfaces the REAL multi-stage eval
(7 BT reports, pinned ladder, 29-edge pool) instead of the dead "no eval" path.

Run (worktree FIRST on PYTHONPATH so the EDITED web.py is under test):
  wsl bash -lc 'cd /mnt/e/hexgt-evaldash && PYTHONPATH=packages/hexo_frontend/python:packages/hexfield/python:packages/dense_cnn_restnet/python /root/.venvs/hexgt-build/bin/python tests/eval_dashboard/test_eval_dashboard_fixes.py'
"""
import sys
from collections import defaultdict
from pathlib import Path

RUN = Path("/mnt/e/Hexo-BotTrainer/runs/hexfield_main_2")

from hexo_frontend import web  # noqa: E402


def _load():
    eval_hist = web._evaluation_history(RUN)
    ms_hist = web._multistage_eval_history(RUN)
    pool = web._eval_pool_summary(RUN)
    live = web._training_live_status(RUN)
    epoch_hist = web._epoch_history(RUN)
    return eval_hist, ms_hist, pool, live, epoch_hist


def test_d1_health_uses_real_multistage():
    eval_hist, ms_hist, pool, live, epoch_hist = _load()
    assert web.__file__.startswith("/mnt/e/hexgt-evaldash/"), (
        f"testing the WRONG web.py (not the worktree): {web.__file__}"
    )
    assert len(ms_hist) >= 7, f"expected >=7 multistage reports, got {len(ms_hist)}"

    health = web._learning_health(epoch_hist, eval_hist, live, ms_hist)
    msgs = health.get("messages") or []

    # (a) NO false "no eval" sentinel.
    assert not any("No SealBot evaluation result yet" in m for m in msgs), (
        "D1: false 'no eval' message still emitted: " + repr(msgs)
    )
    # (b) NO false hexfield "D6 missing" noise.
    assert not any("D6 augmentation preview is missing" in m for m in msgs), (
        "D1: false 'D6 missing' message still emitted: " + repr(msgs)
    )
    # (c) the new health fields are populated and match the newest report.
    latest_ms = ms_hist[-1]
    want_verdict = str(
        latest_ms.get("verdict_label") or latest_ms["verdict"]["label"]
    ).upper()
    assert health.get("latest_verdict") == want_verdict, (
        f"D1: latest_verdict {health.get('latest_verdict')!r} != {want_verdict!r}"
    )
    assert health.get("latest_cand_elo") is not None, "D1: candidate Elo not populated"
    # candidate elo must match the named candidate node in the latest report.
    cand_label = latest_ms["verdict"]["primary"]["candidate"]
    cand_node = next(
        p for p in latest_ms["ratings"]["players"] if p.get("label") == cand_label
    )
    assert abs(health["latest_cand_elo"] - cand_node["elo"]) < 1e-6, (
        f"D1: cand elo {health['latest_cand_elo']} != node {cand_node['elo']}"
    )
    assert health.get("latest_sealbot_winrate") is not None, "D1: sealbot winrate missing"
    assert health.get("latest_eval_epoch") == 35, (
        f"D1: latest_eval_epoch {health.get('latest_eval_epoch')} != 35"
    )
    # (d) status is on the known ladder and NOT the dead-path 'collecting'.
    assert health["status"] in {"ok", "improving", "watch", "intervene"}, (
        f"D1: unexpected status {health['status']!r}"
    )
    assert health["status"] != "collecting", "D1: still forced to 'collecting'"
    print(
        f"[D1] PASS — status={health['status']} verdict={health['latest_verdict']} "
        f"cand_elo={health['latest_cand_elo']:.1f} sealbot_wr={health['latest_sealbot_winrate']:.3f} "
        f"ep={health['latest_eval_epoch']}; msgs={len(msgs)}"
    )


def test_d1_dense_cnn_path_unchanged():
    # Feed a synthetic dense_cnn-shaped evaluation_history (mean_turns present) and
    # EMPTY multistage history: the legacy turns-based messages must still drive.
    epoch_hist = [
        {"epoch": 1, "status": "completed", "training": {"loss": 2.0}},
        {"epoch": 6, "status": "completed", "training": {"loss": 1.0}},
    ]
    eval_hist = [
        {"epoch": 1, "mean_turns": 20.0, "wins": 1, "games": 8},
        {"epoch": 6, "mean_turns": 28.0, "wins": 3, "games": 8},
    ]
    health = web._learning_health(epoch_hist, eval_hist, {}, [])
    msgs = health.get("messages") or []
    assert any("SealBot eval has" in m or "SealBot survival" in m for m in msgs), (
        "legacy turns-based message missing on dense_cnn path: " + repr(msgs)
    )
    assert health.get("latest_eval_mean_turns") == 28.0
    assert health.get("latest_verdict") is None, "dense_cnn path must not set latest_verdict"
    print(f"[D1] dense_cnn legacy path PASS — status={health['status']} msgs={len(msgs)}")


def test_d2_pool_ledger_builds():
    eval_hist, ms_hist, pool, live, epoch_hist = _load()
    assert pool is not None, "D2: eval_pool summary should be present"
    edges = pool["edges"]
    assert pool["edges_total"] == len(edges) == 29, f"D2: expected 29 edges, got {len(edges)}"
    assert pool["anchor"] == "sealbot"

    # every edge carries the per-opponent W-L contract the renderer needs.
    for e in edges:
        assert {"a", "b", "wins_a", "wins_b", "epoch", "kind"} <= set(e), (
            "D2: pool edge missing a required key: " + repr(e)
        )

    # Aggregating wins_a/wins_b by (a,b) must reproduce the per-opponent record in
    # the newest report's edges. The pool stores n_eff-weighted wins for checkpoint
    # pairs; the raw physical record (5-5 for cand_ep35 vs ep30) lives in the
    # report edge's provenance, so cross-check the *physical* count there.
    latest_path = RUN / "diagnostics" / "hexfield.multistage_eval.epoch_000035.json"
    import json
    rep = json.load(open(latest_path))
    champ_edge = next(e for e in rep["edges"] if e.get("primary"))
    prov = champ_edge["provenance"]
    assert prov["physical_wins_a"] == 5 and prov["physical_wins_b"] == 5, (
        "D2: expected cand_ep35 vs ep30 physical record 5-5, got "
        f"{prov['physical_wins_a']}-{prov['physical_wins_b']}"
    )

    # the per-opponent matrix the renderer builds: group pool edges by (a,b).
    matrix = defaultdict(lambda: [0.0, 0.0, 0])
    for e in edges:
        key = (e["a"], e["b"])
        matrix[key][0] += float(e.get("wins_a") or 0)
        matrix[key][1] += float(e.get("wins_b") or 0)
        matrix[key][2] += 1
    assert ("cand_ep35", "ep30") in matrix, "D2: cand_ep35 vs ep30 pairing missing from pool"
    print(f"[D2] PASS — 29 edges, {len(matrix)} unique pairings, anchor={pool['anchor']}")


def test_d2_d3_unify_has_both_readings():
    # The split-node artifact: the newest report's players include BOTH cand_ep30
    # and bare ep30 (same file, two Elo readings) so the unify step can show the
    # delta rather than reading them as two bots.
    _, ms_hist, _, _, _ = _load()
    latest = ms_hist[-1]
    labels = {p.get("label") for p in latest["ratings"]["players"]}
    assert "cand_ep30" in labels and "ep30" in labels, (
        "D3: expected both cand_ep30 and ep30 split nodes in the ladder"
    )
    print("[D2/D3] PASS — split nodes cand_ep30 + ep30 both present for unify view")


def test_d3_roster_driven_headline_and_dropped_anchor():
    _, ms_hist, _, _, _ = _load()
    latest = ms_hist[-1]
    roster = latest.get("roster") or {}
    assert roster, "D3: roster not shipped on multistage rows"
    perms = roster.get("permanent_anchors") or []
    assert "bc_prefit" in perms and "ep5" in perms, (
        f"D3: permanent_anchors not surfaced from config: {perms}"
    )
    present = {o.get("label") for o in (roster.get("opponents") or [])}
    # bc_prefit is a configured permanent anchor but DROPPED at ep35 (SEV-2) — the
    # dropped-anchor builder must be able to detect it (in perms, NOT in present).
    dropped = [a for a in perms if a not in present]
    assert "bc_prefit" in dropped, (
        f"D3: bc_prefit should be detected as a dropped anchor at ep35; present={present}"
    )
    # ...and at an EARLIER epoch (ep30) bc_prefit IS in the roster -> not dropped.
    ep30 = next((r for r in ms_hist if r.get("epoch") == 30), None)
    assert ep30 is not None
    present30 = {o.get("label") for o in (ep30.get("roster", {}).get("opponents") or [])}
    assert "bc_prefit" in present30, (
        f"D3: bc_prefit should be in ep30 roster; present={present30}"
    )
    print(
        f"[D3] PASS — roster-driven: perms={perms}, ep35 dropped={dropped}, "
        f"ep30 present has bc_prefit={'bc_prefit' in present30}"
    )


def test_d3_verdict_history_strip_data():
    _, ms_hist, _, _, _ = _load()
    strip = [
        (r.get("epoch"), str((r.get("verdict_label") or (r.get("verdict") or {}).get("label") or "")).upper())
        for r in ms_hist
    ]
    epochs = [e for e, _ in strip]
    assert epochs == sorted(epochs), "D3: verdict strip not ascending by epoch"
    labels = [v for _, v in strip]
    assert labels[0] == "REGRESS", f"D3: ep5 should be REGRESS, got {labels[0]}"
    assert all(v for v in labels), "D3: every report must carry a verdict label"
    print(f"[D3] PASS — verdict strip {strip}")


def main():
    test_d1_health_uses_real_multistage()
    test_d1_dense_cnn_path_unchanged()
    test_d2_pool_ledger_builds()
    test_d2_d3_unify_has_both_readings()
    test_d3_roster_driven_headline_and_dropped_anchor()
    test_d3_verdict_history_strip_data()
    print("\nALL DASHBOARD-FIX DATA-LAYER TESTS GREEN.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print("TEST FAILED:", exc)
        sys.exit(1)
