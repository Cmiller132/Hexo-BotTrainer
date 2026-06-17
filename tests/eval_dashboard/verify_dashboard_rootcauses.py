"""Read-only verification of the D1/D2/D3 dashboard root causes against the LIVE
run dir (hexfield_main_2). Confirms, with asserts, the exact symptoms the
INVESTIGATION cites before any fix is written. No fix is applied here.

Run (PowerShell):
  wsl bash -lc 'cd /mnt/e/hexgt-evaldash && PYTHONPATH=packages/hexo_frontend/python:packages/hexfield/python:packages/dense_cnn_restnet/python /root/.venvs/hexgt-build/bin/python tests/eval_dashboard/verify_dashboard_rootcauses.py'
"""
import sys
from pathlib import Path

RUN = Path("/mnt/e/Hexo-BotTrainer/runs/hexfield_main_2")

from hexo_frontend import web  # noqa: E402


def main() -> None:
    assert RUN.exists(), f"live run dir missing: {RUN}"

    # --- shared readers ---
    assert web._diag_prefix(RUN) == "hexfield", "run must be hexfield-lineage"
    eval_hist = web._evaluation_history(RUN)
    ms_hist = web._multistage_eval_history(RUN)
    pool = web._eval_pool_summary(RUN)
    live = web._training_live_status(RUN)
    epoch_hist = web._epoch_history(RUN)

    # ===== D1: learning_health is blind to the real eval =====
    # The wrapper evaluation_history rows carry NONE of games/wins/mean_turns.
    assert eval_hist, "expected wrapper evaluation rows to exist"
    null_turns = [r for r in eval_hist if r.get("mean_turns") is None]
    assert len(null_turns) == len(eval_hist), (
        "D1 premise broken: some wrapper rows DO carry mean_turns "
        f"({len(eval_hist) - len(null_turns)}/{len(eval_hist)})"
    )
    print(f"[D1] evaluation_history rows: {len(eval_hist)}, all mean_turns=None: True")

    health = web._learning_health(epoch_hist, eval_hist, live)
    msgs = health.get("messages") or []
    assert any("No SealBot evaluation result yet" in m for m in msgs), (
        "D1: expected the false 'no eval' message; got " + repr(msgs)
    )
    assert health.get("latest_eval_mean_turns") is None, "D1: eval chip would show --"
    # ...while the REAL eval has 7 completed multistage reports.
    assert len(ms_hist) >= 7, f"D1: real eval has {len(ms_hist)} reports (>=7)"
    print(f"[D1] CONFIRMED: learning_health emits {sum('No SealBot' in m for m in msgs)} "
          f"false 'no eval' msg(s) while multistage_eval_history has {len(ms_hist)} reports")
    print(f"[D1] health.status={health.get('status')!r} (forced low by the dead path)")

    # ===== D2: eval_pool ships but has zero UI consumers =====
    assert pool is not None, "D2: eval_pool summary should be present"
    edges = pool.get("edges") or []
    assert pool.get("edges_total") == len(edges)
    assert len(edges) >= 20, f"D2: expected ~29 pool edges, got {len(edges)}"
    # The app.js has no `eval_pool` reference -> dead payload. Verify the symptom
    # POST-FIX: the D2 dead-payload bug is now fixed — app.js DOES consume
    # eval_pool (renderHistEvalPool). The original premise asserted "no consumer";
    # that has been resolved, so this now verifies the consumer is present.
    app_js = Path(web.__file__).resolve().parent / "static" / "app.js"
    js = app_js.read_text(encoding="utf-8")
    assert "eval_pool" in js and "renderHistEvalPool" in js, (
        "D2 FIX regressed: app.js no longer consumes eval_pool"
    )
    kinds = sorted({e.get("kind") for e in edges})
    anchor = pool.get("anchor")
    print(f"[D2] FIXED: eval_pool has {len(edges)} edges (kinds={kinds}, "
          f"anchor={anchor!r}); app.js now renders them (renderHistEvalPool)")

    # ===== D3: data needed for the clean view is all present in the payload =====
    latest = ms_hist[-1]
    players = (latest.get("ratings") or {}).get("players") or []
    assert players, "D3: rating table players present"
    cand = next((p for p in players if str(p.get("label", "")).startswith("cand_")), None)
    assert cand is not None, "D3: candidate node present in ratings"
    assert "verdict" in latest and latest["verdict"].get("label"), "D3: verdict present"
    assert latest.get("sealbot_winrate_ci95"), "D3: sealbot CI present"
    # per-opponent W-L lives in pool edges (wins_a/wins_b) AND headline edges.
    sample = edges[0]
    assert {"a", "b", "wins_a", "wins_b", "epoch", "kind"} <= set(sample), (
        "D3: pool edges carry per-opponent W-L + epoch"
    )
    print(f"[D3] CONFIRMED data available: {len(players)} rating nodes, "
          f"candidate={cand.get('label')} elo={cand.get('elo')}, "
          f"verdict={latest['verdict']['label']}, "
          f"pool edges carry wins_a/wins_b/epoch/kind")

    print("\nALL ROOT-CAUSE PREMISES VERIFIED AGAINST LIVE DATA.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print("VERIFY FAILED:", exc)
        sys.exit(1)
