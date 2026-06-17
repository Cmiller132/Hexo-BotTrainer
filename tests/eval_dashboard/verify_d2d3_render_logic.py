"""INDEPENDENT verifier (reviewer-authored) for the D2/D3 Evaluation-region
render logic. It does NOT trust the implementer's tests: it re-ports the exact
JS branches of renderHistEvalPool against the REAL payload the server hands the
frontend (the _eval_pool_summary / _multistage_eval_history readers run on the
live run dir, READ-ONLY) and checks the rendered numbers against GROUND-TRUTH
physical counts read straight from eval_pool.json's raw blocks.

D2 fix under test: _eval_pool_summary used to STRIP the whole raw block, so the
W-L matrix in app.js (which prefers raw.physical_wins_*) silently fell back to
the n_eff-WEIGHTED top-level wins_a/wins_b and printed distorted records
(e.g. 3-3 for a real 5-5). The fix threads a SLIM raw block carrying only the
integer physical_wins_* counts. This verifier asserts the summary now carries
those counts AND that the UI-rendered W-L matches the physical ground truth for
every edge.

Run (worktree FIRST on PYTHONPATH; cwd-independent -- reads RUN_DIR directly):
  wsl bash -lc 'cd /mnt/e/hexgt-evaldash && PYTHONPATH=packages/hexo_frontend/python:packages/hexfield/python:packages/dense_cnn_restnet/python /root/.venvs/hexgt-build/bin/python tests/eval_dashboard/verify_d2d3_render_logic.py'
"""
import json
import re
import sys
from pathlib import Path

from hexo_frontend import web  # noqa: E402

RUN_DIR = Path("/mnt/e/Hexo-BotTrainer/runs/hexfield_main_2")
POOL_FILE = RUN_DIR / "diagnostics" / "eval_pool.json"


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def render_wl_record(edge):
    """Port of the W-L matrix per-edge record selection in renderHistEvalPool
    (app.js): prefer raw.physical_wins_a/b, then raw.physical_wins_cand/sealbot,
    then fall back to the top-level wins_a/wins_b. Returns (wa, wb) ROUNDED, the
    exact integers the UI prints."""
    raw = edge.get("raw") if isinstance(edge.get("raw"), dict) else {}
    wa = _f(raw.get("physical_wins_a"))
    wb = _f(raw.get("physical_wins_b"))
    wcand = _f(raw.get("physical_wins_cand"))
    wsb = _f(raw.get("physical_wins_sealbot"))
    if wa is not None and wb is not None:
        a, b = wa, wb
    elif wcand is not None and wsb is not None:
        a, b = wcand, wsb
    else:
        a, b = (_f(edge.get("wins_a")) or 0.0), (_f(edge.get("wins_b")) or 0.0)
    return round(a), round(b)


def ground_truth_physical(edge):
    """The TRUE physical head-to-head from the on-disk raw block."""
    raw = edge["raw"]
    if "physical_wins_a" in raw:
        return round(raw["physical_wins_a"]), round(raw["physical_wins_b"])
    return round(raw["physical_wins_cand"]), round(raw["physical_wins_sealbot"])


def main():
    assert web.__file__.startswith("/mnt/e/hexgt-evaldash/"), (
        f"testing the WRONG web.py: {web.__file__}"
    )
    assert POOL_FILE.is_file(), f"missing live pool file: {POOL_FILE}"

    # The EXACT payload the frontend receives (run the real reader on the live
    # dir, read-only). cwd-independent: do not depend on _resolve_run_dir.
    pool = web._eval_pool_summary(RUN_DIR)
    assert pool is not None, "_eval_pool_summary returned None for the live run"
    summary_edges = pool["edges"]
    on_disk = json.loads(POOL_FILE.read_text())["edges"]

    # 1) The D2 FIX: the slim summary now carries the physical_wins_* counts the
    #    matrix renderer prefers (raw block present, but slim -- no pentanomial).
    raw_present = any(isinstance(e.get("raw"), dict) and e["raw"] for e in summary_edges)
    print(f"[D2-fix] summary edges carry slim 'raw' with physical_wins_*: {raw_present}")
    assert raw_present, (
        "D2 fix regressed: summary edges no longer carry physical_wins_* -> the "
        "W-L matrix would fall back to weighted counts and misreport records"
    )
    # And the slim raw is SLIM (the heavy keys were dropped).
    heavy = {"pentanomial", "n_eff", "virtual_batch_size", "wins_a_eff"}
    for e in summary_edges:
        raw = e.get("raw") or {}
        assert not (set(raw) & heavy), f"slim raw leaked heavy keys: {set(raw) & heavy}"

    # 2) For every on-disk edge, the UI-rendered W-L (using the SUMMARY payload)
    #    now MATCHES the physical ground truth -- no more distorted records.
    summary_by_key = {(e["a"], e["b"]): e for e in summary_edges}
    mismatches = []
    checked = 0
    for disk_edge in on_disk:
        key = (disk_edge["a"], disk_edge["b"])
        ui_edge = summary_by_key.get(key)
        if ui_edge is None:
            continue
        checked += 1
        ui_wa, ui_wb = render_wl_record(ui_edge)          # what UI prints (summary)
        gt_a, gt_b = ground_truth_physical(disk_edge)     # true physical
        if (ui_wa, ui_wb) != (gt_a, gt_b):
            mismatches.append((key, (ui_wa, ui_wb), (gt_a, gt_b),
                               disk_edge.get("kind")))
    print(f"[D2-fix] W-L matrix mismatches (UI vs physical): {len(mismatches)}/{checked}")
    for key, ui, gt, kind in mismatches:
        print(f"   {key[0]} vs {key[1]} ({kind}): UI {ui[0]}-{ui[1]} != PHYSICAL {gt[0]}-{gt[1]}")
    assert checked > 0, "no edges checked -- payload empty?"
    assert not mismatches, (
        f"D2 fix incomplete: {len(mismatches)} edges still render weighted (distorted) "
        "records instead of the true physical head-to-head"
    )

    # 3) D2/D3 unify-ladder sanity: both cand_epN and bare epN exist as split
    #    nodes; the ladder collapses them to one row with a delta.
    ms = web._multistage_eval_history(RUN_DIR)
    assert ms, "no multistage_eval_history for the live run"
    latest = ms[-1]
    labels = {p.get("label") for p in latest["ratings"]["players"]}
    ep_re = re.compile(r"^(?:cand_)?ep(\d+)$")
    ckpt_eps = sorted({int(ep_re.match(l).group(1)) for l in labels if ep_re.match(l)})
    print(f"[D3-ladder] checkpoint epochs unified on ladder: {ckpt_eps}")
    assert 30 in ckpt_eps and {"cand_ep30", "ep30"} <= labels, (
        "unify ladder premise: both readings of ep30 must exist"
    )

    # 4) D3 verdict strip is ascending and complete.
    strip = [(r.get("epoch"),
              str(r.get("verdict_label") or (r.get("verdict") or {}).get("label") or "").upper())
             for r in ms]
    assert [e for e, _ in strip] == sorted(e for e, _ in strip)
    assert all(v for _, v in strip), "every report has a verdict label"
    print(f"[D3-strip] {strip}")

    # 5) D3 dropped-anchor: bc_prefit configured-permanent but absent at ep35.
    roster = latest["roster"]
    perms = roster["permanent_anchors"]
    present = {o.get("label") for o in roster["opponents"]}
    dropped = [a for a in perms if a not in present]
    print(f"[D3-anchor] perms={perms} present={sorted(present)} dropped={dropped}")
    assert "bc_prefit" in dropped, "bc_prefit should render as a dropped-anchor pill"

    print("\nRESULT: D2 W-L matrix now renders TRUE physical head-to-head; "
          "D3 ladder/verdict/anchor logic verified GREEN.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print("VERIFY FAILED:", exc)
        sys.exit(1)
