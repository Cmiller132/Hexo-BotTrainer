"""E4 — radius-confound annotation + anchor exclusion.

The support radius is a process-global read once per process, so EVERY opponent
is featurized at the live HEXFIELD_SUPPORT_RADIUS. A radius-8-era anchor
(bc_prefit) forced to radius-4 plays OOD -> weaker -> inflates the candidate's
relative Elo. We cannot vary the radius per net (OnceLock + metadata-less frozen
checkpoints), so we:
  * TAG each radius-8-era edge ``featurized_ood`` + ``featurize_radius`` (flows
    into eval_pool.json rows via provenance),
  * EXCLUDE an OOD anchor from the pinned BT zero-point (kept descriptive),
  * surface ``ratings["fit"]["ood_opponents"]`` + a verdict note.

Because the radius is import-time, the radius-8 control runs in a SUBPROCESS with
HEXFIELD_SUPPORT_RADIUS=8 (cannot be mutated in-process).

Run (radius 4):
  wsl bash -lc 'cd /mnt/e/hexgt-evaldash && HEXFIELD_SUPPORT_RADIUS=4 \
    PYTHONPATH=packages/hexfield/python:packages/dense_cnn_restnet/python \
    /root/.venvs/hexgt-build/bin/python tests/eval_dashboard/test_e4_radius_annotation.py'
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from hexfield import support
from hexfield.config import parse_hexfield_config
from hexfield.multistage_eval import (
    SEALBOT_LABEL,
    Opponent,
    Roster,
    _build_checkpoint_edge_from_match,
    _choose_anchor,
    _stage_d_pool,
)
import hexfield.eval_stats as eval_stats


def _cfg():
    return parse_hexfield_config({}).multi_stage_eval


def _stub_match(a_wins, b_wins):
    decided = a_wins + b_wins
    return {
        "score": {"decided": decided, "a_wins": a_wins},
        "pentanomial": {},
        "meta": {},
    }


def _roster():
    return Roster(
        candidate_label="cand_ep35",
        candidate_epoch=35,
        sealbot=None,
        champion=Opponent(label="ep30", role="champion", ckpt=Path("x"), epoch=30),
        opponents=(
            Opponent(label="bc_prefit", role="anchor", ckpt=Path("bc"), epoch=2),
            Opponent(label="ep30", role="champion", ckpt=Path("x"), epoch=30),
        ),
    )


def test_edge_tagging_radius4():
    assert support._SUPPORT_RADIUS == 4, (
        f"this test must run with HEXFIELD_SUPPORT_RADIUS=4 (got {support._SUPPORT_RADIUS})"
    )
    cfg = _cfg()
    roster = _roster()
    bc = roster.opponents[0]
    ep30 = roster.opponents[1]

    bc_edge = _build_checkpoint_edge_from_match(roster, bc, _stub_match(8, 0), cfg=cfg)
    ep30_edge = _build_checkpoint_edge_from_match(roster, ep30, _stub_match(5, 5), cfg=cfg)

    assert bc_edge["descriptive"]["featurized_ood"] is True, bc_edge["descriptive"]
    assert bc_edge["descriptive"]["featurize_radius"] == 4
    # provenance carries it too (so it lands in the persisted pool row)
    assert bc_edge["descriptive"]["provenance"].get("featurized_ood") is True
    assert ep30_edge["descriptive"]["featurized_ood"] is False, ep30_edge["descriptive"]
    print("[E4] PASS tagging@r4: bc_prefit OOD, ep30 not-OOD, radius=4")


def test_anchor_excludes_ood():
    # bc_prefit (OOD) must NOT be chosen as anchor when a non-OOD ckpt edge exists.
    edges = [
        eval_stats.BTEdge(a="cand_ep35", b="bc_prefit", wins_a=8, wins_b=0),
        eval_stats.BTEdge(a="cand_ep35", b="ep30", wins_a=5, wins_b=5),
    ]
    roster = _roster()
    chosen = _choose_anchor(edges, roster, ood_labels={"bc_prefit"})
    assert chosen != "bc_prefit", f"OOD bc_prefit was picked as anchor: {chosen}"
    assert chosen == "ep30", f"expected non-OOD ep30 anchor, got {chosen}"

    # Without OOD exclusion, bc_prefit (anchor role) would win tier 2.
    chosen_no_excl = _choose_anchor(edges, roster, ood_labels=set())
    assert chosen_no_excl == "bc_prefit", chosen_no_excl
    print(f"[E4] PASS anchor-exclude: OOD->{chosen} (non-OOD->{chosen_no_excl})")


def test_stage_d_surfaces_ood():
    assert support._SUPPORT_RADIUS == 4
    cfg = _cfg()
    roster = _roster()
    pool = {
        "edges": [
            {"epoch": 35, "a": "cand_ep35", "b": "bc_prefit", "wins_a": 8.0,
             "wins_b": 0.0, "weight": 1.0, "kind": "checkpoint", "raw": {}},
            {"epoch": 35, "a": "cand_ep35", "b": "ep30", "wins_a": 5.0,
             "wins_b": 5.0, "weight": 1.0, "kind": "checkpoint", "raw": {}},
        ]
    }
    tmp = Path(tempfile.mkdtemp())
    stage_d, ratings, verdict, _ = _stage_d_pool(
        cfg, roster, [], tmp, pool_doc=pool, append=False
    )
    assert ratings["fit"].get("ood_opponents") == ["bc_prefit"], ratings["fit"]
    assert ratings["fit"].get("anchor") != "bc_prefit", ratings["fit"]
    assert verdict.get("ood_opponents") == ["bc_prefit"], verdict
    assert "ood_note" in verdict, verdict
    print(f"[E4] PASS stage-d: ood_opponents={ratings['fit']['ood_opponents']} "
          f"anchor={ratings['fit']['anchor']}")


_SUBPROC_R8 = """
import sys
from pathlib import Path
from hexfield import support
from hexfield.config import parse_hexfield_config
from hexfield.multistage_eval import Opponent, Roster, _build_checkpoint_edge_from_match, _choose_anchor
import hexfield.eval_stats as eval_stats

assert support._SUPPORT_RADIUS == 8, support._SUPPORT_RADIUS
cfg = parse_hexfield_config({}).multi_stage_eval
roster = Roster(
    candidate_label="cand_ep35", candidate_epoch=35, sealbot=None,
    champion=Opponent("ep30", "champion", Path("x"), 30),
    opponents=(Opponent("bc_prefit", "anchor", Path("bc"), 2),
               Opponent("ep30", "champion", Path("x"), 30)),
)
bc = roster.opponents[0]
match = {"score": {"decided": 8, "a_wins": 8}, "pentanomial": {}, "meta": {}}
edge = _build_checkpoint_edge_from_match(roster, bc, match, cfg=cfg)
assert edge["descriptive"]["featurized_ood"] is False, edge["descriptive"]
assert edge["descriptive"]["featurize_radius"] == 8
# bc_prefit at native radius 8 is NOT OOD -> remains anchor-eligible.
edges = [eval_stats.BTEdge("cand_ep35", "bc_prefit", 8, 0),
         eval_stats.BTEdge("cand_ep35", "ep30", 5, 5)]
chosen = _choose_anchor(edges, roster, ood_labels=set())
assert chosen == "bc_prefit", chosen
print("[E4] PASS@r8 SUBPROCESS: bc_prefit NOT OOD, anchor-eligible, radius=8")
"""


def test_radius8_control_subprocess():
    env = dict(os.environ)
    env["HEXFIELD_SUPPORT_RADIUS"] = "8"
    env["PYTHONPATH"] = "packages/hexfield/python:packages/dense_cnn_restnet/python"
    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROC_R8],
        cwd="/mnt/e/hexgt-evaldash",
        env=env,
        capture_output=True,
        text=True,
    )
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    assert proc.returncode == 0, f"radius-8 subprocess failed:\n{proc.stderr}"
    assert "PASS@r8" in proc.stdout


if __name__ == "__main__":
    test_edge_tagging_radius4()
    test_anchor_excludes_ood()
    test_stage_d_surfaces_ood()
    test_radius8_control_subprocess()
    print("E4 ALL GREEN")
