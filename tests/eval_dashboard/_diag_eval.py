"""Minimal real harness: reproduce the in-run concurrent eval .hxr write and
diagnose why it produces 0-record files. Tiny GPU budget (2 games, 16 visits).

Writes ONLY to /mnt/e/hexgt-evaldash/scratch/diag_eval (NEVER the live run).
"""
import logging
import os
import shutil
import sys
import traceback
from pathlib import Path

logging.basicConfig(level=logging.DEBUG, stream=sys.stderr,
                    format="%(levelname)s %(name)s: %(message)s")

from hexfield import eval_arena
from hexfield.config import parse_hexfield_config

CAND = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_2/checkpoints/epoch_000039.pt"
OPP = "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_2/checkpoints/epoch_000005.pt"

# Scratch diag dir: <scratch>/diag_eval is diagnostics_dir; the writer writes to
# its PARENT/evaluation/... so parent = <scratch>/run.
SCRATCH = Path("/mnt/e/hexgt-evaldash/scratch/diag_eval")
RUN = SCRATCH / "run"
DIAG = RUN / "diagnostics"
if SCRATCH.exists():
    shutil.rmtree(SCRATCH)
DIAG.mkdir(parents=True, exist_ok=True)

# ---- Instrument _write_eval_hxr: log per-game actions length + any exception. --
_orig = eval_arena._write_eval_hxr


def _patched(games, diagnostics_dir, label_a, label_b, *, kind="checkpoint", stats=None):
    glist = list(games)
    print(f"\n[DIAG] _write_eval_hxr called: label_a={label_a!r} label_b={label_b!r} "
          f"n_games={len(glist)}", file=sys.stderr)
    for i, g in enumerate(glist):
        acts = getattr(g, "actions", None)
        print(f"[DIAG]   game[{i}] index={getattr(g,'local_index','?')} "
              f"actions_len={len(acts) if acts is not None else 'NONE'} "
              f"plies={getattr(g,'plies','?')} status={getattr(g,'status','?')} "
              f"winner={getattr(g,'winner','?')} a_is_p0={getattr(g,'a_is_p0','?')}",
              file=sys.stderr)
    try:
        ret = _orig(glist, diagnostics_dir, label_a, label_b, kind=kind, stats=stats)
        print(f"[DIAG] _write_eval_hxr returned {ret!r}; stats={stats!r}", file=sys.stderr)
        return ret
    except Exception:
        print("[DIAG] _write_eval_hxr RAISED (would be swallowed by outer try):",
              file=sys.stderr)
        traceback.print_exc()
        raise


eval_arena._write_eval_hxr = _patched

cfg = parse_hexfield_config({})
print(f"[DIAG] device={cfg.device} radius_env={os.environ.get('HEXFIELD_SUPPORT_RADIUS')}",
      file=sys.stderr)

matches = eval_arena.play_multi_checkpoint_match(
    CAND,
    [("ep5", OPP)],
    2,                      # n_games_per_opponent
    config=cfg,
    candidate_label="cand_ep39",
    visits=16,
    diagnostics_dir=str(DIAG),
)
print(f"\n[DIAG] matches keys: {list(matches.keys())}", file=sys.stderr)
for lbl, m in matches.items():
    meta = m.get("meta", {})
    print(f"[DIAG] {lbl}: hxr_record={meta.get('hxr_record')!r} "
          f"hxr_games_written={meta.get('hxr_games_written')!r}", file=sys.stderr)

# ---- Decode whatever .hxr got written under the scratch run. -----------------
from hexo_runner.records import HexoRecordFile

import glob
produced = sorted(glob.glob(str(RUN / "evaluation" / "**" / "*.hxr"), recursive=True))
print(f"\n[DIAG] produced .hxr files: {produced}", file=sys.stderr)
for p in produced:
    rf = HexoRecordFile.open(p)
    recs = rf.iter_records()
    print(f"[DIAG] DECODED {p}: num_records={len(recs)}", file=sys.stderr)
    for rec in recs:
        print(f"[DIAG]    game_id={rec.game_id} status={rec.status} "
              f"winner={rec.winner} actions={len(rec.action_ids)}", file=sys.stderr)
