"""Read-only bridge: surface the hexgt RL run on the live :8080 dashboard with the
SAME richness as a dense_cnn run — epoch/eval summaries, per-epoch loss/training
detail, AND the actual GAME RECORDS (clickable move replays). Reads ONLY the
worktree RL run's own outputs (eval/*.json, eval/*/*.hxr, rl_train.log); writes
ONLY into derived locations (diagnostics/, and copies of game records) under the
run dirs — never touches dense_cnn data or the RL job's files. Loops every
REFRESH_INTERVAL_SECONDS (default 7s) so the dashboard mirror stays near-live.

Translations:
  - eval/epoch_*_vs_{dense,sealbot}/*.hxr  -> MAIN/evaluation/<same>/*.hxr   (the
    dashboard's game viewer requires .hxr under selfplay/ or evaluation/).
  - eval/epoch_*_examples.json (self-play example traces) -> MAIN/selfplay/
    epoch_*.hxr  (regenerated game records so self-play games are viewable).
  - rl_train.log per-epoch loss -> diagnostics/epoch_*.json (metadata.result.training)
  - eval/*_selfplay.json -> diagnostics/dense_cnn.selfplay.epoch_*.json
  - eval/*_eval.json (vs dense_cnn e24) -> diagnostics/dense_cnn.evaluation.epoch_*.json
"""
from __future__ import annotations
import json, time, sys, shutil, re
from pathlib import Path

# Mirror-refresh cadence. 15s (was 7s): still near-live for the dashboard, but
# halves the scan/disk churn so it does not compete with the CPU-bound MCTS select
# threads during self-play. Each loop only does NEW work (caches keep it cheap).
REFRESH_INTERVAL_SECONDS = 15

WT = Path("/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main3")   # worktree RL run (read) -- the ACTIVE run
EVAL = WT / "eval"
SELFPLAY = WT / "selfplay"                                       # per-game .npz (read)
MAIN = Path("/mnt/e/Hexo-BotTrainer/runs/hexgt_rl_main3")        # main-tree mirror (write) -> :8080 dashboard
# Write diagnostics ONLY to the main-tree mirror (the :8080 dashboard). The
# worktree :8081 dashboard was retired, so we no longer mirror into WT/diagnostics.
DIAG_DIRS = [MAIN / "diagnostics"]

# Caches so the 30s loop only does NEW work: each completed game .npz is parsed
# once into (game_id, moves, winner); each epoch's .hxr is rebuilt only when its
# completed-game count grows (so the in-progress epoch refreshes, finished ones
# are left alone).
_GAME_CACHE: dict[str, tuple] = {}
_EPOCH_HXR_COUNT: dict[int, int] = {}


def _epoch_of(npz_name: str) -> int:
    # epoch_000000_game_000012.npz -> 0
    try:
        return int(npz_name.split("_game_")[0].split("_")[-1])
    except Exception:
        return -1


# The dashboard (app.js) is 1-INDEXED and treats epoch 0 as falsy ("No epoch" /
# "--"), because a working dense_cnn run's first epoch is epoch_000001. hexgt's
# RL loop is 0-indexed (rl_epoch 0 = first). Map every epoch the bridge emits to
# the dashboard's convention so labels/counts/filters agree: RL epoch N -> N+1,
# and the pre-RL baseline (rl_epoch -1) -> 0. Display epoch N == RL epoch N-1.
def _disp(rl_epoch: int) -> int:
    return rl_epoch + 1


def _load(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_all(name: str, text: str):
    for d in DIAG_DIRS:
        try:
            d.mkdir(parents=True, exist_ok=True)
            (d / name).write_text(text, encoding="utf-8")
        except Exception as exc:
            print(f"[bridge] write {d/name} failed: {exc}", flush=True)


_LEN_STATS_CACHE: dict[tuple[int, int], dict] = {}  # (epoch, game_count) -> stats


def _length_stats(ep: int) -> dict:
    """Per-epoch game-length stats (mean/median/max/stdev) computed LIVE from the
    full .hxr action counts when present (true moves-to-terminal), else the shard
    rows/game (which equals the true length). Cached by (epoch, game_count) so it
    refreshes as the in-progress epoch grows but is computed once per state."""
    import statistics
    lengths: list[int] = []
    real = SELFPLAY / f"epoch_{ep:06d}.hxr"
    if real.exists():
        try:
            from hexo_runner.records import HexoRecordFile
            lengths = [len(list(r.action_ids)) for r in HexoRecordFile.open(real).iter_records()]
        except Exception:
            lengths = []
    if not lengths:
        from hexo_models.dense_cnn.compact_io import read_compact_shard
        for npz in SELFPLAY.glob(f"epoch_{ep:06d}_game_*.npz"):
            try:
                rows = read_compact_shard(npz)
            except Exception:
                continue
            if rows:
                lengths.append(len(rows))
    if not lengths:
        return {}
    key = (ep, len(lengths))
    if key in _LEN_STATS_CACHE:
        return _LEN_STATS_CACHE[key]
    stats = {
        "game_length_mean": round(statistics.mean(lengths), 2),
        "game_length_median": float(statistics.median(lengths)),
        "game_length_max": max(lengths),
        "game_length_stdev": round(statistics.pstdev(lengths), 2) if len(lengths) > 1 else 0.0,
    }
    _LEN_STATS_CACHE[key] = stats
    return stats


_BUF_RE = re.compile(
    r"epoch (\d+) train: (\d+) steps \(step=\d+\) total=([\d.]+) policy=([\d.]+) value=([\d.]+)"
    r".*?window=(\d+)ep\[([-\d]+)\] (\d+)sh pool~(\d+)k/(\d+)k decay=([\d.]+)")


def _buffer_stats() -> dict[int, dict]:
    """Per-epoch replay-buffer + training stats parsed from the rl_train.log train
    line, keyed by DISPLAY epoch. Matches the recency/cap log format (live after the
    replay deploy); epochs logged in the older uniform format get no buffer block,
    so the dashboard segment is simply omitted (additive / dense-safe)."""
    log = WT / "rl_train.log"
    out: dict[int, dict] = {}
    try:
        txt = log.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return out
    batch = _log_int(r" batch=(\d+)", 128)
    for m in _BUF_RE.finditer(txt):
        steps = int(m.group(2))
        # Per-head losses live between value= and window= in the SAME matched span
        # (m.group(0)), which _BUF_RE bridges with .*?. opp= is always logged;
        # the stv<h>= terms appear only after the per-head driver-log deploy, so
        # epochs logged before it simply get no stvalue keys (additive/dense-safe).
        span_text = m.group(0)
        opp_m = re.search(r"opp=([\d.]+)", span_text)
        stv = {f"loss_stvalue_{h}": float(v)
               for h, v in re.findall(r"stv(\d+)=([\d.]+)", span_text)}
        out[_disp(int(m.group(1)))] = {
            "samples": int(m.group(9)) * 1000,        # current window positions
            "cap": int(m.group(10)) * 1000,           # pool cap
            "window_epochs": int(m.group(6)),
            "window_span": m.group(7),                # "lo-hi" display epochs
            "decay": float(m.group(11)),
            "train_steps": steps,
            "train_batch": batch,
            "train_samples_per_epoch": steps * batch,
            "loss_total": float(m.group(3)),
            "loss_policy": float(m.group(4)),
            "loss_value": float(m.group(5)),
            "loss_opp": float(opp_m.group(1)) if opp_m else None,
            **stv,
        }
    # Value-head calibration: the per-epoch `calib: optimism_sum_mean=` line (a
    # SEPARATE log line from the train line) — 0 = zero-sum-consistent, >0 =
    # optimistic. Merged onto the matching epoch's buffer block; epochs logged
    # before the calib-probe deploy simply get no key (additive / dense-safe).
    for cm in re.finditer(r"epoch (\d+) calib: optimism_sum_mean=([-+\d.]+)", txt):
        block = out.get(_disp(int(cm.group(1))))
        if block is not None:
            block["optimism_sum_mean"] = float(cm.group(2))
    return out


def bridge_diagnostics():
    """Per-epoch self-play + evaluation summaries + live indicator."""
    bufstats = _buffer_stats()
    latest_sp, latest_ep = None, -1
    for sp_path in sorted(EVAL.glob("epoch_*_selfplay.json")):
        d = _load(sp_path)
        if not isinstance(d, dict):
            continue
        ep = int(d.get("epoch", -1))
        cg = int(d.get("completed_games", 0) or 0)
        tg = int(d.get("truncated_games", 0) or 0)
        out = dict(d)
        out.update(_length_stats(ep))  # live length stats (backfills pre-fix epochs)
        out.update(status="completed", epoch=_disp(ep),
                   games_finished=cg + tg, effective_samples=d.get("raw_samples"))
        b = bufstats.get(_disp(ep))   # recency-buffer + training stats (post replay-deploy)
        if b:
            out["buffer"] = b
        _write_all(f"dense_cnn.selfplay.epoch_{_disp(ep):06d}.json", json.dumps(out))
        if ep >= latest_ep:
            latest_ep, latest_sp = ep, out
    for ev_path in sorted(EVAL.glob("epoch_*_eval.json")):
        d = _load(ev_path)
        if not isinstance(d, dict):
            continue
        try:
            ep = int(ev_path.name.split("_")[1])
        except Exception:
            continue
        vd = d.get("vs_dense_cnn_e24") or {}
        vs = d.get("vs_sealbot") or {}
        # The dashboard's eval panel is the "SealBot Evaluation Trend" and renders
        # games/wins/losses/mean_turns. vs_dense_cnn is SKIPPED whenever the
        # dense_cnn e24 baseline checkpoint is absent (it was lost), so sourcing the
        # display W/L/turns from vd left them all null -> the panel rendered blank.
        # Prefer the SealBot leg (which actually ran) for those fields; fall back to
        # dense_cnn only if SealBot is the one missing. Both win-rate fields kept.
        primary = vs if vs.get("games") is not None else vd
        out = {
            "status": "completed", "epoch": _disp(ep),
            "games": primary.get("games"), "completed": primary.get("completed"),
            "wins": primary.get("wins"), "losses": primary.get("losses"), "mean_turns": primary.get("mean_turns"),
            "win_rate_vs_dense_cnn_e24": vd.get("win_rate"), "win_rate_vs_sealbot": vs.get("win_rate"),
        }
        _write_all(f"dense_cnn.evaluation.epoch_{_disp(ep):06d}.json", json.dumps(out))
    if latest_sp is not None:
        _write_all("dense_cnn.selfplay.live.json", json.dumps({
            "status": "running", "epoch": _disp(latest_ep),
            "completed_games": latest_sp.get("completed_games"),
            "requested_games": latest_sp.get("requested_games"),
            "searched_positions": latest_sp.get("searched_positions"),
            "search_positions_per_second": latest_sp.get("search_positions_per_second"),
            "positions_per_second": latest_sp.get("positions_per_second"),
            # KataGo Playout Cap Randomization: surface the full/fast mix + recorded-row
            # count (recorded ~= half of searched when PCR is on). All .get -> safe/None
            # when absent (pre-PCR shards), so the panel degrades cleanly.
            "pcr_enabled": latest_sp.get("pcr_enabled"),
            "recorded_positions": latest_sp.get("recorded_positions"),
            "full_search_count": latest_sp.get("full_search_count"),
            "fast_search_count": latest_sp.get("fast_search_count"),
            # Non-finite-logit sanitization audit (0 in a healthy run; non-zero is a
            # red flag and drives training exclusion -- surfaced at a glance here).
            "sanitized_logit_events": latest_sp.get("sanitized_logit_events"),
            "sanitized_samples_excluded": latest_sp.get("sanitized_samples_excluded"),
            "timestamp": time.time(),
        }))
    return latest_ep


def bridge_loss_detail():
    """Per-epoch training loss from rl_train.log -> diagnostics/epoch_*.json."""
    log = WT / "rl_train.log"
    if not log.exists():
        return
    txt = log.read_text(encoding="utf-8", errors="ignore")
    pat = re.compile(
        r"epoch (\d+) train: (\d+) steps \(step=(\d+)\) total=([\d.]+) "
        r"policy=([\d.]+) value=([\d.]+) opp=([\d.]+).*?([\d.]+)ms/step", re.S)
    for m in pat.finditer(txt):
        ep = int(m.group(1)); steps = int(m.group(2)); total = float(m.group(4))
        ms = float(m.group(8))
        result = {
            "epoch": _disp(ep),
            "training": {
                "status": "completed", "loss": total, "steps": steps, "batch_size": 128,
                "samples": steps * 128, "samples_per_second": (128.0 / (ms / 1000.0)) if ms else None,
                "elapsed_seconds": steps * ms / 1000.0 if ms else None,
            },
        }
        _write_all(f"epoch_{_disp(ep):06d}.json", json.dumps({"status": "completed", "metadata": {"result": result}}))


def bridge_eval_games():
    """Copy the real eval .hxr into MAIN/evaluation/<epoch_vs_x>/ (the layout the
    dashboard's game viewer scans). Only copies new/changed files."""
    dest_root = MAIN / "evaluation"
    n = 0
    for sub in sorted(EVAL.glob("epoch_*_vs_*")):
        if not sub.is_dir():
            continue
        # Rename the epoch in the dest dir to the 1-indexed display epoch so the
        # game viewer parses a truthy epoch (e.g. epoch_-00001_vs_x -> 0,
        # epoch_000000_vs_x -> 1) — matching the self-play .hxr + diagnostics.
        m = re.match(r"epoch_(-?\d+)_vs_(.+)", sub.name)
        dest_name = f"epoch_{_disp(int(m.group(1))):06d}_vs_{m.group(2)}" if m else sub.name
        dest = dest_root / dest_name
        dest.mkdir(parents=True, exist_ok=True)
        for hxr in sub.glob("*.hxr"):
            d = dest / hxr.name
            try:
                if not d.exists() or d.stat().st_size != hxr.stat().st_size:
                    shutil.copy2(hxr, d)
                    n += 1
            except Exception as exc:
                print(f"[bridge] copy {hxr} failed: {exc}", flush=True)
    return n


def _extract_game(npz_path: Path):
    """Parse one completed self-play game shard into (game_id, moves, winner).

    Self-play samples only SEARCHED (nonterminal) positions, so the deepest row's
    placement_history is the game's move sequence (sans the final unsearched
    ply), and the eventual winner is read from that row's value target (value is
    from `current_player`'s perspective: >0 they won, <0 the opponent). Cached by
    path so each shard is parsed once."""
    key = str(npz_path)
    if key in _GAME_CACHE:
        return _GAME_CACHE[key]
    from hexo_models.dense_cnn.compact_io import read_compact_shard
    rows = read_compact_shard(npz_path)
    if not rows:
        return None
    deepest = max(rows, key=lambda r: len(r.placement_history))
    ph = deepest.placement_history
    if not ph:
        return None
    # placement_history entries are (q, r, player, _, placement_index, _, _);
    # order by placement_index and emit (q, r) per move (mirrors reconstruct_state).
    ordered = sorted(ph, key=lambda h: h[4])
    moves = [(int(e[0]), int(e[1])) for e in ordered]
    cp = deepest.current_player
    other = "player1" if cp == "player0" else "player0"
    val = float(deepest.value)  # final outcome target, from cp's perspective
    winner = cp if val > 0.05 else (other if val < -0.05 else None)
    # Honest label: self-play persists only training shards (sampled NONTERMINAL
    # positions), not full game records — so this replay is every move EXCEPT the
    # final winning move (the terminal position is never sampled). Mark it so it
    # is not mistaken for a complete game. (Real games ARE long + decisive — see
    # the self-play game_length_median; the proper fix is writing .hxr in
    # selfplay.py. We do NOT fabricate the missing move from the partial data.)
    game_id = f"{npz_path.stem} (sampled preview — final move not recorded)"
    result = (game_id, moves, winner)
    _GAME_CACHE[key] = result
    return result


def bridge_selfplay_games():
    """Surface self-play games on the dashboard. For a CLOSED epoch that self-play
    recorded a full .hxr for (the feature added in selfplay.py), copy that real
    record (complete games incl. the terminal/winning move). Otherwise — the
    in-progress epoch, or epochs played before that feature (0-3) — reconstruct a
    "sampled preview" .hxr from the per-game .npz shards (which omit the final
    move), rebuilt only when the epoch's completed-game count grows so the
    in-progress epoch refreshes live."""
    import hexo_engine as engine
    from hexo_engine.types import AxialCoord
    from hexo_runner.records import HexoRecordFile, HexoRecordPlayer
    if not SELFPLAY.exists():
        return 0
    spd = MAIN / "selfplay"; spd.mkdir(parents=True, exist_ok=True)
    players = (HexoRecordPlayer("hexgt-a", "player0", "hexgt A"),
               HexoRecordPlayer("hexgt-b", "player1", "hexgt B"))
    by_epoch: dict[int, list[Path]] = {}
    for npz in SELFPLAY.glob("epoch_*_game_*.npz"):
        by_epoch.setdefault(_epoch_of(npz.name), []).append(npz)

    n = 0
    for ep, paths in by_epoch.items():
        if ep < 0:
            continue
        out = spd / f"epoch_{_disp(ep):06d}.hxr"  # 1-indexed display epoch
        # Prefer the REAL full game record whenever self-play has written one —
        # INCLUDING the in-progress epoch (the record is appended per finished
        # game and is readable mid-write, so the current epoch shows FULL replays
        # live as games land). It contains every move incl. the terminal/winning
        # move. Only epochs played before this feature (0-3, no real .hxr) fall
        # back to the npz-reconstructed "sampled preview" below. The copy goes via
        # a temp + a read-validate so a torn mid-write snapshot is never served.
        real_hxr = SELFPLAY / f"epoch_{ep:06d}.hxr"
        if real_hxr.exists():
            try:
                size = real_hxr.stat().st_size
                if not (out.exists() and out.stat().st_size == size):
                    tmp = out.with_suffix(".hxr.tmp")
                    shutil.copy2(real_hxr, tmp)
                    list(HexoRecordFile.open(tmp).iter_records())  # validate readable
                    tmp.replace(out)
                    n += 1
            except Exception as exc:
                print(f"[bridge] copy real selfplay hxr epoch {ep} failed: {exc}", flush=True)
                try:
                    tmp.unlink()
                except Exception:
                    pass
            continue
        if _EPOCH_HXR_COUNT.get(ep) == len(paths):
            continue  # nothing new for this epoch since last preview build
        games = [g for g in (_extract_game(p) for p in sorted(paths)) if g]
        if not games:
            continue
        tmp = out.with_suffix(".hxr.tmp")
        try:
            with HexoRecordFile.create(tmp, engine.engine_metadata(), players) as rf:
                for game_id, moves, winner in games:
                    w = rf.begin_game(game_id, seed=0)
                    for q, r in moves:
                        w.record_action(engine.PlacementAction(AxialCoord(q=q, r=r)))
                    w.finish_completed(winner, len(moves))
            tmp.replace(out)
            _EPOCH_HXR_COUNT[ep] = len(paths)
            n += 1
        except Exception as exc:
            print(f"[bridge] selfplay hxr epoch {ep} failed: {exc}", flush=True)
            try:
                tmp.unlink()
            except Exception:
                pass
    return n


def bridge_baseline_eval():
    """Surface the pre-RL baseline head-to-head (eval/baseline_bc_eval.json) as the
    epoch -1 evaluation point so the 512-visit anchor vs dense_cnn + SealBot shows
    on the detail page (the regular epoch_*_eval.json are picked up by
    bridge_diagnostics; the baseline file is named differently)."""
    d = _load(EVAL / "baseline_bc_eval.json")
    if not isinstance(d, dict):
        return
    vd = d.get("vs_dense_cnn_e24") or {}
    vs = d.get("vs_sealbot") or {}
    # Pre-RL baseline = display epoch 0 (the seed anchor; RL epochs start at 1).
    # Prefer the SealBot leg for the displayed W/L/turns (see bridge_diagnostics):
    # vs_dense_cnn is skipped when the baseline checkpoint is absent.
    primary = vs if vs.get("games") is not None else vd
    out = {
        "status": "completed", "epoch": _disp(-1), "baseline": True,
        "games": primary.get("games"), "completed": primary.get("completed"),
        "wins": primary.get("wins"), "losses": primary.get("losses"), "mean_turns": primary.get("mean_turns"),
        "win_rate_vs_dense_cnn_e24": vd.get("win_rate"), "win_rate_vs_sealbot": vs.get("win_rate"),
    }
    _write_all(f"dense_cnn.evaluation.epoch_{_disp(-1):06d}.json", json.dumps(out))


def _log_int(pattern: str, default: int) -> int:
    log = WT / "rl_train.log"
    try:
        m = re.search(pattern, log.read_text(encoding="utf-8", errors="ignore"))
        return int(m.group(1)) if m else default
    except Exception:
        return default


def _selfplay_counts() -> dict[int, int]:
    """{epoch: completed-game count} from the accumulating per-game .npz."""
    counts: dict[int, int] = {}
    if not SELFPLAY.exists():
        return counts
    for npz in SELFPLAY.glob("epoch_*_game_*.npz"):
        ep = _epoch_of(npz.name)
        if ep >= 0:
            counts[ep] = counts.get(ep, 0) + 1
    return counts


def bridge_live_progress(counts: dict[int, int]):
    """Live mid-epoch self-play state, derived from the per-game .npz so the
    detail page is useful BEFORE an epoch closes (the epoch_*_selfplay.json
    summary + the events the dashboard normally reads only land at epoch end):

      - events.jsonl: a `stage_started` for the in-progress epoch. THIS is what
        sets `current_epoch` in the dashboard, and the game-history viewer only
        shows games whose epoch is in the live/current-epoch window — without it
        the (epoch-0) self-play games are filtered out and the page looks empty.
      - dense_cnn.selfplay.epoch_N.json: an in-progress per-epoch summary so the
        epoch appears in the epoch table (and the eval-window machinery).
      - dense_cnn.selfplay.live.json: the live games-done/requested + speed panel
        (field names must match app.js: games_finished, requested_games, ...).
    """
    if not counts:
        return
    cur = max(counts)
    requested = _log_int(r"games/epoch=(\d+)", 256)
    active = _log_int(r"active=(\d+)", 64)
    closed = (EVAL / f"epoch_{cur:06d}_selfplay.json").exists()

    # events.jsonl: prior epochs finished; the MAX epoch stays "running" (the run
    # is still on it — self-play may have closed but training/eval follow, and the
    # next epoch's games haven't appeared yet). Keeping it active makes
    # current_epoch = cur, which (a) populates the header and (b) keeps cur in the
    # game-history epoch window so its self-play games render.
    # Epochs are written 1-indexed (_disp) to match the dashboard convention.
    lines = []
    for ep in range(0, cur):
        lines.append(json.dumps({"event": "stage_started", "payload": {"stage": f"epoch_{_disp(ep):06d}"}}))
        lines.append(json.dumps({"event": "stage_finished", "payload": {"stage": f"epoch_{_disp(ep):06d}"}}))
    lines.append(json.dumps({"event": "stage_started", "payload": {"stage": f"epoch_{_disp(cur):06d}"}}))
    _write_all("events.jsonl", "\n".join(lines) + "\n")

    if not closed:
        # In-progress per-epoch self-play summary (epoch-table row) + live length stats.
        _write_all(f"dense_cnn.selfplay.epoch_{_disp(cur):06d}.json", json.dumps({
            "status": "running", "epoch": _disp(cur),
            "games_finished": counts[cur], "completed_games": counts[cur],
            "truncated_games": 0, "requested_games": requested, "active_games": active,
            **_length_stats(cur),
        }))
        # Live speed/progress panel.
        _write_all("dense_cnn.selfplay.live.json", json.dumps({
            "status": "running", "epoch": _disp(cur),
            "games_finished": counts[cur], "requested_games": requested,
            "active_games": active, "timestamp": time.time(),
        }))


def bridge_once():
    ep = bridge_diagnostics()
    counts = _selfplay_counts()
    for fn in (bridge_loss_detail, bridge_eval_games, bridge_baseline_eval,
               bridge_selfplay_games):
        try:
            fn()
        except Exception as exc:
            print(f"[bridge] {fn.__name__} failed: {exc}", flush=True)
    try:
        bridge_live_progress(counts)
    except Exception as exc:
        print(f"[bridge] bridge_live_progress failed: {exc}", flush=True)
    return ep


def main():
    once = "--once" in sys.argv
    while True:
        try:
            ep = bridge_once()
            print(f"[bridge] synced through epoch {ep}", flush=True)
        except Exception as exc:
            print(f"[bridge] error: {exc}", flush=True)
        if once:
            break
        time.sleep(REFRESH_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
