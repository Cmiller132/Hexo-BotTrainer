#!/usr/bin/env python3
"""Autonomous prefit-ladder runner for the hexfield_eq (D6-equivariant) rewrite.

Runs DETACHED in WSL (launch via scripts/run_eq_ladder.sh), chains the prefit
arms sequentially on the single training GPU under a HARD DEADLINE, evaluates
every healthy arm for PLAYING STRENGTH vs SealBot, picks a winner by the
owner's strength-based rules, and launches the hexfield_eq_main_1 self-play
soak warm-started from the winner — all without human intervention.

OWNER DECISION RULES (encoded below; verbatim intent, incl. the 2026-07-09
deadline update):
  * Ladder: arm1_vanilla -> arm2_reglane -> arm3_tokread -> arm4_raylayout ->
    arm4c_georay, sequential (one GPU). DEADLINE REGIME: **1 epoch per arm**
    (the 4/6-epoch plan is dead under the ~6 h deadline), all arms identical:
    ``--batch-rows 256 --lr 2.8e-3 --warmup-steps 200 --policy-target gumbel
    --workers 10 --seed 1`` plus an orchestrator-calibrated ``--limit-steps``
    cap (see REGIME below; every number is env-overridable so the
    orchestrator's calibration wins). ``HEXFIELD_EQ_PAIR_BUDGET`` = 4.0e7 for
    the C/A arms (1-3), 1.6e7 for the L arms (4/4c).
  * **Arm 4c is CONDITIONAL**: run it only if the projected remaining time
    (measured arm durations) leaves >= 50 min of headroom (its own projected
    duration + ARM4C_RESERVE) before the deadline after arm 4 finishes;
    otherwise skip it — blockers-on (arm 4) is then the default ray mode.
  * WINNER = STRENGTH-BASED, not top-1. Each healthy arm's checkpoint plays an
    identical UNPAIRED match vs SealBot (owner update: SealBot, not Strix —
    Strix is far too strong for prefit-level checkpoints to separate arms).
    **60 games/arm** under the deadline (knob); coarser SE is acceptable and
    biases toward keeping arms. Score = decided win rate, SE = binomial (the
    SealBot adapter is unpaired by design — no pentanomial).
  * Ranking and the soak init use the **RAW weights** ("model" key), NOT the
    EMA: with ~2-4k optimizer steps/arm the EMA twin (0.9995) lags most of the
    run. ``ema_*`` metrics are recorded in the status file for reference only.
  * Preference order, fullest-stack-first: arm4 — replaced by arm4c only if 4c
    beats 4 by > 1*SE_of_difference on their SealBot scores (the soak then
    runs RAY_BLOCKERS=0) — > arm3 > arm2 > arm1. Walk down; select the FIRST
    arm NOT unambiguously negative, where unambiguously negative = SealBot
    score < (best arm's score - 2*SE_of_difference) OR catastrophic health
    (NaN death / no checkpoint / value_ece_ema > 0.2). Mild prefit-metric
    regressions do NOT disqualify.
  * DEADLINE GOVERNOR: before each stage its duration is projected from
    measured history (priors until data exists); on projected overrun degrade
    in order: (1) skip arm 4c -> (2) remaining SealBot matches drop to 40
    games -> (3) skip the record-only Strix match -> (4) last resort: stop
    prefits/evals, decide from the arms completed so far. THE SOAK LAUNCH
    ALWAYS HAPPENS BEFORE THE DEADLINE with the best available checkpoint.
    Projected-vs-actual timing per stage is written to LADDER_STATUS.md.
  * ON FAILURE: proceed with best available. Never hard-stop except when NO
    arm produced a loadable checkpoint.
  * After the winner: build a soak-init checkpoint from the winner's prefit
    checkpoint (RAW weights), packaged as ``{"meta": <arch meta>, "model":
    <state dict>}`` (the shape hexfield_eq.checkpoints.HexfieldCheckpointLoader
    warm-starts from via ``initialize_from``); write a run-ready toml copy
    with initialize_from pointed at it; source the WINNER ARM's env file and
    launch scripts/_hexfield_eq_supervise_main1.sh detached; verify it is
    alive at ~2 min and record PID + log paths. The systemd unit is NOT
    installed (manual alternative, see docs).
  * OPTIONAL record-only Strix baseline (~60 games, paired) for the WINNER
    only — no decision weight, skipped on any error or deadline pressure.

Status surfaces (both under the ladder root): LADDER_STATUS.md (human,
append-only) and ladder_state.json (machine, atomically rewritten).

Defensive properties: idempotent (completed arms skip; partial arms resume
from their latest checkpoint; an arm already running — e.g. started directly
by the orchestrator — is detected via /proc and monitored, never
double-launched; ``--limit-steps`` SMOKE artifacts are quarantined via the
steps threshold), stall-watchdogged, retry-with-resume, every stage wrapped so
a failure records the error and proceeds.

Modes (stdlib-only parent — torch is imported ONLY in subprocess modes):
  (default)                     the full autonomous ladder.
  --deadline-ts T | --deadline-in-minutes M
                                hard deadline (unix seconds / minutes from now).
  --dry-run [--mock-root DIR]   walk the state machine; construct commands,
                                execute nothing; a virtual clock charges each
                                stage its projection so deadline degradation
                                is exercised deterministically.
  --make-mock DIR --scenario happy|arm3sick|deadline
  --eval-arm NAME ...           [subprocess] repackage + play_sealbot_match.
  --strix-baseline NAME ...     [subprocess] record-only play_strix_match.
  --repackage ...               [subprocess] build the soak-init only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

# --------------------------------------------------------------------------- #
# Paths (env-overridable).
# --------------------------------------------------------------------------- #
REPO = Path(os.environ.get("EQ_LADDER_REPO", "/mnt/e/Hexo-BotTrainer-hexgt"))
VENV_PY = os.environ.get("EQ_LADDER_VENV_PY", "/root/.venvs/hexgt-build/bin/python")
DATA_DIR = Path(os.environ.get(
    "EQ_LADDER_DATA", "/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_prefit_main11"))
LADDER_ROOT = Path(os.environ.get(
    "EQ_LADDER_ROOT", "/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_prefit"))
CONFIG_TOML = Path(os.environ.get(
    "EQ_LADDER_CONFIG", str(REPO / "configs" / "hexfield_eq_main_1.toml")))
SUPERVISOR_SH = REPO / "scripts" / "_hexfield_eq_supervise_main1.sh"
SOAK_RUNDIR = Path(os.environ.get(
    "EQ_LADDER_SOAK_RUNDIR", "/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main_1"))
SEALBOT_PATH = os.environ.get("SEALBOT_PATH", "/mnt/e/SealBot")
DEFAULT_STRIX_CKPT = "/mnt/e/Hexo-BotTrainer/anchors/strix/checkpoint_00237000.pt"

# --------------------------------------------------------------------------- #
# THE DEADLINE REGIME (owner update 2026-07-09). Single source of truth for
# the per-arm prefit invocation; the orchestrator's calibration overrides any
# entry via the EQ_LADDER_* env before launch.
# --------------------------------------------------------------------------- #
REGIME = {
    "epochs": int(os.environ.get("EQ_LADDER_EPOCHS", "1")),          # 1 epoch/arm (arm 2 included)
    "batch_rows": int(os.environ.get("EQ_LADDER_BATCH_ROWS", "256")),
    "lr": os.environ.get("EQ_LADDER_LR", "2.8e-3"),
    "warmup_steps": int(os.environ.get("EQ_LADDER_WARMUP_STEPS", "200")),
    "workers": int(os.environ.get("EQ_LADDER_WORKERS", "10")),
    "seed": int(os.environ.get("EQ_LADDER_SEED", "1")),
    "policy_target": "gumbel",
    # Orchestrator-calibrated per-epoch step cap (0 = uncapped). MUST be set by
    # the orchestrator so each arm fits its ~35-45 min slot.
    "limit_steps": int(os.environ.get("EQ_LADDER_LIMIT_STEPS", "0")),
    # HEXFIELD_EQ_PAIR_BUDGET per arm class (L bias transients ~2-3x heavier).
    "pair_budget_ca": os.environ.get("EQ_LADDER_PAIR_BUDGET_CA", "4.0e7"),
    "pair_budget_l": os.environ.get("EQ_LADDER_PAIR_BUDGET_L", "1.6e7"),
    # Ranking/soak weights: "raw" (owner: EMA lags at ~2-4k steps) or "ema".
    "weights": os.environ.get("EQ_LADDER_WEIGHTS", "raw"),
}

# Strength-eval sizing (owner: 60 games/arm under the deadline; degradation
# drops remaining matches to 40).
EVAL_GAMES = int(os.environ.get("EQ_LADDER_EVAL_GAMES", "60"))
DEGRADED_EVAL_GAMES = int(os.environ.get("EQ_LADDER_DEGRADED_EVAL_GAMES", "40"))
EVAL_SEED_BASE = int(os.environ.get("EQ_LADDER_EVAL_SEED_BASE", "990001"))
EVAL_MAX_WALL = float(os.environ.get("EQ_LADDER_EVAL_MAX_WALL", "3600"))
EVAL_MAX_STATES = int(os.environ.get("EQ_LADDER_EVAL_MAX_STATES", "262144"))
STRIX_BASELINE_GAMES = int(os.environ.get("EQ_LADDER_STRIX_BASELINE_GAMES", "60"))

# Decision rule constants.
NEG_SE_MULT = 2.0          # unambiguously negative: score < best - 2*SE_diff
HEADTOHEAD_SE_MULT = 1.0   # 4c replaces 4 only if 4c - 4 > 1*SE_diff
CATASTROPHIC_ECE = 0.2     # value_ece_ema above this = catastrophic health

# Deadline governor priors (seconds) — used until measured history exists.
PRIOR_PREFIT_SECONDS = float(os.environ.get("EQ_LADDER_PRIOR_PREFIT_SECONDS", "2400"))
PRIOR_EVAL_SECONDS_PER_GAME = float(os.environ.get("EQ_LADDER_PRIOR_EVAL_PER_GAME", "20"))
PRIOR_STRIX_SECONDS = float(os.environ.get("EQ_LADDER_PRIOR_STRIX_SECONDS", "900"))
FINAL_RESERVE_SECONDS = float(os.environ.get("EQ_LADDER_FINAL_RESERVE", "900"))
ARM4C_RESERVE_SECONDS = float(os.environ.get("EQ_LADDER_4C_RESERVE_SECONDS", str(50 * 60)))

# Robustness knobs. MIN_EPOCH_STEPS separates orchestrator --limit-steps~30
# SMOKES from real (possibly step-capped) epochs; with a configured cap the
# real-row threshold is min(cap, MIN_EPOCH_STEPS), floored at 50.
MIN_EPOCH_STEPS = int(os.environ.get("EQ_LADDER_MIN_EPOCH_STEPS", "200"))
POLL_SECONDS = float(os.environ.get("EQ_LADDER_POLL_SECONDS", "30"))
HEARTBEAT_SECONDS = float(os.environ.get("EQ_LADDER_HEARTBEAT_SECONDS", "300"))
STATUS_HEARTBEAT_EVERY = 6
STALL_SECONDS = float(os.environ.get("EQ_LADDER_STALL_SECONDS", "3600"))
MAX_ATTEMPTS = int(os.environ.get("EQ_LADDER_MAX_ATTEMPTS", "3"))
SOAK_VERIFY_SECONDS = float(os.environ.get("EQ_LADDER_SOAK_VERIFY_SECONDS", "120"))

SOAK_INIT_NAME = "soak_init.pt"

PKG_ROOTS = (
    "hexfield_eq", "hexo_engine", "hexo_utils", "hexo_train",
    "hexo_runner", "hexo_strix", "dense_cnn_restnet",
)


@dataclass(frozen=True)
class Arm:
    name: str
    l_layout: bool  # L-trunk arm (arms 4/4c): smaller PAIR_BUDGET

    @property
    def epochs(self) -> int:
        return REGIME["epochs"]

    @property
    def env_file(self) -> Path:
        return REPO / "scripts" / "prefit_env" / f"hexfield_eq_{self.name}.env"


_DEFAULT_ARMS = (
    Arm("arm1_vanilla", False),
    Arm("arm2_reglane", False),
    Arm("arm3_tokread", False),
    Arm("arm4_raylayout", True),
    Arm("arm4c_georay", True),
)


def _arms_from_env() -> tuple[Arm, ...]:
    """EQ_LADDER_ARMS: comma list of ``name[:l]`` tokens (``:l`` = L-class
    pair budget, i.e. the layout has L blocks). Unset -> the default R/L
    ladder. Each name resolves scripts/prefit_env/hexfield_eq_<name>.env.
    Added for the ray-tap wave-1 arm set (SPEC_RAYTAP_CONV.md §6.3); the
    default ladder is byte-identical with the env unset."""

    spec = os.environ.get("EQ_LADDER_ARMS", "").strip()
    if not spec:
        return _DEFAULT_ARMS
    arms = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        name, _, flag = tok.partition(":")
        arms.append(Arm(name.strip(), flag.strip() == "l"))
    return tuple(arms) or _DEFAULT_ARMS


ARMS = _arms_from_env()
CUSTOM_ARMS = bool(os.environ.get("EQ_LADDER_ARMS", "").strip())
ARM_BY_NAME = {a.name: a for a in ARMS}
# Under deadline pressure the decision-relevant arms are evaluated first. For
# a custom arm set the default priority (and decision preference order) is the
# EQ_LADDER_ARMS order; EQ_LADDER_EVAL_PRIORITY overrides either way.
_PRIO_ENV = os.environ.get("EQ_LADDER_EVAL_PRIORITY", "").strip()
if _PRIO_ENV:
    EVAL_PRIORITY = tuple(n.strip() for n in _PRIO_ENV.split(",") if n.strip())
elif CUSTOM_ARMS:
    EVAL_PRIORITY = tuple(a.name for a in ARMS)
else:
    EVAL_PRIORITY = ("arm4_raylayout", "arm4c_georay", "arm3_tokread",
                     "arm2_reglane", "arm1_vanilla")
# EQ_LADDER_NO_SOAK=1: read-only ladder — prefit + eval + decision + the
# record-only Strix baseline, but NO soak launch (the wave-1 mode: the live
# soak keeps running; the winner feeds the write-up, not a relaunch).
NO_SOAK = os.environ.get("EQ_LADDER_NO_SOAK") == "1"


def real_row_min_steps() -> int:
    cap = REGIME["limit_steps"]
    if cap and cap > 0:
        return max(50, min(MIN_EPOCH_STEPS, cap))
    return MIN_EPOCH_STEPS


# --------------------------------------------------------------------------- #
# Small utilities.
# --------------------------------------------------------------------------- #
def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _json_safe(obj):
    """NaN/Inf -> string so ladder_state.json stays strict-JSON parseable."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    return obj


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def parse_env_file(path: Path) -> dict[str, str]:
    """KEY=VALUE lines; '#' comments and blanks ignored; optional quotes/export."""
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.split("#", 1)[0].strip().strip('"').strip("'")
        if key:
            out[key] = val
    return out


def pkg_pythonpath() -> str:
    return ":".join(str(REPO / "packages" / p / "python") for p in PKG_ROOTS)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def _iter_procs():
    proc = Path("/proc")
    if not proc.is_dir():  # non-Linux (dry-run on Windows): fail-soft
        return
    for p in proc.iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        yield int(p.name), cmd


def find_prefit_process(out_dir: Path) -> tuple[int, str] | None:
    """A live ``hexfield_eq.prefit`` whose cmdline targets ``out_dir``."""
    me = os.getpid()
    for pid, cmd in _iter_procs():
        if pid == me:
            continue
        if "hexfield_eq.prefit" in cmd and str(out_dir) in cmd:
            return pid, cmd
    return None


def find_any_prefit_process(exclude_dir: Path | None = None) -> tuple[int, str] | None:
    me = os.getpid()
    for pid, cmd in _iter_procs():
        if pid == me or "hexfield_eq.prefit" not in cmd:
            continue
        if exclude_dir is not None and str(exclude_dir) in cmd:
            continue
        return pid, cmd
    return None


def _kill_tree(pid: int) -> None:
    """SIGKILL a process group (fallback: the pid) — stall-watchdog action."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (OSError, PermissionError, ProcessLookupError):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Deadline governor: projections, measured history, degradation, virtual clock.
# --------------------------------------------------------------------------- #
class Governor:
    """Projects stage durations from measured history (priors until data
    exists) against a hard deadline. In dry-run a virtual clock charges each
    'executed' stage its projection so degradation paths are deterministic."""

    def __init__(self, deadline_ts: float, dry_run: bool):
        self.deadline_ts = float(deadline_ts or 0.0)
        self.dry_run = dry_run
        self.virtual = 0.0
        self.history: dict[str, list[float]] = {}

    def now(self) -> float:
        return time.time() + self.virtual

    def remaining(self) -> float:
        if not self.deadline_ts:
            return math.inf
        return self.deadline_ts - self.now()

    def fmt_remaining(self) -> str:
        r = self.remaining()
        return "no deadline" if r == math.inf else f"{r / 60:.0f} min to deadline"

    def note(self, kind: str, seconds: float) -> None:
        self.history.setdefault(kind, []).append(float(seconds))

    def project(self, kind: str, fallback: float) -> float:
        h = self.history.get(kind)
        return float(statistics.median(h)) if h else float(fallback)

    def eval_proj(self, games: int) -> float:
        per_game = self.project("eval_per_game", PRIOR_EVAL_SECONDS_PER_GAME)
        return per_game * games

    def charge(self, seconds: float) -> None:
        if self.dry_run:
            self.virtual += float(seconds)


# --------------------------------------------------------------------------- #
# Ladder context: status file + state json + logging.
# --------------------------------------------------------------------------- #
class Ladder:
    def __init__(self, root: Path, dry_run: bool = False, deadline_ts: float = 0.0):
        self.root = root
        self.dry_run = dry_run
        self.gov = Governor(deadline_ts, dry_run)
        self.status_md = root / "LADDER_STATUS.md"
        self.state_json = root / "ladder_state.json"
        self.lock_file = root / "ladder_runner.lock"
        root.mkdir(parents=True, exist_ok=True)
        self.state: dict = {
            "version": 2,
            "mode": "dry-run" if dry_run else "run",
            "started_utc": _utc(),
            "updated_utc": _utc(),
            "heartbeat_utc": None,
            "deadline_ts": deadline_ts or None,
            "stage": "init",
            "runner_pid": os.getpid(),
            "regime": dict(REGIME),
            "arms": {a.name: {} for a in ARMS},
            "timeline": [],
            "degradation": [],
            "decision": None,
            "soak": None,
            "errors": [],
        }
        if not self.status_md.exists():
            self.status_md.write_text(
                "# hexfield_eq prefit-ladder runner — status log\n\n"
                "Machine state: `ladder_state.json` (same dir). "
                "Newest entries at the bottom.\n\n",
                encoding="utf-8",
            )

    # ---- logging / status ----
    def status(self, msg: str) -> None:
        line = f"- [{_utc()}] {msg}"
        print(line, flush=True)
        try:
            with open(self.status_md, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:  # never die on a status write
            print(f"  (status write failed: {exc})", flush=True)
        self.save_state()

    def timeline(self, stage: str, projected: float, actual: float | None) -> None:
        rec = {"utc": _utc(), "stage": stage,
               "projected_min": round(projected / 60, 1),
               "actual_min": round(actual / 60, 1) if actual is not None else None,
               "remaining": self.gov.fmt_remaining()}
        self.state["timeline"].append(rec)
        self.status(f"TIMELINE {stage}: projected {rec['projected_min']}m, "
                    f"actual {rec['actual_min']}m, {rec['remaining']}")

    def degrade(self, msg: str) -> None:
        self.state["degradation"].append({"utc": _utc(), "action": msg})
        self.status(f"DEADLINE DEGRADATION: {msg}")

    def error(self, where: str, exc: BaseException | str) -> None:
        text = f"{type(exc).__name__}: {exc}" if isinstance(exc, BaseException) else str(exc)
        self.state["errors"].append({"utc": _utc(), "where": where, "error": text})
        self.status(f"ERROR in {where}: {text} — proceeding with best available")

    def set_stage(self, stage: str) -> None:
        self.state["stage"] = stage
        self.status(f"STAGE -> {stage} ({self.gov.fmt_remaining()})")

    def save_state(self) -> None:
        self.state["updated_utc"] = _utc()
        try:
            _atomic_write_text(
                self.state_json, json.dumps(_json_safe(self.state), indent=2))
        except OSError as exc:
            print(f"  (state write failed: {exc})", flush=True)

    def heartbeat(self, note: str, *, to_md: bool = False) -> None:
        self.state["heartbeat_utc"] = _utc()
        self.state["heartbeat_note"] = note
        if to_md:
            self.status(f"heartbeat: {note}")
        else:
            self.save_state()

    # ---- single-instance lock ----
    def acquire_lock(self) -> bool:
        if self.lock_file.exists():
            try:
                other = int(self.lock_file.read_text().strip())
            except (ValueError, OSError):
                other = -1
            if other > 0 and pid_alive(other):
                print(f"ABORT: another ladder runner is alive (pid {other}, "
                      f"{self.lock_file})", flush=True)
                return False
            print(f"stale ladder lock (pid {other}) — taking over", flush=True)
        self.lock_file.write_text(str(os.getpid()))
        return True

    def release_lock(self) -> None:
        try:
            if self.lock_file.exists() and self.lock_file.read_text().strip() == str(os.getpid()):
                self.lock_file.unlink()
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Per-arm assessment: diagnostics, checkpoints, completion, health.
# --------------------------------------------------------------------------- #
_CKPT_RE = re.compile(r"^checkpoint_epoch(\d+)\.pt$")


@dataclass
class ArmAssessment:
    arm: Arm
    out_dir: Path
    rows: dict[int, dict] = field(default_factory=dict)        # real epochs only
    smoke_epochs: set[int] = field(default_factory=set)
    ckpts: dict[int, Path] = field(default_factory=dict)
    complete: bool = False
    complete_via_ckpt_only: bool = False
    smoke_only: bool = False

    @property
    def latest_ckpt(self) -> Path | None:
        return self.ckpts[max(self.ckpts)] if self.ckpts else None

    @property
    def final_ckpt(self) -> Path | None:
        return self.ckpts.get(self.arm.epochs - 1)

    def summary(self) -> dict:
        return {
            "out_dir": str(self.out_dir),
            "epochs_required": self.arm.epochs,
            "real_epochs": sorted(self.rows),
            "smoke_epochs": sorted(self.smoke_epochs),
            "checkpoints": {e: str(p) for e, p in sorted(self.ckpts.items())},
            "complete": self.complete,
            "complete_via_ckpt_only": self.complete_via_ckpt_only,
            "smoke_only": self.smoke_only,
        }


def arm_out(arm: Arm, root: Path) -> Path:
    return root / arm.name


def assess_arm(arm: Arm, root: Path) -> ArmAssessment:
    out_dir = arm_out(arm, root)
    a = ArmAssessment(arm=arm, out_dir=out_dir)
    min_steps = real_row_min_steps()
    diag = out_dir / "diagnostics.jsonl"
    if diag.is_file():
        for line in diag.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            try:
                epoch = int(row.get("epoch"))
                steps = int(row.get("steps") or 0)
            except (TypeError, ValueError):
                continue
            if steps >= min_steps:
                a.rows[epoch] = row  # later duplicate rows win (crash-rerun case)
            else:
                a.smoke_epochs.add(epoch)
    if out_dir.is_dir():
        for p in out_dir.iterdir():
            m = _CKPT_RE.match(p.name)
            if m and p.stat().st_size > 0:
                a.ckpts[int(m.group(1))] = p
    a.smoke_only = bool(a.smoke_epochs) and not a.rows
    a.complete = (
        a.final_ckpt is not None
        and set(range(arm.epochs)).issubset(a.rows.keys())
    )
    # Edge: the final ckpt exists but NO diagnostics rows at all (a crash while
    # writing the diag line loses the row but the ckpt write follows it, so
    # this means a torn diag file). The checkpoint is fully trained — treat as
    # complete-with-warning rather than looping a no-op resume forever.
    if not a.complete and a.final_ckpt is not None and not a.rows and not a.smoke_epochs:
        a.complete = True
        a.complete_via_ckpt_only = True
    return a


def quarantine_smoke(a: ArmAssessment, ladder: Ladder) -> None:
    """Rename smoke artifacts (``--limit-steps`` probe runs) so they can never
    be mistaken for a finished arm or resumed from. Only called when no live
    prefit process targets the dir."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    moved = []
    diag = a.out_dir / "diagnostics.jsonl"
    if diag.is_file():
        dst = a.out_dir / f"diagnostics.smoke.{ts}.jsonl"
        os.replace(diag, dst)
        moved.append(dst.name)
    for epoch, p in sorted(a.ckpts.items()):
        dst = a.out_dir / f"smoke.{ts}.checkpoint_epoch{epoch}.notpt"
        os.replace(p, dst)
        moved.append(dst.name)
    ladder.status(
        f"{a.arm.name}: SMOKE-ONLY artifacts quarantined ({', '.join(moved) or 'none'}) "
        f"— all diagnostics rows had steps < {real_row_min_steps()}; starting fresh")


def extract_health(a: ArmAssessment) -> dict:
    """Health snapshot from the newest real diagnostics row. The catastrophic
    gate reads the EMA ece when finite; ema_* metrics are otherwise RECORDED
    ONLY (ranking + soak init use the RAW weights under the deadline regime)."""
    health: dict = {"rows_seen": len(a.rows)}
    row = a.rows[max(a.rows)] if a.rows else None
    if row is not None:
        for key in ("epoch", "steps", "top1", "value_ece", "policy_ce", "value_ce",
                    "ema_top1", "ema_value_ece", "ema_policy_ce", "ema_value_ce",
                    "train_total", "token_stream_max", "grad_norm_mean",
                    "train_val_policy_ce_gap", "train_val_value_ce_gap"):
            if key in row:
                health[key] = row[key]
    ece = None
    if row is not None:
        ece = row.get("ema_value_ece")
        if not _is_finite(ece):
            ece = row.get("value_ece")
    health["gate_value_ece"] = ece

    catastrophic, reason = False, None
    if not a.ckpts:
        catastrophic, reason = True, "no_checkpoint"
    elif a.complete_via_ckpt_only:
        catastrophic, reason = False, None
        health.setdefault("warnings", []).append(
            "no diagnostics rows (torn diag write?) — health unknown, checkpoint kept")
    elif row is None:
        catastrophic, reason = True, "no_real_diagnostics_rows"
    else:
        finite_probe = [row.get("train_total"), ece, row.get("ema_top1", row.get("top1"))]
        if any(v is not None and not _is_finite(v) for v in finite_probe) or ece is None:
            catastrophic, reason = True, "nan_death (non-finite final-epoch metrics)"
        elif float(ece) > CATASTROPHIC_ECE:
            catastrophic, reason = True, f"value_ece_ema {float(ece):.3f} > {CATASTROPHIC_ECE}"
    health["catastrophic"] = catastrophic
    health["catastrophic_reason"] = reason
    warnings = health.setdefault("warnings", [])
    if row is not None:
        tsm = row.get("token_stream_max")
        if _is_finite(tsm) and float(tsm) > 50.0:
            warnings.append(f"token_stream_max {float(tsm):.1f} (register-lane watchdog)")
        if _is_finite(ece) and 0.08 < float(ece) <= CATASTROPHIC_ECE:
            warnings.append(f"value_ece {float(ece):.3f} above the 0.08 gate band (mild; not blocking)")
    return health


# --------------------------------------------------------------------------- #
# Prefit stage: launch / take over / monitor / retry.
# --------------------------------------------------------------------------- #
def build_arm_env(arm: Arm) -> dict[str, str]:
    env = dict(os.environ)
    env.update(parse_env_file(arm.env_file))
    env["PYTHONPATH"] = pkg_pythonpath()
    env.setdefault("HEXFIELD_EQ_SUPPORT_RADIUS", "4")  # belt+braces (checklist B5)
    # Deadline regime: per-class pair budget (env file wins if it ever sets one).
    env.setdefault("HEXFIELD_EQ_PAIR_BUDGET",
                   REGIME["pair_budget_l"] if arm.l_layout else REGIME["pair_budget_ca"])
    env.setdefault("SEALBOT_PATH", SEALBOT_PATH)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("OMP_NUM_THREADS", "8")
    return env


def prefit_cmd(arm: Arm, root: Path, resume_ckpt: Path | None) -> list[str]:
    cmd = [
        VENV_PY, "-u", "-m", "hexfield_eq.prefit",
        "--data", str(DATA_DIR),
        "--out", str(arm_out(arm, root)),
        "--epochs", str(arm.epochs),
        "--workers", str(REGIME["workers"]),
        "--seed", str(REGIME["seed"]),
        "--policy-target", REGIME["policy_target"],
        "--batch-rows", str(REGIME["batch_rows"]),
        "--lr", str(REGIME["lr"]),
        "--warmup-steps", str(REGIME["warmup_steps"]),
    ]
    if REGIME["limit_steps"] and REGIME["limit_steps"] > 0:
        cmd += ["--limit-steps", str(REGIME["limit_steps"])]
    if resume_ckpt is not None:
        cmd += ["--resume", str(resume_ckpt)]
    return cmd


def _progress_signature(out_dir: Path) -> tuple:
    """Anything that should move while a prefit is alive: log size, diag size,
    newest checkpoint mtime."""
    sig = []
    for name in ("prefit.log", "diagnostics.jsonl"):
        p = out_dir / name
        sig.append(p.stat().st_size if p.is_file() else -1)
    newest = 0.0
    if out_dir.is_dir():
        for p in out_dir.glob("checkpoint_epoch*.pt"):
            newest = max(newest, p.stat().st_mtime)
    sig.append(int(newest))
    return tuple(sig)


def _monitor(ladder: Ladder, arm: Arm, *, alive_fn, kill_fn, label: str) -> str:
    """Poll until the watched process exits; heartbeat + stall watchdog.
    Returns "exited" or "stall_killed"."""
    out_dir = arm_out(arm, ladder.root)
    last_sig = _progress_signature(out_dir)
    last_progress = time.time()
    last_beat = 0.0
    beats = 0
    while alive_fn():
        time.sleep(POLL_SECONDS)
        sig = _progress_signature(out_dir)
        if sig != last_sig:
            last_sig, last_progress = sig, time.time()
        now = time.time()
        if now - last_beat >= HEARTBEAT_SECONDS:
            last_beat = now
            beats += 1
            done = len(assess_arm(arm, ladder.root).rows)
            note = (f"{label}: monitoring {arm.name} — {done}/{arm.epochs} epoch rows, "
                    f"idle {int(now - last_progress)}s, {ladder.gov.fmt_remaining()}")
            ladder.heartbeat(note, to_md=(beats % STATUS_HEARTBEAT_EVERY == 0))
        if now - last_progress > STALL_SECONDS:
            ladder.status(
                f"{arm.name}: STALL — no log/diagnostics/checkpoint progress for "
                f"{int(STALL_SECONDS)}s; killing the process ({label})")
            kill_fn()
            time.sleep(5)
            return "stall_killed"
    return "exited"


def ensure_arm_prefit(ladder: Ladder, arm: Arm) -> str:
    """Drive one arm to completion (idempotent). Returns 'complete'/'failed'/
    dry-run markers."""
    out_dir = arm_out(arm, ladder.root)
    out_dir.mkdir(parents=True, exist_ok=True)
    attempts = 0
    while True:
        a = assess_arm(arm, ladder.root)
        ladder.state["arms"][arm.name]["prefit"] = a.summary()
        if a.complete:
            note = " (via final ckpt only — torn diag)" if a.complete_via_ckpt_only else ""
            ladder.status(f"{arm.name}: prefit COMPLETE{note} "
                          f"(epochs {sorted(a.rows)}, final ckpt "
                          f"{a.final_ckpt.name if a.final_ckpt else '?'})")
            return "complete"

        # Take over an externally-launched run (orchestrator may have started
        # arm 1 directly): monitor, never double-launch.
        ext = find_prefit_process(out_dir)
        if ext is not None:
            pid, cmd = ext
            ladder.status(f"{arm.name}: prefit ALREADY RUNNING (pid {pid}) — "
                          f"taking over monitoring, not double-launching. cmd: {cmd[:160]}")
            if ladder.dry_run:
                ladder.status(f"{arm.name}: DRY-RUN — would monitor pid {pid} to exit, "
                              f"then re-assess/resume")
                return "dry-run-external-running"
            outcome = _monitor(
                ladder, arm,
                alive_fn=lambda: pid_alive(pid),
                kill_fn=lambda: _kill_tree(pid),
                label=f"external pid {pid}",
            )
            ladder.status(f"{arm.name}: external prefit {outcome}; re-assessing")
            continue  # re-assess: complete? resume?

        if a.smoke_only:
            if ladder.dry_run:
                ladder.status(f"{arm.name}: DRY-RUN — smoke-only artifacts detected "
                              f"(would quarantine and start fresh)")
            else:
                quarantine_smoke(a, ladder)
                a = assess_arm(arm, ladder.root)

        # GPU exclusivity: never launch while ANOTHER arm's prefit is running.
        other = find_any_prefit_process(exclude_dir=out_dir)
        if other is not None:
            pid, cmd = other
            if ladder.dry_run:
                ladder.status(f"{arm.name}: DRY-RUN — foreign prefit on the GPU "
                              f"(pid {pid}); a real run would wait for it")
            else:
                ladder.status(f"{arm.name}: waiting — another hexfield_eq.prefit is on "
                              f"the GPU (pid {pid}: {cmd[:120]})")
                while pid_alive(pid):
                    time.sleep(POLL_SECONDS)
                    ladder.heartbeat(f"waiting for foreign prefit pid {pid} before {arm.name}")
                continue

        if attempts >= MAX_ATTEMPTS:
            ladder.status(f"{arm.name}: prefit FAILED after {attempts} attempts — "
                          f"proceeding (latest ckpt: {a.latest_ckpt})")
            return "failed"
        attempts += 1
        resume = a.latest_ckpt
        cmd = prefit_cmd(arm, ladder.root, resume)
        env = build_arm_env(arm)
        log_path = out_dir / "prefit.log"
        ladder.status(
            f"{arm.name}: LAUNCH prefit attempt {attempts}/{MAX_ATTEMPTS}"
            + (f" (resume from {resume.name})" if resume else " (fresh)")
            + f" [PAIR_BUDGET={env['HEXFIELD_EQ_PAIR_BUDGET']}] -> {log_path}")
        ladder.status(f"{arm.name}: cmd: {' '.join(cmd)}")
        if ladder.dry_run:
            ladder.status(f"{arm.name}: DRY-RUN — not executing; assuming it would complete")
            return "dry-run-would-run"
        pidfile = out_dir / "prefit.pid"
        try:
            with open(log_path, "ab") as log_fh:
                log_fh.write(f"\n===== ladder-runner launch {_utc()} =====\n".encode())
                log_fh.flush()
                proc = subprocess.Popen(
                    cmd, stdout=log_fh, stderr=subprocess.STDOUT,
                    env=env, cwd=str(REPO), start_new_session=True,
                )
            pidfile.write_text(str(proc.pid))
            _monitor(
                ladder, arm,
                alive_fn=lambda: proc.poll() is None,
                kill_fn=lambda: _kill_tree(proc.pid),
                label=f"attempt {attempts}",
            )
            code = proc.wait()
            ladder.status(f"{arm.name}: prefit attempt {attempts} exited code={code}")
        except Exception as exc:  # noqa: BLE001 - defensive by design
            ladder.error(f"prefit:{arm.name}", exc)
        finally:
            try:
                pidfile.unlink(missing_ok=True)
            except OSError:
                pass
        # loop re-assesses (a nonzero exit may still have produced new epochs)


# --------------------------------------------------------------------------- #
# Eval stage (parent side): knobs from the toml + subprocess construction.
# --------------------------------------------------------------------------- #
def resolve_eval_knobs() -> dict:
    raw = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    mc = ((raw.get("model") or {}).get("config") or {})
    sp = mc.get("selfplay") or {}
    ms = mc.get("multi_stage_eval") or {}
    opp = ms.get("opponents") or {}
    return {
        "visits": int(ms.get("full_search_visits") or sp.get("search_visits", 512)),
        "vbs": int(ms.get("eval_virtual_batch_size", 16)),
        "opening_plies": int(ms.get("opening_plies", 8)),
        "opening_temperature": float(ms.get("opening_temperature", 1.0)),
        "sealbot_variant": str(opp.get("sealbot_variant", "current")),
        "sealbot_time_limit": float(opp.get("sealbot_time_limit", 0.05)),
        "strix_ckpt": str(opp.get("strix_ckpt") or DEFAULT_STRIX_CKPT),
        "strix_sims": int(opp.get("strix_sims", 512)),
        "strix_m": int(opp.get("strix_m", 16)),
        "strix_device": str(opp.get("strix_device", "cuda")),
    }


def eval_cmd(arm: Arm, root: Path, ckpt: Path, knobs: dict, games: int) -> list[str]:
    arm_dir = arm_out(arm, root)
    return [
        VENV_PY, "-u", str(Path(__file__).resolve()),
        "--eval-arm", arm.name,
        "--arm-dir", str(arm_dir),
        "--ckpt", str(ckpt),
        "--out", str(arm_dir / "eval_sealbot.json"),
        "--config", str(CONFIG_TOML),
        "--n-games", str(games),
        "--seed-base", str(EVAL_SEED_BASE),
        "--visits", str(knobs["visits"]),
        "--vbs", str(knobs["vbs"]),
        "--opening-plies", str(knobs["opening_plies"]),
        "--opening-temp", str(knobs["opening_temperature"]),
        "--sealbot-variant", knobs["sealbot_variant"],
        "--sealbot-time-limit", str(knobs["sealbot_time_limit"]),
        "--max-wall-seconds", str(EVAL_MAX_WALL),
        "--max-states", str(EVAL_MAX_STATES),
        "--weights", REGIME["weights"],
    ]


def strix_baseline_cmd(arm: Arm, root: Path, ckpt: Path, knobs: dict) -> list[str]:
    arm_dir = arm_out(arm, root)
    return [
        VENV_PY, "-u", str(Path(__file__).resolve()),
        "--strix-baseline", arm.name,
        "--arm-dir", str(arm_dir),
        "--ckpt", str(ckpt),
        "--out", str(arm_dir / "strix_baseline.json"),
        "--config", str(CONFIG_TOML),
        "--n-games", str(STRIX_BASELINE_GAMES),
        "--seed-base", str(EVAL_SEED_BASE),
        "--visits", str(knobs["visits"]),
        "--vbs", str(knobs["vbs"]),
        "--opening-plies", str(knobs["opening_plies"]),
        "--opening-temp", str(knobs["opening_temperature"]),
        "--strix-ckpt", knobs["strix_ckpt"],
        "--strix-sims", str(knobs["strix_sims"]),
        "--strix-m", str(knobs["strix_m"]),
        "--strix-device", knobs["strix_device"],
        "--max-wall-seconds", str(EVAL_MAX_WALL),
        "--max-states", str(EVAL_MAX_STATES),
        "--weights", REGIME["weights"],
    ]


def _eval_summary_fresh(summary: dict, ckpt: Path, games: int) -> bool:
    """An existing eval json is reusable iff it evaluated THIS checkpoint file
    (path + mtime + size) with at least the currently-required game count and
    the same seed base + weights choice."""
    try:
        st = ckpt.stat()
        return (
            summary.get("source_ckpt") == str(ckpt)
            and summary.get("source_ckpt_mtime") == int(st.st_mtime)
            and summary.get("source_ckpt_size") == st.st_size
            and int(summary.get("n_games") or 0) >= games
            and summary.get("seed_base") == EVAL_SEED_BASE
            and summary.get("weights_requested", REGIME["weights"]) == REGIME["weights"]
            and summary.get("error") is None
        )
    except OSError:
        return False


def run_arm_eval(ladder: Ladder, arm: Arm, ckpt: Path, knobs: dict, games: int) -> dict | None:
    """Run (or reuse) the SealBot strength eval for one arm. Returns the
    summary dict or None on failure."""
    arm_dir = arm_out(arm, ladder.root)
    out_json = arm_dir / "eval_sealbot.json"
    if out_json.is_file():
        try:
            summary = json.loads(out_json.read_text(encoding="utf-8"))
        except ValueError:
            summary = None
        if summary and _eval_summary_fresh(summary, ckpt, games):
            ladder.status(f"{arm.name}: eval RESUME — reusing existing eval_sealbot.json "
                          f"(score {summary.get('score')}, se {summary.get('se')})")
            return summary
        ladder.status(f"{arm.name}: existing eval_sealbot.json is stale/invalid — re-running")

    cmd = eval_cmd(arm, ladder.root, ckpt, knobs, games)
    ladder.status(f"{arm.name}: EVAL vs SealBot ({games} games, unpaired, RAW weights, "
                  f"visits {knobs['visits']}, seed_base {EVAL_SEED_BASE})")
    ladder.status(f"{arm.name}: eval cmd: {' '.join(cmd)}")
    if ladder.dry_run:
        mock = arm_dir / "mock_eval.json"
        if mock.is_file():
            summary = json.loads(mock.read_text(encoding="utf-8"))
            ladder.status(f"{arm.name}: DRY-RUN — using mock_eval.json "
                          f"(score {summary.get('score')}, se {summary.get('se')})")
            ladder.gov.charge(ladder.gov.eval_proj(games))
            return summary
        ladder.status(f"{arm.name}: DRY-RUN — no mock_eval.json; treating eval as unavailable")
        ladder.gov.charge(ladder.gov.eval_proj(games))
        return None

    env = build_arm_env(arm)
    log_path = arm_dir / "eval_sealbot.log"
    for attempt in (1, 2):
        t0 = time.time()
        try:
            with open(log_path, "ab") as fh:
                fh.write(f"\n===== ladder-runner eval attempt {attempt} {_utc()} =====\n".encode())
                fh.flush()
                proc = subprocess.Popen(
                    cmd, stdout=fh, stderr=subprocess.STDOUT,
                    env=env, cwd=str(REPO), start_new_session=True,
                )
            try:
                code = proc.wait(timeout=EVAL_MAX_WALL + 1200)
            except subprocess.TimeoutExpired:
                ladder.status(f"{arm.name}: eval hard-timeout — killing")
                _kill_tree(proc.pid)
                code = -9
            if code == 0 and out_json.is_file():
                summary = json.loads(out_json.read_text(encoding="utf-8"))
                if summary.get("error") is None:
                    ladder.gov.note("eval_per_game", (time.time() - t0) / max(games, 1))
                    ladder.status(
                        f"{arm.name}: eval DONE — score {summary.get('score')} "
                        f"se {summary.get('se')} decided {summary.get('decided')}"
                        f"/{summary.get('n_games')}")
                    return summary
                ladder.error(f"eval:{arm.name}", summary["error"])
            else:
                ladder.error(f"eval:{arm.name}", f"exit code {code} (see {log_path})")
        except Exception as exc:  # noqa: BLE001
            ladder.error(f"eval:{arm.name}", exc)
        if attempt == 1:
            if ladder.gov.remaining() < FINAL_RESERVE_SECONDS + ladder.gov.eval_proj(games):
                ladder.degrade(f"{arm.name}: no time for an eval retry — skipping")
                break
            ladder.status(f"{arm.name}: retrying eval once")
    return None


# --------------------------------------------------------------------------- #
# Decision logic (pure function over plain dicts — exercised by --dry-run).
# --------------------------------------------------------------------------- #
def _se_diff(se_a, se_b) -> float:
    parts = [float(s) ** 2 for s in (se_a, se_b) if _is_finite(s)]
    return math.sqrt(sum(parts)) if parts else float("inf")


def decide(infos: dict[str, dict]) -> dict:
    """Owner decision rules over per-arm ``{catastrophic, catastrophic_reason,
    score, se, decided}``. Returns winner + full ranking + reasoning.

    * fullest-stack pick: arm4, replaced by arm4c ONLY if 4c beats 4 by
      > HEADTOHEAD_SE_MULT * SE_of_difference on their SealBot scores (or if
      arm4 is catastrophic/unscored and 4c is usable).
    * preference order: [4-or-4c, arm3, arm2, arm1]; select the FIRST arm not
      unambiguously negative: score >= best_score - NEG_SE_MULT * SE_diff and
      not catastrophic. Unscored arms are skipped in the primary walk.
    * fallbacks (proceed-with-best doctrine): if no scored arm survives the
      walk -> first non-catastrophic arm in order; if ALL arms are
      catastrophic -> least-bad arm that still has a checkpoint (loud
      warning); if NO arm has a checkpoint -> no winner (the only hard stop).
    """
    d: dict = {"rules": {
        "ranking_opponent": "sealbot (owner update 2026-07-08; Strix too strong to separate prefit arms)",
        "ranking_weights": f"{REGIME['weights']} (owner update 2026-07-09: EMA lags at ~2-4k steps)",
        "unambiguously_negative": f"score < best - {NEG_SE_MULT}*SE_diff OR catastrophic health",
        "head_to_head_4c": f"4c replaces 4 iff 4c - 4 > {HEADTOHEAD_SE_MULT}*SE_diff",
        "catastrophic": f"NaN death / no checkpoint / value_ece_ema > {CATASTROPHIC_ECE}",
    }, "warnings": []}

    def usable(n):  # non-catastrophic
        return not infos[n]["catastrophic"]

    def scored(n):
        return usable(n) and _is_finite(infos[n].get("score"))

    # ---- fullest-stack pick: arm4 vs arm4c -------------------------------
    # Custom arm sets (EQ_LADDER_ARMS) have no 4/4c head-to-head: the
    # preference order is EVAL_PRIORITY verbatim.
    if CUSTOM_ARMS:
        d["head_to_head_4_vs_4c"] = {
            "picked": None,
            "reason": "custom arm set (EQ_LADDER_ARMS) — no 4/4c head-to-head",
        }
        order = [n for n in EVAL_PRIORITY if n in infos]
        d["preference_order"] = order
        return _decide_walk(d, infos, order, usable, scored)
    fullest, hh = None, {}
    u4, u4c = usable("arm4_raylayout"), usable("arm4c_georay")
    if u4 and u4c and scored("arm4_raylayout") and scored("arm4c_georay"):
        s4, s4c = infos["arm4_raylayout"]["score"], infos["arm4c_georay"]["score"]
        sed = _se_diff(infos["arm4_raylayout"].get("se"), infos["arm4c_georay"].get("se"))
        margin = s4c - s4
        pick4c = margin > HEADTOHEAD_SE_MULT * sed
        fullest = "arm4c_georay" if pick4c else "arm4_raylayout"
        hh = {"score_4": s4, "score_4c": s4c, "margin_4c_minus_4": round(margin, 4),
              "se_diff": round(sed, 4), "threshold": round(HEADTOHEAD_SE_MULT * sed, 4),
              "picked": fullest,
              "reason": (f"4c beats 4 by {margin:.4f} > {HEADTOHEAD_SE_MULT}*SE_diff"
                         f"={HEADTOHEAD_SE_MULT * sed:.4f} -> 4c (soak RAY_BLOCKERS=0)"
                         if pick4c else
                         f"4c - 4 = {margin:.4f} <= {HEADTOHEAD_SE_MULT}*SE_diff"
                         f"={HEADTOHEAD_SE_MULT * sed:.4f} -> keep arm4")}
    elif u4 and not u4c:
        fullest, hh = "arm4_raylayout", {"picked": "arm4_raylayout",
                                         "reason": "only arm4 usable among the L arms"}
    elif u4c and not u4:
        fullest, hh = "arm4c_georay", {"picked": "arm4c_georay",
                                       "reason": "arm4 catastrophic; 4c stands in as the fullest stack"}
    elif u4 and u4c:  # both usable but at most one scored -> no head-to-head
        if scored("arm4c_georay") and not scored("arm4_raylayout"):
            fullest, hh = "arm4c_georay", {"picked": "arm4c_georay",
                                           "reason": "only 4c has a strength score; head-to-head unavailable"}
        else:
            fullest, hh = "arm4_raylayout", {"picked": "arm4_raylayout",
                                             "reason": "head-to-head unavailable (4c unscored or both unscored) -> default arm4"}
    else:
        hh = {"picked": None, "reason": "both L arms catastrophic — order starts at arm3"}
    d["head_to_head_4_vs_4c"] = hh

    order = [n for n in (fullest, "arm3_tokread", "arm2_reglane", "arm1_vanilla") if n]
    d["preference_order"] = order
    return _decide_walk(d, infos, order, usable, scored)


def _decide_walk(d: dict, infos: dict, order: list, usable, scored) -> dict:
    """The arm-set-independent tail of decide(): best-score, per-arm verdicts,
    the preference-order walk, and the proceed-with-best fallbacks."""

    # ---- best score over all usable, scored arms (controls included) ------
    scored_arms = [n for n in infos if scored(n)]
    best = max(scored_arms, key=lambda n: infos[n]["score"]) if scored_arms else None
    d["best_arm"] = best
    d["best_score"] = infos[best]["score"] if best else None

    # ---- per-arm verdicts + the walk --------------------------------------
    verdicts: dict[str, dict] = {}
    for n, info in infos.items():
        v: dict = {"score": info.get("score"), "se": info.get("se"),
                   "decided": info.get("decided")}
        if info["catastrophic"]:
            v["verdict"] = f"catastrophic ({info.get('catastrophic_reason')})"
        elif not _is_finite(info.get("score")):
            v["verdict"] = "no strength score (eval failed/unavailable/deadline-skipped)"
        elif best is None or n == best:
            v["verdict"] = "best arm" if n == best else "scored"
        else:
            sed = _se_diff(info.get("se"), infos[best].get("se"))
            gap = infos[best]["score"] - info["score"]
            v["gap_to_best"] = round(gap, 4)
            v["neg_threshold"] = round(NEG_SE_MULT * sed, 4)
            v["verdict"] = ("UNAMBIGUOUSLY NEGATIVE" if gap > NEG_SE_MULT * sed
                            else "within band of best")
        if n not in order and not info["catastrophic"]:
            v["note"] = "not in preference order (4/4c head-to-head control)"
        verdicts[n] = v
    d["verdicts"] = verdicts

    winner, reason = None, None
    for n in order:
        if not usable(n):
            continue
        if not scored(n):
            continue  # cannot verify strength in the primary walk
        if verdicts[n]["verdict"] == "UNAMBIGUOUSLY NEGATIVE":
            continue
        winner = n
        reason = (f"first arm in preference order {order} that is not "
                  f"unambiguously negative (score {infos[n]['score']} vs best "
                  f"{d['best_score']} [{best}])")
        break
    if winner is None:
        for n in order:  # fallback 1: no scored survivor -> first usable arm
            if usable(n):
                winner = n
                reason = ("FALLBACK: no arm in the preference order had a usable "
                          "strength score; picked the first non-catastrophic arm "
                          "by prefit health (proceed-with-best doctrine)")
                d["warnings"].append("winner selected WITHOUT a strength score")
                break
    if winner is None:
        with_ckpt = [n for n, i in infos.items() if i.get("has_checkpoint")]
        if with_ckpt:  # fallback 2: everything catastrophic but ckpts exist
            ranked = sorted(
                with_ckpt,
                key=lambda n: (
                    -(infos[n]["score"] if _is_finite(infos[n].get("score")) else -1.0),
                    infos[n].get("gate_value_ece") if _is_finite(infos[n].get("gate_value_ece")) else 9.9,
                ),
            )
            winner = ranked[0]
            reason = ("FALLBACK: ALL ARMS CATASTROPHIC — picked the least-bad arm "
                      "that still has a loadable checkpoint. REVIEW BEFORE TRUSTING "
                      "THE SOAK.")
            d["warnings"].append("ALL ARMS UNHEALTHY — soak launched from a catastrophic arm")
    d["winner"] = winner
    d["winner_reason"] = reason if winner else (
        "NO ARM PRODUCED A LOADABLE CHECKPOINT — hard stop (the only permitted one)")
    d["soak_ray_blockers_0"] = (winner == "arm4c_georay")
    return d


def _rank_table(infos: dict[str, dict], decision: dict) -> str:
    lines = [
        "| arm | score vs SealBot (raw wts) | SE | decided | health | verdict |",
        "|---|---|---|---|---|---|",
    ]
    def _key(n):
        s = infos[n].get("score")
        return -(s if _is_finite(s) else -1.0)
    for n in sorted(infos, key=_key):
        i, v = infos[n], decision["verdicts"][n]
        score = f"{i['score']:.4f}" if _is_finite(i.get("score")) else "—"
        se = f"{i['se']:.4f}" if _is_finite(i.get("se")) else "—"
        health = "CATASTROPHIC: " + str(i.get("catastrophic_reason")) if i["catastrophic"] else "ok"
        mark = " **<- WINNER**" if n == decision.get("winner") else ""
        lines.append(f"| {n}{mark} | {score} | {se} | {i.get('decided') or '—'} "
                     f"| {health} | {v['verdict']} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Soak launch.
# --------------------------------------------------------------------------- #
def write_launch_toml(init_ckpt: Path, out_path: Path) -> Path:
    """Run-ready copy of the run toml with [checkpoint] initialize_from pointed
    at the soak-init checkpoint (existing initialize_from/resume_from lines in
    that section are commented out)."""
    lines = CONFIG_TOML.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    in_ckpt = inserted = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            in_ckpt = stripped == "[checkpoint]"
            out.append(line)
            if in_ckpt:
                out.append(f'initialize_from = "{init_ckpt}"  # <- prefit-ladder winner (ladder runner)')
                inserted = True
            continue
        if in_ckpt and re.match(r"\s*(initialize_from|resume_from)\s*=", line):
            out.append(f"# (ladder runner superseded) {stripped}")
            continue
        out.append(line)
    if not inserted:
        out += ["", "[checkpoint]", f'initialize_from = "{init_ckpt}"']
    _atomic_write_text(out_path, "\n".join(out) + "\n")
    return out_path


def repackage_cmd(arm: Arm, root: Path, ckpt: Path) -> list[str]:
    arm_dir = arm_out(arm, root)
    return [
        VENV_PY, "-u", str(Path(__file__).resolve()),
        "--repackage", arm.name,
        "--arm-dir", str(arm_dir),
        "--ckpt", str(ckpt),
        "--out", str(arm_dir / SOAK_INIT_NAME),
        "--weights", REGIME["weights"],
    ]


def build_soak_env(arm: Arm, launch_toml: Path) -> dict[str, str]:
    """Winner arm env (arch MUST match the soak-init meta) + the supervisor's
    operational env (mirrors the systemd unit, which is NOT installed)."""
    env = build_arm_env(arm)
    env["ROOT"] = str(REPO)
    env["VENV"] = str(Path(VENV_PY).parent.parent)
    env["CONFIG"] = str(launch_toml)
    env["RUNDIR"] = str(SOAK_RUNDIR)
    env["SEALBOT_PATH"] = SEALBOT_PATH
    env.setdefault("HEXFIELD_ANCHOR_ROOTS", str(REPO))
    # Malloc tunables from the systemd unit (R2 serve-speedup bundle).
    env.setdefault("MALLOC_TRIM_THRESHOLD_", "536870912")
    env.setdefault("MALLOC_MMAP_THRESHOLD_", "536870912")
    env.setdefault("MALLOC_TOP_PAD_", "134217728")
    # The training-side PAIR_BUDGET regime knob is prefit-only; don't leak it
    # into the soak (the trainer has its own defaults).
    env.pop("HEXFIELD_EQ_PAIR_BUDGET", None)
    return env


def launch_soak(ladder: Ladder, winner: Arm, decision: dict, knobs: dict) -> None:
    arm_dir = arm_out(winner, ladder.root)
    soak_init = arm_dir / SOAK_INIT_NAME

    # 1) soak-init checkpoint (RAW weights under the deadline regime). The eval
    # subprocess already builds it; rebuild via the repackage mode if missing.
    a = assess_arm(winner, ladder.root)
    src_ckpt = a.final_ckpt or a.latest_ckpt
    if src_ckpt is None:
        ladder.error("soak", "winner has no checkpoint — cannot build soak init")
        return
    rcmd = repackage_cmd(winner, ladder.root, src_ckpt)
    if not soak_init.is_file():
        ladder.status(f"soak: building soak-init checkpoint: {' '.join(rcmd)}")
        if not ladder.dry_run:
            r = subprocess.run(rcmd, env=build_arm_env(winner), cwd=str(REPO),
                               capture_output=True, text=True, timeout=1200)
            if r.returncode != 0 or not soak_init.is_file():
                ladder.error("soak", f"repackage failed (code {r.returncode}): "
                                     f"{(r.stdout + r.stderr)[-500:]}")
                return
            for line in (r.stdout or "").splitlines()[-8:]:
                ladder.status(f"soak: repackage| {line}")
    else:
        ladder.status(f"soak: soak-init checkpoint present: {soak_init}")

    # 2) run-ready toml.
    launch_toml = ladder.root / "hexfield_eq_main_1.launch.toml"
    if ladder.dry_run:
        ladder.status(f"soak: DRY-RUN — would write {launch_toml} with "
                      f'initialize_from = "{soak_init}"')
    else:
        write_launch_toml(soak_init, launch_toml)
        ladder.status(f"soak: wrote run-ready toml {launch_toml} "
                      f"(initialize_from -> {soak_init})")

    # 3) detached supervisor launch with the WINNER arm's env sourced.
    lock = SOAK_RUNDIR / "supervisor.lock"
    if lock.is_file():
        try:
            other = int(lock.read_text().strip())
        except (ValueError, OSError):
            other = -1
        if other > 0 and pid_alive(other):
            ladder.status(f"soak: supervisor ALREADY RUNNING (pid {other}) — not relaunching")
            ladder.state["soak"] = {"status": "already_running", "pid": other}
            ladder.save_state()
            return
    env = build_soak_env(winner, launch_toml)
    nohup_log = SOAK_RUNDIR / "supervisor_nohup.log"
    arch_echo = {k: env.get(k) for k in (
        "HEXFIELD_EQ_TRUNK", "HEXFIELD_EQ_REG_LANE", "HEXFIELD_EQ_REG_TOK_READ",
        "HEXFIELD_EQ_RAY_BLOCKERS", "HEXFIELD_EQ_SUPPORT_RADIUS",
        "HEXFIELD_EQ_CHANNELS", "HEXFIELD_EQ_GROUP_ORDER", "HEXFIELD_EQ_C_ORBIT")}
    ladder.status(f"soak: launching supervisor detached — env file {winner.env_file.name}, "
                  f"arch {arch_echo}")
    ladder.status(
        "soak: equivalent shell command: "
        f"set -a; source {winner.env_file}; set +a; "
        f"CONFIG={launch_toml} ROOT={REPO} RUNDIR={SOAK_RUNDIR} "
        f"nohup setsid bash {SUPERVISOR_SH} >> {nohup_log} 2>&1 &")
    if ladder.dry_run:
        ladder.status("soak: DRY-RUN — supervisor NOT launched")
        ladder.state["soak"] = {"status": "dry_run", "launch_toml": str(launch_toml),
                                "soak_init": str(soak_init), "arch_env": arch_echo}
        ladder.save_state()
        return

    SOAK_RUNDIR.mkdir(parents=True, exist_ok=True)
    with open(nohup_log, "ab") as fh:
        fh.write(f"\n===== ladder-runner soak launch {_utc()} =====\n".encode())
        fh.flush()
        proc = subprocess.Popen(
            ["bash", str(SUPERVISOR_SH)], stdout=fh, stderr=subprocess.STDOUT,
            env=env, cwd=str(REPO), start_new_session=True,
        )
    ladder.state["soak"] = {
        "status": "launched", "pid": proc.pid, "launch_toml": str(launch_toml),
        "soak_init": str(soak_init), "supervisor_log": str(SOAK_RUNDIR / "supervisor.log"),
        "nohup_log": str(nohup_log), "arch_env": arch_echo, "launched_utc": _utc(),
    }
    ladder.status(f"soak: supervisor launched pid={proc.pid}; verifying in "
                  f"{int(SOAK_VERIFY_SECONDS)}s")
    time.sleep(SOAK_VERIFY_SECONDS)

    # 4) verify: process alive + sane first log lines.
    alive = proc.poll() is None
    checks: dict = {"supervisor_alive": alive}
    sup_log = SOAK_RUNDIR / "supervisor.log"
    tail = ""
    if sup_log.is_file():
        tail = "\n".join(sup_log.read_text(encoding="utf-8", errors="replace")
                         .splitlines()[-15:])
        checks["supervisor_log_tail"] = tail
        checks["saw_launch_line"] = "LAUNCH out=" in tail
    train_logs = sorted(SOAK_RUNDIR.glob("train.*.out.log"))
    if train_logs:
        tl = train_logs[-1]
        t_tail = tl.read_text(encoding="utf-8", errors="replace").splitlines()
        checks["train_log"] = str(tl)
        checks["train_log_tail"] = "\n".join(t_tail[-10:])
        bad = [l for l in t_tail[-60:] if "Traceback" in l or "CUDA error" in l]
        checks["train_log_errors"] = bad
    driver_pid_file = SOAK_RUNDIR / "driver.pid"
    if driver_pid_file.is_file():
        try:
            dpid = int(driver_pid_file.read_text().strip())
            checks["trainer_pid"] = dpid
            checks["trainer_alive"] = pid_alive(dpid)
        except (ValueError, OSError):
            pass
    ok = alive and checks.get("saw_launch_line", False) and not checks.get("train_log_errors")
    ladder.state["soak"]["verify"] = _json_safe(checks)
    ladder.state["soak"]["status"] = "verified" if ok else "launched_with_warnings"
    ladder.status(
        f"soak: verification {'OK' if ok else 'WITH WARNINGS'} — supervisor pid "
        f"{proc.pid} alive={alive}, supervisor.log={sup_log}, "
        f"train_log={checks.get('train_log', 'not yet created')}")
    if not ok:
        ladder.status(f"soak: WARNING details: launch_line={checks.get('saw_launch_line')}, "
                      f"errors={checks.get('train_log_errors')}; tail:\n{tail}")
    ladder.save_state()


# --------------------------------------------------------------------------- #
# The full state machine.
# --------------------------------------------------------------------------- #
def run_ladder(ladder: Ladder) -> int:
    gov = ladder.gov
    ladder.status(f"LADDER RUNNER start (pid {os.getpid()}, mode={ladder.state['mode']}) — "
                  f"root {ladder.root}, data {DATA_DIR}, config {CONFIG_TOML}, "
                  f"{gov.fmt_remaining()}")
    ladder.status(f"REGIME: {REGIME}")
    ladder.status(f"arms: {', '.join(f'{a.name}({a.epochs}ep)' for a in ARMS)}; "
                  f"eval: {EVAL_GAMES} games vs SealBot per arm (unpaired, binomial SE, "
                  f"RAW weights), seed_base {EVAL_SEED_BASE}; strix baseline "
                  f"{STRIX_BASELINE_GAMES} games (winner only, record-only)")
    if REGIME["limit_steps"] <= 0:
        ladder.status("WARNING: EQ_LADDER_LIMIT_STEPS is unset (no per-epoch step cap) — "
                      "the orchestrator was to supply the calibrated cap; arms may "
                      "overrun their time slots (the deadline governor will degrade)")

    # ---- sanity: corpus + env files (record, don't die needlessly) ----------
    if not (DATA_DIR / "train").is_dir() or not (DATA_DIR / "val").is_dir():
        ladder.error("init", f"corpus missing train/ or val/ under {DATA_DIR}")
        if not ladder.dry_run:
            ladder.set_stage("fatal")
            return 2
    for arm in ARMS:
        if not arm.env_file.is_file():
            ladder.error("init", f"missing env file {arm.env_file}")

    try:
        knobs = resolve_eval_knobs()
        ladder.status(f"eval knobs from toml: {knobs}")
    except Exception as exc:  # noqa: BLE001
        ladder.error("init:eval-knobs", exc)
        knobs = {"visits": 512, "vbs": 32, "opening_plies": 8, "opening_temperature": 1.0,
                 "sealbot_variant": "current", "sealbot_time_limit": 0.05,
                 "strix_ckpt": DEFAULT_STRIX_CKPT, "strix_sims": 512, "strix_m": 16,
                 "strix_device": "cuda"}

    # ---- stage 1: prefit each arm, sequentially, deadline-gated -------------
    for arm in ARMS:
        ladder.set_stage(f"prefit:{arm.name}")
        pre = assess_arm(arm, ladder.root)
        proj = gov.project("prefit", PRIOR_PREFIT_SECONDS)
        skipped = False
        if not pre.complete:
            any_ckpt = any(assess_arm(x, ladder.root).ckpts for x in ARMS)
            if arm.name == "arm4c_georay":
                # OWNER: 4c is conditional — run only with its projected
                # duration + >= 50 min of post-arm-4 headroom before the deadline.
                need = proj + ARM4C_RESERVE_SECONDS
                if gov.remaining() < need:
                    ladder.degrade(
                        f"SKIP arm4c_georay — remaining {gov.remaining()/60:.0f}m < "
                        f"projected {proj/60:.0f}m + {ARM4C_RESERVE_SECONDS/60:.0f}m reserve; "
                        f"blockers-on (arm 4) is the default ray mode")
                    ladder.state["arms"][arm.name]["prefit_outcome"] = "skipped_deadline_conditional"
                    skipped = True
            elif gov.remaining() < proj + FINAL_RESERVE_SECONDS and any_ckpt:
                ladder.degrade(
                    f"SKIP prefit {arm.name} — remaining {gov.remaining()/60:.0f}m < "
                    f"projected {proj/60:.0f}m + final reserve "
                    f"{FINAL_RESERVE_SECONDS/60:.0f}m (last-resort: decide from "
                    f"completed arms)")
                ladder.state["arms"][arm.name]["prefit_outcome"] = "skipped_deadline"
                skipped = True
        if not skipped:
            t0 = gov.now()
            try:
                outcome = ensure_arm_prefit(ladder, arm)
                ladder.state["arms"][arm.name]["prefit_outcome"] = outcome
                if outcome == "dry-run-would-run":
                    gov.charge(proj)
                actual = gov.now() - t0
                if not pre.complete:  # only time actual work, not skip-completes
                    if not ladder.dry_run and actual > 60:
                        gov.note("prefit", actual)
                    ladder.timeline(f"prefit:{arm.name}", proj, actual)
            except Exception as exc:  # noqa: BLE001
                ladder.error(f"prefit:{arm.name}", exc)
                ladder.state["arms"][arm.name]["prefit_outcome"] = "error"
        # health extraction (always, even on failure/skip — partial arms count)
        try:
            a = assess_arm(arm, ladder.root)
            health = extract_health(a)
            ladder.state["arms"][arm.name]["prefit"] = a.summary()
            ladder.state["arms"][arm.name]["health"] = health
            warn = f"; warnings: {health['warnings']}" if health.get("warnings") else ""
            ladder.status(
                f"{arm.name}: health — catastrophic={health['catastrophic']}"
                f" ({health.get('catastrophic_reason')}) value_ece={health.get('gate_value_ece')}"
                f" top1={health.get('top1')} (ema_top1={health.get('ema_top1')}, "
                f"recorded only){warn}")
        except Exception as exc:  # noqa: BLE001
            ladder.error(f"health:{arm.name}", exc)
            ladder.state["arms"][arm.name]["health"] = {
                "catastrophic": True, "catastrophic_reason": f"health extraction failed: {exc}"}

    # ---- stage 2: strength eval vs SealBot (deadline-planned, priority order) -
    eligible: list[Arm] = []
    for name in EVAL_PRIORITY:
        arm = ARM_BY_NAME[name]
        st = ladder.state["arms"][arm.name]
        health = st.get("health") or {"catastrophic": True, "catastrophic_reason": "no health"}
        if health.get("catastrophic"):
            ladder.status(f"{arm.name}: SKIP eval — catastrophic health "
                          f"({health.get('catastrophic_reason')})")
            continue
        if (assess_arm(arm, ladder.root).final_ckpt
                or assess_arm(arm, ladder.root).latest_ckpt) is None:
            ladder.status(f"{arm.name}: SKIP eval — no checkpoint")
            continue
        eligible.append(arm)

    # Degradation ladder for the eval stage (owner order): full 60-game plan ->
    # 40-game matches -> drop the Strix baseline -> partial evals as they fit.
    games, strix_planned = EVAL_GAMES, True
    if gov.remaining() != math.inf:
        for level, g, s in ((0, EVAL_GAMES, True), (1, DEGRADED_EVAL_GAMES, True),
                            (2, DEGRADED_EVAL_GAMES, False)):
            total = len(eligible) * gov.eval_proj(g) \
                + (PRIOR_STRIX_SECONDS if s else 0) + FINAL_RESERVE_SECONDS
            games, strix_planned = g, s
            if total <= gov.remaining():
                if level:
                    ladder.degrade(f"eval stage plan at level {level}: {g} games/arm, "
                                   f"strix baseline={'on' if s else 'OFF'} "
                                   f"(projected {total/60:.0f}m fits {gov.fmt_remaining()})")
                break
        else:
            ladder.degrade(f"eval stage plan at level 3: {games} games/arm, strix "
                           f"baseline OFF, PARTIAL evals only (per-arm gate)")

    for arm in eligible:
        st = ladder.state["arms"][arm.name]
        a = assess_arm(arm, ladder.root)
        ckpt = a.final_ckpt or a.latest_ckpt
        # Reuse a fresh existing eval before spending time on the gate.
        out_json = arm_out(arm, ladder.root) / "eval_sealbot.json"
        reusable = False
        if out_json.is_file():
            try:
                reusable = _eval_summary_fresh(
                    json.loads(out_json.read_text(encoding="utf-8")), ckpt, games)
            except ValueError:
                reusable = False
        if not reusable and gov.remaining() < gov.eval_proj(games) + FINAL_RESERVE_SECONDS:
            ladder.degrade(f"SKIP eval {arm.name} (and any remaining) — projected "
                           f"{gov.eval_proj(games)/60:.0f}m does not fit before the "
                           f"final reserve; deciding from evals done so far")
            break
        ladder.set_stage(f"eval:{arm.name}")
        t0 = gov.now()
        try:
            st["eval"] = run_arm_eval(ladder, arm, ckpt, knobs, games)
        except Exception as exc:  # noqa: BLE001
            ladder.error(f"eval:{arm.name}", exc)
            st["eval"] = None
        ladder.timeline(f"eval:{arm.name}", gov.eval_proj(games), gov.now() - t0)
        ladder.save_state()

    # ---- stage 3: decision ---------------------------------------------------
    ladder.set_stage("decision")
    infos: dict[str, dict] = {}
    for arm in ARMS:
        st = ladder.state["arms"][arm.name]
        health = st.get("health") or {}
        ev = st.get("eval") or {}
        a = assess_arm(arm, ladder.root)
        infos[arm.name] = {
            "catastrophic": bool(health.get("catastrophic", True)),
            "catastrophic_reason": health.get("catastrophic_reason"),
            "score": ev.get("score"),
            "se": ev.get("se"),
            "decided": ev.get("decided"),
            "gate_value_ece": health.get("gate_value_ece"),
            "has_checkpoint": bool(a.ckpts),
        }
    decision = decide(infos)
    ladder.state["decision"] = decision
    ladder.status("DECISION inputs: " + json.dumps(_json_safe(infos)))
    ladder.status(f"DECISION: head-to-head 4 vs 4c: {decision['head_to_head_4_vs_4c'].get('reason')}")
    ladder.status(f"DECISION: preference order: {decision['preference_order']}; "
                  f"best arm: {decision['best_arm']} (score {decision['best_score']})")
    for n, v in decision["verdicts"].items():
        ladder.status(f"DECISION: {n}: {v['verdict']}"
                      + (f" (gap {v.get('gap_to_best')} vs threshold {v.get('neg_threshold')})"
                         if "gap_to_best" in v else ""))
    ladder.status(f"DECISION: WINNER = {decision['winner']} — {decision['winner_reason']}")
    for w in decision["warnings"]:
        ladder.status(f"DECISION WARNING: {w}")
    try:
        with open(ladder.status_md, "a", encoding="utf-8") as fh:
            fh.write("\n## Final ranking (SealBot strength eval, RAW weights)\n\n"
                     + _rank_table(infos, decision) + "\n\n")
    except OSError:
        pass
    ladder.save_state()

    if decision["winner"] is None:
        ladder.set_stage("fatal")
        ladder.status("HARD STOP: no arm produced a loadable checkpoint. Nothing to soak from.")
        return 3
    winner = ARM_BY_NAME[decision["winner"]]

    # ---- stage 3b: OPTIONAL record-only Strix baseline for the winner --------
    strix_proj = gov.project("strix_baseline", PRIOR_STRIX_SECONDS)
    if not strix_planned:
        ladder.status("strix baseline: SKIPPED by the deadline degradation plan "
                      "(record-only, no decision weight)")
    elif gov.remaining() < strix_proj + FINAL_RESERVE_SECONDS:
        ladder.degrade(f"SKIP strix baseline — projected {strix_proj/60:.0f}m does "
                       f"not fit before the final reserve")
    else:
        ladder.set_stage("strix-baseline")
        try:
            a = assess_arm(winner, ladder.root)
            ckpt = a.final_ckpt or a.latest_ckpt
            scmd = strix_baseline_cmd(winner, ladder.root, ckpt, knobs)
            ladder.status(f"strix baseline (record-only, no decision weight): {' '.join(scmd)}")
            if ladder.dry_run:
                ladder.status("strix baseline: DRY-RUN — not executed")
                gov.charge(strix_proj)
            else:
                t0 = time.time()
                log_path = arm_out(winner, ladder.root) / "strix_baseline.log"
                with open(log_path, "ab") as fh:
                    fh.write(f"\n===== strix baseline {_utc()} =====\n".encode())
                    fh.flush()
                    proc = subprocess.Popen(scmd, stdout=fh, stderr=subprocess.STDOUT,
                                            env=build_arm_env(winner), cwd=str(REPO),
                                            start_new_session=True)
                try:
                    code = proc.wait(timeout=min(EVAL_MAX_WALL,
                                                 max(300.0, gov.remaining() - FINAL_RESERVE_SECONDS)))
                except subprocess.TimeoutExpired:
                    _kill_tree(proc.pid)
                    code = -9
                gov.note("strix_baseline", time.time() - t0)
                out_json = arm_out(winner, ladder.root) / "strix_baseline.json"
                if code == 0 and out_json.is_file():
                    sb = json.loads(out_json.read_text(encoding="utf-8"))
                    ladder.state["arms"][winner.name]["strix_baseline"] = sb
                    ladder.status(
                        f"STRIX BASELINE ({winner.name}, record-only): score "
                        f"{sb.get('score')} se {sb.get('se')} decided {sb.get('decided')}"
                        f"/{sb.get('n_games')} (pentanomial {sb.get('pentanomial')})")
                else:
                    ladder.status(f"strix baseline skipped (exit {code}) — record-only, not blocking")
        except Exception as exc:  # noqa: BLE001
            ladder.error("strix-baseline (record-only, non-blocking)", exc)

    # ---- stage 4: soak launch — ALWAYS happens (owner instruction), UNLESS
    # this is a read-only ladder (EQ_LADDER_NO_SOAK=1, the ray-tap wave mode:
    # the live soak keeps running; the winner feeds the write-up).
    ladder.set_stage("soak")
    if NO_SOAK:
        ladder.status("soak: SKIPPED — EQ_LADDER_NO_SOAK=1 (read-only ladder; "
                      "winner recorded for the write-up, no supervisor launch)")
        ladder.state["soak"] = {"status": "skipped_no_soak"}
        ladder.save_state()
    else:
        if decision.get("soak_ray_blockers_0"):
            ladder.status("soak: winner is arm4c — RAY_BLOCKERS=0 rides in via its env file")
        try:
            launch_soak(ladder, winner, decision, knobs)
        except Exception as exc:  # noqa: BLE001
            ladder.error("soak", exc)
    ladder.set_stage("done")
    ladder.status("LADDER RUNNER done.")
    return 0


# --------------------------------------------------------------------------- #
# Subprocess modes (torch / hexfield_eq imported HERE only).
# --------------------------------------------------------------------------- #
def _ensure_eq_paths() -> None:
    for pkg in PKG_ROOTS:
        root = REPO / "packages" / pkg / "python"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))


def _load_run_config(config_path: str):
    _ensure_eq_paths()
    from hexfield_eq.config import parse_hexfield_config
    raw = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    return parse_hexfield_config(((raw.get("model") or {}).get("config")) or {})


def build_soak_init(prefit_ckpt: Path, out_path: Path, weights: str = "raw") -> dict:
    """Repackage a prefit checkpoint into the warm-start shape: {"meta": <arch
    meta>, "model": <state dict>}. ``weights='raw'`` (deadline default) takes
    the "model" key — with ~2-4k optimizer steps the EMA lags most of the run;
    ``weights='ema'`` prefers "ema_model". This exact file is BOTH the eval
    candidate and the soak initialize_from payload (loader:
    hexfield_eq.checkpoints.HexfieldCheckpointLoader meta-shape warm start).
    Also cross-checks arch meta vs this process's HEXFIELD_EQ_* env."""
    import torch

    payload = torch.load(prefit_ckpt, map_location="cpu", weights_only=False)
    if weights == "ema" and "ema_model" in payload:
        weights_key = "ema_model"
    else:
        weights_key = "model"
    sd = payload[weights_key]
    meta = dict(payload.get("meta") or {})
    meta.setdefault("lineage", "hexfield_eq")
    info = {
        "source_checkpoint": str(prefit_ckpt),
        "source_ckpt_mtime": int(prefit_ckpt.stat().st_mtime),
        "source_ckpt_size": prefit_ckpt.stat().st_size,
        "weights": weights_key,
        "weights_requested": weights,
        "built_utc": _utc(),
    }
    meta["ladder_soak_init"] = info
    mismatches = []
    env_map = {
        "trunk_layout": "HEXFIELD_EQ_TRUNK",
        "reg_lane": "HEXFIELD_EQ_REG_LANE",
        "reg_tok_read": "HEXFIELD_EQ_REG_TOK_READ",
        "ray_blockers": "HEXFIELD_EQ_RAY_BLOCKERS",
        "support_radius": "HEXFIELD_EQ_SUPPORT_RADIUS",
        "channels": "HEXFIELD_EQ_CHANNELS",
        "group_order": "HEXFIELD_EQ_GROUP_ORDER",
        "c_orbit": "HEXFIELD_EQ_C_ORBIT",
        "attention_heads": "HEXFIELD_EQ_ATTENTION_HEADS",
    }
    for mk, ek in env_map.items():
        got_env = os.environ.get(ek)
        want = meta.get(mk)
        if got_env is None or want is None:
            continue
        env_val = got_env if mk == "trunk_layout" else str(int(float(got_env)))
        want_val = str(want) if mk == "trunk_layout" else str(int(want))
        if env_val != want_val:
            mismatches.append(f"{mk}: meta={want!r} env {ek}={got_env!r}")
    if mismatches:
        print(f"[WARN] soak-init arch meta vs env MISMATCH: {mismatches} — the "
              "warm start will silently drop mismatched shapes; fix the env "
              "before trusting the soak", flush=True)
    info["env_mismatches"] = mismatches
    tmp = out_path.with_name(out_path.name + ".tmp")
    torch.save({"meta": meta, "model": sd}, tmp)
    os.replace(tmp, out_path)
    _atomic_write_text(out_path.with_suffix(".meta.json"),
                       json.dumps(_json_safe({k: v for k, v in meta.items()
                                              if k != "train_state"}), indent=2))
    print(f"soak-init written: {out_path} (weights={weights_key}, "
          f"meta keys={sorted(meta)})", flush=True)
    return info


def _binomial_summary(result: dict) -> dict:
    sc = result.get("score") or {}
    decided = int(sc.get("decided") or 0)
    a_wins = int(sc.get("a_wins") or 0)
    if decided > 0:
        p = a_wins / decided
        se = math.sqrt(p * (1.0 - p) / decided)
        ci = sc.get("a_winrate_ci95") or [None, None]
        if se == 0.0 and ci[0] is not None:  # p in {0,1}: Wilson-derived SE floor
            se = max(se, (float(ci[1]) - float(ci[0])) / (2 * 1.959964))
    else:
        p, se, ci = None, None, [None, None]
    return {
        "score": p, "se": se, "wilson_ci95": ci, "decided": decided,
        "a_wins": a_wins, "b_wins": int(sc.get("b_wins") or 0),
        "completed": sc.get("completed"), "truncated": sc.get("truncated"),
        "aborted_budget": sc.get("aborted_budget"),
    }


def cmd_eval_arm(args) -> int:
    """[subprocess] repackage (raw weights by default) + play_sealbot_match;
    writes the summary json."""
    _ensure_eq_paths()
    arm_dir = Path(args.arm_dir)
    out_json = Path(args.out)
    summary: dict = {
        "arm": args.eval_arm, "opponent": "sealbot", "variant": args.sealbot_variant,
        "n_games": args.n_games, "seed_base": args.seed_base, "visits": args.visits,
        "vbs": args.vbs, "weights_requested": args.weights, "error": None,
        "written_utc": None,
    }
    try:
        src = Path(args.ckpt)
        soak_init = arm_dir / SOAK_INIT_NAME
        info = build_soak_init(src, soak_init, weights=args.weights)
        summary.update({
            "source_ckpt": str(src),
            "source_ckpt_mtime": info["source_ckpt_mtime"],
            "source_ckpt_size": info["source_ckpt_size"],
            "weights": info["weights"],
            "eval_ckpt": str(soak_init),
        })
        cfg = _load_run_config(args.config)
        from hexfield_eq.eval_arena import play_sealbot_match
        t0 = time.time()
        result = play_sealbot_match(
            str(soak_init),
            args.n_games,
            config=cfg,
            label=args.eval_arm,
            sealbot_variant=args.sealbot_variant,
            sealbot_time_limit=args.sealbot_time_limit,
            sealbot_path=None,  # resolves $SEALBOT_PATH
            visits=args.visits,
            virtual_batch_size=args.vbs,
            opening_plies=args.opening_plies,
            opening_temperature=args.opening_temp,
            diagnostics_dir=str(arm_dir / "eval_sealbot"),
            game_seed_base=args.seed_base,
            max_wall_seconds=args.max_wall_seconds,
            max_states=args.max_states,
        )
        summary.update(_binomial_summary(result))
        summary["elapsed_seconds"] = round(time.time() - t0, 1)
        _atomic_write_text(arm_dir / "eval_sealbot_full.json",
                           json.dumps(_json_safe(result), indent=2))
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        summary["error"] = f"{type(exc).__name__}: {exc}"
    summary["written_utc"] = _utc()
    _atomic_write_text(out_json, json.dumps(_json_safe(summary), indent=2))
    print(json.dumps(_json_safe(summary)), flush=True)
    return 0 if summary["error"] is None else 1


def cmd_strix_baseline(args) -> int:
    """[subprocess] record-only Strix baseline for the winner (paired match)."""
    _ensure_eq_paths()
    arm_dir = Path(args.arm_dir)
    out_json = Path(args.out)
    summary: dict = {
        "arm": args.strix_baseline, "opponent": "strix", "n_games": args.n_games,
        "seed_base": args.seed_base, "visits": args.visits, "strix_sims": args.strix_sims,
        "error": None,
    }
    try:
        soak_init = arm_dir / SOAK_INIT_NAME
        if not soak_init.is_file():
            build_soak_init(Path(args.ckpt), soak_init, weights=args.weights)
        cfg = _load_run_config(args.config)
        from hexfield_eq.eval_arena import play_strix_match
        result = play_strix_match(
            str(soak_init),
            args.strix_ckpt,
            args.n_games,
            config=cfg,
            label=args.strix_baseline,
            strix_label="strix",
            visits=args.visits,
            virtual_batch_size=args.vbs,
            opening_plies=args.opening_plies,
            opening_temperature=args.opening_temp,
            strix_sims=args.strix_sims,
            strix_m_actions=args.strix_m,
            strix_device=args.strix_device,
            paired_openings=True,
            diagnostics_dir=str(arm_dir / "eval_strix_baseline"),
            game_seed_base=args.seed_base,
            max_wall_seconds=args.max_wall_seconds,
            max_states=args.max_states,
        )
        summary.update(_binomial_summary(result))
        pent = result.get("pentanomial") or {}
        summary["pentanomial"] = pent.get("histogram_a_wins")
        summary["pair_winrate_mean"] = pent.get("pair_winrate_mean")
        summary["pair_winrate_se"] = pent.get("pair_winrate_se")
        _atomic_write_text(arm_dir / "strix_baseline_full.json",
                           json.dumps(_json_safe(result), indent=2))
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        summary["error"] = f"{type(exc).__name__}: {exc}"
    summary["written_utc"] = _utc()
    _atomic_write_text(out_json, json.dumps(_json_safe(summary), indent=2))
    print(json.dumps(_json_safe(summary)), flush=True)
    return 0 if summary["error"] is None else 1


def cmd_repackage(args) -> int:
    _ensure_eq_paths()
    build_soak_init(Path(args.ckpt), Path(args.out), weights=args.weights)
    return 0


# --------------------------------------------------------------------------- #
# Mock runs dir for --dry-run.
# --------------------------------------------------------------------------- #
def make_mock(root: Path, scenario: str) -> None:
    """Fabricate a runs dir for the mandated dry-run scenarios (1-epoch,
    step-capped deadline regime rows: steps=1400).

    happy:    all 5 arms complete + healthy; arm4 best; 4c within 1 SE of 4.
    arm3sick: arm3 catastrophic (ema_value_ece 0.35); arm4c beats arm4 by
              >1 SE -> winner arm4c with RAY_BLOCKERS=0.
    deadline: arms 1-4 complete; arm4c NOT run yet; only arm4 has a mock eval.
              Drive with --deadline-in-minutes 45 to watch the governor skip
              4c, degrade the eval plan, run partial evals, skip strix, and
              still reach the soak launch.
    """
    def rows(arm: Arm, n_epochs: int, ece: float, top1: float, *, nan=False):
        out = []
        for e in range(n_epochs):
            frac = (e + 1) / max(arm.epochs, 1)
            r = {
                "epoch": e, "steps": 1400, "global_step": 1400 * (e + 1),
                "seconds": 2300.0, "lr": 2.8e-3 * (1 - 0.45 * frac),
                "train_total": float("nan") if nan else 2.9 - 0.8 * frac,
                "train_policy": 1.9 - 0.5 * frac, "train_value": 0.62 - 0.1 * frac,
                "grad_norm_mean": 1.9, "amp_scale": 32768.0, "token_stream_max": 6.5,
                "val_rows": 50000, "top1": top1 - 0.02 * (1 - frac),
                "policy_ce": 1.85 - 0.4 * frac, "value_ce": 0.6 - 0.08 * frac,
                "value_ece": ece + 0.02 * (1 - frac), "value_optimism": 0.01,
                "ema_top1": top1 - 0.015 * (1 - frac),
                "ema_policy_ce": 1.83 - 0.4 * frac, "ema_value_ce": 0.59 - 0.08 * frac,
                "ema_value_ece": float("nan") if nan else ece,
                "train_val_policy_ce_gap": -0.05, "train_val_value_ce_gap": -0.02,
                "probe_entropy": 2.1, "probe_ev_mean": 0.02,
                "probe_policy_kl_prev": 0.08 if e else None,
            }
            out.append(r)
        return out

    # (n_epochs_done, ece, top1, score) — n_epochs_done 0 = arm not started.
    if scenario == "happy":
        spec = {
            "arm1_vanilla": (1, 0.045, 0.44, 0.52),
            "arm2_reglane": (1, 0.040, 0.46, 0.58),
            "arm3_tokread": (1, 0.038, 0.47, 0.61),
            "arm4_raylayout": (1, 0.036, 0.48, 0.66),
            "arm4c_georay": (1, 0.037, 0.475, 0.64),
        }
    elif scenario == "arm3sick":
        spec = {
            "arm1_vanilla": (1, 0.045, 0.44, 0.51),
            "arm2_reglane": (1, 0.040, 0.46, 0.57),
            "arm3_tokread": (1, 0.35, 0.30, None),     # catastrophic ECE
            "arm4_raylayout": (1, 0.041, 0.465, 0.55),
            "arm4c_georay": (1, 0.037, 0.48, 0.66),    # beats 4 by > 1 SE
        }
    elif scenario == "deadline":
        spec = {
            "arm1_vanilla": (1, 0.045, 0.44, None),    # no mock eval: eval costs time
            "arm2_reglane": (1, 0.040, 0.46, None),
            "arm3_tokread": (1, 0.038, 0.47, None),
            "arm4_raylayout": (1, 0.036, 0.48, 0.60),
            "arm4c_georay": (0, 0.0, 0.0, None),       # never started
        }
    else:
        raise SystemExit(f"unknown scenario {scenario!r} (happy|arm3sick|deadline)")

    for arm in ARMS:
        n_done, ece, top1, score = spec[arm.name]
        d = root / arm.name
        if n_done == 0 and score is None:
            continue  # arm not started at all
        d.mkdir(parents=True, exist_ok=True)
        if n_done:
            with open(d / "diagnostics.jsonl", "w", encoding="utf-8") as fh:
                for r in rows(arm, n_done, ece, top1):
                    fh.write(json.dumps(r) + "\n")
            for e in range(n_done):
                (d / f"checkpoint_epoch{e}.pt").write_bytes(b"mock-prefit-checkpoint")
        if score is not None:
            decided = EVAL_GAMES
            se = math.sqrt(score * (1 - score) / decided)
            (d / "mock_eval.json").write_text(json.dumps({
                "arm": arm.name, "opponent": "sealbot", "mock": True,
                "n_games": EVAL_GAMES, "seed_base": EVAL_SEED_BASE,
                "weights_requested": REGIME["weights"],
                "score": score, "se": round(se, 4), "decided": decided,
                "a_wins": int(round(score * decided)),
                "b_wins": decided - int(round(score * decided)),
            }, indent=2), encoding="utf-8")
    print(f"mock runs dir written: {root} (scenario {scenario})")


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--deadline-ts", type=float,
                   default=float(os.environ.get("EQ_LADDER_DEADLINE_TS", "0") or 0),
                   help="hard deadline as unix seconds (0 = none); the soak launch "
                        "always happens before it")
    p.add_argument("--deadline-in-minutes", type=float, default=None,
                   help="convenience: deadline = now + N minutes (overrides --deadline-ts)")
    p.add_argument("--dry-run", action="store_true",
                   help="walk the state machine; construct but never execute")
    p.add_argument("--mock-root", default=None,
                   help="with --dry-run: use this dir as the ladder root")
    p.add_argument("--make-mock", default=None, metavar="DIR",
                   help="fabricate a mock runs dir for --dry-run and exit")
    p.add_argument("--scenario", default="happy", choices=("happy", "arm3sick", "deadline"))
    # subprocess modes
    p.add_argument("--eval-arm", default=None, help="[subprocess] SealBot eval for one arm")
    p.add_argument("--strix-baseline", default=None,
                   help="[subprocess] record-only Strix baseline for the winner")
    p.add_argument("--repackage", default=None, help="[subprocess] build soak-init only")
    p.add_argument("--arm-dir", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--config", default=str(CONFIG_TOML))
    p.add_argument("--n-games", type=int, default=EVAL_GAMES)
    p.add_argument("--seed-base", type=int, default=EVAL_SEED_BASE)
    p.add_argument("--visits", type=int, default=512)
    p.add_argument("--vbs", type=int, default=32)
    p.add_argument("--opening-plies", type=int, default=8)
    p.add_argument("--opening-temp", type=float, default=1.0)
    p.add_argument("--sealbot-variant", default="current")
    p.add_argument("--sealbot-time-limit", type=float, default=0.05)
    p.add_argument("--strix-ckpt", default=DEFAULT_STRIX_CKPT)
    p.add_argument("--strix-sims", type=int, default=512)
    p.add_argument("--strix-m", type=int, default=16)
    p.add_argument("--strix-device", default="cuda")
    p.add_argument("--max-wall-seconds", type=float, default=EVAL_MAX_WALL)
    p.add_argument("--max-states", type=int, default=EVAL_MAX_STATES)
    p.add_argument("--weights", default=REGIME["weights"], choices=("raw", "ema"),
                   help="which prefit weights rank + seed the soak (owner: raw)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.make_mock:
        make_mock(Path(args.make_mock), args.scenario)
        return 0
    if args.eval_arm:
        return cmd_eval_arm(args)
    if args.strix_baseline:
        return cmd_strix_baseline(args)
    if args.repackage:
        return cmd_repackage(args)

    deadline_ts = args.deadline_ts
    if args.deadline_in_minutes is not None:
        deadline_ts = time.time() + args.deadline_in_minutes * 60.0
    root = Path(args.mock_root) if (args.dry_run and args.mock_root) else LADDER_ROOT
    ladder = Ladder(root, dry_run=args.dry_run, deadline_ts=deadline_ts)
    if not args.dry_run and not ladder.acquire_lock():
        return 1
    try:
        return run_ladder(ladder)
    except Exception as exc:  # noqa: BLE001 - last-resort recorder
        ladder.error("run_ladder (unhandled)", exc)
        import traceback
        traceback.print_exc()
        return 4
    finally:
        if not args.dry_run:
            ladder.release_lock()


if __name__ == "__main__":
    raise SystemExit(main())
