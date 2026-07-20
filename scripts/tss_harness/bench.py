"""Production-shaped throughput benchmark for the TSS self-play solver.

The full tier holds 256 seeded games active at the production 256 visits,
warms the persistent session for 60 seconds, then measures a 240-second wall
window.  It loads main_3's production config and ep90 checkpoint; ``--config-json``
overlays only TSS fields in the self-play section.  The smoke tier is the later
orchestrator check (8 games, 16 visits, 20 measured seconds, no warmup).

Runtime environment: ``/root/.venvs/hexo-bottrainer-wsl`` with this worktree on
``PYTHONPATH`` following ``scripts/_v1_soak/arch_env.py`` conventions.  This
module deliberately imports torch and ``hexfield_eq`` only inside the live run,
so its accounting/refusal tests run under plain Python without that venv.

Examples::

    python scripts/tss_harness/bench.py --full --config-json '{}' \
        --out harness_runs/bench.json
    python scripts/tss_harness/bench.py --smoke \
        --config-json '{"tss_solver_horizon":0}' --out /tmp/tss-smoke.json

Throughput adoption is report-only.  With ``--baseline``, the scorecard labels
a result within the configured tolerance as comparable, above it as improved,
and below it as a regression to investigate; the label never changes exit
status.  Soundness is different: any verifier failure is fatal and exits
nonzero after preserving the scorecard.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import queue
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

try:
    from .bench_seeds import DEFAULT_OUTPUT as DEFAULT_SEED_SET
    from .bench_seeds import load_and_verify
    from .contract import SCHEMA_VERSION
except ImportError:  # Direct script execution.
    from bench_seeds import DEFAULT_OUTPUT as DEFAULT_SEED_SET
    from bench_seeds import load_and_verify
    from contract import SCHEMA_VERSION


RUN_DIR = Path("/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main_3")
CONFIG_PATH = RUN_DIR / "_resume_config.toml"
CHECKPOINT = RUN_DIR / "checkpoints" / "epoch_000090.pt"
BASE_SEED = 20260720
QUIET_GPU_UTIL_MAX = 10.0
QUIET_OTHER_PROCESS_MIB_MAX = 1024.0
TOLERANCE_PERCENT_DEFAULT = 5.0


@dataclass(frozen=True)
class BenchProfile:
    name: str
    games_active: int
    visits: int
    warmup_seconds: float
    window_seconds: float


FULL = BenchProfile("full", 256, 256, 60.0, 240.0)
SMOKE = BenchProfile("smoke", 8, 16, 0.0, 20.0)


class BenchmarkError(RuntimeError):
    pass


class QuietMachineRefusal(BenchmarkError):
    pass


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    used_memory_mib: float


@dataclass(frozen=True)
class GpuState:
    utilization: tuple[float, ...]
    processes: tuple[GpuProcess, ...]


@dataclass(frozen=True)
class SeedPosition:
    """Small object matching ContinuousDriver's existing ``.move_prefix`` API."""

    move_prefix: tuple[tuple[int, int], ...]


def percentile(values: Sequence[float | int], q: float) -> float | None:
    """Linear percentile (the usual p50/p90/p99 scorecard convention)."""

    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    fraction = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


def latency_summary(latencies_nanos: Sequence[int]) -> dict[str, int | None]:
    def rounded(q: float) -> int | None:
        value = percentile(latencies_nanos, q)
        return int(round(value)) if value is not None else None

    return {"p50": rounded(50), "p90": rounded(90), "p99": rounded(99)}


def gpu_util_summary(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "mean": float(statistics.fmean(values)) if values else None,
        "p10": percentile(values, 10),
        "p90": percentile(values, 90),
    }


@dataclass
class WindowAccounting:
    """Exact callback-time accounting for the measured half-open window."""

    start_ns: int
    end_ns: int
    decisions: int = 0
    games_finished: int = 0
    latencies_nanos: list[int] | None = None

    def __post_init__(self) -> None:
        if self.end_ns <= self.start_ns:
            raise ValueError("measurement window must have positive duration")
        if self.latencies_nanos is None:
            self.latencies_nanos = []

    def contains(self, timestamp_ns: int) -> bool:
        return self.start_ns <= timestamp_ns < self.end_ns

    def record(self, timestamp_ns: int, *, latency_nanos: int, game_finished: bool) -> bool:
        if not self.contains(timestamp_ns):
            return False
        self.decisions += 1
        self.games_finished += int(game_finished)
        assert self.latencies_nanos is not None
        self.latencies_nanos.append(max(int(latency_nanos), 0))
        return True

    @property
    def window_seconds(self) -> float:
        return (self.end_ns - self.start_ns) / 1_000_000_000.0

    @property
    def moves_per_min(self) -> float:
        return self.decisions * 60.0 / self.window_seconds


def _run_command(args: list[str]) -> str:
    completed = subprocess.run(
        args, check=True, capture_output=True, text=True, timeout=10
    )
    return completed.stdout


def _parse_numbers(output: str) -> tuple[float, ...]:
    values: list[float] = []
    for line in output.splitlines():
        token = line.strip().split(",", 1)[0].strip()
        if not token or token.lower() in {"n/a", "[n/a]"}:
            continue
        values.append(float(token))
    return tuple(values)


def _parse_processes(output: str) -> tuple[GpuProcess, ...]:
    processes: list[GpuProcess] = []
    for line in output.splitlines():
        if not line.strip() or "no running" in line.lower():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            raise BenchmarkError(f"unexpected nvidia-smi process row: {line!r}")
        try:
            processes.append(GpuProcess(int(parts[0]), float(parts[1])))
        except ValueError as exc:
            raise BenchmarkError(f"unexpected nvidia-smi process row: {line!r}") from exc
    return tuple(processes)


def query_gpu_state(command: Callable[[list[str]], str] = _run_command) -> GpuState:
    try:
        utilization = _parse_numbers(
            command(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ]
            )
        )
        processes = _parse_processes(
            command(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ]
            )
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise BenchmarkError(f"cannot sample nvidia-smi: {exc}") from exc
    if not utilization:
        raise BenchmarkError("nvidia-smi returned no GPU utilization values")
    return GpuState(utilization, processes)


def quiet_machine_problem(
    state: GpuState,
    *,
    current_pid: int | None = None,
    utilization_limit: float = QUIET_GPU_UTIL_MAX,
    other_process_memory_limit_mib: float = QUIET_OTHER_PROCESS_MIB_MAX,
) -> str | None:
    """Return the refusal reason, or ``None`` when the machine is quiet."""

    current_pid = os.getpid() if current_pid is None else current_pid
    peak_util = max(state.utilization, default=0.0)
    if peak_util > utilization_limit:
        return (
            f"GPU utilization is {peak_util:.1f}% (quiet limit {utilization_limit:.1f}%)"
        )
    contenders = [
        process
        for process in state.processes
        if process.pid != current_pid
        and process.used_memory_mib > other_process_memory_limit_mib
    ]
    if contenders:
        detail = ", ".join(
            f"pid {process.pid}: {process.used_memory_mib:.0f} MiB"
            for process in contenders
        )
        return (
            "another process holds more than "
            f"{other_process_memory_limit_mib:.0f} MiB GPU memory ({detail})"
        )
    return None


class GpuSampler:
    """One-second nvidia-smi sampler with timestamped values."""

    def __init__(self, command: Callable[[list[str]], str] = _run_command):
        self._command = command
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[tuple[int, float]] = []
        self.errors: queue.SimpleQueue[BaseException] = queue.SimpleQueue()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="tss-bench-gpu", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                values = _parse_numbers(
                    self._command(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ]
                    )
                )
                if not values:
                    raise BenchmarkError("nvidia-smi sampler returned no utilization")
                self.samples.append((time.monotonic_ns(), max(values)))
            except BaseException as exc:  # surfaced on the owner thread
                self.errors.put(exc)
                return
            self._stop.wait(1.0)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if not self.errors.empty():
            raise BenchmarkError(f"GPU utilization sampling failed: {self.errors.get()}")

    def values_between(self, start_ns: int, end_ns: int) -> list[float]:
        return [value for timestamp, value in self.samples if start_ns <= timestamp < end_ns]


def seed_plies_summary(seed_plies: Sequence[int]) -> dict[str, int | float | None]:
    return {
        "count": len(seed_plies),
        "min": min(seed_plies) if seed_plies else None,
        "mean": float(statistics.fmean(seed_plies)) if seed_plies else None,
        "p50": percentile(seed_plies, 50),
        "p90": percentile(seed_plies, 90),
        "max": max(seed_plies) if seed_plies else None,
    }


def adoption_report(
    moves_per_min: float,
    *,
    baseline_moves_per_min: float | None,
    tolerance_percent: float,
) -> dict[str, Any]:
    """Report-only tolerance semantics; never used as an exit gate."""

    result: dict[str, Any] = {
        "hard_gate": False,
        "tolerance_percent": float(tolerance_percent),
        "baseline_moves_per_min": baseline_moves_per_min,
        "delta_percent": None,
        "classification": "baseline_not_supplied",
    }
    if baseline_moves_per_min is None:
        return result
    if baseline_moves_per_min <= 0:
        raise ValueError("baseline moves_per_min must be positive")
    delta = (moves_per_min / baseline_moves_per_min - 1.0) * 100.0
    result["delta_percent"] = delta
    if delta > tolerance_percent:
        result["classification"] = "improved"
    elif delta < -tolerance_percent:
        result["classification"] = "regression_to_investigate"
    else:
        result["classification"] = "within_tolerance"
    return result


def build_scorecard(
    *,
    set_hash: str,
    checkpoint: str,
    arm_config: dict[str, Any],
    effective_tss: dict[str, Any] | None = None,
    profile: BenchProfile,
    accounting: WindowAccounting,
    games_finished: int,
    games_seeded: int,
    seed_plies: Sequence[int],
    tss: dict[str, Any],
    gpu_values: Sequence[float],
    load_fingerprint: dict[str, Any],
    verify_failed_total: int,
    baseline_moves_per_min: float | None = None,
    tolerance_percent: float = TOLERANCE_PERCENT_DEFAULT,
) -> dict[str, Any]:
    moves_per_min = accounting.moves_per_min
    return {
        "schema": SCHEMA_VERSION,
        "set_hash": set_hash,
        "checkpoint": checkpoint,
        "arm_config": arm_config,
        "effective_tss": effective_tss or {},
        "window_seconds": profile.window_seconds,
        "warmup_seconds": profile.warmup_seconds,
        "games_active": profile.games_active,
        "visits": profile.visits,
        "moves_per_min": moves_per_min,
        "decisions": accounting.decisions,
        "games_finished": int(games_finished),
        "games_seeded": int(games_seeded),
        "seed_plies_summary": seed_plies_summary(seed_plies),
        "tss": tss,
        "per_move_latency": latency_summary(accounting.latencies_nanos or []),
        "gpu_util": gpu_util_summary(gpu_values),
        "load_fingerprint": load_fingerprint,
        "verify_failed_total": int(verify_failed_total),
        "adoption": adoption_report(
            moves_per_min,
            baseline_moves_per_min=baseline_moves_per_min,
            tolerance_percent=tolerance_percent,
        ),
    }


def _load_arm_config(raw_json: str) -> dict[str, Any]:
    try:
        value = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"--config-json is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkError("--config-json must decode to an object")
    invalid = sorted(
        key
        for key in value
        if not isinstance(key, str)
        or not key.startswith("tss_")
        or key == "tss_policy_target_sharpen"
    )
    if invalid:
        raise BenchmarkError(
            "--config-json may override solver TSS fields only; invalid keys: "
            + ", ".join(map(str, invalid))
        )
    return value


def _production_config(arm_config: dict[str, Any]):
    """Lazy config import/parse, after the main_3 architecture has been primed."""

    with CONFIG_PATH.open("rb") as handle:
        raw = tomllib.load(handle)
    model_config = copy.deepcopy(raw["model"]["config"])
    selfplay = dict(model_config.get("selfplay", {}))
    selfplay.update(arm_config)
    model_config["selfplay"] = selfplay
    from hexfield_eq.config import parse_hexfield_config

    return parse_hexfield_config(model_config)


def _load_fingerprint(initial_gpu: GpuState) -> dict[str, Any]:
    try:
        loadavg: list[float] | None = list(os.getloadavg())
    except (AttributeError, OSError):
        loadavg = None
    current_pid = os.getpid()
    return {
        "cpu_count": os.cpu_count(),
        "other_gpu_procs": len(
            {process.pid for process in initial_gpu.processes if process.pid != current_pid}
        ),
        "loadavg": loadavg,
    }


def _reset_tss_window(driver: Any) -> int:
    """Reset only driver TSS telemetry at the warmup/measurement boundary."""

    warmup_verify = int(getattr(driver, "tss_deep_verify_failed", 0)) + int(
        getattr(driver, "tss_zone_verify_failed", 0)
    )
    for name, value in vars(driver).items():
        if not name.startswith("tss_") or name == "tss_sharpen":
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            setattr(driver, name, type(value)(0))
        elif isinstance(value, list):
            value.clear()
    return warmup_verify


class TimedDriver:
    """Window/refill wrapper around the unmodified production ContinuousDriver."""

    def __init__(
        self,
        driver: Any,
        accounting: WindowAccounting,
        initial_keys: Iterable[int],
        run_start_ns: int,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ):
        self.driver = driver
        self.accounting = accounting
        self.clock_ns = clock_ns
        self.started_ns = {int(key): run_start_ns for key in initial_keys}
        self.window_reset = False
        self.warmup_verify_failed = 0
        self.drain_verify_failed = 0

    @staticmethod
    def _payload_verify_failed(payload: dict[str, Any]) -> int:
        tss = ((payload.get("diagnostics") or {}).get("tss") or {})
        return int(tss.get("deep_verify_failed", 0) or 0) + int(
            tss.get("zone_verify_failed", 0) or 0
        )

    def _ensure_window_reset(self) -> None:
        if not self.window_reset:
            self.warmup_verify_failed = _reset_tss_window(self.driver)
            self.window_reset = True

    def __call__(self, game_key: int, payload: dict[str, Any]):
        now_ns = self.clock_ns()
        key = int(game_key)
        if now_ns >= self.accounting.end_ns:
            self._ensure_window_reset()
            self.drain_verify_failed += self._payload_verify_failed(payload)
            self.started_ns.pop(key, None)
            return None  # cleanly retire/abort this unfinished native slot

        if now_ns >= self.accounting.start_ns:
            self._ensure_window_reset()
        previous_start = self.started_ns.get(key, now_ns)
        finished_before = int(self.driver.games_finished)
        response = self.driver(game_key, payload)
        finished = int(self.driver.games_finished) > finished_before
        if finished:
            # ContinuousDriver always hands finished tapes to its writer queue.
            # The benchmark intentionally has no writer: discard that one item
            # immediately so a five-minute run does not retain finalized games
            # or emit training shards/records.
            try:
                self.driver._write_queue.get_nowait()
            except queue.Empty as exc:
                raise BenchmarkError("finished game was not queued by ContinuousDriver") from exc
            else:
                self.driver._write_queue.task_done()

        if self.accounting.contains(now_ns):
            latency_start = max(previous_start, self.accounting.start_ns)
            self.accounting.record(
                now_ns,
                latency_nanos=now_ns - latency_start,
                game_finished=finished,
            )

        self.started_ns.pop(key, None)
        if response is not None:
            if response[0] == "advance":
                self.started_ns[key] = now_ns
            elif response[0] == "replace":
                self.started_ns[int(response[1])] = now_ns
        return response


def _baseline_mpm(path: Path | None) -> float | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = float(payload["moves_per_min"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise BenchmarkError(f"cannot read baseline scorecard {path}: {exc}") from exc
    if value <= 0:
        raise BenchmarkError(f"baseline scorecard {path} has non-positive moves_per_min")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def run_benchmark(
    profile: BenchProfile,
    arm_config: dict[str, Any],
    *,
    seed_set_path: Path,
    initial_gpu: GpuState,
    baseline_moves_per_min: float | None,
    tolerance_percent: float,
) -> dict[str, Any]:
    manifest, seed_rows = load_and_verify(seed_set_path)
    seeds = [
        SeedPosition(tuple((int(q), int(r)) for q, r in row["moves"]))
        for row in seed_rows
    ]

    # Architecture environment must precede every hexfield_eq import.
    soak_dir = Path(__file__).resolve().parents[1] / "_v1_soak"
    if str(soak_dir) not in sys.path:
        sys.path.insert(0, str(soak_dir))
    import arch_env  # noqa: F401
    # The GPU venv's .pth entries point at a stale checkout. The solver under
    # test (hexfield_eq, compiled .so in-tree) must come from THIS worktree;
    # the infrastructure packages need their compiled _rust extensions, which
    # only the main checkout carries — worktrees hold source only.
    worktree_pkgs = Path(__file__).resolve().parents[2] / "packages"
    main_pkgs = Path("/mnt/e/Hexo-BotTrainer-hexgt/packages")
    paths = [worktree_pkgs / "hexfield_eq" / "python"]
    paths += [main_pkgs / pkg / "python"
              for pkg in ("hexo_engine", "hexo_utils", "hexo_models",
                          "hexo_runner", "hexo_train")]
    for p in reversed([str(p) for p in paths]):
        if p not in sys.path:
            sys.path.insert(0, p)
    from hexfield_eq.serve_env import prime_serve_env

    prime_serve_env()
    from hexfield_eq import _rust, eval_arena
    from hexfield_eq.config import build_divergence_overrides, build_fast_divergence_overrides
    from hexfield_eq.inference import build_serve_evaluator
    from hexfield_eq.selfplay import ContinuousDriver
    import torch

    if not CONFIG_PATH.is_file():
        raise BenchmarkError(f"production config not found: {CONFIG_PATH}")
    if not CHECKPOINT.is_file():
        raise BenchmarkError(f"ep90 checkpoint not found: {CHECKPOINT}")
    cfg = _production_config(arm_config)
    sp = cfg.selfplay
    model = eval_arena._load_hexfield_net(CHECKPOINT)
    evaluator = build_serve_evaluator(model, cfg, role="selfplay", auto_match_serve_env=True)

    sampler = GpuSampler()
    scheduler_stats: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="tss-bench-") as tmp:
        # A very high target keeps every finished slot refilled for the entire
        # timebox. record_file=None and no writer thread disable disk/training-row
        # output while retaining the production callback and telemetry path.
        driver = ContinuousDriver(
            epoch=90,
            games_target=1_000_000,
            max_plies=sp.max_game_plies,
            out_dir=Path(tmp),
            record_file=None,
            diag_dir=None,
            active_limit=profile.games_active,
            blunder_seeds=seeds,
            blunder_seed_fraction=1.0,
            blunder_base_seed=BASE_SEED,
            # This writer-side lever is not an arm override, but retaining its
            # production value keeps callback CPU work production-shaped even
            # though record_file=None prevents rows reaching disk.
            tss_sharpen=sp.tss_policy_target_sharpen,
        )
        tapes = driver.start_games(profile.games_active)
        session = _rust.HexfieldMctsSession(max_states=sp.cache_max_states)
        noise_kwargs: dict[str, Any] = {}
        if sp.root_dirichlet_noise_fraction > 0:
            noise_kwargs = {
                "root_dirichlet_total_alpha": sp.root_dirichlet_total_alpha,
                "root_dirichlet_noise_fraction": sp.root_dirichlet_noise_fraction,
            }

        torch.cuda.synchronize()
        run_start_ns = time.monotonic_ns()
        window_start_ns = run_start_ns + int(profile.warmup_seconds * 1_000_000_000)
        window_end_ns = window_start_ns + int(profile.window_seconds * 1_000_000_000)
        accounting = WindowAccounting(window_start_ns, window_end_ns)
        timed_driver = TimedDriver(
            driver, accounting, (tape.key for tape in tapes), run_start_ns
        )
        sampler.start()
        try:
            scheduler_stats = session.run_continuous(
                [tape.key for tape in tapes],
                tuple(tape.state for tape in tapes),
                evaluator=evaluator,
                on_move=timed_driver,
                visits=profile.visits,
                c_puct=sp.c_puct,
                base_seed=BASE_SEED,
                virtual_batch_size=sp.virtual_batch_size,
                flush_target=sp.flush_target,
                active_root_limit=profile.games_active,
                temperature_by_ply=cfg.temperature_by_ply(),
                root_policy_temperature=sp.root_policy_temperature,
                root_policy_temperature_early=sp.root_policy_temperature_early or None,
                root_policy_temperature_halflife=sp.root_policy_temperature_halflife or None,
                fpu_reduction=sp.fpu_reduction,
                virtual_loss=sp.virtual_loss,
                widening_policy_mass=sp.widening_policy_mass,
                widening_max_children=sp.widening_max_children,
                widening_min_children=sp.widening_min_children,
                forced_playout_k=sp.forced_playout_k,
                pcr_full_proportion=sp.pcr_full_proportion,
                pcr_fast_visits=sp.pcr_fast_visits,
                pcr_fast_temperature=sp.pcr_fast_temperature,
                policy_init_fraction=sp.policy_init_fraction,
                policy_init_avg_plies=sp.policy_init_avg_plies,
                policy_init_max_plies=sp.policy_init_max_plies,
                policy_init_temperature=sp.policy_init_temperature,
                tss_enabled=sp.tss_enabled,
                root_fpu_reduction=sp.root_fpu_reduction,
                root_fpu_zero_under_noise=sp.root_fpu_zero_under_noise,
                search_parity_mode=sp.search_parity_mode,
                divergence_overrides=build_divergence_overrides(sp),
                fast_divergence_overrides=build_fast_divergence_overrides(sp),
                **noise_kwargs,
            )
            torch.cuda.synchronize()
        finally:
            sampler.stop()
        if not timed_driver.window_reset:
            timed_driver._ensure_window_reset()
        # Native slots were retired without finalizing partial Python tapes.
        driver.games.clear()
        measured_stats = driver.stats()

    tss = dict(measured_stats.get("tss") or {})
    measured_verify = int(tss.get("deep_verify_failed", 0) or 0) + int(
        tss.get("zone_verify_failed", 0) or 0
    )
    tail_verify = int(scheduler_stats.get("tss_async_verify_failed_tail", 0) or 0)
    verify_failed_total = (
        timed_driver.warmup_verify_failed
        + measured_verify
        + timed_driver.drain_verify_failed
        + tail_verify
    )
    gpu_values = sampler.values_between(accounting.start_ns, accounting.end_ns)
    if not gpu_values:
        raise BenchmarkError("no 1-second GPU utilization samples landed in measured window")
    # Echo the RESOLVED solver fields (not the requested overlay) so the
    # runner can gate bench-arm identity against the coverage arm's manifest
    # — caught live 2026-07-20: config {} silently benched the engine-default
    # h16 while the coverage sweep ran unbounded.
    effective_tss = {
        k: getattr(sp, k)
        for k in (
            "tss_enabled", "tss_solver_mode", "tss_solver_node_cap",
            "tss_solver_horizon", "tss_solver_horizon_ladder", "tss_zone",
        )
    }
    return build_scorecard(
        set_hash=str(manifest["sha256"]),
        checkpoint=str(CHECKPOINT),
        arm_config=arm_config,
        effective_tss=effective_tss,
        profile=profile,
        accounting=accounting,
        games_finished=accounting.games_finished,
        games_seeded=driver.games_seeded,
        seed_plies=driver.seed_plies,
        tss=tss,
        gpu_values=gpu_values,
        load_fingerprint=_load_fingerprint(initial_gpu),
        verify_failed_total=verify_failed_total,
        baseline_moves_per_min=baseline_moves_per_min,
        tolerance_percent=tolerance_percent,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    tier = parser.add_mutually_exclusive_group(required=True)
    tier.add_argument("--smoke", action="store_true", help="8 games, 16 visits, 20s")
    tier.add_argument("--full", action="store_true", help="256 games, 256 visits, 60s+240s")
    parser.add_argument("--config-json", default="{}", help="JSON object of TSS overrides")
    parser.add_argument("--out", type=Path, required=True, help="BenchScorecard JSON path")
    parser.add_argument("--seed-set", type=Path, default=DEFAULT_SEED_SET)
    parser.add_argument("--baseline", type=Path, help="prior BenchScorecard for report-only comparison")
    parser.add_argument(
        "--tolerance-percent", type=float, default=TOLERANCE_PERCENT_DEFAULT
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    profile = SMOKE if args.smoke else FULL
    try:
        arm_config = _load_arm_config(args.config_json)
        if args.tolerance_percent < 0:
            raise BenchmarkError("--tolerance-percent must be nonnegative")
        baseline = _baseline_mpm(args.baseline)
        # Refusal happens before torch/model loading because this metric is wall-based.
        initial_gpu = query_gpu_state()
        refusal = quiet_machine_problem(initial_gpu)
        if refusal:
            raise QuietMachineRefusal(
                "REFUSED: throughput benchmark requires a quiet machine; " + refusal
            )
        scorecard = run_benchmark(
            profile,
            arm_config,
            seed_set_path=args.seed_set,
            initial_gpu=initial_gpu,
            baseline_moves_per_min=baseline,
            tolerance_percent=args.tolerance_percent,
        )
        _write_json(args.out, scorecard)
    except QuietMachineRefusal as exc:
        print(str(exc), file=sys.stderr)
        return 4
    except (BenchmarkError, OSError, ValueError) as exc:
        print(f"TSS BENCH ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        f"TSS BENCH {profile.name}: {scorecard['moves_per_min']:.2f} moves/min, "
        f"{scorecard['decisions']} decisions -> {args.out}"
    )
    print(
        "adoption (report only): "
        f"{scorecard['adoption']['classification']} "
        f"(tolerance {scorecard['adoption']['tolerance_percent']:.1f}%)"
    )
    if scorecard["verify_failed_total"]:
        print(
            f"FATAL: verify_failed_total={scorecard['verify_failed_total']}",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
