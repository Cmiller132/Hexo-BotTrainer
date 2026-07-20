from __future__ import annotations

import unittest

from scripts.tss_harness.bench import (
    BenchProfile,
    GpuProcess,
    GpuState,
    TimedDriver,
    WindowAccounting,
    adoption_report,
    build_scorecard,
    query_gpu_state,
    quiet_machine_problem,
)


class WindowAccountingTests(unittest.TestCase):
    def test_half_open_window_and_throughput(self):
        accounting = WindowAccounting(1_000, 11_000)
        self.assertFalse(
            accounting.record(999, latency_nanos=10, game_finished=True)
        )
        self.assertTrue(
            accounting.record(1_000, latency_nanos=100, game_finished=False)
        )
        self.assertTrue(
            accounting.record(10_999, latency_nanos=300, game_finished=True)
        )
        self.assertFalse(
            accounting.record(11_000, latency_nanos=10, game_finished=True)
        )
        self.assertEqual(accounting.decisions, 2)
        self.assertEqual(accounting.games_finished, 1)
        self.assertAlmostEqual(accounting.window_seconds, 0.00001)
        self.assertAlmostEqual(accounting.moves_per_min, 12_000_000.0)

    def test_timed_driver_excludes_warmup_and_deadline_drain(self):
        class FakeDriver:
            def __init__(self):
                self.games_finished = 0
                self.tss_sharpen = False
                self.tss_moves = 0
                self.tss_deep_verify_failed = 0
                self.tss_zone_verify_failed = 0
                self.tss_values = [1.0]
                self.calls = 0

            def __call__(self, game_key, payload):
                self.calls += 1
                self.tss_moves += 1
                tss = payload["diagnostics"]["tss"]
                self.tss_deep_verify_failed += int(tss.get("deep_verify_failed", 0))
                return ("advance", object())

        times = iter((50, 120, 200))
        driver = FakeDriver()
        accounting = WindowAccounting(100, 200)
        timed = TimedDriver(driver, accounting, [7], 0, clock_ns=lambda: next(times))
        warm = {"diagnostics": {"tss": {"deep_verify_failed": 1}}}
        clean = {"diagnostics": {"tss": {"deep_verify_failed": 0}}}
        drain = {"diagnostics": {"tss": {"deep_verify_failed": 2}}}

        timed(7, warm)
        timed(7, clean)
        response = timed(7, drain)

        self.assertIsNone(response)
        self.assertEqual(driver.calls, 2)
        self.assertEqual(timed.warmup_verify_failed, 1)
        self.assertEqual(timed.drain_verify_failed, 2)
        self.assertEqual(driver.tss_moves, 1)  # warmup telemetry was reset
        self.assertEqual(accounting.decisions, 1)
        self.assertEqual(accounting.latencies_nanos, [20])


class ScorecardTests(unittest.TestCase):
    def test_scorecard_math_and_report_only_adoption(self):
        profile = BenchProfile("test", 8, 16, 5.0, 10.0)
        accounting = WindowAccounting(0, 10_000_000_000)
        for index in range(20):
            accounting.record(
                index * 100_000_000,
                latency_nanos=(index + 1) * 10,
                game_finished=index < 3,
            )
        scorecard = build_scorecard(
            set_hash="abc",
            checkpoint="epoch_000090.pt",
            arm_config={"tss_solver_horizon": 0},
            profile=profile,
            accounting=accounting,
            games_finished=accounting.games_finished,
            games_seeded=8,
            seed_plies=[30, 40, 50, 60],
            tss={"deep_calls": 12, "deep_verify_failed": 0},
            gpu_values=[10.0, 20.0, 30.0],
            load_fingerprint={"cpu_count": 4, "other_gpu_procs": 0, "loadavg": [1, 2, 3]},
            verify_failed_total=0,
            baseline_moves_per_min=115.0,
            tolerance_percent=5.0,
        )
        self.assertEqual(scorecard["moves_per_min"], 120.0)
        self.assertEqual(scorecard["decisions"], 20)
        self.assertEqual(scorecard["games_finished"], 3)
        self.assertEqual(scorecard["per_move_latency"]["p50"], 105)
        self.assertEqual(scorecard["gpu_util"]["mean"], 20.0)
        self.assertEqual(scorecard["adoption"]["classification"], "within_tolerance")
        self.assertFalse(scorecard["adoption"]["hard_gate"])

    def test_tolerance_classifications(self):
        self.assertEqual(
            adoption_report(
                106.0, baseline_moves_per_min=100.0, tolerance_percent=5.0
            )["classification"],
            "improved",
        )
        self.assertEqual(
            adoption_report(
                94.0, baseline_moves_per_min=100.0, tolerance_percent=5.0
            )["classification"],
            "regression_to_investigate",
        )


class QuietMachineTests(unittest.TestCase):
    def test_query_uses_stubbed_nvidia_smi(self):
        def fake_command(args):
            if "--query-gpu=utilization.gpu" in args:
                return "4\n7\n"
            return "123, 900\n456, 1100\n"

        state = query_gpu_state(fake_command)
        self.assertEqual(state.utilization, (4.0, 7.0))
        self.assertEqual(state.processes[1], GpuProcess(456, 1100.0))

    def test_refuses_busy_gpu(self):
        reason = quiet_machine_problem(GpuState((10.1,), ()), current_pid=1)
        self.assertIn("utilization", reason or "")

    def test_refuses_large_other_process_but_ignores_self(self):
        busy = GpuState((2.0,), (GpuProcess(22, 1025.0),))
        self.assertIn("another process", quiet_machine_problem(busy, current_pid=11) or "")
        self.assertIsNone(quiet_machine_problem(busy, current_pid=22))

    def test_boundary_is_allowed(self):
        state = GpuState((10.0,), (GpuProcess(22, 1024.0),))
        self.assertIsNone(quiet_machine_problem(state, current_pid=11))


if __name__ == "__main__":
    unittest.main()
