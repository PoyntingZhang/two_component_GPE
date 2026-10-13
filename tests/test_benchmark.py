"""CPU checks of benchmark configuration, timing boundaries and reporting."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from config.yaml_parser import normalize_config
from gpe.benchmark import VARIANTS, benchmark_config, benchmark_stage, measure, write_report


class BenchmarkTests(unittest.TestCase):
    def test_configs_cover_all_valid_variants_and_disable_outputs(self):
        config = normalize_config(benchmark_config(64, 20, 0.002))
        combinations = set()
        for stage in config["stages"]:
            combinations.add((stage["evolution"]["type"],
                              stage["potential1"]["static"]["enabled"],
                              stage["potential1"]["dynamic"]["enabled"],
                              stage["k_cut"]["enabled"]))
            self.assertEqual(stage["initial_state"]["type"], "gaussian")
            self.assertFalse(any(stage["monitor"].values()))
            self.assertFalse(any(stage["save"].values()))
            self.assertFalse(any(stage[key]["enabled"] for key in ("box", "noise", "absorber")))
        expected = {("imaginary", static, False, False) for static in (False, True)}
        expected |= {("real", static, dynamic, cutoff) for static in (False, True)
                     for dynamic in (False, True) for cutoff in (False, True)}
        self.assertEqual(combinations, expected)
        self.assertEqual(len(config["stages"]), 10)

    def test_wall_timer_waits_for_gpu_completion(self):
        calls = []
        start = SimpleNamespace(record=lambda: calls.append("start event"))
        end = SimpleNamespace(record=lambda: calls.append("end event"),
                              synchronize=lambda: calls.append("wait for end"))
        cp = SimpleNamespace(cuda=SimpleNamespace(
            Event=Mock(side_effect=[start, end]),
            get_current_stream=lambda: SimpleNamespace(synchronize=lambda: calls.append("drain stream")),
            get_elapsed_time=lambda a, b: 12.5))
        times = iter((1.0, 1.02))
        def clock():
            calls.append("wall clock")
            return next(times)
        with patch("gpe.benchmark.perf_counter", clock):
            wall, gpu = measure(cp, lambda: calls.append("work"))
        self.assertAlmostEqual(wall, 0.02)
        self.assertEqual(gpu, 12.5)
        self.assertEqual(calls, ["drain stream", "wall clock", "start event", "work",
                                 "end event", "wait for end", "wall clock"])

    def test_repeats_reset_state_and_report_per_step_times(self):
        stage = SimpleNamespace(name="imaginary_none", N_grids=2, x_array=np.arange(2.),
                                initial_state={"type": "gaussian", "sigma1": 10, "sigma2": 10})
        stage.prepare = lambda: [setattr(stage, key, np.ones((2, 2, 2), dtype=np.complex128)) for key in ("psi1", "psi2")]
        stage.release = Mock()
        calls = []
        def launch(item):
            calls.append(item.steps)
            # The first step and each timed repeat must start from the same input.
            if item.steps != 2:
                np.testing.assert_array_equal(item.psi1, np.ones((2, 2, 2)))
            item.psi1 += 1
        def gaussian(x, sigma):
            self.assertIsNone(stage.psi2)
            self.assertEqual(sigma, 10)
            return np.ones((2, 2, 2), dtype=np.complex128)
        timings = iter(((0.02, 19), (0.01, 9), (0.008, 7), (0.012, 11), (0.016, 15)))
        def fake_measure(cp, function):
            function()
            return next(timings)
        cp = SimpleNamespace(isfinite=np.isfinite, cuda=SimpleNamespace(
            get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None)))
        modules = {"gpe.launch": SimpleNamespace(launch_stage=launch),
                   "gpe.core.initial_states": SimpleNamespace(gaussian=gaussian),
                   "gpe.core.helper": SimpleNamespace(get_N=lambda psi, x: np.sum(np.abs(psi)**2))}
        with patch.dict(sys.modules, modules), patch("gpe.benchmark.measure", fake_measure):
            row = benchmark_stage(cp, stage, steps=4, warmup=2, repeats=3)
        self.assertEqual(calls, [1, 2, 4, 4, 4])
        stage.release.assert_called_once()
        self.assertEqual(row["setup_ms"], 20)
        self.assertEqual(row["first_step_ms"], 10)
        self.assertEqual(row["wall_ms_per_step"], 3)
        self.assertEqual(row["wall_min_ms_per_step"], 2)
        self.assertEqual(row["wall_max_ms_per_step"], 4)
        self.assertEqual(row["cuda_ms_per_step"], 2.75)
        self.assertAlmostEqual(row["steps_per_second"], 1000/3)
        self.assertEqual(len(row["samples"]), 3)

    def test_failed_setup_releases_stage(self):
        stage = SimpleNamespace(prepare=Mock(side_effect=RuntimeError("allocation failed")), release=Mock())
        modules = {"gpe.launch": SimpleNamespace(launch_stage=Mock()),
                   "gpe.core.initial_states": SimpleNamespace(gaussian=Mock()),
                   "gpe.core.helper": SimpleNamespace(get_N=Mock())}
        with patch.dict(sys.modules, modules), patch("gpe.benchmark.measure", lambda cp, fn: fn()):
            with self.assertRaisesRegex(RuntimeError, "allocation failed"):
                benchmark_stage(None, stage, 4, 2, 3)
        stage.release.assert_called_once()

    def test_report_preserves_samples_and_marks_partial_results(self):
        metadata = dict(expected_cases=10, started_utc="test", gpu="CPU test placeholder", gpu_memory_gib=0,
                        python="test", cupy="test", cuda_runtime=0, cuda_driver=0,
                        options=dict(steps=4, warmup=2, repeats=3, dt=0.002))
        row = dict(grid=2, variant=VARIANTS[0][0], wall_ms_per_step=3., wall_min_ms_per_step=2.,
                   wall_max_ms_per_step=4., cuda_ms_per_step=2.75, steps_per_second=1000/3,
                   setup_ms=20., first_step_ms=10., samples=[dict(wall_seconds=0.012)])
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            write_report(folder, metadata, [row])
            payload = json.loads((folder / "results.json").read_text())
            self.assertFalse(payload["complete"])
            self.assertEqual(payload["results"][0]["samples"], row["samples"])
            self.assertIn("Completed cases: 1/10", (folder / "report.md").read_text())
            self.assertIn("wall_ms_per_step", (folder / "summary.csv").read_text())


if __name__ == "__main__":
    unittest.main()
