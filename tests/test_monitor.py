"""Saved-time selection, plotting units and monitor history, without a GPU."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from matplotlib.figure import Figure
import numpy as np

from gui_periphery.monitor_plots import (
    BEC, PHASE, ENERGY_NK_39K, image_plot, render_plot, scalar_value, spectrum_plot,
)
from gui_periphery.viewer import read_monitor
from gui_periphery.workflow import MONITOR_FILES


class MonitorTests(unittest.TestCase):
    def test_each_panel_selects_its_own_saved_time_within_chosen_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("stage_a", "stage_b"):
                stage = root / "monitor_data" / name
                stage.mkdir(parents=True)
                for step in (10, 20, 30):
                    np.savez(stage / f"nk_{step:08d}.npz", step=step, time=step/10,
                             k=[0, 1, 2], nk=[step, 2, 1], complete_shells=[True, True, False])
                    np.savez(stage / f"density_xy_{step:08d}.npz", step=step, time=step/10,
                             x=[-1, 0, 1], y=[-1, 0, 1], values=np.full((3, 3), step))
            result = read_monitor(root, "stage_a", "nk_00000010.npz", image_name="density_xy_00000020.npz",
                                  radial_name="nk_00000030.npz")
            self.assertEqual(result["stage"], "stage_a")
            self.assertEqual(result["spectrum"]["step"], 10)
            self.assertEqual(result["radial"]["step"], 30)
            self.assertEqual(result["image"]["step"], 20)
            self.assertEqual(len(result["image_samples"]), 3)
            # A pinned file never silently substitutes a different time.
            (root / "monitor_data/stage_a/density_xy_00000020.npz").write_bytes(b"partial")
            pinned = read_monitor(root, "stage_a", image_name="density_xy_00000020.npz")
            self.assertIsNone(pinned["image"])
            self.assertTrue(pinned["warnings"])

    def test_radial_curve_is_four_pi_k_squared_nk_without_atom_number(self):
        data = dict(k=[0, .1, 1, 2], nk=[4, 3, 2, 1], complete=[True, True, True, False])
        fig = Figure(); log, linear = fig.subplots(1, 2)
        spectrum_plot(log, data, False, {"k_max": 5})
        spectrum_plot(linear, data, True, {"k_max": 5})
        self.assertEqual((log.get_xscale(), log.get_yscale()), ("log", "log"))
        self.assertEqual((linear.get_xscale(), linear.get_yscale()), ("linear", "linear"))
        self.assertEqual(log.get_xlim(), (.01, 5))
        np.testing.assert_allclose(linear.lines[0].get_ydata(), 4*np.pi*np.array(data["k"])**2*data["nk"])
        self.assertTrue(np.isnan(linear.lines[1].get_ydata()[-1]))
        np.testing.assert_allclose(linear.get_xticks(), np.arange(0, 5.01, .5))

    def test_image_orientation_fixed_colour_limits_and_phase_colours(self):
        fig = Figure(); ax = fig.subplots()
        data = dict(kind="TOF_xy", axes=["kx", "ky"], x=[-1, 0, 1], y=[-1, 1],
                    values=[[1, 2], [3, 4], [5, 6]])
        image_plot(fig, ax, data, dict(log=True, vmin=.1, vmax=10, image_limit=5))
        mesh = ax.collections[0]
        self.assertEqual((mesh.norm.vmin, mesh.norm.vmax), (.1, 10))
        np.testing.assert_equal(mesh.get_array().reshape(2, 3), np.array(data["values"]).T)
        np.testing.assert_allclose(mesh.cmap(0.), BEC(0.))
        self.assertEqual(ax.get_xlim(), (-5, 5))
        data.update(kind="phase_xy", axes=["x", "y"], values=[[-2, -1], [0, 1], [2, 3]])
        phase = Figure(); phase_ax = phase.subplots()
        image_plot(phase, phase_ax, data, {"log": True})
        self.assertEqual(phase_ax.collections[0].norm.vmin, -2)
        np.testing.assert_allclose(phase_ax.collections[0].cmap(.5), PHASE(.5))
        with self.assertRaisesRegex(ValueError, "vmin"):
            image_plot(Figure(), Figure().subplots(), dict(data, kind="density_xy"), dict(log=True, vmin=0))

    def test_mean_angular_momentum_and_energy_unit(self):
        row = dict(norm=.5, E_kin=2, Lz=3)
        self.assertEqual(scalar_value(row, "Lz"), 6)
        self.assertEqual(scalar_value(row, "E_kin"), 2)  # Preserve the saved integral.
        self.assertEqual(scalar_value(row, "norm"), .5)
        self.assertTrue(np.isnan(scalar_value(dict(norm=0, Lz=0), "Lz")))
        hbar = 6.62607015e-34 / (2 * np.pi)
        expected = hbar**2 / (2 * 38.9637064864 * 1.66053906892e-27 * 1e-12 * 1.380649e-23) * 1e9
        self.assertAlmostEqual(ENERGY_NK_39K, expected, places=10)

    def test_svg_math_labels_and_empty_view_are_self_contained(self):
        data = dict(k=[0, .1, 1, 2], nk=[4, 3, 2, 1], complete=[True]*4)
        svg = render_plot("radial", data)
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn(r"4\pi k^2 n(k)", svg)
        self.assertNotIn("<script", svg)
        self.assertIn("No saved data", render_plot("image", None))

    def test_remote_manifest_includes_history_but_never_wavefunctions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = root / "monitor_data/stage_a"; stage.mkdir(parents=True)
            for name in ("nk_00000010.npz", "nk_00000020.npz", "TOF_xy_00000010.npz", "scalars.csv"):
                (stage / name).write_bytes(b"fixture")
            saved = root / "saved_data"; saved.mkdir()
            (saved / "psi_00000020.npy").write_bytes(b"never transfer")
            result = json.loads(subprocess.check_output([sys.executable, "-c", MONITOR_FILES, directory]))
            self.assertEqual(len(result), 4)
            self.assertIn("monitor_data/stage_a/nk_00000010.npz", result)
            self.assertTrue(all(name.startswith("monitor_data/") for name in result))


if __name__ == "__main__":
    unittest.main()
