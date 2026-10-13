"""Portable runtime and local editor boundaries, without a CUDA installation."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from config.yaml_parser import dump_yaml, new_config, parse_yaml
from gui_periphery.prepare_run import prepare_run


class RunBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        guard=patch("gui_periphery.prepare_run.require_drive_path", side_effect=lambda path: Path(path).resolve())
        guard.start(); self.addCleanup(guard.stop)
        self.config = new_config()
        self.path = self.root / "original.yaml"
        self.write_config()

    def write_config(self):
        self.path.write_text(dump_yaml(self.config))

    def build(self, **kwargs):
        return prepare_run(self.path, self.root / "runs", "simulation_trial", **kwargs)

    def test_bundle_preserves_stage_settings_and_original_yaml(self):
        original = self.path.read_bytes()
        folder = self.build(notes="Study a rotating cloud. Save spectra and final states.")
        loaded = parse_yaml(folder / "config/run.yaml")
        self.assertEqual(loaded["stages"], self.config["stages"])
        self.assertEqual(loaded["simulation"]["N_grids"], self.config["simulation"]["N_grids"])
        self.assertEqual(loaded["simulation"]["simulation_range"], self.config["simulation"]["simulation_range"])
        self.assertEqual(loaded["simulation"]["data_root"], "data")
        self.assertEqual(loaded["simulation"]["monitor_root"], "monitor")
        self.assertEqual(self.path.read_bytes(), original)
        self.assertTrue((folder / "notes.txt").read_text().startswith("Study a rotating cloud."))
        for name in ("data", "monitor", "logs"):
            self.assertTrue((folder / name).is_dir())
        for name in ("gui_periphery", "config_editor", "tests", ".git", ".venv"):
            self.assertFalse((folder / name).exists())
        self.assertFalse((folder / "gpe/benchmark.py").exists())
        self.assertEqual((folder / ".gitignore").read_text(), "*\n")
        manifest = json.loads((folder / "run.json").read_text())
        for name, expected in manifest["files_sha256"].items():
            self.assertEqual(hashlib.sha256((folder / name).read_bytes()).hexdigest(), expected)

    def test_isolated_check_and_runtime_imports_without_gui(self):
        folder = self.build()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        env.pop("PYTHONPATH", None)
        result = subprocess.run([sys.executable, "-m", "gpe.main", "config/run.yaml", "--check"],
                                cwd=folder, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Valid configuration", result.stdout)
        # All runtime modules must be present; CUDA calls themselves are tested
        # by test_propagators on a GPU. Importing must never require a GUI.
        script = '''
import builtins
import sys
from types import SimpleNamespace
import numpy as np
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.split('.')[0] in {'gui_periphery', 'config_editor', 'matplotlib'}:
        raise AssertionError('Runtime imported a GUI dependency: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
from config.yaml_parser import parse_yaml
parse_yaml('config/run.yaml')
assert 'cupy' not in sys.modules
kernel = lambda *args, **kwargs: None
sys.modules['cupy'] = SimpleNamespace(ndarray=np.ndarray, RawKernel=kernel,
    ReductionKernel=kernel, ElementwiseKernel=kernel)
sys.modules['cupyx'] = SimpleNamespace()
sys.modules['cupyx.scipy'] = SimpleNamespace()
sys.modules['cupyx.scipy.fft'] = SimpleNamespace(get_fft_plan=kernel, fftn=kernel, ifftn=kernel)
import gpe.stage_factory
import gpe.launch
'''
        result = subprocess.run([sys.executable, "-c", script], cwd=folder, env=env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_existing_run_is_not_overwritten(self):
        folder = self.build()
        (folder / "notes.txt").write_text("Keep this run.")
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual((folder / "notes.txt").read_text(), "Keep this run.")

    @unittest.skipIf(os.name == "nt", "Generated launch scripts target Bash on Linux/macOS")
    def test_launch_script_logs_validation_and_preserves_failure_status(self):
        folder = self.build()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
        env.pop("PYTHONPATH", None)
        result = subprocess.run(["bash", "run.sh", "--check"], cwd=folder, env=env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Valid configuration", (folder / "logs/run.log").read_text())
        (folder / "config/run.yaml").write_text("stages: []\n")
        result = subprocess.run(["bash", "run.sh", "--check"], cwd=folder, env=env,
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Add at least one stage", (folder / "logs/run.log").read_text())

    def test_snapshot_reference_is_explicit_and_never_copied(self):
        self.config["stages"][0]["initial_state"]["component1"].update(type="load", path="previous/psi1_00000010.npy")
        self.write_config()
        with self.assertRaisesRegex(ValueError, "absolute target snapshot"):
            self.build()
        folder = self.build(input_paths={(1, 1): "/rds/project/previous/psi1_00000010.npy"})
        initial = parse_yaml(folder / "config/run.yaml")["stages"][0]["initial_state"]
        self.assertEqual(initial["component1"]["path"], "/rds/project/previous/psi1_00000010.npy")
        self.assertEqual(initial["component2"]["type"], "gaussian")
        self.assertEqual(json.loads((folder / 'run.json').read_text())['external_snapshots'],
                         ["/rds/project/previous/psi1_00000010.npy"])
        self.assertFalse(list(folder.rglob("*.npy")))
        self.assertEqual(parse_yaml(self.path)["stages"][0]["initial_state"]["component1"]["path"], "previous/psi1_00000010.npy")

    def test_legacy_pair_path_override_still_sets_both_components(self):
        self.config["stages"][0]["initial_state"] = dict(type="load", path="previous/psi_00000010.json")
        self.write_config()
        folder = self.build(input_paths={1: "/Drive/previous/psi_00000010.json"})
        initial = parse_yaml(folder / "config/run.yaml")["stages"][0]["initial_state"]
        self.assertEqual(initial["component1"]["path"], "/Drive/previous/psi_00000010.json")
        self.assertEqual(initial["component2"]["path"], initial["component1"]["path"])

    def test_run_outputs_cannot_escape_the_named_folder(self):
        with self.assertRaisesRegex(ValueError, "folder name"):
            prepare_run(self.path, self.root / "runs", "../outside")
        self.config["stages"][0]["save"]["directory"] = "../../outside"
        self.write_config()
        with self.assertRaisesRegex(ValueError, "relative save.directory"):
            self.build()


if __name__ == "__main__":
    unittest.main()
