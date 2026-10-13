import copy
from pathlib import Path
import tempfile
import unittest

from config.yaml_parser import ConfigError, dump_yaml, load_yaml, new_config, new_stage, normalize_config, save_yaml


class ConfigurationTests(unittest.TestCase):
    def test_roundtrip_preserves_custom_potential_and_stage_settings(self):
        config = new_config()
        real = new_stage()
        real["name"] = "No"  # YAML must preserve strings resembling booleans.
        real["potential1"]["static"].update(enabled=True, expression="0.03*x")
        real["potential1"]["dynamic"].update(enabled=True, expression="0.125*cos(0.065*t)*y**2")
        real["noise"].update(enabled=True, seed=12345, fraction_of_trap_depth=0.01)
        real["save"].update(save_initial=True, psi_interval=10000, directory="results with spaces")
        config["stages"].append(real)
        text = dump_yaml(config)
        self.assertEqual(load_yaml(text), config)
        self.assertNotIn("precision:", text)

    def test_first_stage_cannot_continue(self):
        config = new_config()
        config["stages"][0]["initial_state"]["type"] = "continue"
        with self.assertRaises(ConfigError) as result:
            normalize_config(config)
        self.assertEqual(result.exception.errors[0]["path"], "stages.0.initial_state.type")

    def test_reordering_requires_an_initial_state_at_the_front(self):
        config = new_config()
        config["stages"].insert(0, new_stage())
        with self.assertRaises(ConfigError):
            dump_yaml(config)
        config["stages"][0]["initial_state"].update(type="load", path="/rds/project/state.npy")
        self.assertEqual(load_yaml(dump_yaml(config)), config)

    def test_noise_tracks_stage_depth_without_changing_seed(self):
        config = new_config()
        config["stages"][0]["noise"].update(enabled=True, seed=31)
        second = copy.deepcopy(config["stages"][0])
        second["box"]["trap_depth"] = 80
        config["stages"].append(second)
        loaded = load_yaml(dump_yaml(config))
        self.assertEqual(loaded["stages"][0]["noise"], loaded["stages"][1]["noise"])
        self.assertEqual(loaded["stages"][1]["box"]["trap_depth"], 80)

    def test_disabled_box_with_trap_noise_is_explained(self):
        config = new_config()
        config["stages"][0]["box"]["enabled"] = False
        config["stages"][0]["noise"]["enabled"] = True
        with self.assertRaisesRegex(ConfigError, "Trap-dependent noise needs"):
            normalize_config(config)

    def test_correlated_noise_is_rejected_before_running(self):
        config = new_config()
        config["stages"][0]["noise"].update(enabled=True, correlation_length=1)
        with self.assertRaisesRegex(ConfigError, "Correlated disorder is not implemented"):
            normalize_config(config)

    def test_continued_state_keeps_reference_atom_number(self):
        config = new_config()
        second = new_stage()
        second["cloud"]["N1"] += 1
        config["stages"].append(second)
        with self.assertRaisesRegex(ConfigError, "reference N1"):
            normalize_config(config)
        second["initial_state"]["type"] = "gaussian"
        normalize_config(config)  # A newly generated cloud can use a new reference.

    def test_missing_load_path_and_invalid_numbers(self):
        config = new_config()
        config["stages"][0]["initial_state"]["type"] = "load"
        config["stages"][0]["evolution"]["steps"] = 1.5
        config["simulation"]["N_grids"] = True
        config["stages"][0]["evolution"]["dt"] = float("nan")
        with self.assertRaises(ConfigError) as result:
            normalize_config(config)
        paths = {item["path"] for item in result.exception.errors}
        self.assertTrue({"stages.0.initial_state.path", "stages.0.evolution.steps",
                         "simulation.N_grids", "stages.0.evolution.dt"} <= paths)

    def test_time_dependence_requires_real_evolution(self):
        config = new_config()
        config["stages"][0]["potential1"]["dynamic"].update(enabled=True, expression="cos(t)*y")
        with self.assertRaisesRegex(ConfigError, "Imaginary time evolution does not accept a time dependent potential"):
            normalize_config(config)

    def test_imaginary_absorber_is_rejected_on_import(self):
        with self.assertRaisesRegex(ConfigError, "does not accept an absorber"):
            load_yaml("schema_version: 3\nstages: [{absorber: {enabled: true}}]")

    def test_zero_disables_potential_and_disabled_formulas_are_cleared(self):
        config = new_config()
        config["stages"][0]["potential1"]["static"].update(enabled=True, expression="(-0.0)")
        result = normalize_config(config)
        self.assertFalse(result["stages"][0]["potential1"]["static"]["enabled"])
        config["stages"][0]["potential1"]["static"].update(enabled=False, expression="0.03*y")
        self.assertEqual(load_yaml(dump_yaml(config))["stages"][0]["potential1"]["static"],
                         {"enabled": False, "expression": "0"})

    def test_invalid_formulas_cannot_be_exported_or_imported(self):
        for formula in ("A*y", "t*x", "sin(", "__import__('os')"):
            config = new_config()
            config["stages"][0]["potential1"]["static"].update(enabled=True, expression=formula)
            with self.subTest(formula=formula), self.assertRaises(ConfigError) as result:
                dump_yaml(config)
            self.assertEqual(result.exception.errors[0]["path"], "stages.0.potential1.static.expression")
        with self.assertRaisesRegex(ConfigError, "Unrecognised variable A"):
            load_yaml('schema_version: 3\nstages: [{potential1: {dynamic: {enabled: true, expression: "A*y"}}}]')

    def test_legacy_output_root_and_checkpoint_import_as_snapshots(self):
        result = load_yaml("schema_version: 3\nsimulation: {output_root: results}\n"
                           "stages: [{save: {checkpoint_interval: 50}}]")
        self.assertEqual(result["simulation"]["data_root"], "results/data")
        self.assertEqual(result["simulation"]["monitor_root"], "results/monitor")
        self.assertEqual(result["stages"][0]["save"]["psi_interval"], 50)
        self.assertNotIn("checkpoint_interval", result["stages"][0]["save"])
        with self.assertRaisesRegex(ConfigError, "Choose one wavefunction snapshot interval"):
            load_yaml("schema_version: 3\nstages: [{save: {checkpoint_interval: 50, psi_interval: 20}}]")

    def test_cutoff_is_positive_and_real_time_only(self):
        config = new_config()
        config["stages"][0]["k_cut"].update(enabled=True, value=2)
        with self.assertRaisesRegex(ConfigError, "only for real time"):
            normalize_config(config)
        config["stages"][0]["evolution"]["type"] = "real"
        self.assertEqual(load_yaml(dump_yaml(config)), config)
        config["stages"][0]["k_cut"]["value"] = 0
        with self.assertRaisesRegex(ConfigError, "positive"):
            normalize_config(config)

    def test_old_format_and_noise_region_are_not_silently_accepted(self):
        with self.assertRaisesRegex(ConfigError, "schema_version: 3"):
            load_yaml("schema_version: 1\nstages: [{}]")
        with self.assertRaisesRegex(ConfigError, "Unknown setting: stages.0.noise.region"):
            load_yaml("schema_version: 3\nstages: [{noise: {region: inside_box}}]")

    def test_unsafe_tags_duplicates_and_unknown_fields_are_rejected(self):
        for source in ("!!python/object/apply:os.system ['echo should-not-run']",
                       "schema_version: 1\nschema_version: 3\n",
                       "schema_version: 3\nstages: [{}]\nsimulation: {precision: single}\n"):
            with self.subTest(source=source), self.assertRaises(ConfigError):
                load_yaml(source)

    def test_malformed_groups_report_errors_instead_of_crashing(self):
        for key in ("initial_state", "box", "evolution", "potential1", "monitor"):
            config = new_config()
            config["stages"][0][key] = None
            with self.subTest(key=key), self.assertRaises(ConfigError):
                normalize_config(config)

    def test_save_requires_explicit_overwrite_and_preserves_valid_yaml(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "experiment.yaml"
            config = new_config()
            save_yaml(config, path)
            config["simulation"]["N_grids"] = 128
            with self.assertRaises(FileExistsError):
                save_yaml(config, path)
            self.assertNotEqual(load_yaml(path.read_text())["simulation"]["N_grids"], 128)
            save_yaml(config, path, overwrite=True)
            self.assertEqual(load_yaml(path.read_text()), config)

    def test_invalid_save_does_not_truncate_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "experiment.yaml"
            config = new_config()
            save_yaml(config, path)
            before = path.read_bytes()
            config["stages"][0]["evolution"]["dt"] = -1
            with self.assertRaises(ConfigError):
                save_yaml(config, path, overwrite=True)
            self.assertEqual(path.read_bytes(), before)

    def test_example_is_complete_and_roundtrips(self):
        path = Path(__file__).resolve().parents[1] / "config" / "examples" / "kick_and_hold.yaml"
        config = load_yaml(path.read_text())
        self.assertEqual(len(config["stages"]), 3)
        self.assertEqual(load_yaml(dump_yaml(config)), config)
        self.assertEqual(config["stages"][1]["potential1"]["static"]["expression"], "0.24*y")


if __name__ == "__main__":
    unittest.main()
