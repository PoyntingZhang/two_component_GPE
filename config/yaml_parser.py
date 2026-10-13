"""CPU-only YAML parsing and validation shared by the runner and local editor."""

import copy
import math
from pathlib import Path

import yaml

from . import defaults
from .expressions import is_zero, parse_expression

MONITOR_FIELDS = (
    "print_scalar_interval", "show_nk_interval", "show_TOF_image_interval",
    "show_verticle_TOF_interval", "show_TOF_slice_interval",
    "show_verticle_TOF_slice_interval", "show_magnifier_interval",
    "show_magnifier_slice_interval", "show_phase_interval",
)
SAVE_INTERVALS = ("scalar_interval", "nk_interval", "psi_interval")


def project_defaults():
    names = ("N_grids", "simulation_range", "R", "L", "U_D", "wallthickness",
             "r_abs", "A_abs", "a", "N_atom", "dt")
    return {name: getattr(defaults, name) for name in names}


def new_stage(imaginary=False):
    cfg = project_defaults()
    monitor = dict.fromkeys(MONITOR_FIELDS, 0)
    monitor["print_scalar_interval"] = 1000
    return {
        "name": "",
        "evolution": {"type": "imaginary" if imaginary else "real", "steps": 2000, "dt": cfg["dt"]},
        "initial_state": {"type": "gaussian" if imaginary else "continue", "sigma1": 3.0, "sigma2": 3.0, "path": ""},
        "potential1": {
            "static": {"enabled": False, "expression": "0"},
            "dynamic": {"enabled": False, "expression": "0"},
        },
        "potential2": {
            "static": {"enabled": False, "expression": "0"},
            "dynamic": {"enabled": False, "expression": "0"},
        },
        "k_cut": {"enabled": False, "value": 0.0},
        "box": {"enabled": True, "R": cfg["R"], "L": cfg["L"],
                "trap_depth": cfg["U_D"], "wall_thickness": cfg["wallthickness"]},
        "absorber": {"enabled": not imaginary, "activation_distance": cfg["r_abs"], "strength": cfg["A_abs"]},
        "noise": {"enabled": False, "fraction_of_trap_depth": 0.01, "seed": 42,
                  "correlation_length": 0.0},
        "cloud": {"a11": cfg["a"], "a22": cfg["a"], "a12": 0.0,
                  "N1": cfg["N_atom"], "N2": cfg["N_atom"]},
        "monitor": monitor,
        "save": {"directory": "", "scalar_interval": 1000, "nk_interval": 0,
                 "psi_interval": 0,
                 "save_initial": False, "save_final": True},
    }


def new_config():
    cfg = project_defaults()
    stage = new_stage(imaginary=True)
    stage["name"] = "Preparation"
    return {"schema_version": 3,
            "simulation": {"simulation_range": cfg["simulation_range"],
                           "N_grids": cfg["N_grids"], "data_root": "data", "monitor_root": "monitor"},
            "stages": [stage]}


class ConfigError(ValueError):
    def __init__(self, errors):
        self.errors = errors
        super().__init__("; ".join(item["message"] for item in errors))


def normalize_config(raw):
    """Fill optional settings, reject unknown fields, and validate physical inputs."""
    errors = []

    def error(path, message):
        errors.append({"path": path, "message": message})

    def merge(value, default, path):
        if not isinstance(value, dict):
            error(path, f"{path or 'Configuration'} must be a mapping.")
            return copy.deepcopy(default)
        result = copy.deepcopy(default)
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if key not in default:
                error(child, f"Unknown setting: {child}.")
            elif isinstance(default[key], dict):
                result[key] = merge(item, default[key], child)
            else:
                result[key] = copy.deepcopy(item)
        return result

    if not isinstance(raw, dict):
        raise ConfigError([{"path": "", "message": "Configuration must be a YAML mapping."}])
    raw = copy.deepcopy(raw)
    # Older schema-2 files used one output root for everything.
    simulation = raw.get("simulation", {})
    if isinstance(simulation, dict) and "output_root" in simulation:
        root = simulation.pop("output_root")
        if not isinstance(root, str) or not root.strip():
            raise ConfigError([{"path": "simulation.output_root", "message": "Output root must be non-empty text."}])
        simulation.setdefault("data_root", str(Path(root) / "data"))
        simulation.setdefault("monitor_root", str(Path(root) / "monitor"))
    base = new_config()
    config = merge(raw, base, "")
    if type(config["schema_version"]) is not int or config["schema_version"] != 3:
        error("schema_version", "Use schema_version: 3 for two components; single-component configurations need explicit conversion.")

    def number(container, key, path, minimum=None, positive=False, integer=False):
        value = container[key]
        valid = type(value) in (int, float) and math.isfinite(value)
        if integer:
            valid = valid and type(value) is int and abs(value) <= 2**53 - 1
        if not valid or (positive and value <= 0) or (minimum is not None and value < minimum):
            qualifier = "positive " if positive else "non-negative " if minimum == 0 else ""
            error(f"{path}.{key}", f"{path}.{key} must be a {qualifier}{'integer' if integer else 'finite number'}" +
                  (f" ≥ {minimum}." if minimum not in (None, 0) else "."))

    def text(container, key, path, required=False):
        value = container[key]
        if not isinstance(value, str) or (required and not value.strip()):
            error(f"{path}.{key}", f"{path}.{key} must be {'non-empty ' if required else ''}text.")

    def choice(container, key, path, choices):
        if container[key] not in choices:
            error(f"{path}.{key}", f"{path}.{key} must be one of: {', '.join(choices)}.")

    def boolean(container, key, path):
        if type(container[key]) is not bool:
            error(f"{path}.{key}", f"{path}.{key} must be true or false.")

    sim = config["simulation"]
    number(sim, "N_grids", "simulation", minimum=2, integer=True)
    number(sim, "simulation_range", "simulation", positive=True)
    text(sim, "data_root", "simulation", required=True)
    text(sim, "monitor_root", "simulation", required=True)
    stages = raw.get("stages")
    if not isinstance(stages, list) or not stages:
        error("stages", "Add at least one stage.")
        stages = []
    config["stages"] = []
    for index, raw_stage in enumerate(stages):
        path = f"stages.{index}"
        # The old checkpoint field never had restart support. Import it as a snapshot cadence.
        if isinstance(raw_stage, dict) and isinstance(raw_stage.get("save"), dict):
            old_save = raw_stage["save"]
            checkpoint_interval = old_save.pop("checkpoint_interval", 0)
            if checkpoint_interval:
                if old_save.get("psi_interval", 0) not in (0, checkpoint_interval):
                    error(f"{path}.save.psi_interval", "Choose one wavefunction snapshot interval; restart checkpoints are not supported.")
                else:
                    old_save["psi_interval"] = checkpoint_interval
        stage = merge(raw_stage, new_stage(imaginary=index == 0), path)
        config["stages"].append(stage)
        text(stage, "name", path)
        evo = stage["evolution"]
        choice(evo, "type", f"{path}.evolution", ("imaginary", "real"))
        number(evo, "steps", f"{path}.evolution", positive=True, integer=True)
        number(evo, "dt", f"{path}.evolution", positive=True)
        initial = stage["initial_state"]
        choice(initial, "type", f"{path}.initial_state", ("gaussian", "load", "continue"))
        text(initial, "path", f"{path}.initial_state", required=initial["type"] == "load")
        for key in ("sigma1", "sigma2"):
            number(initial, key, f"{path}.initial_state", positive=True)
        if index == 0 and initial["type"] == "continue":
            error(f"{path}.initial_state.type", "Stage 1 needs a Gaussian or a loaded state; there is no previous stage.")
        for group in ("potential1", "potential2"):
            potential = stage[group]
            for kind in ("static", "dynamic"):
                component = potential[kind]
                component_path = f"{path}.{group}.{kind}"
                boolean(component, "enabled", component_path)
                if component["enabled"] is False:
                    component["expression"] = "0"
                try:
                    expression = parse_expression(component["expression"], time_dependent=kind == "dynamic")
                    if is_zero(expression):
                        component["enabled"] = False
                        component["expression"] = "0"
                except ValueError as exc:
                    error(f"{component_path}.expression", str(exc))
            if evo["type"] == "imaginary" and potential["dynamic"]["enabled"]:
                error(f"{path}.{group}.dynamic.enabled", "Imaginary time evolution does not accept a time dependent potential.")
        cutoff = stage["k_cut"]
        boolean(cutoff, "enabled", f"{path}.k_cut")
        number(cutoff, "value", f"{path}.k_cut", minimum=0, positive=cutoff["enabled"] is True)
        if evo["type"] == "imaginary" and cutoff["enabled"]:
            error(f"{path}.k_cut.enabled", "The momentum cutoff is available only for real time evolution.")
        for group in ("box", "absorber", "noise"):
            boolean(stage[group], "enabled", f"{path}.{group}")
        if evo["type"] == "imaginary" and stage["absorber"]["enabled"]:
            error(f"{path}.absorber.enabled", "Imaginary time evolution does not accept an absorber.")
        for key in ("R", "L", "wall_thickness"):
            number(stage["box"], key, f"{path}.box", positive=True)
        number(stage["box"], "trap_depth", f"{path}.box", minimum=0)
        for key in ("activation_distance", "strength"):
            number(stage["absorber"], key, f"{path}.absorber", minimum=0)
        for key in ("fraction_of_trap_depth", "correlation_length"):
            number(stage["noise"], key, f"{path}.noise", minimum=0)
        number(stage["noise"], "seed", f"{path}.noise", minimum=0, integer=True)
        if stage["noise"]["enabled"] and not stage["box"]["enabled"]:
            error(f"{path}.noise.enabled", "Trap-dependent noise needs an enabled box to define its depth.")
        if stage["noise"]["enabled"] and stage["noise"]["correlation_length"] != 0:
            error(f"{path}.noise.correlation_length", "Correlated disorder is not implemented; use correlation_length: 0.")
        for key in ("a11", "a22", "a12"):
            number(stage["cloud"], key, f"{path}.cloud")
        for key in ("N1", "N2"):
            number(stage["cloud"], key, f"{path}.cloud", positive=True, integer=True)
            if index > 0 and initial["type"] == "continue" and stage["cloud"][key] != config["stages"][index - 1]["cloud"][key]:
                error(f"{path}.cloud.{key}", f"A continued state must keep the previous stage's reference {key}. Change scattering lengths to adjust interactions.")
        for key in MONITOR_FIELDS:
            number(stage["monitor"], key, f"{path}.monitor", minimum=0, integer=True)
        for key in SAVE_INTERVALS:
            number(stage["save"], key, f"{path}.save", minimum=0, integer=True)
        for key in ("save_initial", "save_final"):
            boolean(stage["save"], key, f"{path}.save")
        text(stage["save"], "directory", f"{path}.save")
    if errors:
        raise ConfigError(errors)
    return config


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, (str, int, float, bool, type(None))):
            raise ValueError("YAML mapping keys must be scalar values.")
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def load_yaml(source):
    try:
        return normalize_config(yaml.load(source, Loader=UniqueLoader))
    except (yaml.YAMLError, ValueError, RecursionError) as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError([{"path": "", "message": f"Cannot read YAML: {exc}"}]) from exc


def dump_yaml(config):
    config = normalize_config(config)
    heading = ("# Two-component GPE experiment configuration — schema 3\n"
               "# Double precision. Lengths and times are dimensionless; a11, a22, a12 are in Bohr radii.\n"
               "# Every stage starts at t = 0. Intervals count completed steps; 0 disables.\n"
               "# Gaussian disorder covers the whole grid. Potential expressions use Python syntax.\n")
    return heading + yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=100)


def save_yaml(config, path, overwrite=False):
    destination = Path(path).expanduser().resolve()
    if destination.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("Choose a filename ending in .yaml or .yml.")
    document = dump_yaml(config)
    with destination.open("w" if overwrite else "x", encoding="utf-8") as file:
        file.write(document)
    return str(destination)


def parse_yaml(path):
    """Read and validate a YAML file without importing the GPU solver."""
    return load_yaml(Path(path).expanduser().read_text())
