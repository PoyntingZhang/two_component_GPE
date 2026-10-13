"""Read an experiment YAML file and return its ordered stage configurations."""

import copy
from datetime import datetime, timezone
from pathlib import Path
import re

from config.yaml_parser import parse_yaml as read_config
from .core.expressions import cuda_expression
from .core.grids import get_grids
from .stage import stage
from .io.drive_storage import check_storage


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_stages(path):
    path = Path(path).expanduser().resolve()
    config = read_config(path)
    simulation = config["simulation"]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    def output_root(value):
        folder = Path(value).expanduser()
        return folder if folder.is_absolute() else PROJECT_ROOT / folder

    data_root = output_root(simulation["data_root"])
    monitor_root = output_root(simulation["monitor_root"])
    check_storage(data_root)
    check_storage(monitor_root)
    x, k = get_grids(simulation["N_grids"], simulation["simulation_range"])
    stages = []
    data_dirs, monitor_dirs = set(), set()
    for index, settings in enumerate(config["stages"], 1):
        initial = copy.deepcopy(settings["initial_state"])
        if initial["type"] == "load":
            source = Path(initial["path"]).expanduser()
            initial["path"] = str(source if source.is_absolute() else path.parent / source)

        def active(group):
            values = dict(settings[group])
            values["activate"] = values.pop("enabled")
            return values

        potentials = {}
        for group in ("potential1", "potential2"):
            potentials[group] = {}
            for kind, values in settings[group].items():
                potentials[group][kind] = dict(activate=values["enabled"], expression=cuda_expression(
                    values["expression"] if values["enabled"] else "0", time_dependent=(kind == "dynamic")))
        noise = active("noise")
        noise["fraction"] = noise.pop("fraction_of_trap_depth")
        item = stage(simulation_range=simulation["simulation_range"], x_array=x, k_array=k,
                     N_grids=simulation["N_grids"], **settings["evolution"],
                     initial_state=initial, box_params=active("box"), absorber_params=active("absorber"),
                     noise_params=noise, cloud_params=settings["cloud"], monitor_params=settings["monitor"],
                     save_params=settings["save"], potential1_params=potentials["potential1"], potential2_params=potentials["potential2"],
                     k_cut_params=dict(activate=settings["k_cut"]["enabled"], k_cut=settings["k_cut"]["value"]))
        item.name = settings["name"].strip() or f"Stage {index}"
        item.config = copy.deepcopy(settings)
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", settings["name"].strip()).strip("_")
        default_folder = f"stage_{index:02d}" + (f"_{slug}" if slug else "")
        folder = Path(settings["save"]["directory"] or default_folder).expanduser()
        item.data_dir = folder / run_id if folder.is_absolute() else data_root / run_id / folder
        item.monitor_dir = monitor_root / run_id / (default_folder if folder.is_absolute() else folder)
        if not folder.is_absolute():
            if not (item.data_dir.resolve().is_relative_to((data_root / run_id).resolve()) and
                    item.monitor_dir.resolve().is_relative_to((monitor_root / run_id).resolve())):
                raise ValueError("Relative save.directory must stay inside the run folder; use an absolute path for another location.")
        if item.data_dir.resolve() in data_dirs or item.monitor_dir.resolve() in monitor_dirs:
            raise ValueError(f"Stage {index} reuses another stage's output directory; choose a distinct save.directory.")
        data_dirs.add(item.data_dir.resolve())
        monitor_dirs.add(item.monitor_dir.resolve())
        stages.append(item)
    return stages
