"""Prepare a portable simulation folder without a GPU, SSH, or job submission."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import os
import shlex
from gpe.io.drive_storage import DEFAULT_OUTPUT, require_drive_path

from config.yaml_parser import dump_yaml, normalize_config, parse_yaml


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = Path(__file__).resolve().parent / "templates"


def runtime_sources():
    """Only the files needed by `python -m gpe.main`; no GUI or study tooling."""
    sources = [ROOT / "requirements.txt"]
    for name in ("__init__", "main", "launch", "stage", "stage_factory"):
        sources.append(ROOT / "gpe" / f"{name}.py")
    for folder in ("gpe/core", "config"):
        sources.extend(sorted((ROOT / folder).glob("*.py")))
    for name in ("__init__", "save", "state_io", "diagnostics", "nk_extraction", "atomic", "drive_storage"):
        sources.append(ROOT / "gpe/io" / f"{name}.py")
    return sources


def prepare_run(config_path, output=DEFAULT_OUTPUT, name=None, notes="", *,
                cupy_package="cupy-cuda12x", input_paths=None):
    """Copy runtime sources and a validated YAML into a new, named run folder.

    `input_paths` maps (stage, component) tuples to absolute target paths. A
    stage number alone overrides both components, for legacy pair manifests.
    Wavefunctions themselves are never copied by this function.
    """
    created = datetime.now(timezone.utc)
    name = name or created.strftime("simulation_%Y%m%dT%H%M%S%fZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError("Use letters, numbers, underscores, dots, or hyphens for the run folder name.")
    if cupy_package not in ("cupy-cuda12x", "cupy-cuda13x"):
        raise ValueError("Choose cupy-cuda12x or cupy-cuda13x to match the target CUDA environment.")
    config = normalize_config(config_path) if isinstance(config_path, dict) else parse_yaml(config_path)
    input_paths = input_paths or {}
    for key, path in input_paths.items():
        index, components = (key[0], (key[1],)) if isinstance(key, tuple) else (key, (1, 2))
        if not 1 <= index <= len(config["stages"]):
            raise ValueError(f"No stage {index} exists for --input-path.")
        for component in components:
            if component not in (1, 2):
                raise ValueError("The input component must be 1 or 2.")
            initial = config["stages"][index - 1]["initial_state"][f"component{component}"]
            if initial["type"] != "load":
                raise ValueError(f"Stage {index}, component {component} does not load a wavefunction.")
            initial["path"] = str(path)
    for index, stage in enumerate(config["stages"], 1):
        for component in (1, 2):
            initial = stage["initial_state"][f"component{component}"]
            if initial["type"] == "load" and not Path(initial["path"]).is_absolute():
                raise ValueError(f"Stage {index}, component {component}: give an absolute target snapshot path with --input-path {index}.{component}=/path/to/psi{component}_00001000.npy.")
        folder = Path(stage["save"]["directory"])
        if folder.is_absolute() or ".." in folder.parts:
            raise ValueError(f"Stage {index}: use a relative save.directory inside the run folder, or leave it blank.")
    config["simulation"].update(data_root="data", monitor_root="monitor")
    yaml_text = dump_yaml(config)

    destination = require_drive_path(output) / name
    destination.mkdir(parents=True, exist_ok=False)
    for path in runtime_sources():
        target = destination / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for path in (TEMPLATES / "run.sh",):
        shutil.copy2(path, destination / path.name)
    for folder in ("data", "monitor", "logs"):
        (destination / folder).mkdir()
    mount = os.environ.get("GPE_DRIVE_MOUNT", "/content/drive")
    (destination / "environment.sh").write_text("#!/usr/bin/env bash\nexport GPE_DRIVE_MOUNT=" + shlex.quote(mount) + "\n")
    (destination / "config/run.yaml").write_text(yaml_text)
    (destination / "notes.txt").write_text(notes.rstrip() + "\n")
    (destination / "requirements-gpu.txt").write_text(cupy_package + "\n")
    # Also ignore custom-named bundles placed outside the default runs/ folder.
    (destination / ".gitignore").write_text("*\n")
    hashes = {str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(destination.rglob("*")) if path.is_file()}
    manifest = dict(deployment="colab", name=name, created_utc=created.isoformat(), config="config/run.yaml",
                    cupy_package=cupy_package, files_sha256=hashes,
                    external_snapshots=[initial["path"] for stage in config["stages"]
                                        for initial in stage["initial_state"].values() if initial["type"] == "load"])
    (destination / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="YAML file made by the editor or written by hand")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Parent folder on mounted Google Drive")
    parser.add_argument("--name", help="Run folder name; defaults to simulation_<UTC timestamp>")
    notes = parser.add_mutually_exclusive_group()
    notes.add_argument("--notes", default="", help="Purpose and other comments to save in notes.txt")
    notes.add_argument("--notes-file", type=Path, help="Read the run description from this text file")
    parser.add_argument("--cupy-package", choices=("cupy-cuda12x", "cupy-cuda13x"), default="cupy-cuda12x")
    parser.add_argument("--input-path", action="append", default=[], metavar="STAGE.COMPONENT=/TARGET/PATH",
                        help="Absolute target NPY path for one component; STAGE=/manifest.json sets both. Repeat as needed.")
    args = parser.parse_args()
    try:
        paths = {}
        for value in args.input_path:
            index, path = value.split("=", 1)
            parts = tuple(int(part) for part in index.split("."))
            if len(parts) not in (1, 2):
                raise ValueError("Use STAGE.COMPONENT=/path/to/psi_STEP.npy or STAGE=/path/to/manifest.json.")
            paths[parts[0] if len(parts) == 1 else parts] = path
        description = args.notes_file.read_text() if args.notes_file else args.notes
        folder = prepare_run(args.config, args.output, args.name, description,
                             cupy_package=args.cupy_package, input_paths=paths)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Prepared: {folder}\nPrepared on Google Drive; no simulation launched.")
    print("After transfer and environment setup, run from that folder: bash run.sh")


if __name__ == "__main__":
    main()
