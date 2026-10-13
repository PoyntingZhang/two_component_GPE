"""Run an experiment: python -m gpe.main experiment.yaml"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Run the stages in a two-component GPE experiment YAML file.")
    parser.add_argument("config", help="Path to the experiment YAML file")
    parser.add_argument("--check", action="store_true", help="Validate YAML settings without loading CuPy or running a simulation")
    args = parser.parse_args()
    from config.yaml_parser import ConfigError, parse_yaml
    try:
        config = parse_yaml(args.config)
    except (OSError, ConfigError) as exc:
        parser.error(str(exc))
    if args.check:
        print(f"Valid configuration: {len(config['stages'])} stages. No simulation was run.")
        return

    try:
        from .launch import launch_stage
        from .stage_factory import make_stages
    except ModuleNotFoundError as exc:
        if exc.name == "cupy":
            parser.error("Simulation requires CuPy and a CUDA GPU. Install a matching cupy-cuda12x or cupy-cuda13x build; --check works without CuPy.")
        raise
    os.environ.setdefault("GPE_DRIVE_MOUNT", "/content/drive")
    stages = make_stages(args.config)
    psi = None
    for current in stages:
        print(f"Starting {current.name} ({current.type}, {current.steps} steps)", flush=True)
        if psi is not None:
            # Release replaced components before allocating their new fields.
            psi = tuple(value if current.initial_state[f"component{i}"]["type"] == "continue" else None
                        for i, value in enumerate(psi, 1))
        current.prepare(psi)
        psi = None  # Do not retain the first step's input throughout the whole stage.
        psi = launch_stage(current)
        current.release()
        print(f"Finished {current.name}", flush=True)


if __name__ == "__main__":
    main()
