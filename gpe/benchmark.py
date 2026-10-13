"""Compare the existing propagators: python -m gpe.benchmark --help."""

import argparse
from contextlib import redirect_stdout
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import random
from statistics import median
from time import perf_counter


# name, evolution, external static potential, dynamic potential, momentum cutoff
VARIANTS = [
    ("imaginary_none", "imaginary", False, False, False),
    ("imaginary_static", "imaginary", True, False, False),
    ("real_none", "real", False, False, False),
    ("real_static", "real", True, False, False),
    ("real_dynamic", "real", False, True, False),
    ("real_static_dynamic", "real", True, True, False),
    ("real_none_cutoff", "real", False, False, True),
    ("real_static_cutoff", "real", True, False, True),
    ("real_dynamic_cutoff", "real", False, True, True),
    ("real_static_dynamic_cutoff", "real", True, True, True),
]


def benchmark_config(n, steps, dt):
    from config.yaml_parser import new_config, new_stage

    config = new_config()
    config["simulation"].update(N_grids=n, simulation_range=70)
    config["stages"] = []
    for name, evolution, static, dynamic, cutoff in VARIANTS:
        stage = new_stage(imaginary=evolution == "imaginary")
        stage["name"] = name
        stage["evolution"].update(type=evolution, steps=steps, dt=dt)
        stage["initial_state"].update(type="gaussian", sigma1=10, sigma2=10)
        stage["cloud"].update(a11=100, a22=80, a12=20, N1=150000, N2=100000)
        for group in ("box", "absorber", "noise"):
            stage[group]["enabled"] = False
        stage["potential1"]["static"].update(
            enabled=static, expression="0.002*(x*x + y*y + z*z)" if static else "0")
        stage["potential1"]["dynamic"].update(
            enabled=dynamic, expression="0.1*cos(0.5*t)*y" if dynamic else "0")
        stage["k_cut"].update(enabled=cutoff, value=0.7*math.pi*n/70 if cutoff else 0)
        stage["monitor"] = dict.fromkeys(stage["monitor"], 0)
        stage["save"].update(directory="", scalar_interval=0, nk_interval=0,
                             psi_interval=0, save_initial=False, save_final=False)
        config["stages"].append(stage)
    return config


def measure(cp, function):
    """Synchronized wall seconds and CUDA-event milliseconds for one batch."""
    start, end = cp.cuda.Event(), cp.cuda.Event()
    cp.cuda.get_current_stream().synchronize()
    before = perf_counter()
    start.record()
    function()
    end.record()
    end.synchronize()
    wall_seconds = perf_counter() - before
    return wall_seconds, cp.cuda.get_elapsed_time(start, end)


def benchmark_stage(cp, stage, steps, warmup, repeats):
    from .core.helper import get_N
    from .core.initial_states import gaussian
    from .launch import launch_stage

    try:
        setup_seconds, _ = measure(cp, stage.prepare)
        stage.steps = 1
        first_seconds, _ = measure(cp, lambda: launch_stage(stage))
        stage.steps = warmup
        launch_stage(stage)
        cp.cuda.get_current_stream().synchronize()

        samples = []
        stage.steps = steps
        for repeat in range(1, repeats + 1):
            # Regenerate outside the timer instead of keeping a full GPU backup.
            # Release the old state first; a 1024^3 complex128 array is 16 GiB.
            stage.psi1 = stage.psi2 = None
            stage.psi1 = gaussian(stage.x_array, stage.initial_state["sigma1"])
            stage.psi2 = gaussian(stage.x_array, stage.initial_state["sigma2"])
            wall_seconds, gpu_ms = measure(cp, lambda: launch_stage(stage))
            norms = []
            for psi in (stage.psi1, stage.psi2):
                if not bool(cp.isfinite(psi).all()):
                    raise RuntimeError(f"{stage.name}: non-finite wavefunction after repeat {repeat}")
                norm = float(get_N(psi, stage.x_array))
                if not math.isfinite(norm) or norm <= 0:
                    raise RuntimeError(f"{stage.name}: invalid norm {norm}")
                norms.append(norm)
            samples.append(dict(repeat=repeat, wall_seconds=wall_seconds,
                                cuda_ms=gpu_ms, final_norm1=norms[0], final_norm2=norms[1]))

        wall = [sample["wall_seconds"]*1000/steps for sample in samples]
        return dict(grid=stage.N_grids, variant=stage.name, steps=steps,
                    setup_ms=setup_seconds*1000, first_step_ms=first_seconds*1000,
                    wall_ms_per_step=median(wall), wall_min_ms_per_step=min(wall),
                    wall_max_ms_per_step=max(wall),
                    cuda_ms_per_step=median(sample["cuda_ms"]/steps for sample in samples),
                    steps_per_second=1000/median(wall), final_norm1=samples[-1]["final_norm1"], final_norm2=samples[-1]["final_norm2"],
                    samples=samples)
    finally:
        stage.release()


def write_report(folder, metadata, results):
    """Update the report after each completed case, retaining all raw samples."""
    order = {variant[0]: index for index, variant in enumerate(VARIANTS)}
    results = sorted(results, key=lambda row: (row["grid"], order[row["variant"]]))
    payload = dict(metadata=metadata, completed_cases=len(results),
                   complete=len(results) == metadata["expected_cases"], results=results)
    (folder / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    if results:
        with (folder / "summary.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=[key for key in results[0] if key != "samples"])
            writer.writeheader()
            writer.writerows({key: value for key, value in row.items() if key != "samples"}
                             for row in results)

    args = metadata["options"]
    lines = [
        "# GPE propagator benchmark", "",
        f"Completed cases: {len(results)}/{metadata['expected_cases']}. Started: {metadata['started_utc']}.",
        f"GPU: {metadata['gpu']} ({metadata['gpu_memory_gib']:.2f} GiB). "
        f"Python {metadata['python']}; CuPy {metadata['cupy']}; "
        f"CUDA runtime {metadata['cuda_runtime']}; driver API {metadata['cuda_driver']}.",
        "", f"Each repeat runs {args['steps']} steps; {args['repeats']} repeats per case. "
        f"One first step plus {args['warmup']} warm-up steps precede measurement. "
        f"dt={args['dt']}; complex128 wavefunctions and float64 coordinates.",
        "", "The primary metric is synchronized wall time per step through the existing "
        "launch_stage loop, including its disabled-output checks. Initial-state resets, "
        "validation, setup, warm-up and report writing are outside the repeated timings. "
        "All snapshots and diagnostics are disabled. The Gaussian is regenerated outside "
        "the timer before each repeat, without retaining a full GPU backup. CUDA events cover the same batch; "
        "their interval can include GPU idle time waiting for CPU submission.",
        "", "All cases start each repeat from the same normalized Gaussian (sigma1=10, sigma2=10, box length=70), "
        "with a11=100, a22=80, a12=20 (Bohr radii), N1=150000 and N2=100000. Box, absorber and noise are off. "
        "Component 1 static expression: `0.002*(x*x + y*y + z*z)`. "
        "Component 1 dynamic expression: `0.1*cos(0.5*t)*y`. Cutoff: 0.7 times the axis Nyquist wavenumber. "
        "Imaginary time includes per-step normalization; real time does not. "
        "These are different workloads, not interchangeable numerical methods.",
        "", "## Warmed-up stepping", "",
        "Values are per-step times averaged within each batch, then summarized across repeats. "
        "Min/max show the observed spread, not a confidence interval. "
        "Potential complexity, grid, GPU load and clock speeds affect these measurements.", "",
        "| Grid | Variant | Wall median ms/step | Wall min–max | CUDA median ms/step | Steps/s |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(f"| {row['grid']}³ | {row['variant']} | {row['wall_ms_per_step']:.4f} | "
                     f"{row['wall_min_ms_per_step']:.4f}–{row['wall_max_ms_per_step']:.4f} | "
                     f"{row['cuda_ms_per_step']:.4f} | {row['steps_per_second']:.1f} |")
    lines += [
        "", "## Observed setup and first step", "",
        "Each value is a single observation. Setup times stage.prepare; YAML parsing and "
        "axis creation are excluded. The first step may include compilation and FFT planning. "
        "Cases share in-process and on-disk caches, so these are not controlled cold-start timings. "
        "Case order is shuffled with a recorded seed. FFT plans and unused memory pools are "
        "cleared between grid sizes, not between cases or repeats.", "",
        "| Grid | Variant | Setup ms | First step ms |", "| --- | --- | ---: | ---: |",
    ]
    for row in results:
        lines.append(f"| {row['grid']}³ | {row['variant']} | {row['setup_ms']:.3f} | {row['first_step_ms']:.3f} |")
    lines += [
        "", "Exact generated configurations are in config_N*.yaml. summary.csv holds the tables; "
        "results.json includes individual repeat timings, norms, execution order and source hashes. "
        "cupy_config.txt records the CUDA library configuration.", "",
        "Timing method: [CuPy performance guidance](https://docs.cupy.dev/en/stable/user_guide/performance.html).",
        "",
    ]
    (folder / "report.md").write_text("\n".join(lines))


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", nargs="+", type=positive_int, default=[64, 128, 256])
    parser.add_argument("--steps", type=positive_int, default=50, help="steps per timed repeat (default: 50)")
    parser.add_argument("--warmup", type=positive_int, default=5, help="warm-up steps after first step (default: 5)")
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42, help="shuffle case order with this seed")
    parser.add_argument("--output", default="benchmark_results", help="parent folder for a new timestamped report")
    args = parser.parse_args()
    if min(args.grids) < 2 or len(set(args.grids)) != len(args.grids):
        parser.error("grids must be distinct integers of at least 2")
    if not math.isfinite(args.dt) or args.dt <= 0:
        parser.error("dt must be finite and positive")
    try:
        import cupy as cp
    except ModuleNotFoundError as exc:
        if exc.name != "cupy":
            raise
        parser.error("Benchmarking requires CuPy and a CUDA GPU; see gpe/run.md.")
    from config.yaml_parser import dump_yaml
    from .stage_factory import make_stages

    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    gpu_name = properties["name"]
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    device.synchronize()
    started = datetime.now(timezone.utc)
    folder = Path(args.output).expanduser().resolve() / started.strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True)
    config_text = io.StringIO()
    with redirect_stdout(config_text):
        cp.show_config()
    (folder / "cupy_config.txt").write_text(config_text.getvalue())
    root = Path(__file__).resolve().parents[1]
    sources = sorted(path for package in ("gpe", "config")
                     for path in (root / package).rglob("*.py"))
    metadata = dict(started_utc=started.isoformat(), options=vars(args), gpu=gpu_name,
                    gpu_memory_gib=properties["totalGlobalMem"]/2**30,
                    python=platform.python_version(), platform=platform.platform(), cupy=cp.__version__,
                    cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                    cuda_driver=cp.cuda.runtime.driverGetVersion(),
                    expected_cases=len(args.grids)*len(VARIANTS), execution_order=[],
                    source_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in sources})
    results = []
    write_report(folder, metadata, results)
    print(f"GPU: {gpu_name}\nReports: {folder}", flush=True)
    rng = random.Random(args.seed)
    for n in args.grids:
        path = folder / f"config_N{n}.yaml"
        path.write_text(dump_yaml(benchmark_config(n, args.steps, args.dt)))
        stages = make_stages(path)
        rng.shuffle(stages)
        for stage in stages:
            print(f"Benchmarking {n}³ {stage.name} ...", flush=True)
            metadata["execution_order"].append(dict(grid=n, variant=stage.name))
            result = benchmark_stage(cp, stage, args.steps, args.warmup, args.repeats)
            results.append(result)
            write_report(folder, metadata, results)
            print(f"  {result['wall_ms_per_step']:.4f} ms/step; "
                  f"{result['steps_per_second']:.1f} steps/s", flush=True)
        cp.cuda.get_current_stream().synchronize()
        cp.fft.config.get_plan_cache().clear()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    print(f"Finished {len(results)} cases. Report: {folder / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
