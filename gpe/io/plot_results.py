"""Plot saved diagnostics on the CPU: python -m gpe.io.plot_results --help."""

import argparse
from html import escape
from math import isfinite
from pathlib import Path
import shutil


# Filename prefix, horizontal coordinate, vertical coordinate, title, colorbar.
IMAGES = [
    ("density_xy", "x", "y", "Density integrated over z", r"$\int |\psi|^2\,dz$"),
    ("density_slice_xy", "x", "y", "Density slice", r"$|\psi|^2$"),
    ("phase_xy", "x", "y", "Phase slice", "Phase (rad)"),
    ("TOF_xy", "kx", "ky", "TOF integrated over kz", r"$\int n(\mathbf{k})\,dk_z$"),
    ("TOF_yz", "ky", "kz", "TOF integrated over kx", r"$\int n(\mathbf{k})\,dk_x$"),
    ("TOF_slice_xy", "kx", "ky", "TOF slice at kz=0", r"$n(\mathbf{k})$"),
    ("TOF_slice_yz", "ky", "kz", "TOF slice at kx=0", r"$n(\mathbf{k})$"),
]


def latest_file(folder, prefix):
    paths = folder.glob(f"{prefix}_[0-9]*.npz")
    return max(paths, key=lambda path: int(path.stem.rsplit("_", 1)[1]), default=None)


def draw_spectrum(ax, data, np, k_max):
    """Draw positive spectrum samples on logarithmic axes; zero stays in the NPZ."""
    k, nk, counts = data["k"], data["nk"], data["counts"]
    visible = (k > 0) & (k <= k_max) & (counts > 0)
    positive = visible & (nk > 0)
    complete = data["complete_shells"]
    ax.set_xscale("log")
    ax.set_yscale("log")
    for mask, label, style in ((positive & complete, "Complete shells", ".-"),
                               (positive & ~complete, "Incomplete outer shells", ".")):
        if mask.any():
            ax.plot(k[mask], nk[mask], style, label=label)
    if positive.any():
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, f"No positive samples in 0 < k ≤ {k_max:g}",
                ha="center", transform=ax.transAxes)
    lower = float(k[visible].min()) if visible.any() else k_max / 10
    sample = f"step={int(data['step'])}, t={float(data['time']):g}"
    ax.set(title=f"Momentum spectrum\n{sample}", xlabel="k", ylabel="n(k), shell average",
           xlim=(lower if lower < k_max else k_max / 10, k_max))
    ax.grid(which="both", alpha=0.2)


def plot_spectrum_series(stage, output, np, plt, k_max):
    """Export one PNG per saved spectrum, in completed-step order."""
    sources = sorted(stage.glob("nk_[0-9]*.npz"),
                     key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    if sources:
        output.mkdir(parents=True, exist_ok=True)
    generated = []
    for path in sources:
        fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
        with np.load(path, allow_pickle=False) as data:
            draw_spectrum(ax, data, np, k_max)
        fig.suptitle(stage.name.replace("_", " "))
        name = path.stem + ".png"
        fig.savefig(output / name, dpi=150)
        plt.close(fig)
        generated.append(name)
    return generated, [path.name for path in sources]


def plot_stage(stage, output, np, plt, tof_limit=None, k_max=5.0):
    """Use the newest saved sample of each diagnostic; retain its own time label."""
    output.mkdir(parents=True, exist_ok=True)
    title = stage.name.replace("_", " ")
    generated, sources = [], []
    scalar_path = stage / "scalars.csv"
    if scalar_path.exists():
        values = np.genfromtxt(scalar_path, delimiter=",", names=True, ndmin=1)
        if values.size:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
            for ax, key, label in zip(axes, ("norm", "E_kin", "Lz"),
                                      ("Wavefunction norm", "Kinetic energy", "Integrated Lz")):
                ax.plot(values["time"], values[key], ".-", linewidth=1)
                ax.set(xlabel="Stage time", ylabel=label)
                ax.ticklabel_format(axis="y", style="plain", useOffset=False)
                ax.grid(alpha=0.2)
            fig.suptitle(f"{title} · saved scalar history")
            fig.savefig(output / "scalars.png", dpi=150)
            plt.close(fig)
            generated.append("scalars.png")
            sources.append(scalar_path.name)

    available = [(spec, latest_file(stage, spec[0])) for spec in IMAGES]
    available = [(spec, path) for spec, path in available if path is not None]
    spectrum = latest_file(stage, "nk")
    panels = len(available) + (2 if spectrum else 0)
    if panels:
        columns = min(3, panels)
        rows = (panels + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(5.5*columns, 4.5*rows),
                                 squeeze=False, layout="constrained")
        axes = axes.ravel()
        for ax, (spec, path) in zip(axes, available):
            prefix, horizontal, vertical, label, colorbar = spec
            with np.load(path, allow_pickle=False) as data:
                options = dict(cmap="twilight", vmin=-np.pi, vmax=np.pi) if prefix == "phase_xy" else dict(cmap="magma")
                mesh = ax.pcolormesh(data[horizontal], data[vertical], data["values"].T,
                                     shading="auto", **options)
                sample = f"step={int(data['step'])}, t={float(data['time']):g}"
                if "z" in data:
                    sample += f", z={float(data['z']):g}"
                ax.set(title=f"{label}\n{sample}", xlabel=horizontal, ylabel=vertical)
                if prefix.startswith("TOF_") and tof_limit is not None:
                    ax.set(xlim=(-tof_limit, tof_limit), ylim=(-tof_limit, tof_limit))
                ax.set_aspect("equal")
                fig.colorbar(mesh, ax=ax, label=colorbar, shrink=0.85)
            sources.append(path.name)

        if spectrum:
            average, population = axes[len(available):len(available)+2]
            with np.load(spectrum, allow_pickle=False) as data:
                draw_spectrum(average, data, np, k_max)
                k, counts = data["k"], data["counts"]
                visible = (k >= 0) & (k <= k_max) & (counts > 0)
                sample = f"step={int(data['step'])}, t={float(data['time']):g}"
                population.plot(k[visible], data["Nk"][visible], ".-", linewidth=1)
                population.set(title=f"Shell populations\n{sample}", xlabel="k",
                               ylabel="Nk, population per shell", xlim=(0, k_max))
                population.grid(alpha=0.2)
            sources.append(spectrum.name)
        for ax in axes[panels:]:
            ax.set_visible(False)
        fig.suptitle(f"{title} · latest saved diagnostics")
        fig.savefig(output / "diagnostics.png", dpi=150)
        plt.close(fig)
        generated.append("diagnostics.png")
    return generated, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="timestamped monitor run folder, or monitor root with --latest")
    parser.add_argument("--latest", action="store_true", help="select the newest run under the supplied monitor root")
    parser.add_argument("--tof-limit", type=float, help="show both axes of TOF images and slices from -LIMIT to +LIMIT in dimensionless momentum")
    parser.add_argument("--k-max", type=float, default=5.0, help="upper k limit for radial spectra; n(k) uses log-log axes and omits k=0 (default: 5)")
    parser.add_argument("--all-spectra", action="store_true", help="also export a log-log PNG for every saved nk sample and include the series in the gallery")
    args = parser.parse_args()
    if args.tof_limit is not None and (not isfinite(args.tof_limit) or args.tof_limit <= 0):
        parser.error("--tof-limit must be a positive finite number")
    if not isfinite(args.k_max) or args.k_max <= 0:
        parser.error("--k-max must be a positive finite number")
    folder = Path(args.directory).expanduser().resolve()
    if not folder.is_dir():
        parser.error(f"Folder does not exist: {folder}")
    if args.latest:
        runs = sorted(path for path in folder.iterdir() if path.is_dir() and any(path.glob("stage_*")))
        if not runs:
            parser.error(f"No stage output folders found under {folder}")
        folder = runs[-1]
    stages = sorted(path for path in folder.glob("stage_*") if path.is_dir())
    if not stages:
        parser.error("No stage_* folders found. Pass the monitor run folder, or its parent with --latest.")
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        parser.error(f"Missing {exc.name}. Install plotting dependencies with: python -m pip install numpy matplotlib")

    output = folder / "plots"
    output.mkdir(exist_ok=True)
    html = [
        '<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>GPE results</title>',
        '<style>body{font:16px system-ui,sans-serif;margin:2rem auto;padding:0 1rem;max-width:1200px;'
        'background:#fafafa;color:#202020}img{width:100%;height:auto;background:white}section{margin:3rem 0}'
        'p{line-height:1.5}summary{cursor:pointer}li{overflow-wrap:anywhere}</style>',
        f'<h1>GPE results</h1><p>Run: {escape(folder.name)}</p>',
        '<p>Each diagnostic uses its latest saved sample, which may precede the end of its stage. '
        'Read the step and time printed on each panel. Times are local to each stage. '
        'Image color scales are independent; compare colorbar values across stages. '
        'Energy is kinetic energy only, and the angular momentum shown is Lz.</p>',
        f'<p>Momentum spectra use log-log axes for 0 &lt; k ≤ {args.k_max:g}, '
        'omitting k=0 and nonpositive n(k). Shell populations use linear axes from '
        f'k=0 to {args.k_max:g}. Saved spectra retain all bins.</p>',
    ]
    if args.tof_limit is not None:
        html.append(f'<p>TOF images and slices show dimensionless momentum from '
                    f'{-args.tof_limit:g} to {args.tof_limit:g} on both axes. '
                    'Saved arrays retain their full momentum range.</p>')
    print(f"Plotting run: {folder}", flush=True)
    for stage in stages:
        generated, sources = plot_stage(stage, output / stage.name, np, plt, args.tof_limit, args.k_max)
        html.append(f"<section><h2>{escape(stage.name.replace('_', ' '))}</h2>")
        for name in generated:
            image = escape(f"{stage.name}/{name}", quote=True)
            html.append(f'<a href="{image}"><img src="{image}" alt="{escape(stage.name)} {name}" loading="lazy"></a>')
        if not generated:
            html.append('<p>No saved diagnostics in this stage.</p>')
        elif "diagnostics.png" not in generated:
            html.append('<p>No image or spectrum diagnostics were saved in this stage.</p>')
        if args.all_spectra:
            spectra, spectrum_sources = plot_spectrum_series(stage, output / stage.name / "spectra", np, plt, args.k_max)
            if spectra:
                html.append(f'<details><summary>Momentum spectrum history ({len(spectra)} plots)</summary>')
                for name in spectra:
                    image = escape(f"{stage.name}/spectra/{name}", quote=True)
                    html.append(f'<a href="{image}"><img src="{image}" alt="{escape(stage.name)} {name}" loading="lazy"></a>')
                html.append('</details>')
                sources.extend(name for name in spectrum_sources if name not in sources)
                print(f"  {stage.name}: exported {len(spectra)} spectrum plots", flush=True)
        html.append('<details><summary>Source files</summary><ul>')
        html.extend(f"<li>{escape(name)}</li>" for name in sources)
        html.append('</ul></details></section>')
        print(f"  {stage.name}: {', '.join(generated) or 'no saved diagnostics'}", flush=True)
    html.append('</html>')
    (output / "index.html").write_text("\n".join(html), encoding="utf-8")
    archive = shutil.make_archive(str(output), "zip", root_dir=output)
    print(f"Images and index: {output}\nDownloadable gallery: {archive}", flush=True)


if __name__ == "__main__":
    main()
