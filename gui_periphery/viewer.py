"""Read existing raw monitor files for the GUI. No CuPy and no simulation work."""

import csv
from pathlib import Path
import re
from zipfile import BadZipFile

import numpy as np


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Choose a file inside the monitor folder.")
    return path


def serial(values):
    array = np.asarray(values)
    return np.where(np.isfinite(array), array, None).tolist()


def read_monitor(folder, stage_name="", spectrum_name="", image_kind="density_xy", image_limit=5,
                 image_name="", radial_name=""):
    folder = Path(folder).expanduser().resolve()
    root = folder / "monitor" if (folder / "monitor").is_dir() else folder
    candidates = sorted({path.parent for path in root.rglob("*.npz") if path.name != "grid.npz"}
                        | {path.parent for path in root.rglob("scalars.csv")}) if root.exists() else []
    stages = [str(path.relative_to(root)) for path in candidates]
    result = dict(stages=stages, stage=stage_name, spectra=[], spectrum=None, radial=None, image=None,
                  images=[], image_samples=[], scalars=[], warnings=[], log="")
    if not candidates:
        result["warnings"].append("No saved diagnostics yet. They appear at the intervals chosen in the experiment.")
    else:
        chosen = inside(root, stage_name) if stage_name in stages else max(candidates, key=lambda path: path.stat().st_mtime_ns)
        result["stage"] = str(chosen.relative_to(root))
        spectra = sorted(chosen.glob("nk_*.npz"))
        result["spectra"] = [path.name for path in spectra]
        for key, name in (("spectrum", spectrum_name), ("radial", radial_name)):
            if key == "radial" and name == spectrum_name:
                result[key] = result["spectrum"]
                continue
            selected = [path for path in spectra if path.name == name] if name else list(reversed(spectra))
            for path in selected:
                try:
                    with np.load(path, allow_pickle=False) as data:
                        result[key] = dict(file=path.name, time=float(data["time"]), step=int(data["step"]),
                            k=serial(data["k"]), nk=serial(data["nk"]),
                            complete=data["complete_shells"].astype(bool).tolist())
                    break
                except (OSError, ValueError, EOFError, KeyError, BadZipFile):
                    result["warnings"].append(f"{path.name} is not complete yet; waiting for a readable sample.")
        image_files = {}
        for path in sorted(chosen.glob("*.npz")):
            match = re.fullmatch(r"(.+)_(\d+)\.npz", path.name)
            if match and match[1] != "nk":
                image_files.setdefault(match[1], []).append(path)
        result["images"] = sorted(image_files)
        kind = image_kind if image_kind in image_files else next(iter(image_files), None)
        samples = image_files.get(kind, [])
        result["image_samples"] = [path.name for path in samples]
        selected = [path for path in samples if path.name == image_name] if image_name else list(reversed(samples))
        for path in selected:
            try:
                with np.load(path, allow_pickle=False) as data:
                    axes = [name for name in ("kx", "ky", "kz", "x", "y") if name in data and data[name].ndim == 1]
                    if len(axes) != 2 or data["values"].ndim != 2:
                        continue
                    x, y = (data[name] for name in axes)
                    ix, iy = np.arange(x.size), np.arange(y.size)
                    if kind.startswith("TOF") and image_limit > 0:
                        ix, iy = ix[np.abs(x) <= image_limit], iy[np.abs(y) <= image_limit]
                    if not ix.size or not iy.size:
                        continue
                    ix, iy = ix[::max(1, int(np.ceil(ix.size/256)))], iy[::max(1, int(np.ceil(iy.size/256)))]
                    result["image"] = dict(kind=kind, file=path.name, time=float(data["time"]), step=int(data["step"]),
                        x=serial(x[ix]), y=serial(y[iy]), axes=axes,
                        values=serial(data["values"][np.ix_(ix, iy)]), display_stride_only=True)
                break
            except (OSError, ValueError, EOFError, KeyError, BadZipFile):
                result["warnings"].append(f"{path.name} is not complete yet; waiting for a readable image.")
        scalar_path = chosen / "scalars.csv"
        if scalar_path.exists():
            with scalar_path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    try:
                        values = {key: float(value) for key, value in row.items()}
                        if values and all(np.isfinite(value) for value in values.values()):
                            result["scalars"].append(values)
                    except (TypeError, ValueError):
                        pass  # The final CSV row may still be being appended.
    for name in ("remote_run.log", "run.log", "launcher.log"):
        path = folder / "logs" / name
        if path.exists():
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 16000))
                result["log"] = stream.read().decode(errors="replace")
            break
    return result
