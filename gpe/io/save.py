"""Save snapshots and small diagnostics after a completed simulation step."""

import csv
from types import SimpleNamespace

import cupy as cp
import numpy as np

from .diagnostics import get_N, get_E, get_L, is_due, sum_abs2
from .nk_extraction import nk_extraction
from .state_io import save_wavefunctions
from .atomic import atomic_npz


def save_outputs(stage, step, *, initial=False, final=False):
    save, monitor = stage.save_params, stage.monitor_params
    if (is_due(step, save["psi_interval"]) or
            (initial and save["save_initial"]) or (final and save["save_final"])):
        save_wavefunctions(stage.data_dir / f"psi_{step:08d}.json", stage, step)

    for component in (1, 2):
        view = SimpleNamespace(**vars(stage))
        view.psi = getattr(stage, f"psi{component}")
        view.N_atoms = getattr(stage, f"N{component}")
        view.monitor_dir = stage.monitor_dir / f"component_{component}"
        view.name = f"{stage.name} / component {component}"
        _save_component_outputs(view, step, initial=initial)


def _save_component_outputs(stage, step, *, initial=False):
    save, monitor = stage.save_params, stage.monitor_params

    # Initial/final flags request wavefunction snapshots only.
    if initial:
        return
    scalar_save = is_due(step, save["scalar_interval"])
    scalar_print = is_due(step, monitor["print_scalar_interval"])
    spectrum_due = (is_due(step, save["nk_interval"]) or
                    is_due(step, monitor["show_nk_interval"]))
    due = {name: is_due(step, interval) for name, interval in monitor.items()}
    momentum_due = scalar_save or scalar_print or spectrum_due or any(
        due[name] for name in ("show_TOF_image_interval", "show_verticle_TOF_interval",
                              "show_TOF_slice_interval", "show_verticle_TOF_slice_interval"))
    psi = stage.psi
    psi_k = cp.fft.fftn(psi) if momentum_due else None
    time = step * stage.dt

    def save_npz(name, **arrays):
        stage.monitor_dir.mkdir(parents=True, exist_ok=True)
        atomic_npz(stage.monitor_dir / f"{name}_{step:08d}.npz", step=step, time=time,
                 **{key: cp.asnumpy(value) for key, value in arrays.items()})

    if scalar_save or scalar_print:
        norm = float(get_N(psi, stage.x_array))
        energy = float(get_E(psi_k, stage.x_array, stage.k_array))
        angular_momentum = float(get_L(psi, stage.x_array, stage.k_array))
        if scalar_print:
            print(f"{stage.name}: step={step} t={time:g} N={norm:.8g} "
                  f"E_kin={energy:.8g} Lz={angular_momentum:.8g}", flush=True)
        if scalar_save:
            stage.monitor_dir.mkdir(parents=True, exist_ok=True)
            path = stage.monitor_dir / "scalars.csv"
            new_file = not path.exists()
            with path.open("a", newline="") as file:
                writer = csv.writer(file)
                if new_file:
                    writer.writerow(["step", "time", "norm", "N_atoms", "E_kin", "Lz"])
                writer.writerow([step, time, norm, stage.N_atoms * norm, energy, angular_momentum])

    if spectrum_due:
        save_npz("nk", **nk_extraction(psi_k, stage.x_array, stage.k_array))

    if any(due[name] for name in ("show_TOF_image_interval", "show_verticle_TOF_interval",
                                  "show_TOF_slice_interval", "show_verticle_TOF_slice_interval")):
        dx = float(stage.x_array[1] - stage.x_array[0])
        dk = abs(float(stage.k_array[1] - stage.k_array[0]))
        scale = dx**3 / (psi.size * dk**3)
        k = cp.fft.fftshift(stage.k_array)
        if due["show_TOF_image_interval"]:
            values = cp.fft.fftshift(sum_abs2(psi_k, axis=2)) * (scale * dk)
            save_npz("TOF_xy", values=values, kx=k, ky=k)
        if due["show_verticle_TOF_interval"]:
            values = cp.fft.fftshift(sum_abs2(psi_k, axis=0)) * (scale * dk)
            save_npz("TOF_yz", values=values, ky=k, kz=k)
        if due["show_TOF_slice_interval"]:
            values = cp.fft.fftshift(cp.abs(psi_k[:, :, 0])**2) * scale
            save_npz("TOF_slice_xy", values=values, kx=k, ky=k)
        if due["show_verticle_TOF_slice_interval"]:
            values = cp.fft.fftshift(cp.abs(psi_k[0, :, :])**2) * scale
            save_npz("TOF_slice_yz", values=values, ky=k, kz=k)

    x = stage.x_array
    if due["show_magnifier_interval"]:
        values = sum_abs2(psi, axis=2) * float(x[1] - x[0])
        save_npz("density_xy", values=values, x=x, y=x)
    if due["show_magnifier_slice_interval"] or due["show_phase_interval"]:
        middle = stage.N_grids // 2
        plane = psi[:, :, middle]
        density = cp.abs(plane)**2
        if due["show_magnifier_slice_interval"]:
            save_npz("density_slice_xy", values=density, x=x, y=x, z=x[middle])
        if due["show_phase_interval"]:
            phase = cp.where(density > density.max() * 1e-12, cp.angle(plane), cp.nan)
            save_npz("phase_xy", values=phase, x=x, y=x, z=x[middle])
