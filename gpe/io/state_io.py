"""Slab-wise paired snapshots. The JSON manifest commits a complete pair."""
import json
from pathlib import Path
import re
import cupy as cp
import numpy as np
from .atomic import atomic_file, atomic_json, atomic_npz

SLAB_SIZE = 16

def save_wavefunctions(path, stage, step):
    path = Path(path)
    if path.suffix != ".json":
        raise ValueError("A snapshot pair must use a .json manifest path.")
    if path.exists():
        raise FileExistsError(f"Snapshot already exists: {path}")
    names = {f"psi{i}": f"psi{i}_{step:08d}.npy" for i in (1, 2)}
    # No full host copy and no memory-mapped write on the Drive filesystem.
    for key, name in names.items():
        psi = getattr(stage, key)
        if psi.dtype != cp.complex128 or psi.shape != (stage.N_grids,) * 3:
            raise ValueError("Snapshots require two complex128 cubes matching the grid.")
        with atomic_file(path.parent / name) as stream:
            np.lib.format.write_array_header_2_0(stream, dict(
                descr=np.lib.format.dtype_to_descr(np.dtype("complex128")),
                fortran_order=False, shape=psi.shape))
            for start in range(0, stage.N_grids, SLAB_SIZE):
                slab = np.ascontiguousarray(cp.asnumpy(psi[start:start + SLAB_SIZE]))
                stream.write(memoryview(slab).cast("B"))
    atomic_npz(path.parent / "grid.npz", x_array=cp.asnumpy(stage.x_array),
               k_array=cp.asnumpy(stage.k_array))
    # Publish last. A snapshot is reusable only when this manifest exists.
    atomic_json(path, dict(format="two-component-gpe-snapshot-v1", files=names,
        grid="grid.npz", step=step, time=step * stage.dt, evolution=stage.type,
        N1=stage.N1, N2=stage.N2, simulation_range=stage.simulation_range,
        N_grids=stage.N_grids, stage=stage.config))

def _member(root, name):
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("Snapshot members must be filenames alongside their manifest.")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Snapshot member escapes its directory.")
    return path

def _header(stream, stage):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError("Unsupported snapshot NPY version.")
    if shape != (stage.N_grids,) * 3 or dtype != np.complex128 or fortran:
        raise ValueError("The loaded state must be a C-order complex128 cube matching N_grids.")
    return shape, dtype

def inspect_wavefunction(path, stage, component):
    """Resolve one saved field and validate its metadata without allocating a cube.

    NPY inputs identify the source component from their saved filename. JSON
    inputs select the destination component, preserving old pair-load YAMLs.
    The completed manifest and grid must remain beside the selected NPY file;
    the other component's NPY file is not needed for an independent load.
    """
    if component not in (1, 2):
        raise ValueError("The destination component must be 1 or 2.")
    path = Path(path).expanduser().resolve()
    if path.suffix == ".json":
        manifest, source_component = path, component
    else:
        match = re.fullmatch(r"psi([12])_(\d+)\.npy", path.name)
        if not match:
            raise ValueError("Choose a saved psi1_STEP.npy or psi2_STEP.npy file, keeping its original filename.")
        source_component = int(match[1])
        manifest = path.with_name(f"psi_{match[2]}.json")
    if not manifest.is_file():
        raise ValueError(f"Completed snapshot metadata is missing: {manifest}. Keep the JSON manifest and grid.npz beside the selected wavefunction.")
    metadata = json.loads(manifest.read_text())
    if metadata.get("format") != "two-component-gpe-snapshot-v1":
        raise ValueError("Choose a wavefunction saved by the two-component solver.")
    for key in ("N_grids", "simulation_range"):
        if metadata.get(key) != getattr(stage, key):
            raise ValueError(f"The loaded field has a different {key}.")
    if metadata.get(f"N{source_component}") != getattr(stage, f"N{component}"):
        raise ValueError(f"Component {component}: reference N{component} must match the source field's N{source_component} ({metadata.get(f'N{source_component}')}). Loading preserves its saved norm.")
    files = [_member(manifest.parent, metadata["files"][f"psi{i}"]) for i in (1, 2)]
    if files[0] == files[1]:
        raise ValueError("The two snapshot components must be distinct files.")
    member = files[source_component - 1]
    if path.suffix == ".npy" and member != path:
        raise ValueError("The selected wavefunction does not match its snapshot metadata.")
    with np.load(_member(manifest.parent, metadata["grid"]), allow_pickle=False) as grid:
        for key in ("x_array", "k_array"):
            expected = cp.asnumpy(getattr(stage, key))
            if grid[key].shape != expected.shape or not np.allclose(grid[key], expected, rtol=1e-12, atol=1e-12):
                raise ValueError("The loaded field's grid does not match this simulation.")
    with member.open("rb") as stream:
        shape, dtype = _header(stream, stage)
        if member.stat().st_size != stream.tell() + int(np.prod(shape)) * dtype.itemsize:
            raise ValueError("Incomplete or malformed wavefunction snapshot.")
    return member


def read_wavefunction(member, stage):
    """Read a validated field in bounded host-memory slabs, preserving its norm."""
    with Path(member).open("rb") as stream:
        shape, dtype = _header(stream, stage)
        psi = cp.empty(shape, dtype=cp.complex128)
        for start in range(0, stage.N_grids, SLAB_SIZE):
            planes = min(SLAB_SIZE, stage.N_grids - start)
            size = planes * stage.N_grids**2 * dtype.itemsize
            data = stream.read(size)
            if len(data) != size:
                raise ValueError("Snapshot was truncated while loading.")
            slab = np.frombuffer(data, dtype=dtype).reshape(planes, stage.N_grids, stage.N_grids)
            psi[start:start + planes] = cp.asarray(slab, blocking=True)
    return psi


def load_wavefunction(path, stage, component):
    """Load one component without normalization or continuation of the clock."""
    return read_wavefunction(inspect_wavefunction(path, stage, component), stage)


def load_wavefunctions(path, stage):
    """Compatibility entry point for loading a complete pair from one manifest."""
    if Path(path).suffix != ".json":
        raise ValueError("A pair load needs its .json manifest; use load_wavefunction for an individual NPY.")
    members = [inspect_wavefunction(path, stage, i) for i in (1, 2)]
    return tuple(read_wavefunction(member, stage) for member in members)
