"""Slab-wise paired snapshots. The JSON manifest commits a complete pair."""
import json
from pathlib import Path
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

def load_wavefunctions(path, stage):
    """Load both fields without normalization or continuation of the saved clock."""
    path = Path(path)
    metadata = json.loads(path.read_text())
    if metadata.get("format") != "two-component-gpe-snapshot-v1":
        raise ValueError("Choose a two-component snapshot-pair JSON manifest.")
    for key in ("N1", "N2", "N_grids", "simulation_range"):
        if metadata.get(key) != getattr(stage, key):
            raise ValueError(f"The loaded pair has a different reference {key}.")
    files = [_member(path.parent, metadata["files"][f"psi{i}"]) for i in (1, 2)]
    if files[0] == files[1]:
        raise ValueError("The two snapshot components must be distinct files.")
    with np.load(_member(path.parent, metadata["grid"]), allow_pickle=False) as grid:
        for key in ("x_array", "k_array"):
            expected = cp.asnumpy(getattr(stage, key))
            if grid[key].shape != expected.shape or not np.allclose(grid[key], expected, rtol=1e-12, atol=1e-12):
                raise ValueError("The loaded pair's grid does not match this simulation.")
    # Validate both sizes before allocating either GPU field.
    for member in files:
        with member.open("rb") as stream:
            shape, dtype = _header(stream, stage)
            if member.stat().st_size != stream.tell() + int(np.prod(shape)) * dtype.itemsize:
                raise ValueError("Incomplete or malformed wavefunction snapshot.")
    fields = []
    for member in files:
        with member.open("rb") as stream:
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
            fields.append(psi)
    return tuple(fields)
