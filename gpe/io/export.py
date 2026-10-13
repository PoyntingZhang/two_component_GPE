import csv
from pathlib import Path
import cupy as cp
import numpy as np
from .diagnostics import get_N, is_due
from .nk_extraction import nk_extraction


def export(i,psi,k_array,x_array,psi_interval=0,nk_interval=0,N_interval=0,
           output_dir="output",psi_k=None):
    """i counts completed steps. Zero/None intervals disable saving; i=0 is skipped.

    Optional psi_k must be raw fftn(psi) of this completed state.
    """
    save_psi=is_due(i,psi_interval)
    save_nk=is_due(i,nk_interval)
    save_N=is_due(i,N_interval)
    if not (save_psi or save_nk or save_N):
        return
    folder=Path(output_dir)
    folder.mkdir(parents=True,exist_ok=True)

    if save_psi or save_N:
        N=float(get_N(psi,x_array))
    if save_psi:
        # Write slabs to NPY without bringing the entire state into CPU memory.
        saved=np.lib.format.open_memmap(folder/f"psi_{i:08d}.npy",mode="w+",
                                       dtype=psi.dtype,shape=psi.shape)
        for start in range(0,psi.shape[0],16):
            saved[start:start+16]=cp.asnumpy(psi[start:start+16])
        saved.flush()
        del saved
        np.savez(folder/f"psi_{i:08d}_grid.npz",i=i,N=N,
                 x_array=cp.asnumpy(x_array),k_array=cp.asnumpy(k_array))
    if save_N:
        path=folder/"N.csv"
        new_file=not path.exists() or path.stat().st_size==0
        with path.open("a",newline="") as file:
            writer=csv.writer(file)
            if new_file:
                writer.writerow(["i","N"])
            writer.writerow([i,N])
    if save_nk:
        if psi_k is None:
            psi_k=cp.fft.fftn(psi)
        spectrum=nk_extraction(psi_k,x_array,k_array)
        np.savez(folder/f"nk_{i:08d}.npz",i=i,
                 **{name:cp.asnumpy(value) for name,value in spectrum.items()})
