import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from .diagnostics import sum_abs2, get_N, get_E, get_L, is_due
from .nk_extraction import nk_extraction


def _show_image(data,axis1,axis2,title,xlabel,ylabel,label,phase=False):
    data=cp.asnumpy(data)
    axis1=cp.asnumpy(axis1)
    axis2=cp.asnumpy(axis2)
    d1=axis1[1]-axis1[0]
    d2=axis2[1]-axis2[0]
    extent=(axis1[0]-d1/2,axis1[-1]+d1/2,axis2[0]-d2/2,axis2[-1]+d2/2)
    fig,ax=plt.subplots(figsize=(6,5),layout="constrained")
    options=dict(cmap="twilight",vmin=-np.pi,vmax=np.pi) if phase else dict(cmap="magma")
    image=ax.imshow(data.T,origin="lower",extent=extent,**options)
    ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
    fig.colorbar(image,ax=ax,label=label)
    plt.show()
    plt.close(fig)


def monitor(i,psi,k_array,x_array,print_scalar_interval=0,show_nk_interval=0,
            show_TOF_image_interval=0,show_verticle_TOF_interval=0,
            show_TOF_slice_interval=0,show_verticle_TOF_slice_interval=0,
            show_magnifier_interval=0,show_magnifier_slice_interval=0,
            show_phase_interval=0,psi_k=None):
    """N is norm, E is kinetic energy, L is Lz; all are dimensionless integrals.

    i counts completed steps; zero/None intervals and i=0 produce no output.
    Optional psi_k must be raw fftn(psi) of this completed state.
    """
    scalars=is_due(i,print_scalar_interval)
    spectrum_due=is_due(i,show_nk_interval)
    tof_z=is_due(i,show_TOF_image_interval)
    tof_x=is_due(i,show_verticle_TOF_interval)
    slice_kz=is_due(i,show_TOF_slice_interval)
    slice_kx=is_due(i,show_verticle_TOF_slice_interval)
    density_z=is_due(i,show_magnifier_interval)
    density_slice=is_due(i,show_magnifier_slice_interval)
    phase_due=is_due(i,show_phase_interval)
    momentum_due=scalars or spectrum_due or tof_z or tof_x or slice_kz or slice_kx
    if not (momentum_due or density_z or density_slice or phase_due):
        return {}
    results={}
    dx=float(x_array[1]-x_array[0])
    dk=abs(float(k_array[1]-k_array[0]))
    if momentum_due and psi_k is None:
        psi_k=cp.fft.fftn(psi)
    if scalars:
        results.update(N=float(get_N(psi,x_array)),E=float(get_E(psi_k,x_array,k_array)),
                       L=float(get_L(psi,x_array,k_array)))
        print(f"i={i}  N={results['N']:.8g}  E_kin={results['E']:.8g}  Lz={results['L']:.8g}")
    if spectrum_due:
        spectrum=nk_extraction(psi_k,x_array,k_array)
        spectrum={name:cp.asnumpy(value) for name,value in spectrum.items()}
        results['spectrum']=spectrum
        k,nk,Nk=spectrum['k'],spectrum['nk'],spectrum['Nk']
        positive=(k>0)&(nk>0)
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
        axes[0].loglog(k[positive],nk[positive])
        axes[0].set(xlabel="k",ylabel="n(k), shell average")
        axes[1].plot(k,Nk)
        axes[1].set(xlabel="k",ylabel="Nk, population per shell")
        for ax in axes:
            ax.axvline(psi.shape[0]*dk/2,color="grey",ls="--",label="axis Nyquist")
            ax.legend()
        fig.suptitle(f"i={i}")
        plt.show()
        plt.close(fig)
    if tof_z or tof_x or slice_kz or slice_kx:
        k_axis=cp.fft.fftshift(k_array)
        scale=dx**3/(psi.size*dk**3)
        if tof_z:
            image=cp.fft.fftshift(sum_abs2(psi_k,axis=2))*(scale*dk)
            _show_image(image,k_axis,k_axis,f"i={i}: TOF integrated over kz","kx","ky","integral n(k) dkz")
        if tof_x:
            image=cp.fft.fftshift(sum_abs2(psi_k,axis=0))*(scale*dk)
            _show_image(image,k_axis,k_axis,f"i={i}: TOF integrated over kx","ky","kz","integral n(k) dkx")
        if slice_kz:
            image=cp.fft.fftshift(cp.abs(psi_k[:,:,0])**2)*scale
            _show_image(image,k_axis,k_axis,f"i={i}: TOF at kz=0","kx","ky","n(k)")
        if slice_kx:
            image=cp.fft.fftshift(cp.abs(psi_k[0,:,:])**2)*scale
            _show_image(image,k_axis,k_axis,f"i={i}: TOF at kx=0","ky","kz","n(k)")
    if density_z:
        image=sum_abs2(psi,axis=2)*dx
        _show_image(image,x_array,x_array,f"i={i}: density integrated over z","x","y","integral |psi|² dz")
    if density_slice or phase_due:
        mid=psi.shape[2]//2
        plane=psi[:,:,mid]
        density=cp.abs(plane)**2
        z=float(x_array[mid])
        if density_slice:
            _show_image(density,x_array,x_array,f"i={i}: density at z={z:g}","x","y","|psi|²")
        if phase_due:
            phase=cp.where(density>density.max()*1e-12,cp.angle(plane),cp.nan)
            _show_image(phase,x_array,x_array,f"i={i}: phase at z={z:g}","x","y","phase (rad)",phase=True)
    return results
