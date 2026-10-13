import math
import cupy as cp


shell_modes = cp.ElementwiseKernel(
    'complex128 z, float64 kx, float64 ky, float64 kz, float64 dk',
    'int32 shell, float64 occupation',
    '''
    shell = (int)floor(sqrt(kx*kx+ky*ky+kz*kz)/dk+0.5);
    occupation = z.real()*z.real()+z.imag()*z.imag();
    ''',
    'shell_modes',
)


def nk_extraction(psi_k,x_array,k_array,slab_size=16):
    """Shell averages from raw fftn(psi); Nk sums to the wavefunction norm."""
    N_grid=psi_k.shape[0]
    dx=float(x_array[1]-x_array[0])
    dk=abs(float(k_array[1]-k_array[0]))
    nbins=int(math.floor(math.sqrt(3)*(N_grid//2)+0.5))+1
    counts=cp.zeros(nbins,dtype=cp.int64)
    occupation=cp.zeros(nbins,dtype=cp.float64)
    planes=min(slab_size,N_grid)
    bins=cp.empty((planes,N_grid,N_grid),dtype=cp.int32)
    weights=cp.empty((planes,N_grid,N_grid),dtype=cp.float64)

    for start in range(0,N_grid,planes):
        stop=min(start+planes,N_grid)
        shell=bins[:stop-start]
        values=weights[:stop-start]
        shell_modes(psi_k[start:stop],k_array[start:stop,None,None],
                    k_array[None,:,None],k_array[None,None,:],dk,shell,values)
        counts+=cp.bincount(shell.ravel(),minlength=nbins)
        occupation+=cp.bincount(shell.ravel(),weights=values.ravel(),minlength=nbins)

    Nk=occupation*(dx**3/psi_k.size)
    nk=Nk/(cp.maximum(counts,1)*dk**3)
    k=cp.arange(nbins,dtype=cp.float64)*dk
    edges=(cp.arange(nbins+1,dtype=cp.float64)-0.5)*dk
    edges[0]=0
    # Keep the cube corners for normalization; outer spherical shells are incomplete.
    complete_shells=edges[1:]<=N_grid*dk/2
    return dict(k=k,nk=nk,Nk=Nk,counts=counts,edges=edges,
                complete_shells=complete_shells)
