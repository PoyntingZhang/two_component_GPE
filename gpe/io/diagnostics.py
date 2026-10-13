"""Numerical diagnostics used by the output writers."""

import cupy as cp
from ..core.helper import get_N

sum_abs2 = cp.ReductionKernel(
    'complex128 z',  # input params
    'float64 result',  # output params
    'z.real()*z.real()+z.imag()*z.imag()',  # map
    'a + b',  # reduce
    'result=a',  # post-reduction map
    '0',  # identity value
    'sum_abs2' , # kernel name
    reduce_type='double',
)

energy_sum = cp.ReductionKernel(
    'complex128 z, float64 kx, float64 ky, float64 kz',
    'float64 result',
    '(z.real()*z.real()+z.imag()*z.imag())*(kx*kx+ky*ky+kz*kz)',
    'a + b', 'result=a', '0', 'energy_sum', reduce_type='double',
)

angular_sum = cp.ReductionKernel(
    'complex128 z, complex128 momentum, float64 coordinate',
    'float64 result',
    'coordinate*(z.real()*momentum.real()+z.imag()*momentum.imag())',
    'a + b', 'result=a', '0', 'angular_sum', reduce_type='double',
)

def is_due(i,interval):
    return i>0 and interval is not None and interval>0 and i%interval==0

def get_E(psi_k,x_array,k_array):
    # psi_k is the unnormalized fftn of the completed state.
    dx=x_array[1]-x_array[0]
    energy=energy_sum(psi_k,k_array[:,None,None],
                      k_array[None,:,None],k_array[None,None,:])
    return energy*dx**3/psi_k.size

def get_L(psi,x_array,k_array,slab_size=16):
    # Lz = Re integral psi* (x py - y px) psi dV; spectral derivatives.
    dx=x_array[1]-x_array[0]
    L=cp.zeros((),dtype=cp.float64)
    for start in range(0,psi.shape[2],slab_size):
        slab=psi[:,:,start:start+slab_size]
        slab_k=cp.fft.fftn(slab,axes=(0,1))
        momentum=cp.fft.ifftn(slab_k*k_array[None,:,None],axes=(0,1))
        L+=angular_sum(slab,momentum,x_array[:,None,None])
        del momentum
        momentum=cp.fft.ifftn(slab_k*k_array[:,None,None],axes=(0,1))
        L-=angular_sum(slab,momentum,x_array[None,:,None])
        del momentum,slab_k
    return L*dx**3
