import cupy as cp
from .helper import normalise

def gaussian(x_array,sigma):
    psix=cp.exp(-x_array**2/(2*sigma**2),dtype=cp.complex128)
    psiX=psix[:,None,None]
    psiy=cp.exp(-x_array**2/(2*sigma**2),dtype=cp.complex128)
    psiY=psiy[None,:,None]   
    psiz=cp.exp(-x_array**2/(2*sigma**2),dtype=cp.complex128)
    psiZ=psiz[None,None,:]  
    psi=psiX*psiY*psiZ
    normalise(psi,x_array)

    return psi
