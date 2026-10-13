import cupy as cp
from .make_kernel import *
from .helper import *
from cupyx.scipy.fft import get_fft_plan, fftn, ifftn


class InPlaceFFTError(RuntimeError):
    """An FFT returned a different data buffer despite the in-place requirement."""


def _checked_fft(psi, *, axes, plan, component, inverse=False):
    """Check each transform while retaining its input allocation for comparison."""
    transform = ifftn if inverse else fftn
    result = transform(psi, axes=axes, plan=plan, overwrite_x=True)
    if result.data.ptr != psi.data.ptr:
        direction = "Inverse" if inverse else "Forward"
        raise InPlaceFFTError(
            f"{direction} FFT for {component} returned a different GPU data buffer "
            "despite overwrite_x=True. The in-place memory requirement was not met; "
            "this does not by itself indicate an out-of-memory error. "
            "The timestep may already be partially applied; restart from a saved state."
        )
    return result


def make_real_propagator(stage):
    # this propagator maker is a funciton that take stage object as an input
    # the output is a funciton that also requires a stage object as an input
    spatial_kernel=make_real_spatial_kernel(stage)
    momentum_kernel=make_real_momentum_kernel(stage)
    axes=(0,1,2)
    
    
    plan=get_fft_plan(stage.psi1,axes=axes,value_type="C2C")

    if not (stage.potential1_params["dynamic"]["activate"] or stage.potential2_params["dynamic"]["activate"]):
        

        def time_independent_propagator(stage):
            psi1=stage.psi1
            psi2=stage.psi2
            V_static=stage.staticV
            beta_11=stage.beta_11
            beta_12=stage.beta_12
            beta_21=stage.beta_21
            beta_22=stage.beta_22
            k_array=stage.k_array
            x_array=stage.x_array
            dt=stage.dt
            N_grids=stage.N_grids
            assert isinstance(psi1,cp.ndarray)
            assert isinstance(psi2,cp.ndarray)
            assert isinstance(V_static, cp.ndarray)
        
            assert psi1.dtype==cp.complex128
            assert psi2.dtype==cp.complex128
            assert V_static.dtype==cp.complex128
            assert psi1.flags.c_contiguous
            assert psi2.flags.c_contiguous
            assert V_static.flags.c_contiguous
            assert psi1.shape==(N_grids,N_grids,N_grids)
            assert psi2.shape==(N_grids,N_grids,N_grids)
            assert V_static.shape==psi1.shape
            assert V_static.shape==psi2.shape
            assert isinstance(k_array,cp.ndarray)
            assert k_array.dtype==cp.float64
            assert k_array.shape==(N_grids,)
            assert k_array.flags.c_contiguous

            assert x_array.dtype==cp.float64
            assert x_array.shape==(N_grids,)
            assert x_array.flags.c_contiguous

            



            threads=256
            blocks=(psi1.size+threads-1)//threads
            grid=(blocks,)
            block=(threads,)
            beta_11=float(beta_11)
            beta_12=float(beta_12)
            beta_21=float(beta_21)
            beta_22=float(beta_22)
            dt=float(dt)
            N_grids=int(N_grids)
            spatial_kernel(grid,block,(psi1,psi2,V_static,x_array,beta_11,beta_12,beta_21,beta_22,dt,N_grids))
            psi1=_checked_fft(psi1,axes=axes,plan=plan,component="psi1")
            psi2=_checked_fft(psi2,axes=axes,plan=plan,component="psi2")
  
            momentum_kernel(grid,block,(psi1,psi2, k_array,dt,N_grids))
            psi1=_checked_fft(psi1,axes=axes,plan=plan,component="psi1",inverse=True)
            psi2=_checked_fft(psi2,axes=axes,plan=plan,component="psi2",inverse=True)
            spatial_kernel(grid,block,(psi1,psi2,V_static,x_array,beta_11,beta_12,beta_21,beta_22,dt,N_grids))
            return psi1, psi2
        return time_independent_propagator
    else:
        def time_dependent_propagator(stage,t):
            psi1=stage.psi1
            psi2=stage.psi2
            V_static=stage.staticV
            beta_11=stage.beta_11
            beta_12=stage.beta_12
            beta_21=stage.beta_21
            beta_22=stage.beta_22
            k_array=stage.k_array
            x_array=stage.x_array
            dt=stage.dt
            N_grids=stage.N_grids
            assert isinstance(psi1,cp.ndarray)
            assert isinstance(psi2,cp.ndarray)
            assert isinstance(V_static, cp.ndarray)
        
            assert psi1.dtype==cp.complex128
            assert psi2.dtype==cp.complex128
            assert V_static.dtype==cp.complex128
            assert psi1.flags.c_contiguous
            assert psi2.flags.c_contiguous
            assert V_static.flags.c_contiguous
            assert psi1.shape==(N_grids,N_grids,N_grids)
            assert psi2.shape==(N_grids,N_grids,N_grids)
            assert V_static.shape==psi1.shape
            assert V_static.shape==psi2.shape
            assert isinstance(k_array,cp.ndarray)
            assert k_array.dtype==cp.float64
            assert k_array.shape==(N_grids,)
            assert k_array.flags.c_contiguous

            assert x_array.dtype==cp.float64
            assert x_array.shape==(N_grids,)
            assert x_array.flags.c_contiguous
            


            threads=256
            blocks=(psi1.size+threads-1)//threads
            beta_11=float(beta_11)
            beta_12=float(beta_12)
            beta_21=float(beta_21)
            beta_22=float(beta_22)
            grid=(blocks,)
            block=(threads,)
            dt=float(dt)
            N_grids=int(N_grids)
        
            t_mid=float(t)+dt/2


            spatial_kernel(grid,block,(psi1,psi2,V_static,x_array,beta_11,beta_12,beta_21,beta_22,dt,N_grids,t_mid))
            psi1=_checked_fft(psi1,axes=axes,plan=plan,component="psi1")
            psi2=_checked_fft(psi2,axes=axes,plan=plan,component="psi2")
            momentum_kernel(grid,block,(psi1,psi2,k_array,dt,N_grids))
            psi1=_checked_fft(psi1,axes=axes,plan=plan,component="psi1",inverse=True)
            psi2=_checked_fft(psi2,axes=axes,plan=plan,component="psi2",inverse=True)
            spatial_kernel(grid,block,(psi1,psi2,V_static,x_array,beta_11,beta_12,beta_21,beta_22,dt,N_grids,t_mid))
            
            return psi1, psi2
        return time_dependent_propagator

def make_imag_propagator(stage):
    spatial_kernel=make_imag_spatial_kernel(stage)
    momentum_kernel=make_imag_momentum_kernel(stage)
    axes=(0,1,2)
    
    
    plan=get_fft_plan(stage.psi1,axes=axes,value_type="C2C")

    def standard_imaginary_step(stage):
        psi1=stage.psi1
        psi2=stage.psi2
        V_static=stage.staticV
        beta_11=stage.beta_11
        beta_22=stage.beta_22
        beta_12=stage.beta_12
        beta_21=stage.beta_21
        k_array=stage.k_array
        dt=stage.dt
        N_grids=stage.N_grids
        x_array=stage.x_array
        assert isinstance(psi1,cp.ndarray)
        assert isinstance(psi2,cp.ndarray)
        assert isinstance(V_static,cp.ndarray)
        assert psi1.dtype==cp.complex128
        assert psi2.dtype==cp.complex128
        assert V_static.dtype==cp.float64 #here for imaginary time evolution we require
        # potential being real
        assert psi1.flags.c_contiguous
        assert psi2.flags.c_contiguous
        assert V_static.flags.c_contiguous
        assert psi1.shape==(N_grids,N_grids,N_grids)
        assert psi2.shape==(N_grids,N_grids,N_grids)
        assert V_static.shape==psi1.shape 
        assert isinstance(x_array,cp.ndarray)
        assert x_array.flags.c_contiguous
        assert x_array.shape==(N_grids,)
        assert x_array.dtype==cp.float64
        assert isinstance(k_array, cp.ndarray)
        assert k_array.dtype == cp.float64
        assert k_array.shape == (N_grids,)
        assert k_array.flags.c_contiguous
        
        
        threads=256;
        blocks=(psi1.size+threads-1)//threads
        grid=(blocks,)
        block=(threads,)
        beta_11=float(beta_11)
        beta_12=float(beta_12)
        beta_21=float(beta_21)
        beta_22=float(beta_22)
        dt=float(dt)
        N_grids=int(N_grids)
        spatial_kernel(grid,block,(psi1,psi2,V_static,x_array,beta_11,beta_12,beta_21,beta_22,dt,N_grids))
        psi1=_checked_fft(psi1,axes=axes,plan=plan,component="psi1")
        psi2=_checked_fft(psi2,axes=axes,plan=plan,component="psi2")
        momentum_kernel(grid,block,(psi1,psi2,k_array,dt,N_grids))
        psi1=_checked_fft(psi1,axes=axes,plan=plan,component="psi1",inverse=True)
        psi2=_checked_fft(psi2,axes=axes,plan=plan,component="psi2",inverse=True)

        spatial_kernel(grid,block,(psi1,psi2,V_static,x_array,beta_11,beta_12,beta_21,beta_22,dt,N_grids))
        normalise(psi1,x_array)
        normalise(psi2,x_array)
        return psi1,psi2
    return standard_imaginary_step
    

                
