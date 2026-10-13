import cupy as cp

kinetic_full_step=cp.RawKernel(
    r'''
#include<cupy/complex.cuh>
extern "C" __global__
void kinetic_full_step(complex<double>* psi1_k,complex<double>* psi2_k, const double* k_array,const double dt, const long long N_grid )
{  
    const long long tid=(long long)blockDim.x*blockIdx.x+threadIdx.x;
    long long count=(long long)N_grid*N_grid*N_grid;
    if (tid>=count) return;
    long long k=tid%N_grid;
    long long j=(tid%((long long)N_grid*N_grid))/N_grid;
    long long i=tid/((long long)N_grid*N_grid);
    double K_2=k_array[i]*k_array[i]+k_array[j]*k_array[j]+k_array[k]*k_array[k];
    double phase=K_2*dt;
    complex<double> propagator=exp(complex<double>(0,-phase));
    psi1_k[tid]=psi1_k[tid]*propagator;
    psi2_k[tid]=psi2_k[tid]*propagator;
}
'''
,'kinetic_full_step')


masked_kinetic_full_step=cp.RawKernel(
    r'''
#include<cupy/complex.cuh>
extern "C" __global__
void masked_kinetic_full_step(complex<double>* psi1_k,complex<double>* psi2_k, const double* k_array,const double dt, const long long N_grid, double k_mask )
{  
    const long long tid=(long long)blockDim.x*blockIdx.x+threadIdx.x;
    long long count=(long long)N_grid*N_grid*N_grid;
    if (tid>=count) return;
    long long k=tid%N_grid;
    long long j=(tid%((long long)N_grid*N_grid))/N_grid;
    long long i=tid/((long long)N_grid*N_grid);
    double K_2=k_array[i]*k_array[i]+k_array[j]*k_array[j]+k_array[k]*k_array[k];
    if (K_2>k_mask*k_mask){
    psi1_k[tid]=complex<double>(0,0);
    psi2_k[tid]=complex<double>(0,0);
    return;}
    double phase=K_2*dt;
    complex<double> propagator=exp(complex<double>(0,-phase));
    psi1_k[tid]=psi1_k[tid]*propagator;
    psi2_k[tid]=psi2_k[tid]*propagator;
}
'''
,'masked_kinetic_full_step')


imag_kinetic_full_step=cp.RawKernel(
    r'''
#include<cupy/complex.cuh>
extern "C" __global__
void imag_kinetic_full_step(complex<double>* psi1_k, complex<double>* psi2_k, const double* k_array,const double dt, const long long N_grid )
{  
    const long long tid=(long long)blockDim.x*blockIdx.x+threadIdx.x;
    long long count=(long long)N_grid*N_grid*N_grid;
    if (tid>=count) return;
    long long k=tid%N_grid;
    long long j=(tid%((long long)N_grid*N_grid))/N_grid;
    long long i=tid/((long long)N_grid*N_grid);
    double K_2=k_array[i]*k_array[i]+k_array[j]*k_array[j]+k_array[k]*k_array[k];
    double phase=K_2*dt;
    complex<double> propagator=exp(complex<double>(-phase,0));
    psi1_k[tid]=psi1_k[tid]*propagator;
    psi2_k[tid]=psi2_k[tid]*propagator;
}
'''
,'imag_kinetic_full_step')


def make_masked_kinetic_full_step(*, k):
    raw_kernel = masked_kinetic_full_step  # Your existing RawKernel
    fixed_k = cp.float64(k)

    def launch(grid, block, args, **kwargs):
        return raw_kernel(
            grid,
            block,
            (*args, fixed_k),
            **kwargs,
        )

    return launch
