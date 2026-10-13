import cupy as cp

sum_partial=cp.RawKernel(
    r'''
#include<cupy/complex.cuh>
extern "C" __global__
void sum_partial(complex<double>* psi, double* partial, const long long count)
{
    
    __shared__ double sums[256];
    const int t=threadIdx.x;
    const long long tid=(long long)blockIdx.x*blockDim.x+t;
    const long long stride=(long long)blockDim.x*gridDim.x;
    double local_sum=0;
    for(long long j=tid;j<count;j+=stride)
    {
        local_sum=local_sum+psi[j].real()*psi[j].real()+psi[j].imag()*psi[j].imag();
    }
    sums[t]=local_sum;
    __syncthreads();
    for (int offset=blockDim.x/2;offset>0;offset/=2)
    {
        if (t<offset)
        {
            sums[t]+=sums[t+offset];
        }
        __syncthreads();
        
    }
    
    if (t==0)
    {   
        partial[blockIdx.x]=sums[0];
    }
}

''',"sum_partial"
)

sum_finish=cp.RawKernel(r'''

    extern "C" __global__
    void sum_finish(double* partial, double* x_array, double* norm, long long previous_block_nums)
    {
        const int t=threadIdx.x;
        __shared__ double sums[256];
        double local_sum=0;
        for (long long j=t;j<previous_block_nums;j+=blockDim.x)
        {
            local_sum+=partial[j];
        }
        sums[t]=local_sum;
        __syncthreads();
        for (int offset=blockDim.x/2;offset>0;offset/=2)
        {
            if(t<offset)
            {
                sums[t]+=sums[t+offset];
            }
            __syncthreads();
        }
        if(t==0)
        {
            double dx=x_array[1]-x_array[0];
            norm[0]=dx*dx*dx*sums[0];
        }

    }
''','sum_finish')

def get_N(psi,x_array):
    threads=256
    blocks=min((psi.size+threads-1)//256,4096)
    partial=cp.empty(blocks,dtype=cp.float64)
    norm=cp.empty((),dtype=cp.float64)
    sum_partial((blocks,),(threads,),(psi,partial,int(psi.size)))
    sum_finish((1,),(threads,),(partial,x_array, norm,int(blocks)))
    return norm

def normalise(psi,x_array):
    norm=get_N(psi,x_array)
    psi/=cp.sqrt(norm)

def norm_fft(psi,x_array,k_array):
    dx=x_array[1]-x_array[0]
    dk=abs(k_array[1]-k_array[0])
    psi_k=cp.fft.fftn(psi,norm='ortho')
    psi_k*=(dx/dk)**(3/2)
    return psi_k

def norm_ifft(psi_k,x_array,k_array):
    dx=x_array[1]-x_array[0]
    dk=abs(k_array[1]-k_array[0])
    psi=cp.fft.ifftn(psi_k,norm='ortho')
    psi*=(dk/dx)**(3/2)
    return psi

def get_beta(N1,N2,a_11,a_22,a_12):
    a0=5.29177210544e-11
    l=1e-6
    beta_11=8*cp.pi*N1*(a_11*a0)/l
    beta_12=8*cp.pi*N2*(a_12*a0)/l
    beta_21=8*cp.pi*N1*(a_12*a0)/l
    beta_22=8*cp.pi*N2*(a_22*a0)/l
    return beta_11, beta_12, beta_21, beta_22

