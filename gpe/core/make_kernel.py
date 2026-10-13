import cupy as cp
from .kernels import *

def make_real_spatial_kernel(stage)->cp.RawKernel:
        if stage.potential1_params["static"]["activate"]:
            time_independent_expression1=stage.potential1_params["static"]["expression"]
        else:
            time_independent_expression1="0"
        if stage.potential2_params["static"]["activate"]:
            time_independent_expression2=stage.potential2_params["static"]["expression"]
        else:
            time_independent_expression2="0"
        if stage.potential1_params["dynamic"]["activate"] or stage.potential2_params["dynamic"]["activate"]:
            if stage.potential1_params["dynamic"]["activate"]:
                time_dependent_expression1=stage.potential1_params["dynamic"]["expression"]
            else:
                time_dependent_expression1="0"
            if stage.potential2_params["dynamic"]["activate"]:
                time_dependent_expression2=stage.potential2_params["dynamic"]["expression"]
            else:
                time_dependent_expression2="0"


            dynamic_real_halfstep=cp.RawKernel(
                r'''
                #include<cupy/complex.cuh>
                extern "C" __global__
                void dynamic_real_halfstep(complex<double>* psi1, complex<double>* psi2, const complex<double>* V_static,
                const double* x_array, const double beta_11, const double beta_12, const double beta_21, const double beta_22, const double dt,
                long long N_grid, double t )
                {
                const long long tid =(long long)blockDim.x*blockIdx.x+threadIdx.x;
                const long long count=(long long) N_grid*N_grid*N_grid;
                
                if(tid>=count) return;

                long long k=tid%N_grid;
                long long j=(tid%((long long)N_grid*N_grid))/N_grid;
                long long i=(tid/((long long)N_grid*N_grid));
                const double x=x_array[i];
                const double y=x_array[j];
                const double z=x_array[k];
                const double time_potential1=__TIME_POTENTIAL_1;
                const double time_potential2=__TIME_POTENTIAL_2;

                const double time_independent_potential1=__TIME_INDEPENDENT_POTENTIAL_1;
                const double time_independent_potential2=__TIME_INDEPENDENT_POTENTIAL_2;
                double abs2psi1=psi1[tid].real()*psi1[tid].real()+psi1[tid].imag()*psi1[tid].imag();
                double abs2psi2=psi2[tid].real()*psi2[tid].real()+psi2[tid].imag()*psi2[tid].imag();
                double V_static_real=V_static[tid].real();
                double V_static_imag=V_static[tid].imag();
                double real_phase1=(time_potential1+time_independent_potential1+beta_11*abs2psi1+beta_12*abs2psi2+V_static_real)*(dt/2);
                double real_phase2=(time_potential2+time_independent_potential2+beta_22*abs2psi2+beta_21*abs2psi1+V_static_real)*(dt/2);

                double imag_phase=V_static_imag*(dt/2);
                complex<double> propagator1=exp(complex<double>(imag_phase,-real_phase1));
                complex<double> propagator2=exp(complex<double>(imag_phase,-real_phase2));
                psi1[tid]=psi1[tid]*propagator1;
                psi2[tid]=psi2[tid]*propagator2;
                }

                '''.replace("__TIME_POTENTIAL_1",time_dependent_expression1).replace("__TIME_POTENTIAL_2",time_dependent_expression2).replace("__TIME_INDEPENDENT_POTENTIAL_1",time_independent_expression1).replace("__TIME_INDEPENDENT_POTENTIAL_2",time_independent_expression2)
            ,'dynamic_real_halfstep')
            return dynamic_real_halfstep
        else:
            static_real_halfstep=cp.RawKernel(
                r'''
                #include<cupy/complex.cuh>
                extern "C" __global__ 
                void static_real_halfstep(complex<double>* psi1, complex<double>* psi2, const complex<double>* V_static,
                const double* x_array, const double beta_11, const double beta_12, const double beta_21, const double beta_22, const double dt,
                long long N_grid)
                {
                const long long tid =(long long)blockDim.x*blockIdx.x+threadIdx.x;
                const long long count=(long long) N_grid*N_grid*N_grid;
                
                if(tid>=count) return;

                long long k=tid%N_grid;
                long long j=(tid%((long long)N_grid*N_grid))/N_grid;
                long long i=(tid/((long long)N_grid*N_grid));
                const double x=x_array[i];
                const double y=x_array[j];
                const double z=x_array[k];
  

                const double time_independent_potential1=__TIME_INDEPENDENT_POTENTIAL_1;
                const double time_independent_potential2=__TIME_INDEPENDENT_POTENTIAL_2;
                double abs2psi1=psi1[tid].real()*psi1[tid].real()+psi1[tid].imag()*psi1[tid].imag();
                double abs2psi2=psi2[tid].real()*psi2[tid].real()+psi2[tid].imag()*psi2[tid].imag();
                double V_static_real=V_static[tid].real();
                double V_static_imag=V_static[tid].imag();
                double real_phase1=(time_independent_potential1+beta_11*abs2psi1+beta_12*abs2psi2+V_static_real)*(dt/2);
                double real_phase2=(time_independent_potential2+beta_22*abs2psi2+beta_21*abs2psi1+V_static_real)*(dt/2);

                double imag_phase=V_static_imag*(dt/2);
                complex<double> propagator1=exp(complex<double>(imag_phase,-real_phase1));
                complex<double> propagator2=exp(complex<double>(imag_phase,-real_phase2));
                psi1[tid]=psi1[tid]*propagator1;
                psi2[tid]=psi2[tid]*propagator2;
                }
                '''.replace("__TIME_INDEPENDENT_POTENTIAL_1",time_independent_expression1).replace("__TIME_INDEPENDENT_POTENTIAL_2",time_independent_expression2),
                "static_real_halfstep"
            )
            return static_real_halfstep
def make_imag_spatial_kernel(stage)->cp.RawKernel:
        if stage.potential1_params["static"]["activate"]:
            time_independent_expression1=stage.potential1_params["static"]["expression"]
        else:
            time_independent_expression1="0"
        if stage.potential2_params["static"]["activate"]:
            time_independent_expression2=stage.potential2_params["static"]["expression"]
        else:
            time_independent_expression2="0"
        imag_static_real_halfstep = cp.RawKernel(
        r'''
        #include <cupy/complex.cuh>
        extern "C" __global__
        void imag_static_real_halfstep(complex<double>* psi1,complex<double>* psi2, const double* V_static,const double* x_array, 
        const double beta_11, const double beta_12, const double beta_21, const double beta_22, const double dt,const long long N_grid ) 

        {
            const long long tid =(long long) blockDim.x * blockIdx.x + threadIdx.x;
            const long long count=(long long)N_grid*N_grid*N_grid;
            if (tid>=count) return;
            long long k=tid%N_grid;
            long long j=(tid%((long long)N_grid*N_grid))/N_grid;
            long long i=tid/((long long)N_grid*N_grid);
            const double x=x_array[i];
            const double y=x_array[j];
            const double z=x_array[k];
            const double time_independent_potential1=__TIME_INDEPENDENT_POTENTIAL_1;
            const double time_independent_potential2=__TIME_INDEPENDENT_POTENTIAL_2;
            double abs2psi1=psi1[tid].real()*psi1[tid].real()+psi1[tid].imag()*psi1[tid].imag();
            double abs2psi2=psi2[tid].real()*psi2[tid].real()+psi2[tid].imag()*psi2[tid].imag();
            double V_static_real=V_static[tid];
            double real_phase1=(time_independent_potential1+beta_11*abs2psi1+beta_12*abs2psi2+V_static_real)*(dt/2);
            double real_phase2=(time_independent_potential2+beta_22*abs2psi2+beta_21*abs2psi1+V_static_real)*(dt/2);
            complex<double> propagator1=exp(complex<double>(-real_phase1,0));
            complex<double> propagator2=exp(complex<double>(-real_phase2,0));
            psi1[tid]=psi1[tid]*propagator1;
            psi2[tid]=psi2[tid]*propagator2;
        }
        '''.replace("__TIME_INDEPENDENT_POTENTIAL_1",time_independent_expression1).replace("__TIME_INDEPENDENT_POTENTIAL_2",time_independent_expression2), 'imag_static_real_halfstep')
        return imag_static_real_halfstep


def make_real_momentum_kernel(stage)->cp.RawKernel:
     if stage.k_cut_params["activate"]:
          k_cut=stage.k_cut_params["k_cut"]
          return make_masked_kinetic_full_step(k=k_cut)
     else:
          return kinetic_full_step
def make_imag_momentum_kernel(stage)->cp.RawKernel:
    return imag_kinetic_full_step     
                                

      
