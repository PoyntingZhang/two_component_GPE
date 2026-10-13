import cupy as cp

def get_grids(N_grids,simulation_range):
    x_grid=cp.linspace(-simulation_range/2,simulation_range/2,N_grids,endpoint=False)
    k_grid=2*cp.pi*cp.fft.fftfreq(N_grids,simulation_range/N_grids)
    return x_grid, k_grid


