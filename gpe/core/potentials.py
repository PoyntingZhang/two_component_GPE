import cupy as cp

def box_potential(x_grid,box_radius,box_length,wall_thickness,trap_depth):
    X=x_grid[:,None,None]
    Y=x_grid[None,:,None]
    Z=x_grid[None,None,:]
    tube=cp.exp(-(cp.sqrt(X**2+Y**2)-box_radius)**2/(2*wall_thickness**2))
    sheet=cp.exp(-(cp.abs(Z)-box_length/2)**2/(2*wall_thickness**2))
    box_trap= cp.maximum(tube,sheet)
    box_trap *= trap_depth
    return box_trap

def absorption_potential(activation_distance, amplitude,x_grid,simulation_range):
    X=x_grid[:,None,None]
    Y=x_grid[None,:,None]
    Z=x_grid[None,None,:]
    x1=cp.maximum(activation_distance-(X+simulation_range/2),0)
    x2=cp.maximum(activation_distance-(simulation_range/2-X),0)
    y1=cp.maximum(activation_distance-(Y+simulation_range/2),0)
    y2=cp.maximum(activation_distance-(simulation_range/2-Y),0)
    z1=cp.maximum(activation_distance-(Z+simulation_range/2),0)
    z2=cp.maximum(activation_distance-(simulation_range/2-Z),0)

    xy_absorber=x1+x2+y1+y2
    z_absorber=z1+z2
    absorber=xy_absorber+z_absorber
    absorber*=amplitude
    return absorber


def get_total_static_potential(real_grid,imag_grid):
    total_potential=cp.empty(real_grid.shape,dtype=cp.complex128)
    total_potential.real[:,:,:]=real_grid
    cp.negative(imag_grid,out=total_potential.imag)
    del real_grid
    del imag_grid
    return total_potential

def add_white_noise(
    N_grid, trap_depth, fraction, total_potential, seed=42
):
    rng = cp.random.default_rng(seed=seed)
    sigma = float(fraction * trap_depth)

    planes = min(16, N_grid)
    buffer = cp.empty(
        (planes, N_grid, N_grid),
        dtype=cp.float64,
    )

    real_potential = total_potential.real

    for start in range(0, N_grid, planes):
        stop = min(start + planes, N_grid)
        noise = buffer[:stop - start]

        rng.standard_normal(dtype=cp.float64, out=noise)
        noise *= sigma

        target = real_potential[start:stop]
        cp.add(target, noise, out=target)





