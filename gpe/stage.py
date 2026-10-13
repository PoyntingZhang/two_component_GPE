import cupy as cp
from .core.potentials import *
from .core.helper import *
from .core.make_propagator import *
from .core.initial_states import gaussian
from .io.state_io import load_wavefunctions




class stage:
    def __init__ (self,simulation_range:float,x_array:cp.ndarray, k_array:cp.ndarray, N_grids:int,
                  type:str, steps:int, dt:float,
                  initial_state:dict,box_params:dict, absorber_params:dict,
                   noise_params: dict,cloud_params:dict,monitor_params:dict,save_params:dict,potential1_params:dict,potential2_params:dict,k_cut_params:dict ):
        if not isinstance(x_array,cp.ndarray):
            raise TypeError("x_array should be a Cupy array")
        if not isinstance(k_array,cp.ndarray):
            raise TypeError("k_array should be a Cupy array")
        if not x_array.dtype==cp.float64:
            raise TypeError("x_array should be a real array")
        if not k_array.dtype==cp.float64:
            raise TypeError("k_array should be a real array")
        if not isinstance(type,str):
            raise TypeError("type of the stage should be a string")
        if type not in {"imaginary","real"}:
            raise KeyError("type of the stage should either be 'real' or 'imaginary")
        self.absorber_params=absorber_params
        self.noise_params=noise_params
        self.simulation_range=simulation_range
        self.x_array=x_array
        self.k_array=k_array
        self.N_grids=N_grids
        self.type=type
        self.steps=steps
        self.dt=dt
        self.initial_state=initial_state
        self.box_params=box_params
        
        self.N1=cloud_params["N1"]
        self.N2=cloud_params["N2"]
        self.save_params=save_params
        self.monitor_params=monitor_params
        self.potential1_params=potential1_params
        self.potential2_params=potential2_params
        self.beta_11,self.beta_12,self.beta_21,self.beta_22=get_beta(
            self.N1,self.N2,cloud_params["a11"],cloud_params["a22"],cloud_params["a12"])
        self.k_cut_params=k_cut_params
        self.psi1=None
        self.psi2=None
        self.staticV=None
        self.propagator=None
        self.name=""
        self.config={}

    def prepare(self, previous_psi=None):
        """Allocate the large arrays only when this stage is about to run."""
        self.psi1,self.psi2=self._get_initial_state(previous_psi)
        if self.type=="imaginary":
            self.staticV=self._get_box()
        else:
            self.staticV=get_total_static_potential(self._get_box(),self._get_absorber())
        if self.noise_params["activate"]:
            add_white_noise(self.N_grids,self.box_params["trap_depth"],
                            self.noise_params["fraction"],self.staticV,self.noise_params["seed"])
        if self.type=="real":
            self.propagator=make_real_propagator(self)
        else:
            self.propagator=make_imag_propagator(self)

    def release(self):
        """The runner keeps the returned pair; this configuration keeps no 3D arrays."""
        self.psi1=None
        self.psi2=None
        self.staticV=None
        self.propagator=None

    def _get_initial_state(self, previous_psi):
        kind=self.initial_state["type"]
        if kind=="continue":
            if previous_psi is None:
                raise ValueError("A continue stage needs the preceding stage's two wavefunctions.")
            return previous_psi
        if kind=="load":
            return load_wavefunctions(self.initial_state["path"],self)
        if kind=="gaussian":
            return (gaussian(self.x_array,self.initial_state["sigma1"]),
                    gaussian(self.x_array,self.initial_state["sigma2"]))
        raise ValueError(f"Unknown initial state: {kind}")
    def _get_box(self):
        if self.box_params["activate"]==False:
            return cp.zeros((self.N_grids,self.N_grids,self.N_grids),dtype=cp.float64)
        if self.box_params["activate"]==True:
            x_grid=self.x_array
            box_radius=self.box_params["R"]
            box_length=self.box_params["L"]
            wall_thickness=self.box_params["wall_thickness"]
            trap_depth=self.box_params["trap_depth"]
            box=box_potential(x_grid,box_radius,box_length,wall_thickness,trap_depth)
            return box
          
    def _get_absorber(self):
        if self.absorber_params["activate"]==True:
            x_grid=self.x_array
            activation_distance=self.absorber_params["activation_distance"]
            amplitude=self.absorber_params["strength"]
            simulation_range=self.simulation_range
            absorber=absorption_potential(activation_distance,amplitude,x_grid, simulation_range)
            return absorber # this is a real array
        else:
            return cp.zeros((self.N_grids,self.N_grids,self.N_grids),dtype=cp.float64)
