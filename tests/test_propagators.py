"""Actual CuPy/cuFFT tests; run on the target GPU before research simulations."""
import itertools
from pathlib import Path
import tempfile
import unittest
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp=None

class PropagatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if cp is None: raise unittest.SkipTest('CuPy/CUDA required')
        try:
            if cp.cuda.runtime.getDeviceCount()==0: raise unittest.SkipTest('CUDA GPU required')
        except cp.cuda.runtime.CUDARuntimeError as exc:
            raise unittest.SkipTest(str(exc))

    def build(self,kind='real',dynamic=(False,False),cutoff=False):
        from config.yaml_parser import new_config,dump_yaml
        from gpe.stage_factory import make_stages
        config=new_config();config['simulation'].update(N_grids=8,simulation_range=2*np.pi)
        s=config['stages'][0];s['evolution'].update(type=kind,steps=3,dt=.01)
        s['box']['enabled']=False
        s['cloud'].update(N1=100,N2=200,a11=11,a22=23,a12=7)
        s['save'].update(scalar_interval=0,save_final=False)
        s['monitor']=dict.fromkeys(s['monitor'],0)
        s['k_cut'].update(enabled=cutoff,value=2.5)
        for i,on in enumerate(dynamic,1):
            s[f'potential{i}']['dynamic'].update(enabled=on,expression=f'{i}*t/4')
            s[f'potential{i}']['static'].update(enabled=True,expression=f'{i}/2')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'config.yaml';path.write_text(dump_yaml(config));item=make_stages(path)[0]
        item.prepare();return item

    def test_coupled_plane_wave_phases_and_in_place_fft_all_real_branches(self):
        from gpe.launch import launch_stage
        for d1,d2,cutoff in itertools.product((False,True),repeat=3):
            with self.subTest(dynamic=(d1,d2),cutoff=cutoff):
                s=self.build(dynamic=(d1,d2),cutoff=cutoff)
                x=cp.asnumpy(s.x_array)
                fields=[np.broadcast_to(np.exp(1j*m*x[:,None,None]),(8,)*3).copy()/(2*np.pi)**1.5 for m in (1,2)]
                s.psi1=cp.asarray(fields[0]);s.psi2=cp.asarray(fields[1])
                pointers=(s.psi1.data.ptr,s.psi2.data.ptr)
                result=launch_stage(s);t=s.steps*s.dt
                energies=(1+.5+(s.beta_11+s.beta_12)/(2*np.pi)**3,
                          4+1+(s.beta_21+s.beta_22)/(2*np.pi)**3)
                for i in (0,1):
                    phase=energies[i]*t+((i+1)*t*t/8 if (d1,d2)[i] else 0)
                    np.testing.assert_allclose(cp.asnumpy(result[i]),fields[i]*np.exp(-1j*phase),rtol=2e-11,atol=2e-12)
                self.assertEqual(pointers,(s.psi1.data.ptr,s.psi2.data.ptr));s.release()

    def test_imaginary_separate_normalization_and_fft_storage(self):
        from gpe.launch import launch_stage
        from gpe.core.helper import get_N
        s=self.build(kind='imaginary');pointers=(s.psi1.data.ptr,s.psi2.data.ptr)
        a,b=launch_stage(s)
        self.assertAlmostEqual(float(get_N(a,s.x_array)),1,places=12)
        self.assertAlmostEqual(float(get_N(b,s.x_array)),1,places=12)
        self.assertEqual(pointers,(a.data.ptr,b.data.ptr));s.release()

if __name__=='__main__': unittest.main()
