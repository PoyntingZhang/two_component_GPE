"""CPU tests for workflow plumbing; GPU kernels are deliberately not executed."""

import csv
import importlib
import json
import io
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from config.yaml_parser import dump_yaml, new_config, new_stage
from gpe.core.expressions import cuda_expression


def norm(psi, x):
    return np.sum(np.abs(psi)**2) * (x[1] - x[0])**3


class ExpressionTests(unittest.TestCase):
    def test_cuda_arithmetic_preserves_powers_division_and_functions(self):
        self.assertEqual(cuda_expression("1/2*y**2"), "((1.0 / 2.0) * pow(y, 2.0))")
        self.assertEqual(cuda_expression("abs(-x) + cos(t)", True), "(fabs((-x)) + cos(t))")
        self.assertEqual(cuda_expression("pi"), repr(np.pi))
        with self.assertRaises(ValueError):
            cuda_expression("t*y")
        with self.assertRaises(ValueError):
            cuda_expression("x; return 0")


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        # Scope the substitute to these tests, including when collected with GPU tests.
        modules = patch.dict(sys.modules)
        modules.start()
        self.addCleanup(modules.stop)
        for name in list(sys.modules):
            if name == "gpe" or name.startswith("gpe."):
                del sys.modules[name]
        cupy = ModuleType("cupy")
        cupy.__getattr__ = lambda name: getattr(np, name)
        cupy.asnumpy = np.asarray
        cupy.asarray = lambda value, blocking=False, **kwargs: np.asarray(value, **kwargs)
        cupy.fft = SimpleNamespace(fftn=Mock(wraps=np.fft.fftn), fftfreq=np.fft.fftfreq,
                                  fftshift=np.fft.fftshift, ifftn=np.fft.ifftn)
        def kernel(*args, **kwargs):
            return Mock(side_effect=AssertionError("A GPU kernel cannot execute in this CPU test."))
        cupy.RawKernel = cupy.ReductionKernel = cupy.ElementwiseKernel = kernel
        sys.modules["cupy"] = cupy
        sys.modules["cupyx"] = ModuleType("cupyx")
        sys.modules["cupyx.scipy"] = ModuleType("cupyx.scipy")
        sys.modules["cupyx.scipy.fft"] = SimpleNamespace(get_fft_plan=Mock(), fftn=Mock(), ifftn=Mock())
        self.cp = cupy
        self.parser = importlib.import_module("gpe.stage_factory")
        self.io = importlib.import_module("gpe.io.state_io")
        self.save = importlib.import_module("gpe.io.save")
        self.launch = importlib.import_module("gpe.launch")
        self.stage_module = importlib.import_module("gpe.stage")

    def config(self):
        config = new_config()
        config["simulation"].update(N_grids=4, simulation_range=8,
                                    data_root=str(self.root / "data"),
                                    monitor_root=str(self.root / "monitor"))
        first = config["stages"][0]
        first["evolution"].update(type="real", steps=6, dt=0.125)
        first["box"]["enabled"] = False
        first["monitor"] = dict.fromkeys(first["monitor"], 0)
        first["save"].update(scalar_interval=0, save_final=False)
        return config

    def parse(self, config):
        path = self.root / "experiment.yaml"
        path.write_text(dump_yaml(config))
        return self.parser.make_stages(path)

    def ready_stage(self):
        item = self.parse(self.config())[0]
        item.psi1 = np.full((4, 4, 4), .1+.2j, dtype=np.complex128)
        item.psi2 = np.full((4, 4, 4), .3-.1j, dtype=np.complex128)
        return item

    def test_stage_factory_preserves_component_order_and_compiles_both_potentials(self):
        c = self.config()
        s = c['stages'][0]
        s['cloud'].update(N1=100, N2=230, a11=10, a12=-5, a22=80)
        s['potential1']['static'].update(enabled=True, expression='1/2*y**2')
        s['potential2']['dynamic'].update(enabled=True, expression='cos(t)*x')
        item = self.parse(c)[0]
        self.assertIsNone(item.psi1)
        self.assertIsNone(item.psi2)
        factor = 8*np.pi*5.29177210544e-11/1e-6
        np.testing.assert_allclose([item.beta_11,item.beta_12,item.beta_21,item.beta_22],
                                   factor*np.array([100*10,230*(-5),100*(-5),230*80]))
        self.assertEqual(item.potential1_params['static']['expression'], '((1.0 / 2.0) * pow(y, 2.0))')
        self.assertTrue(item.potential2_params['dynamic']['activate'])

    def test_pair_snapshot_roundtrip_preserves_both_norms(self):
        item=self.ready_stage(); path=item.data_dir/'psi_00000006.json'
        with patch.object(self.io,'SLAB_SIZE',3):
            self.io.save_wavefunctions(path,item,6)
            pair=self.io.load_wavefunctions(path,item)
        for saved,original in zip(pair,(item.psi1,item.psi2)):
            np.testing.assert_array_equal(saved,original)
        self.assertFalse(np.shares_memory(*pair))
        self.assertEqual(json.loads(path.read_text())['step'],6)
        self.assertFalse(list(path.parent.glob('*.partial')))
        with self.assertRaises(FileExistsError): self.io.save_wavefunctions(path,item,6)

    def test_pair_load_rejects_reference_changes_missing_and_truncated_component(self):
        item=self.ready_stage();path=item.data_dir/'psi_00000006.json'
        self.io.save_wavefunctions(path,item,6)
        item.N2 += 1
        with self.assertRaisesRegex(ValueError,'N2'): self.io.load_wavefunctions(path,item)
        item.N2 -= 1
        member=path.parent/'psi2_00000006.npy'
        member.write_bytes(member.read_bytes()[:-16])
        with self.assertRaisesRegex(ValueError,'Incomplete'): self.io.load_wavefunctions(path,item)

    def test_failed_second_write_does_not_publish_pair_manifest(self):
        item=self.ready_stage();path=item.data_dir/'psi_00000006.json'
        calls=[]
        def fail_second(value):
            calls.append(None)
            if len(calls)==2: raise OSError('disconnected')
            return np.asarray(value)
        with patch.object(self.cp,'asnumpy',side_effect=fail_second):
            with self.assertRaisesRegex(OSError,'disconnected'): self.io.save_wavefunctions(path,item,6)
        self.assertFalse(path.exists())
        self.assertTrue((path.parent/'psi2_00000006.npy.partial').exists())

    def test_continue_keeps_both_fields_and_component2_driving_reaches_loop(self):
        c=self.config();first=c['stages'][0];second=new_stage()
        second['potential2']['dynamic'].update(enabled=True,expression='t*x')
        second['box']['enabled']=False
        c['stages'].append(second)
        items=self.parse(c);pair=(np.ones((4,4,4),dtype=complex)*.2,np.ones((4,4,4),dtype=complex)*.4)
        with patch.object(self.stage_module,'make_real_propagator',return_value=Mock()):
            items[1].prepare(pair)
        self.assertIs(items[1].psi1,pair[0]);self.assertIs(items[1].psi2,pair[1])
        times=[]
        def step(s,t):
            times.append(t)
            return s.psi1*.9,s.psi2*.8
        items[1].propagator=step;items[1].steps=2;items[1].dt=.125
        with patch.object(self.launch,'save_outputs') as save:
            result=self.launch.launch_stage(items[1])
        self.assertEqual(times,[0,.125]);self.assertEqual(save.call_count,3)
        np.testing.assert_allclose(result[0],pair[0]*.81)
        np.testing.assert_allclose(result[1],pair[1]*.64)
        items[1].release();self.assertIsNone(items[1].psi1);self.assertIsNone(items[1].psi2)

    def test_component_diagnostics_have_separate_files_and_correct_reference_populations(self):
        item=self.ready_stage();item.save_params.update(scalar_interval=2,psi_interval=2)
        item.monitor_params.update(show_magnifier_interval=2)
        with patch.object(self.save,'get_N',side_effect=norm), \
             patch.object(self.save,'get_E',return_value=7), \
             patch.object(self.save,'get_L',return_value=3), \
             patch.object(self.save,'sum_abs2',side_effect=lambda a,axis:np.sum(abs(a)**2,axis=axis)):
            self.save.save_outputs(item,1)
            self.assertFalse(item.monitor_dir.exists())
            self.save.save_outputs(item,2)
        self.assertTrue((item.data_dir/'psi_00000002.json').exists())
        for i in (1,2):
            folder=item.monitor_dir/f'component_{i}'
            with (folder/'scalars.csv').open() as f: row=list(csv.DictReader(f))[0]
            self.assertAlmostEqual(float(row['N_atoms']),getattr(item,f'N{i}')*norm(getattr(item,f'psi{i}'),item.x_array))
            with np.load(folder/'density_xy_00000002.npz') as saved:
                self.assertAlmostEqual(saved['values'].sum()*4,norm(getattr(item,f'psi{i}'),item.x_array))
        from gui_periphery.viewer import read_monitor
        self.assertEqual(len(read_monitor(item.monitor_dir)['stages']),2)

    def test_stage_output_escape_and_collisions_are_rejected(self):
        c=self.config();c['stages'][0]['save']['directory']='../escape'
        with self.assertRaisesRegex(ValueError,'inside'): self.parse(c)
        c['stages'][0]['save']['directory']='same'
        c['stages'].append(copy := new_stage())
        copy['save']['directory']='same'
        with self.assertRaisesRegex(ValueError,'reuses'): self.parse(c)

if __name__ == '__main__':
    unittest.main()
