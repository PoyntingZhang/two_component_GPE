"""Colab launch/persistence regressions; no network or GPU job is started."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
from config.yaml_parser import new_config,parse_yaml
from gui_periphery.workflow import Workflow,archive_bundle
from gpe.io.drive_storage import require_drive_path

class ColabWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.mount=Path(self.temp.name).resolve();(self.mount/'MyDrive').mkdir()
        self.parent=self.mount/'MyDrive/simulations'
        for p in (patch.dict(os.environ,{'GPE_DRIVE_MOUNT':str(self.mount)}),
                  patch('gpe.io.drive_storage.os.path.ismount',side_effect=lambda x:Path(x)==self.mount)):
            p.start();self.addCleanup(p.stop)
        self.workflow=Workflow(output_parent=self.parent)

    def prepare(self):
        self.workflow.prepare(dict(config=new_config(),name='mixture',notes='Test demixing',output=str(self.parent)))
        return self.workflow.folder

    def test_named_run_is_persistent_self_contained_and_refuses_overwrite(self):
        folder=self.prepare();config=parse_yaml(folder/'config/run.yaml')
        self.assertEqual(config['simulation']['data_root'],'data')
        self.assertEqual(config['simulation']['monitor_root'],'monitor')
        self.assertEqual((folder/'notes.txt').read_text(),'Test demixing\n')
        self.assertTrue((folder/'data').is_dir());self.assertTrue((folder/'monitor').is_dir())
        self.assertTrue((folder/'gpe/io/drive_storage.py').is_file())
        with self.assertRaises(FileExistsError):self.prepare()
        self.assertTrue(archive_bundle(folder).startswith(b'\x1f\x8b'))

    def test_ephemeral_paths_missing_mount_and_read_only_reference_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'temporary runtime'):require_drive_path('/content/simulations')
        with self.assertRaisesRegex(ValueError,'read-only'):require_drive_path(self.mount/'MyDrive/GPE_simulator/runs')
        with patch('gpe.io.drive_storage.os.path.ismount',return_value=False):
            with self.assertRaisesRegex(ValueError,'not mounted'):self.prepare()
        link=self.mount/'MyDrive/escape';link.symlink_to(self.mount)
        with self.assertRaisesRegex(ValueError,'temporary runtime'):require_drive_path(link/'elsewhere')

    def test_reopen_after_runtime_loss_keeps_saved_diagnostics_and_does_not_claim_running(self):
        folder=self.prepare();(folder/'launch.json').write_text('{}')
        monitor=folder/'monitor/stage/component_2';monitor.mkdir(parents=True)
        np.savez(monitor/'density_xy_00000001.npz',step=1,time=.1,x=[0,1],y=[0,1],values=np.ones((2,2)))
        other=Workflow(output_parent=self.parent);other.open(folder)
        self.assertIn('unknown',other.run['status'])
        from gui_periphery.viewer import read_monitor
        result=read_monitor(folder)
        self.assertEqual(result['image']['step'],1)
        self.assertIn('component_2',result['stage'])

    def test_launch_uses_copied_sources_not_original_repo_and_records_completion(self):
        folder=self.prepare();process=Mock(pid=23456);process.wait.return_value=0;process.poll.return_value=0
        with patch('gui_periphery.workflow.subprocess.run') as check, \
             patch('gui_periphery.workflow.subprocess.Popen',return_value=process) as launch, \
             patch('gui_periphery.workflow.threading.Thread') as thread:
            self.workflow.submit()
            kwargs=launch.call_args.kwargs
            self.assertEqual(kwargs['cwd'],folder)
            self.assertNotIn('PYTHONPATH',kwargs['env'])
            self.assertEqual(kwargs['env']['GPE_DRIVE_MOUNT'],str(self.mount))
            self.assertEqual(launch.call_args.args[0][-3:],['-m','gpe.main','config/run.yaml'])
            thread.call_args.kwargs['target']()
        self.assertEqual(json.loads((folder/'logs/completion.json').read_text())['exit_code'],0)
        with self.assertRaisesRegex(ValueError,'already been launched'):self.workflow.submit()
        reopened=Workflow(output_parent=self.parent);reopened.open(folder)
        self.assertEqual(reopened.run['status'],'Completed')

    def test_runtime_write_fails_when_drive_disappears(self):
        folder=self.prepare()
        from gpe.io.atomic import atomic_npz
        with patch('gpe.io.drive_storage.os.path.ismount',return_value=False):
            with self.assertRaisesRegex(ValueError,'not mounted'):atomic_npz(folder/'monitor/sample.npz',values=[1])
        self.assertFalse((folder/'monitor/sample.npz').exists())

    def test_failure_before_launch_keeps_run_available(self):
        folder=self.prepare()
        with patch('gui_periphery.workflow.subprocess.run',side_effect=RuntimeError('GPU unavailable')):
            with self.assertRaisesRegex(RuntimeError,'GPU unavailable'):self.workflow.submit()
        self.assertFalse((folder/'launch.json').exists())

    def test_manually_escaped_output_is_rejected_before_gpu_launch(self):
        folder=self.prepare();path=folder/'config/run.yaml'
        path.write_text(path.read_text().replace('data_root: data','data_root: /content/lost'))
        with self.assertRaisesRegex(ValueError,'output roots'):self.workflow.submit()
