"""GUI workflow controls, using fake SSH responses and saved diagnostic fixtures."""

import copy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from config.yaml_parser import new_config, parse_yaml
from gui_periphery.connection import Connection, Prompts
from gui_periphery.viewer import read_monitor
from gui_periphery.workflow import Workflow, archive_bundle, extract_monitors, launch_settings


class GuiWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workflow = Workflow("http://127.0.0.1:8765")
        self.addCleanup(self.workflow.close)
        self.config = new_config()

    def prepare(self, target="local"):
        self.workflow.prepare(dict(config=self.config, output=str(self.root), name="simulation_gui",
            notes="Inspect the spectrum", launch=dict(target=target, account="PROJECT-GPU",
            modules="python/3.12 cuda/12.4" if target == "slurm" else "")))
        return self.workflow.folder

    def test_gui_preparation_and_download_preserve_physics_and_exclude_outputs(self):
        original = copy.deepcopy(self.config)
        folder = self.prepare("slurm")
        self.assertEqual(self.config, original)
        self.assertEqual(parse_yaml(folder / "config/run.yaml")["stages"], original["stages"])
        self.assertIn("module load python/3.12 cuda/12.4", (folder / "environment.sh").read_text())
        self.assertIn("--account=PROJECT-GPU", (folder / "submit.sh").read_text())
        (folder / "saved_data/psi.npy").write_bytes(b"never upload this")
        (folder / "logs/run.log").write_text("old log")
        with tarfile.open(fileobj=io.BytesIO(archive_bundle(folder)), mode="r:gz") as archive:
            names = archive.getnames()
        self.assertIn("config/run.yaml", names)
        self.assertNotIn("saved_data/psi.npy", names)
        self.assertNotIn("logs/run.log", names)
        self.assertNotIn("gui_job.json", names)

    def test_environment_fields_do_not_accept_shell_commands(self):
        for fields in (dict(modules="cuda; touch /tmp/bad"), dict(python="python3 && echo bad"), dict(account="x\n#SBATCH --nodes=50")):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                launch_settings(dict(target="slurm", account="PROJECT-GPU") | fields)

    def test_local_preflight_failure_does_not_launch_a_process(self):
        self.prepare()
        with patch("gui_periphery.workflow.platform.system", return_value="Linux"), \
             patch.object(self.workflow, "stream", side_effect=ValueError("CuPy unavailable")), \
             patch("gui_periphery.workflow.subprocess.Popen") as popen:
            with self.assertRaisesRegex(ValueError, "CuPy unavailable"):
                self.workflow.submit()
            popen.assert_not_called()
        self.assertNotIn("pid", self.workflow.run)

    def test_local_launch_is_detached_and_completion_is_read_on_reopen(self):
        folder = self.prepare()
        process = Mock(pid=98765); process.poll.return_value = None
        with patch("gui_periphery.workflow.platform.system", return_value="Linux"), \
             patch.object(self.workflow, "stream"), patch("gui_periphery.workflow.subprocess.Popen", return_value=process) as popen:
            self.workflow.submit()
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.workflow.open(folder)
        self.assertIs(self.workflow.local_process, process)
        (folder / "logs/exit_code").write_text("0\n")
        self.workflow.refresh()
        self.assertEqual(self.workflow.run["status"], "Completed")

    def test_macos_can_prepare_but_cannot_install_cuda_or_launch_locally(self):
        with patch("gui_periphery.workflow.platform.system", return_value="Darwin"):
            folder = self.prepare()
            self.assertTrue((folder / "config/run.yaml").is_file())
            self.assertIn("macOS", self.workflow.state()["local_execution_error"])
            with patch.object(self.workflow, "stream") as stream, \
                 patch("gui_periphery.workflow.subprocess.Popen") as popen:
                for action in (self.workflow.setup, self.workflow.submit):
                    with self.subTest(action=action.__name__), self.assertRaisesRegex(ValueError, "Colab"):
                        action()
                stream.assert_not_called()
                popen.assert_not_called()
            self.assertEqual(self.workflow.run["status"], "Prepared")
            self.assertNotIn("pid", self.workflow.run)

    def test_macos_hpc_setup_still_runs_over_ssh(self):
        self.prepare("slurm")
        self.workflow.run.update(remote_path="/rds/test", ssh_target="user@host")
        self.workflow.connection.target = "user@host"
        with patch("gui_periphery.workflow.platform.system", return_value="Darwin"), \
             patch.object(self.workflow.connection, "check", return_value=True), \
             patch.object(self.workflow, "stream") as stream:
            self.workflow.setup()
        arguments = stream.call_args.args[0]
        self.assertEqual(arguments[0], "ssh")
        self.assertIn("cd /rds/test && bash setup_environment.sh", arguments)
        self.assertEqual(self.workflow.run["status"], "Environment ready")

    def test_linux_local_setup_is_still_available(self):
        self.prepare()
        with patch("gui_periphery.workflow.platform.system", return_value="Linux"), \
             patch.object(self.workflow, "stream") as stream:
            self.assertEqual(self.workflow.state()["local_execution_error"], "")
            self.workflow.setup()
        stream.assert_called_once_with(["bash", "setup_environment.sh"], cwd=self.workflow.folder)
        self.assertEqual(self.workflow.run["status"], "Environment ready")

    def test_upload_and_submission_use_a_new_folder_and_persist_job_id(self):
        self.prepare("slurm")
        connection = self.workflow.connection
        connection.target = "user@login.example.org"
        with patch.object(connection, "run", return_value=b"") as run:
            self.workflow.upload("/rds/a project/simulations")
            command = run.call_args.args[0]
            self.assertIn("mkdir -- '/rds/a project/simulations/simulation_gui'", command)
            self.assertIn("input", run.call_args.kwargs)
        with patch.object(connection, "run", return_value=b"Valid configuration\n123456\n") as run:
            self.workflow.submit()
            self.assertIn("sbatch --parsable", run.call_args.args[0])
            self.assertIn("logs/job_id.txt", run.call_args.args[0])
            with self.assertRaisesRegex(ValueError, "already been launched"):
                self.workflow.submit()
        self.assertEqual(json.loads((self.workflow.folder / "gui_job.json").read_text())["job_id"], "123456")

    def test_failed_submission_does_not_claim_a_running_job(self):
        self.prepare("slurm")
        self.workflow.run.update(remote_path="/rds/test", ssh_target="user@host")
        self.workflow.connection.target = "user@host"
        with patch.object(self.workflow.connection, "run", side_effect=ValueError("Invalid account")):
            with self.assertRaisesRegex(ValueError, "Invalid account"):
                self.workflow.submit()
        self.assertNotIn("job_id", self.workflow.run)

    def test_monitor_transfer_rejects_wavefunctions_and_path_escape(self):
        folder = self.prepare()
        for name in ("saved_data/psi.npy", "../escape"):
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode="w") as archive:
                info = tarfile.TarInfo(name); info.size = 4
                archive.addfile(info, io.BytesIO(b"test"))
            with self.subTest(name=name), self.assertRaises(ValueError):
                extract_monitors(stream.getvalue(), folder, ["monitor_data/stage/scalars.csv"])

    def test_refresh_only_transfers_changed_monitor_files(self):
        folder = self.prepare("slurm")
        self.workflow.run.update(remote_path="/rds/test", ssh_target="user@host", job_id="123")
        self.workflow.connection.target = "user@host"
        name = "monitor_data/run/stage/scalars.csv"
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            value = b"step,time,norm\n1,0.1,1\n"
            info = tarfile.TarInfo(name); info.size = len(value)
            archive.addfile(info, io.BytesIO(value))
        manifest = json.dumps({name: [len(value), 12345]}).encode()
        replies = [b"RUNNING\n", manifest, stream.getvalue(), manifest, b"step=1\n",
                   b"COMPLETED\n", manifest, b"finished\n"]
        with patch.object(self.workflow.connection, "run", side_effect=replies) as run:
            self.workflow.refresh()
            self.workflow.refresh()
            commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(sum("tar -cf" in command for command in commands), 1)
        self.assertTrue(all("saved_data" not in command for command in commands))
        self.assertEqual((folder / name).read_bytes(), value)
        self.assertEqual(self.workflow.run["status"], "COMPLETED")

    def test_background_errors_are_reported_to_the_gui(self):
        def fail():
            raise ValueError("Readable failure")
        self.workflow.start("Example", fail)
        # Wait on a bounded event rather than requiring a remote service.
        for _ in range(100):
            if not self.workflow.operation["busy"]:
                break
            threading.Event().wait(.01)
        self.assertEqual(self.workflow.operation["error"], "Readable failure")
        self.assertFalse(self.workflow.operation["busy"])


class AuthenticationTests(unittest.TestCase):
    def test_answers_are_delivered_once_and_not_exposed_in_status(self):
        prompts = Prompts()
        answers = []
        thread = threading.Thread(target=lambda: answers.append(prompts.request("One-time code:", timeout=2)))
        thread.start()
        with prompts.condition:
            prompts.condition.wait_for(lambda: prompts.pending is not None, timeout=1)
        pending = prompts.status()
        prompts.respond(pending["id"], "654321")
        thread.join(timeout=2)
        self.assertEqual(answers, ["654321"])
        self.assertIsNone(prompts.status())
        self.assertIsNone(prompts.answer)
        with self.assertRaises(ValueError):
            prompts.respond(pending["id"], "again")

    def test_openssh_keeps_host_checks_and_uses_private_prompt_bridge(self):
        connection = Connection("http://127.0.0.1:8765")
        self.addCleanup(connection.close)
        process = Mock(); process.poll.return_value = 0
        with patch("gui_periphery.connection.subprocess.Popen", return_value=process) as popen, \
             patch("gui_periphery.connection.threading.Thread"):
            connection.start("login.example.org", "user")
        arguments = popen.call_args.args[0]
        env = popen.call_args.kwargs["env"]
        self.assertIn("StrictHostKeyChecking=ask", arguments)
        self.assertEqual(env["SSH_ASKPASS_REQUIRE"], "force")
        self.assertEqual(Path(connection.directory.name).stat().st_mode & 0o777, 0o700)
        self.assertNotIn("password", " ".join(arguments).lower())
        with self.assertRaisesRegex(ValueError, "Invalid SSH prompt"):
            connection.authenticate("wrong token", "Password:")


class ViewerTests(unittest.TestCase):
    def test_read_raw_data_and_fall_back_while_newest_file_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); stage = root / "monitor_data/run/stage_01_test"
            stage.mkdir(parents=True)
            np.savez(stage / "nk_00000010.npz", step=10, time=1., k=[0,1,2], nk=[5,2,1], complete_shells=[True,True,False])
            (stage / "nk_00000020.npz").write_bytes(b"unfinished")
            values = np.arange(12).reshape(3,4).astype(float); values[0,0] = np.nan
            np.savez(stage / "density_xy_00000010.npz", step=10, time=1., values=values, x=[-1,0,1], y=[-2,-1,0,1])
            (stage / "scalars.csv").write_text("step,time,norm,E_kin,Lz\n10,1,1,2,3\n20,2,")
            result = read_monitor(root)
            self.assertEqual(result["spectrum"]["step"], 10)
            self.assertEqual(result["spectrum"]["complete"], [True,True,False])
            self.assertEqual(result["image"]["values"][2][3], 11)
            self.assertIsNone(result["image"]["values"][0][0])
            self.assertEqual(len(result["scalars"]), 1)
            self.assertTrue(result["warnings"])
            json.dumps(result, allow_nan=False)

    def test_tof_crop_changes_display_only(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            axis = np.linspace(-10,10,21); values = np.ones((21,21))
            path = stage / "TOF_xy_00000001.npz"
            np.savez(path, time=.1, step=1, kx=axis, ky=axis, values=values)
            original = path.read_bytes()
            result = read_monitor(stage, image_kind="TOF_xy", image_limit=5)
            self.assertEqual(result["image"]["x"], list(np.linspace(-5,5,11)))
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
