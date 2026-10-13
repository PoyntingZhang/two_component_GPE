"""Run preparation, upload, job control and monitor transfers for the local GUI."""

from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import signal
import subprocess
import tarfile
import threading

from .connection import Connection
from .prepare_run import ROOT, prepare_run


def local_execution_error():
    if platform.system() == "Darwin":
        return ("This GUI is running on macOS, where this simulator's CUDA environment cannot run. "
                "You can prepare and download a run here. For Colab, open the GUI inside a GPU runtime; "
                "for HPC, prepare a CSD3 run, connect and upload before setting up the environment.")
    return ""


def launch_settings(values):
    result = dict(target=values.get("target", "slurm"), account=values.get("account", "").strip(),
                  partition=values.get("partition", "ampere").strip(),
                  walltime=values.get("walltime", "01:00:00").strip(),
                  modules=values.get("modules", "").split(), python=values.get("python", "python3").strip())
    if result["target"] not in ("slurm", "local"):
        raise ValueError("Choose CSD3 or this GPU machine.")
    for key in ("partition", "account"):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", result[key]) and (key != "account" or result["target"] == "slurm"):
            raise ValueError(f"Enter a valid {key} for the job.")
    if not re.fullmatch(r"\d{1,3}:[0-5]\d:[0-5]\d", result["walltime"]):
        raise ValueError("Wall time must be HH:MM:SS.")
    if result["walltime"] == "00:00:00":
        raise ValueError("Wall time must be positive.")
    for name in [*result["modules"], result["python"]]:
        if not re.fullmatch(r"[A-Za-z0-9_/][A-Za-z0-9_./+-]*", name):
            raise ValueError("Enter module names and a Python executable, without shell commands.")
    return result


def configure_bundle(folder, settings):
    environment = "#!/usr/bin/env bash\n# Generated from the GUI's environment settings.\n"
    if settings["modules"]:
        environment += "source /etc/profile.d/modules.sh\nmodule load " + shlex.join(settings["modules"]) + "\n"
    environment += 'if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi\n'
    (folder / "environment.sh").write_text(environment)
    setup = (folder / "setup_environment.sh").read_text()
    (folder / "setup_environment.sh").write_text(setup.replace("python3 -m venv", shlex.quote(settings["python"]) + " -m venv"))
    submit = (folder / "submit.sh").read_text()
    submit = submit.replace("CHANGE_ME_GPU_ACCOUNT", settings["account"] or "UNUSED_LOCAL_JOB")
    submit = submit.replace("--partition=ampere", "--partition=" + settings["partition"])
    submit = submit.replace("--time=01:00:00", "--time=" + settings["walltime"])
    (folder / "submit.sh").write_text(submit)
    manifest = json.loads((folder / "run.json").read_text())
    manifest["launch"] = settings
    manifest["files_sha256"] = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                                for name in manifest["files_sha256"]}
    (folder / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")


def archive_bundle(folder):
    """Archive prepared files only, never output trees or external snapshots."""
    manifest = json.loads((folder / "run.json").read_text())
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name in [*manifest["files_sha256"], "run.json"]:
            path = folder / name
            if not path.resolve().is_relative_to(folder.resolve()) or path.is_symlink() or not path.is_file():
                raise ValueError("The run manifest contains an invalid file path.")
            if PurePosixPath(name).parts[0] in ("saved_data", "monitor_data", "logs"):
                raise ValueError("Output files must not be included in the runtime manifest.")
            archive.add(path, arcname=name, recursive=False)
        for name in ("saved_data", "monitor_data", "logs"):
            item = tarfile.TarInfo(name)
            item.type, item.mode = tarfile.DIRTYPE, 0o755
            archive.addfile(item)
    return output.getvalue()


def extract_monitors(content, folder, requested):
    """Accept only the selected small diagnostic files from an SSH transfer."""
    allowed = set(requested)
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as archive:
        for item in archive:
            path = folder / item.name
            if item.name not in allowed or not item.isfile() or not path.resolve().is_relative_to(folder.resolve()):
                raise ValueError("Unexpected file in monitor transfer.")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".partial")
            with archive.extractfile(item) as source, temporary.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
            temporary.replace(path)


# Runs using the remote system Python, without importing the solver or CuPy.
MONITOR_FILES = r'''
import json, pathlib, re, sys
root = pathlib.Path(sys.argv[1])
result = {}
for path in (root / 'monitor_data').rglob('*'):
    if not path.is_file() or path.is_symlink(): continue
    match = re.fullmatch(r'(.+)_(\d+)\.npz', path.name)
    if not match and path.name != 'scalars.csv': continue
    stat = path.stat()
    result[str(path.relative_to(root))] = [stat.st_size, stat.st_mtime_ns]
print(json.dumps(result))
'''


class Workflow:
    def __init__(self, base_url):
        self.connection = Connection(base_url)
        self.folder = None
        self.run = {}
        self.operation = dict(busy=False, name="", output="", error="")
        self.lock = threading.Lock()
        self.local_process = None
        self.local_processes = {}
        self.synced = {}

    def state(self):
        with self.lock:
            operation = dict(self.operation)
        return dict(connection=self.connection.status(), operation=operation,
                    run=dict(self.run), folder=str(self.folder) if self.folder else None,
                    local_execution_error=local_execution_error())

    def start(self, name, function):
        with self.lock:
            if self.operation["busy"]:
                raise ValueError("Wait for the current operation to finish.")
            self.operation = dict(busy=True, name=name, output="", error="")
        def work():
            try:
                function()
            except Exception as exc:
                with self.lock:
                    self.operation["error"] = str(exc)
            finally:
                with self.lock:
                    self.operation["busy"] = False
        threading.Thread(target=work, daemon=True).start()

    def log(self, value):
        with self.lock:
            self.operation["output"] = (self.operation["output"] + value)[-64000:]

    def save_state(self):
        (self.folder / "gui_job.json").write_text(json.dumps(self.run, indent=2) + "\n")

    def open(self, folder):
        path = Path(folder).expanduser().resolve()
        manifest = json.loads((path / "run.json").read_text())
        if not (path / "config/run.yaml").is_file():
            raise ValueError("Choose a prepared simulation folder.")
        self.folder = path
        state = path / "gui_job.json"
        self.run = json.loads(state.read_text()) if state.exists() else dict(
            name=manifest["name"], target=manifest.get("launch", {}).get("target", "local"), status="Prepared")
        self.local_process = self.local_processes.get(self.run.get("pid"))
        self.synced = {}

    def prepare(self, data):
        settings = launch_settings(data["launch"])
        folder = prepare_run(data["config"], data.get("output") or ROOT / "runs",
                             data.get("name") or None, data.get("notes", ""),
                             cupy_package=data.get("cupy_package", "cupy-cuda12x"))
        configure_bundle(folder, settings)
        self.open(folder)
        self.save_state()
        self.log(f"Prepared {folder}\nPhysics settings and sampling intervals preserved.\n")

    def remote(self):
        if not self.run.get("remote_path"):
            raise ValueError("Upload this run first.")
        if self.run.get("ssh_target") != self.connection.target:
            raise ValueError("Connect to the host and username used for this run.")
        return self.run["remote_path"]

    def upload(self, parent):
        if not self.folder or self.run.get("target") != "slurm":
            raise ValueError("Prepare or reopen an HPC run first.")
        parent = PurePosixPath(parent)
        if not parent.is_absolute() or ".." in parent.parts:
            raise ValueError("Choose an absolute remote parent directory.")
        remote = str(parent / self.folder.name)
        payload = archive_bundle(self.folder)
        command = (f"mkdir -p -- {shlex.quote(str(parent))} && mkdir -- {shlex.quote(remote)} && "
                   f"tar -xzf - -C {shlex.quote(remote)}")
        self.connection.run(command, input=payload, timeout=180)
        self.run.update(remote_path=remote, ssh_target=self.connection.target, status="Uploaded")
        self.save_state()
        self.log(f"Uploaded runtime bundle ({len(payload) / 1024:.0f} KiB) to {remote}\n")

    def stream(self, arguments, cwd=None, log_name="environment_setup.log"):
        with (self.folder / "logs" / log_name).open("a") as log:
            process = subprocess.Popen(arguments, cwd=cwd, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for chunk in iter(lambda: process.stdout.read1(4096), b""):
                value = chunk.decode(errors="replace")
                log.write(value)
                log.flush()
                self.log(value)
            process.stdout.close()
            if process.wait():
                raise ValueError("The operation failed. See its output above.")

    def setup(self):
        if not self.folder:
            raise ValueError("Prepare or reopen a run first.")
        if self.run["target"] == "slurm":
            command = f"cd {shlex.quote(self.remote())} && bash setup_environment.sh"
            self.stream(self.connection.command(command))
        else:
            if error := local_execution_error():
                raise ValueError(error)
            self.stream(["bash", "setup_environment.sh"], cwd=self.folder)
        self.run["status"] = "Environment ready"
        self.save_state()

    def submit(self):
        if not self.folder:
            raise ValueError("Prepare or reopen a run first.")
        if self.run.get("job_id") or self.run.get("pid"):
            raise ValueError("This run has already been launched. Prepare a new named run to launch again.")
        if self.run["target"] == "slurm":
            remote = shlex.quote(self.remote())
            command = (f"cd {remote} && if [ -s logs/job_id.txt ]; then cat logs/job_id.txt; "
                       "else bash run.sh --check && sbatch --parsable submit.sh > logs/job_id.txt "
                       "&& cat logs/job_id.txt; fi")
            output = self.connection.run(command).decode()
            job_id = output.strip().splitlines()[-1].split(";")[0]
            if not re.fullmatch(r"\d+", job_id):
                raise ValueError("Submission returned an unexpected response: " + output)
            self.run.update(job_id=job_id, status="Submitted")
            self.log(output)
        else:
            if error := local_execution_error():
                raise ValueError(error)
            # The existing runtime environment is useful for Colab. GPU validation
            # runs before the stage loop, without altering any simulation settings.
            check = ('source ./environment.sh; python -c '
                     + shlex.quote('import cupy as cp; assert cp.cuda.runtime.getDeviceCount() > 0')
                     + '; bash run.sh --check')
            self.stream(["bash", "-ec", check], cwd=self.folder, log_name="preflight.log")
            wrapper = 'bash run.sh; code=$?; printf "%s\\n" "$code" > logs/exit_code; exit "$code"'
            with (self.folder / "logs/launcher.log").open("ab") as log:
                self.local_process = subprocess.Popen(["bash", "-c", wrapper], cwd=self.folder,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.run.update(pid=self.local_process.pid, status="Running")
            self.local_processes[self.local_process.pid] = self.local_process
            self.log("Simulation launched on this GPU machine.\n")
        self.run["submitted_utc"] = datetime.now(timezone.utc).isoformat()
        self.save_state()

    def refresh(self):
        if not self.folder:
            return
        if self.run["target"] == "slurm":
            remote = shlex.quote(self.remote())
            job_id = self.run.get("job_id")
            if job_id:
                output = self.connection.run(f"squeue -h -j {shlex.quote(job_id)} -o %T").decode().strip()
                if not output:
                    output = self.connection.run(f"sacct -n -X -j {shlex.quote(job_id)} --format=State --parsable2").decode().strip()
                self.run["status"] = output.splitlines()[0].rstrip("|") if output else "Awaiting scheduler accounting"
            manifest = json.loads(self.connection.run(f"python3 -c {shlex.quote(MONITOR_FILES)} {remote}"))
            changed = [path for path, stat in manifest.items() if self.synced.get(path) != stat]
            for path in changed:
                rel = PurePosixPath(path)
                if rel.is_absolute() or ".." in rel.parts or rel.parts[0] != "monitor_data":
                    raise ValueError("Invalid monitor path returned by remote host.")
            if changed:
                # Historical images can accumulate. Bound each in-memory tar transfer.
                batches, batch, size = [], [], 0
                for path in changed:
                    if batch and (size + manifest[path][0] > 32 * 1024**2 or len(batch) >= 128):
                        batches.append(batch)
                        batch, size = [], 0
                    batch.append(path)
                    size += manifest[path][0]
                batches.append(batch)
                for batch in batches:
                    command = f"cd {remote} && tar -cf - -- " + shlex.join(batch)
                    extract_monitors(self.connection.run(command, timeout=180), self.folder, batch)
                # If a writer changed a file during transfer, fetch it again next time.
                after = json.loads(self.connection.run(f"python3 -c {shlex.quote(MONITOR_FILES)} {remote}"))
                for path in changed:
                    if manifest[path] == after.get(path):
                        self.synced[path] = manifest[path]
            log = self.connection.run(f"cd {remote} && if [ -f logs/run.log ]; then tail -c 16000 logs/run.log; fi")
            (self.folder / "logs/remote_run.log").write_bytes(log)
            self.log(f"Refreshed {len(changed)} diagnostic files. Full wavefunctions stay remote.\n")
        else:
            code = self.folder / "logs/exit_code"
            if code.exists():
                self.run["status"] = "Completed" if code.read_text().strip() == "0" else "Failed"
            elif self.local_process and self.local_process.poll() is not None and self.run.get("status") != "Cancelled":
                self.run["status"] = "Stopped"
            elif not self.local_process and self.run.get("pid") and self.run.get("status") != "Cancelled":
                self.run["status"] = "Reopened; awaiting completion record"
        self.run["last_refresh_utc"] = datetime.now(timezone.utc).isoformat()
        self.save_state()

    def cancel(self):
        if self.run.get("job_id"):
            self.remote()
            self.connection.run("scancel " + shlex.quote(self.run["job_id"]))
        elif self.local_process and self.local_process.poll() is None:
            os.killpg(self.local_process.pid, signal.SIGTERM)
        else:
            raise ValueError("No active job controlled by this session.")
        self.run["status"] = "Cancelled"
        self.save_state()

    def close(self):
        # Batch jobs and local detached processes keep running.
        self.connection.close()
