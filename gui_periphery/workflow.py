"""Colab-only launch and monitoring. All prepared runs live on mounted Drive."""
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import tarfile
import threading

from gpe.io.atomic import atomic_json
from gpe.io.drive_storage import DEFAULT_OUTPUT, require_drive_path
from .prepare_run import prepare_run


def archive_bundle(folder):
    manifest = json.loads((folder / 'run.json').read_text())
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:gz') as archive:
        for name in [*manifest['files_sha256'], 'run.json']:
            path = folder / name
            if not path.resolve().is_relative_to(folder.resolve()) or path.is_symlink() or not path.is_file():
                raise ValueError('Invalid runtime manifest member.')
            if PurePosixPath(name).parts[0] in ('data','monitor','logs'):
                raise ValueError('Output files must not be included in the runtime manifest.')
            archive.add(path, arcname=name, recursive=False)
    return output.getvalue()


class Workflow:
    def __init__(self, base_url=None, output_parent=None):
        self.output_parent = str(output_parent or DEFAULT_OUTPUT)
        self.folder = None
        self.run = {}
        self.operation = dict(busy=False, name='', output='', error='')
        self.lock = threading.Lock()
        self.local_process = None
        self.local_processes = {}

    def state(self):
        with self.lock:
            operation = dict(self.operation)
        try:
            require_drive_path(self.output_parent)
            storage_error = ''
        except ValueError as exc:
            storage_error = str(exc)
        return dict(operation=operation, run=dict(self.run),
                    folder=str(self.folder) if self.folder else None,
                    output_parent=self.output_parent, storage_error=storage_error)

    def start(self, name, function):
        with self.lock:
            if self.operation['busy']:
                raise ValueError('Wait for the current operation to finish.')
            self.operation = dict(busy=True, name=name, output='', error='')
        def work():
            try:
                function()
            except Exception as exc:
                with self.lock:
                    self.operation['error'] = str(exc)
            finally:
                with self.lock:
                    self.operation['busy'] = False
        threading.Thread(target=work, daemon=True).start()

    def log(self, value):
        with self.lock:
            self.operation['output'] = (self.operation['output'] + value)[-64000:]

    def save_state(self):
        require_drive_path(self.folder)
        atomic_json(self.folder / 'gui_job.json', self.run)

    def open(self, folder):
        path = require_drive_path(folder)
        manifest = json.loads((path / 'run.json').read_text())
        if manifest.get('deployment') != 'colab' or not (path / 'config/run.yaml').is_file():
            raise ValueError('Choose a Colab two-component simulation folder.')
        self.folder = path
        state = path / 'gui_job.json'
        self.run = json.loads(state.read_text()) if state.exists() else dict(
            name=manifest['name'], target='local', status='Prepared')
        self.local_process = self.local_processes.get(str(path))
        self.refresh()

    def prepare(self, data):
        output = require_drive_path(data.get('output') or self.output_parent)
        folder = prepare_run(data['config'], output, data.get('name') or None,
                             data.get('notes',''), cupy_package=data.get('cupy_package','cupy-cuda12x'))
        self.open(folder)
        self.log(f'Prepared {folder}\nSnapshots, monitoring and logs will be written directly to this Drive folder.\n')

    def setup(self):
        if not self.folder:
            raise ValueError('Prepare or reopen a run first.')
        require_drive_path(self.folder)
        if self.run.get('pid'):
            raise ValueError('This run has already been launched.')
        # Use the notebook interpreter, not a new environment on Drive.
        manifest = json.loads((self.folder / 'run.json').read_text())
        wheel = manifest['cupy_package']
        if wheel not in ('cupy-cuda12x','cupy-cuda13x'):
            raise ValueError('Unsupported CuPy package in run manifest.')
        try:
            import cupy
        except ModuleNotFoundError:
            cupy = None
        if cupy is not None:
            self.log('CuPy is already installed. Reusing the notebook environment.\n')
            return
        with (self.folder / 'logs/environment_setup.log').open('ab') as log:
            subprocess.run([sys.executable,'-m','pip','install',wheel], stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        self.log(f'Installed {wheel} in the notebook environment.\n')

    def submit(self):
        if not self.folder:
            raise ValueError('Prepare or reopen a run first.')
        require_drive_path(self.folder)
        if self.run.get('pid') or (self.folder / 'launch.json').exists():
            raise ValueError('This run has already been launched. Prepare a new named run.')
        # Freeze output containment even if someone manually edited the copied YAML.
        from config.yaml_parser import parse_yaml
        config = parse_yaml(self.folder / 'config/run.yaml')
        if config['simulation']['data_root'] != 'data' or config['simulation']['monitor_root'] != 'monitor':
            raise ValueError('Colab run output roots must remain data/ and monitor/ inside the launch folder.')
        for stage in config['stages']:
            path = Path(stage['save']['directory'])
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Stage output directories must stay inside the launch folder.')
        # A failed preflight must not consume the run's one launch.
        subprocess.run([sys.executable,'-c',
                        'import cupy as cp; assert cp.cuda.runtime.getDeviceCount() > 0'], check=True)
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        environment.setdefault('GPE_DRIVE_MOUNT','/content/drive')
        environment.pop('PYTHONPATH', None)
        with (self.folder / 'launch.json').open('x') as stream:
            json.dump(dict(started_utc=datetime.now(timezone.utc).isoformat()),stream)
            stream.flush(); os.fsync(stream.fileno())
        try:
            with (self.folder / 'logs/run.log').open('ab') as log:
                self.local_process = subprocess.Popen(
                    [sys.executable,'-u','-m','gpe.main','config/run.yaml'], cwd=self.folder,
                    env=environment, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
        except Exception:
            self.run['status'] = 'Launch failed; prepare a new run'
            self.save_state()
            raise
        self.local_processes[str(self.folder)] = self.local_process
        self.run.update(pid=self.local_process.pid, status='Running',
                        submitted_utc=datetime.now(timezone.utc).isoformat())
        self.save_state()
        folder, process = self.folder, self.local_process
        def finish():
            code = process.wait()
            try:
                require_drive_path(folder)
                atomic_json(folder / 'logs/completion.json', dict(exit_code=code,
                            finished_utc=datetime.now(timezone.utc).isoformat()))
            except (OSError, ValueError):
                pass  # A destroyed/disconnected runtime cannot promise a completion record.
        threading.Thread(target=finish,daemon=True).start()
        self.log('Simulation launched. Each scheduled output goes directly to Drive.\n')

    def refresh(self):
        if not self.folder:
            return
        require_drive_path(self.folder)
        completed = self.folder / 'logs/completion.json'
        if completed.exists():
            code = json.loads(completed.read_text())['exit_code']
            if self.run.get('status') != 'Cancelled':
                self.run['status'] = 'Completed' if code == 0 else 'Failed'
        elif self.local_process is not None:
            code = self.local_process.poll()
            if code is not None and self.run.get('status') != 'Cancelled':
                self.run['status'] = 'Completed' if code == 0 else 'Failed'
        elif (self.folder / 'launch.json').exists() and self.run.get('status') != 'Cancelled':
            self.run['status'] = 'Reopened; runtime status unknown (saved outputs available)'
        self.run['last_refresh_utc'] = datetime.now(timezone.utc).isoformat()
        self.save_state()

    def cancel(self):
        if self.local_process is None or self.local_process.poll() is not None:
            raise ValueError('No running process controlled by this notebook session.')
        os.killpg(self.local_process.pid,signal.SIGTERM)
        self.run['status'] = 'Cancelled'
        self.save_state()

    def close(self):
        # Closing the interface does not terminate a running GPU process.
        pass
