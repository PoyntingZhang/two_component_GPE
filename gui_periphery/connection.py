"""Normal OpenSSH authentication with GUI prompts and a private control socket."""

import hmac
import os
from pathlib import Path
import re
import secrets
import shlex
import subprocess
import sys
import tempfile
import threading


class Prompts:
    def __init__(self):
        self.condition = threading.Condition()
        self.pending = None
        self.answer = None
        self.cancelled = False

    def request(self, prompt, timeout=170):
        with self.condition:
            self.pending = {"id": secrets.token_hex(12), "text": prompt}
            self.answer = None
            self.cancelled = False
            self.condition.notify_all()
            try:
                ready = self.condition.wait_for(lambda: self.answer is not None or self.cancelled, timeout)
                if not ready or self.cancelled:
                    raise ValueError("SSH authentication was cancelled or timed out.")
                return self.answer
            finally:
                self.pending = self.answer = None

    def respond(self, prompt_id, answer):
        with self.condition:
            if not self.pending or self.pending["id"] != prompt_id:
                raise ValueError("That SSH prompt has expired.")
            if not isinstance(answer, str) or any(c in answer for c in "\r\n\0"):
                raise ValueError("Enter one response to the SSH prompt.")
            self.answer = answer
            self.condition.notify_all()

    def cancel(self):
        with self.condition:
            self.cancelled = True
            self.condition.notify_all()

    def status(self):
        with self.condition:
            return dict(self.pending) if self.pending else None


class Connection:
    def __init__(self, base_url):
        self.directory = tempfile.TemporaryDirectory(prefix="gpe-ssh-")
        self.socket = str(Path(self.directory.name) / "c")
        self.token = secrets.token_urlsafe(32)
        self.base_url = base_url
        self.prompts = Prompts()
        self.process = None
        self.target = None
        self.port = 22
        self.output = ""
        self.ready = False
        self.lock = threading.Lock()
        helper = Path(self.directory.name) / "askpass"
        helper.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) +
                          ' -m gui_periphery.ssh_askpass "$@"\n')
        helper.chmod(0o700)
        self.helper = str(helper)

    def start(self, host, username, port=22):
        if os.name == "nt":
            raise ValueError("The SSH integration currently supports macOS and Linux.")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", host or ""):
            raise ValueError("Enter a hostname or an SSH host alias.")
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", username or ""):
            raise ValueError("Enter your SSH username.")
        port = int(port)
        if not 1 <= port <= 65535:
            raise ValueError("SSH port must be between 1 and 65535.")
        self.disconnect()
        self.target, self.port = f"{username}@{host}", port
        self.output = ""
        env = dict(os.environ, SSH_ASKPASS=self.helper, SSH_ASKPASS_REQUIRE="force",
                   DISPLAY=os.environ.get("DISPLAY", ":0"),
                   GPE_AUTH_URL=self.base_url + "/api/ssh/askpass", GPE_AUTH_TOKEN=self.token)
        args = ["ssh", "-M", "-N", "-S", self.socket, "-p", str(port),
                "-o", "ControlPersist=900", "-o", "StrictHostKeyChecking=ask",
                "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3", self.target]
        self.process = subprocess.Popen(args, cwd=Path(__file__).resolve().parents[1], env=env,
                                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.PIPE, start_new_session=True)
        threading.Thread(target=self._read_output, args=(self.process,), daemon=True).start()

    def _read_output(self, process):
        while True:
            chunk = os.read(process.stderr.fileno(), 4096)
            if not chunk:
                break
            with self.lock:
                self.output = (self.output + chunk.decode(errors="replace"))[-16000:]
        process.stderr.close()

    def authenticate(self, token, prompt):
        if not hmac.compare_digest(token or "", self.token):
            raise ValueError("Invalid SSH prompt request.")
        if not self.process or self.process.poll() is not None:
            raise ValueError("No SSH authentication is active.")
        return self.prompts.request(prompt)

    def check(self):
        if not self.target or not Path(self.socket).exists():
            self.ready = False
        else:
            result = subprocess.run(["ssh", "-S", self.socket, "-O", "check", self.target],
                                    capture_output=True, timeout=5)
            self.ready = result.returncode == 0
        return self.ready

    def status(self):
        ready = self.check()
        with self.lock:
            output = self.output
        return dict(connected=ready, target=self.target, prompt=self.prompts.status(), output=output,
                    connecting=bool(not ready and self.process and self.process.poll() is None))

    def command(self, command):
        if not self.check():
            raise ValueError("Connect to the HPC first.")
        return ["ssh", "-S", self.socket, "-p", str(self.port), "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=yes", self.target, command]

    def run(self, command, *, timeout=60, input=None):
        result = subprocess.run(self.command(command), input=input, capture_output=True, timeout=timeout)
        if result.returncode:
            raise ValueError(result.stderr.decode(errors="replace")[-8000:] or
                             result.stdout.decode(errors="replace")[-8000:] or "Remote command failed.")
        return result.stdout

    def disconnect(self):
        self.prompts.cancel()
        if self.target and Path(self.socket).exists():
            subprocess.run(["ssh", "-S", self.socket, "-O", "exit", self.target],
                           capture_output=True, timeout=5)
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None
        self.ready = False

    def close(self):
        self.disconnect()
        self.directory.cleanup()
