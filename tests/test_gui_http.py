"""HTTP integration and SSH prompt routing; no HPC connection or GPU execution."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from config.yaml_parser import new_config
from gui_periphery.editor.app import create_server


class GuiHttpTests(unittest.TestCase):
    def setUp(self):
        try:
            self.server = create_server(0)
        except PermissionError:
            self.skipTest("Local test ports are blocked by the execution sandbox")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.workflow.connection.process = None
        self.server.workflow.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, path, data=None, headers=None):
        headers = dict({"X-GPE-Session":self.server.session_token}, **(headers or {}))
        if data is not None:
            headers["Content-Type"] = "application/json"
        return urlopen(Request(self.url + path, data=json.dumps(data).encode() if data is not None else None,
                               headers=headers), timeout=5)

    def test_session_and_origin_checks_cover_local_and_proxy_access(self):
        with self.assertRaises(HTTPError):
            urlopen(self.url + "/")
        with self.request("/") as response:
            html = response.read().decode()
            self.assertIn("view-launch", html)
            self.assertIn("view-monitor", html)
            self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])
        for headers in ({"X-GPE-Session":"incorrect"}, {"Origin":"https://unrelated.example"}):
            with self.subTest(headers=headers), self.assertRaises(HTTPError) as error:
                self.request("/api/defaults", headers=headers)
            self.assertEqual(error.exception.code, 403)
        self.server.colab = True
        with self.assertRaises(HTTPError):
            urlopen(self.url + "/")
        with self.request("/?token=" + self.server.session_token, headers={"Host":"proxy.example"}) as response:
            self.assertIn(b"GPE_SESSION", response.read())
        with self.request("/api/defaults", headers={"Host":"proxy.example"}) as response:
            self.assertIn("config", json.load(response))

    def test_prepare_and_download_from_api_without_saving_yaml_manually(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = dict(config=new_config(), output=directory, name="simulation_buttons",
                           notes="Prepared through GUI", launch={"target":"local"})
            with self.request("/api/run/prepare", payload) as response:
                self.assertTrue(json.load(response)["ok"])
            for _ in range(100):
                with self.request("/api/workflow") as response:
                    state = json.load(response)
                if not state["operation"]["busy"]:
                    break
                time.sleep(.01)
            self.assertFalse(state["operation"]["error"])
            self.assertEqual(state["run"]["status"], "Prepared")
            self.assertTrue((Path(directory) / "simulation_buttons/config/run.yaml").exists())
            with self.request("/api/run/download") as response:
                self.assertEqual(response.headers["Content-Type"], "application/gzip")
                self.assertTrue(response.read().startswith(b"\x1f\x8b"))

    def test_monitor_plot_endpoint_and_assets(self):
        with self.request("/monitor.js") as response:
            self.assertIn(b"monitorTimes", response.read())
        payload = {"kind":"radial", "data":{"k":[0, .1, 1], "nk":[1, 2, 3], "complete":[True]*3}}
        with self.request("/api/monitor/plot", payload) as response:
            self.assertIn(r"4\pi k^2 n(k)", json.load(response)["svg"])
        with self.assertRaises(HTTPError) as error:
            self.request("/api/monitor/plot", payload, headers={"X-GPE-Session":"wrong"})
        self.assertEqual(error.exception.code, 403)

    def test_askpass_answer_reaches_only_the_helper_pipe(self):
        connection = self.server.workflow.connection
        connection.process = Mock(); connection.process.poll.return_value = None
        env = dict(os.environ, GPE_AUTH_URL=self.url + "/api/ssh/askpass", GPE_AUTH_TOKEN=connection.token,
                   PYTHONDONTWRITEBYTECODE="1")
        helper = subprocess.Popen([sys.executable, "-m", "gui_periphery.ssh_askpass", "One-time code:"],
                                  env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.addCleanup(lambda: helper.kill() if helper.poll() is None else None)
        for _ in range(100):
            with self.request("/api/workflow") as response:
                state = json.load(response)
            prompt = state["connection"]["prompt"]
            if prompt:
                break
            time.sleep(.01)
        self.assertEqual(prompt["text"], "One-time code:")
        with self.request("/api/ssh/answer", {"id":prompt["id"], "answer":"654321"}) as response:
            self.assertTrue(json.load(response)["ok"])
        output, error = helper.communicate(timeout=5)
        self.assertEqual(output, b"654321\n")
        self.assertEqual(error, b"")
        with self.request("/api/workflow") as response:
            state = json.load(response)
        self.assertNotIn("654321", json.dumps(state))
        self.assertIsNone(connection.prompts.answer)


if __name__ == "__main__":
    unittest.main()
