"""Local configuration editor. Run: python -m gui_periphery.editor.app"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import hmac
from pathlib import Path
import secrets
import subprocess
import threading
from urllib.parse import parse_qs, urlsplit
import webbrowser

from config.expressions import preview_expression
from gui_periphery.workflow import Workflow, archive_bundle
from gui_periphery.viewer import read_monitor
from gpe.io.drive_storage import require_drive_path

try:
    from config.yaml_parser import ConfigError, dump_yaml, load_yaml, new_config, new_stage, save_yaml
except ModuleNotFoundError as exc:
    if exc.name != "yaml":
        raise
    raise SystemExit("Install the editor dependency: python -m pip install -r gui_periphery/requirements.txt") from exc


ROOT = Path(__file__).resolve().parent


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        if str(args[1]) not in ("200", "304"):
            super().log_message(format, *args)

    def reply(self, value, status=200):
        payload = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def local_request(self):
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        origin = self.headers.get("Origin")
        route = urlsplit(self.path).path
        token = self.headers.get("X-GPE-Session", "")
        if not self.server.colab and (self.headers.get("Host") not in hosts or
                (origin is not None and origin not in {"http://" + host for host in hosts})):
            return False
        if route in ("/app.js", "/style.css", "/workflow.js", "/workflow.css", "/monitor.js"):
            return True  # Static assets contain no session or user data.
        if route == "/":
            token = parse_qs(urlsplit(self.path).query).get("token", [token])[0]
        return hmac.compare_digest(token, self.server.session_token)

    def do_GET(self):
        if not self.local_request():
            return self.reply({"error": "This editor accepts local requests only."}, 403)
        route = urlsplit(self.path).path
        if route == "/api/defaults":
            return self.reply({"config": new_config(), "stage": new_stage(),
                               "save_path": str(Path(self.server.workflow.output_parent) / "experiment.yaml")})
        if route == "/api/example":
            return self.reply({"config": load_yaml((ROOT.parents[1] / "config" / "examples" / "kick_and_hold.yaml").read_text())})
        if route == "/api/workflow":
            try:
                return self.reply(self.server.workflow.state())
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                return self.reply({"error": str(exc)}, 400)
        if route == "/api/run/download":
            try:
                folder = self.server.workflow.folder
                if folder is None:
                    raise ValueError("Prepare a run first.")
                payload = archive_bundle(folder)
                self.send_response(200)
                self.send_header("Content-Type", "application/gzip")
                self.send_header("Content-Disposition", f'attachment; filename="{folder.name}.tar.gz"')
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            except (OSError, ValueError) as exc:
                return self.reply({"error": str(exc)}, 400)
        files = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css",
                 "/workflow.js": "workflow.js", "/workflow.css": "workflow.css", "/monitor.js": "monitor.js"}
        if route not in files:
            return self.reply({"error": "Not found."}, 404)
        path = ROOT / "static" / files[route]
        payload = path.read_bytes()
        if route == "/":
            payload = payload.replace(b"<!-- SESSION -->", ('<script>window.GPE_SESSION=' +
                json.dumps(self.server.session_token) + ';</script>').encode())
        self.send_response(200)
        self.send_header("Content-Type", (mimetypes.guess_type(path)[0] or "text/plain") + "; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Referrer-Policy", "no-referrer")
        if not self.server.colab:
            self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        if not self.local_request() or self.headers.get_content_type() != "application/json":
            return self.reply({"error": "Use a local JSON request."}, 403)
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 2 * 1024 * 1024:
                return self.reply({"error": "Configuration must be smaller than 2 MiB."}, 413)
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError("Request must be an object.")
            route = urlsplit(self.path).path
            workflow = self.server.workflow
            if route == "/api/monitor":
                folder = data.get("folder") or workflow.folder
                if not folder:
                    raise ValueError("Prepare a run or open a monitor folder first.")
                return self.reply(read_monitor(folder, data.get("stage", ""), data.get("spectrum", ""),
                    data.get("image", "density_xy"), float(data.get("image_limit", 5)),
                    data.get("image_sample", ""), data.get("radial", "")))
            if route == "/api/monitor/plot":
                from gui_periphery.monitor_plots import render_plot
                return self.reply({"svg": render_plot(data["kind"], data.get("data"), data.get("options", {}))})
            actions = {
                "/api/run/prepare": ("Prepare run", lambda: workflow.prepare(data)),
                "/api/run/open": ("Open run", lambda: workflow.open(data["folder"])),
                "/api/run/setup": ("Set up environment", workflow.setup),
                "/api/run/submit": ("Launch simulation", workflow.submit),
                "/api/run/refresh": ("Refresh job and monitor files", workflow.refresh),
                "/api/run/cancel": ("Cancel job", workflow.cancel),
            }
            if route in actions:
                workflow.start(*actions[route])
                return self.reply({"ok": True})
            if route == "/api/formula":
                if type(data.get("time_dependent", False)) is not bool:
                    raise ValueError("time_dependent must be true or false.")
                return self.reply(preview_expression(data["expression"], data.get("time_dependent", False)))
            if route == "/api/render":
                return self.reply({"yaml": dump_yaml(data["config"]), "errors": []})
            if route == "/api/import":
                return self.reply({"config": load_yaml(data["yaml"])})
            if route == "/api/save":
                if type(data.get("overwrite", False)) is not bool:
                    raise ValueError("overwrite must be true or false.")
                destination = require_drive_path(data["path"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                return self.reply({"path": save_yaml(data["config"], destination, data.get("overwrite", False))})
            self.reply({"error": "Not found."}, 404)
        except ConfigError as exc:
            self.reply({"errors": exc.errors}, 422)
        except FileExistsError:
            self.reply({"error": "That file already exists. Choose another name or enable Replace existing file."}, 409)
        except (ValueError, TypeError, KeyError, OSError, RecursionError) as exc:
            self.reply({"error": str(exc)}, 400)


def create_server(port=8765, colab=False, output_parent=None):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.session_token = secrets.token_urlsafe(32)
    server.colab = colab
    server.workflow = Workflow(f"http://127.0.0.1:{server.server_port}", output_parent=output_parent)
    return server


def main():
    parser = argparse.ArgumentParser(description="Open the GPE experiment, launch and monitor interface.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--colab", action="store_true", help="Allow a token-protected Colab proxy; use gui_periphery.colab.show()")
    parser.add_argument("--output-parent", help="Simulation parent folder on mounted Google Drive")
    args = parser.parse_args()
    try:
        server = create_server(args.port, args.colab, args.output_parent)
    except OSError as exc:
        parser.exit(1, f"Could not start the editor: {exc}. Try --port 8766.\n")
    url = f"http://127.0.0.1:{server.server_port}/?token=" + server.session_token
    print(f"GPE simulation workspace: {url}\nPress Ctrl+C to stop the GUI.", flush=True)
    if not args.no_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.workflow.close()
        server.server_close()


if __name__ == "__main__":
    main()
