"""Show the Colab-only GUI after the notebook has mounted Google Drive."""
import os
import threading
from gpe.io.drive_storage import DEFAULT_MOUNT, DEFAULT_OUTPUT, require_drive_path
from .editor.app import create_server

_server = None

def show(output_parent=DEFAULT_OUTPUT, port=8765, drive_mount=DEFAULT_MOUNT):
    from google.colab import output
    global _server
    parent = require_drive_path(output_parent, drive_mount)
    os.environ['GPE_DRIVE_MOUNT'] = str(drive_mount)
    parent.mkdir(parents=True, exist_ok=True)
    if _server is None:
        _server = create_server(port, colab=True, output_parent=str(parent))
        threading.Thread(target=_server.serve_forever, daemon=True).start()
    else:
        _server.workflow.output_parent = str(parent)
    output.serve_kernel_port_as_iframe(_server.server_port,
        path='/?token=' + _server.session_token, width='100%', height=1000)
