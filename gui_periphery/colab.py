"""Display the same GUI inside a Colab cell using Colab's authenticated proxy."""

import threading

from .editor.app import create_server

_server = None


def show(port=8765):
    from google.colab import output

    global _server
    if _server is None:
        _server = create_server(port, colab=True)
        threading.Thread(target=_server.serve_forever, daemon=True).start()
    output.serve_kernel_port_as_iframe(_server.server_port,
        path="/?token=" + _server.session_token, width="100%", height=1000)
