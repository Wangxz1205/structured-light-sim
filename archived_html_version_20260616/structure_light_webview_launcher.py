"""
Desktop launcher for the structure-light simulation.

Run this file in VSCode:

    python structure_light_desktop_app.py

It opens a native desktop window and renders index_modified.html through
the system Chromium/WebView2 engine, preserving WebGL GPU acceleration for
the Three.js scene and point-cloud renderer.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

import webview
from flask import Flask, Response, send_file, send_from_directory
from werkzeug.serving import make_server


PROJECT_DIR = Path(__file__).resolve().parent
APP_TITLE = "结构光物理仿真与三维点云重建"
HTML_FILE = PROJECT_DIR / "index_modified.html"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5020


def find_available_port(host: str, start_port: int) -> int:
    for port in range(start_port, start_port + 100):
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((host, port)) != 0:
                return port
    raise RuntimeError(f"No available port found from {start_port} to {start_port + 99}.")


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> Response:
        if not HTML_FILE.exists():
            return Response(
                f"Missing required page file: {HTML_FILE.name}",
                status=500,
                mimetype="text/plain; charset=utf-8",
            )
        return send_file(HTML_FILE, mimetype="text/html; charset=utf-8")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "page": HTML_FILE.name}

    @app.get("/vendor/<path:filename>")
    def vendor(filename: str) -> Response:
        return send_from_directory(PROJECT_DIR / "vendor", filename)

    return app


class FlaskServerThread(threading.Thread):
    def __init__(self, host: str, port: int) -> None:
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.server = make_server(host, port, create_app(), threaded=True)

    def run(self) -> None:
        self.server.serve_forever()

    def shutdown(self) -> None:
        self.server.shutdown()


def wait_for_server(url: str, timeout_seconds: float = 5.0) -> None:
    import urllib.request

    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_error = exc
            time.sleep(0.1)
    raise RuntimeError(f"Local desktop server did not start in time: {last_error}")


def main() -> None:
    os.environ.setdefault(
        "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS",
        "--ignore-gpu-blocklist --enable-gpu-rasterization --enable-zero-copy",
    )

    parser = argparse.ArgumentParser(description="Run the structure-light desktop demo.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind, default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Preferred port, default: 5000")
    parser.add_argument("--fullscreen", action="store_true", help="Open the desktop window fullscreen")
    parser.add_argument("--debug", action="store_true", help="Enable WebView developer tools")
    args = parser.parse_args()

    port = find_available_port(args.host, args.port)
    url = f"http://{args.host}:{port}"

    server = FlaskServerThread(args.host, port)
    server.start()
    wait_for_server(f"{url}/health")

    window = webview.create_window(
        APP_TITLE,
        url,
        width=1440,
        height=920,
        min_size=(1200, 760),
        resizable=True,
        fullscreen=args.fullscreen,
        background_color="#020617",
        text_select=True,
    )

    try:
        webview.start(gui="edgechromium", debug=args.debug, private_mode=False)
    finally:
        server.shutdown()
        if window is not None:
            window.destroy()


if __name__ == "__main__":
    main()
