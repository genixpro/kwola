"""Interactive mitmproxy certificate installation."""

import shutil
import socket
import subprocess
from contextlib import closing

from playwright.sync_api import sync_playwright


def install_certificate() -> None:
    executable = shutil.which("mitmdump")
    if executable is None:
        raise RuntimeError("mitmdump is not available on PATH")
    port = _free_port()
    process = subprocess.Popen(
        [
            executable,
            "--listen-port",
            str(port),
            "--set",
            "http2=false",
            "--set",
            "ssl_insecure=true",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=False,
                proxy={"server": f"http://127.0.0.1:{port}"},
            )
            try:
                page = browser.new_page()
                page.goto("http://mitm.it/", wait_until="domcontentloaded")
                input("Install the certificate in Chromium, then press Enter to finish: ")
            finally:
                browser.close()
    finally:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as connection:
        connection.bind(("127.0.0.1", 0))
        return int(connection.getsockname()[1])
