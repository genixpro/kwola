#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#

import asyncio
import socket
from contextlib import closing
from mitmproxy.tools.dump import DumpMaster
from playwright.sync_api import sync_playwright
import threading
import time
import sys
from ..config.logger import getLogger, setupLocalLogging
import logging

def findFreePort():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def runProxy(port):
    from mitmproxy import options

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    opts = options.Options(listen_port=port, http2=False, ssl_insecure=True)
    m = DumpMaster(opts, loop=loop, with_termlog=False, with_dumper=False)
    loop.run_until_complete(m.run())


def main():
    """
        This is the entry for the command which makes it convenient to install the proxy certificate
    """
    setupLocalLogging()
    commandArgs = sys.argv[1:]

    proxyPort = findFreePort()

    proxyThread = threading.Thread(target=runProxy, args=[proxyPort], daemon=True)
    proxyThread.start()

    # Use Playwright's bundled Chromium; no system Chrome/WebDriver is needed.
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=len(commandArgs) > 0, args=["--no-sandbox"], proxy={"server": f"http://127.0.0.1:{proxyPort}"})
        try:
            page = browser.new_page()
            page.goto("http://mitm.it/", wait_until="domcontentloaded")
            timeout_seconds = int(str(commandArgs[0])) if commandArgs else 600
            print(f"Install the mitmproxy certificate shown in Playwright Chromium, then stop this command. Timeout in {timeout_seconds} seconds...")
            time.sleep(timeout_seconds)
        finally:
            browser.close()
