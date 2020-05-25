#
#     Kwola is an AI algorithm that learns how to use other programs
#     automatically so that it can find bugs in them.
#
#     Copyright (C) 2020 Kwola Software Testing Inc.
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU Affero General Public License as
#     published by the Free Software Foundation, either version 3 of the
#     License, or (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import asyncio
import socket
from contextlib import closing
from mitmproxy.tools.dump import DumpMaster
from selenium import webdriver
from selenium.webdriver.common.proxy import Proxy, ProxyType
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
    from mitmproxy import proxy, options

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    opts = options.Options(listen_port=port)
    pconf = proxy.config.ProxyConfig(opts)

    m = DumpMaster(opts, with_termlog=False, with_dumper=False)
    m.server = proxy.server.ProxyServer(pconf)

    m.run()


def main():
    """
        This is the entry for the command which makes it convenient to install the proxy certificate
    """
    setupLocalLogging()
    commandArgs = sys.argv[1:]

    proxyPort = findFreePort()

    proxyThread = threading.Thread(target=runProxy, args=[proxyPort], daemon=True)
    proxyThread.start()

    capabilities = webdriver.DesiredCapabilities.CHROME
    capabilities['loggingPrefs'] = {'browser': 'ALL'}
    proxyConfig = Proxy()
    proxyConfig.proxy_type = ProxyType.MANUAL
    proxyConfig.http_proxy = f"localhost:{proxyPort}"
    proxyConfig.add_to_capabilities(capabilities)

    chrome_options = webdriver.chrome.options.Options()
    if len(commandArgs) > 0:
        chrome_options.headless = True
    chrome_options.add_argument(f"--no-sandbox")

    driver = webdriver.Chrome(desired_capabilities=capabilities, chrome_options=chrome_options)

    driver.get("http://mitm.it/")

    print("Please kill the command with Ctrl-C or (Cmd-C on macOS) when you are finished installing the certificates. Timeout in 600 seconds...")

    timeout = 600
    if len(commandArgs) > 0:
        timeout = int(str(commandArgs[0]))
    
    time.sleep(timeout)



