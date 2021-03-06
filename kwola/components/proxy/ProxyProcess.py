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

from ...components.proxy.JSRewriteProxy import JSRewriteProxy
from ...components.proxy.PathTracer import PathTracer
from contextlib import closing
from mitmproxy.tools.dump import DumpMaster
from threading import Thread
import asyncio
import multiprocessing
import socket
import time


class ProxyProcess:
    """
        This class is used to run and manage the proxy subprocess
    """

    def __init__(self, config):
        self.config = config
        self.port = self.findFreePort()
        self.commandQueue = multiprocessing.Queue()
        self.resultQueue = multiprocessing.Queue()

        self.proxyProcess = multiprocessing.Process(target=self.runProxyServerSubprocess, args=(self.config, self.port, self.commandQueue, self.resultQueue))
        self.proxyProcess.start()

        # Wait for the result indicating that the proxy process is ready
        self.resultQueue.get()
        time.sleep(0.1)

    def __del__(self):
        if self.proxyProcess is not None:
            self.proxyProcess.terminate()

    def findFreePort(self):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    def getPathTrace(self):
        self.commandQueue.put("getPathTrace")
        return self.resultQueue.get()

    def resetPathTrace(self):
        self.commandQueue.put("resetPathTrace")

    @staticmethod
    def runProxyServerSubprocess(config, port, commandQueue, resultQueue):
        codeRewriter = JSRewriteProxy(config)
        pathTracer = PathTracer()

        proxyThread = Thread(target=ProxyProcess.runProxyServerThread, args=(port, codeRewriter, pathTracer, resultQueue), daemon=True)
        proxyThread.start()

        while True:
            message = commandQueue.get()

            if message == "resetPathTrace":
                pathTracer.recentPaths = set()

            if message == "getPathTrace":
                pathTrace = {
                    "seen": pathTracer.seenPaths,
                    "recent": pathTracer.recentPaths
                }

                pathTracer.recentPaths = set()

                resultQueue.put(pathTrace)


    @staticmethod
    def runProxyServerThread(port, codeRewriter, pathTracer, resultQueue):
        from mitmproxy import proxy, options

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        opts = options.Options(listen_port=port)
        pconf = proxy.config.ProxyConfig(opts)

        m = DumpMaster(opts, with_termlog=False, with_dumper=False)
        m.server = proxy.server.ProxyServer(pconf)
        m.addons.add(codeRewriter)
        m.addons.add(pathTracer)

        resultQueue.put("ready")
        m.run()
