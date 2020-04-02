from .BaseEnvironment import BaseEnvironment
import time
import numpy as np
from mitmproxy.tools.dump import DumpMaster
from kwola.components.proxy.JSRewriteProxy import JSRewriteProxy
from kwola.components.proxy.PathTracer import PathTracer
from threading import Thread
import asyncio
import concurrent.futures
import socket
from contextlib import closing
from .WebEnvironmentSession import WebEnvironmentSession


class WebEnvironment(BaseEnvironment):
    """
        This class represents web / browser based environments. It will boot up a headless browser and use it to communicate
        with the software.
    """
    def __init__(self, config, sessionLimit=None):
        self.config = config

        self.startProxyServer()

        def createSession(number):
            return WebEnvironmentSession(config, number, self.proxyPort, self.pathTracer)

        with concurrent.futures.ThreadPoolExecutor(max_workers=config['web_session_max_startup_workers']) as executor:
            sessionCount = config['web_session_parallel_execution_sessions']
            if sessionLimit is not None:
                sessionCount = min(sessionLimit, sessionCount)

            sessionFutures = [
                executor.submit(createSession, sessionNumber) for sessionNumber in range(sessionCount)
            ]

            self.sessions = [
                future.result() for future in sessionFutures
            ]

    def shutdown(self):
        for session in self.sessions:
            session.shutdown()

    def startProxyServer(self):
        self.proxyPort = self.findFreePort()

        self.proxyThread = Thread(target=lambda: self.runProxyServer(), daemon=True)
        self.proxyThread.start()

        # Hack, wait for proxy thread to start
        time.sleep(1)

    def findFreePort(self):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    def runProxyServer(self):
        from mitmproxy import proxy, options

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        self.codeRewriter = JSRewriteProxy(self.config)
        self.pathTracer = PathTracer()

        opts = options.Options(listen_port=self.proxyPort)
        pconf = proxy.config.ProxyConfig(opts)

        m = DumpMaster(opts)
        m.server = proxy.server.ProxyServer(pconf)
        m.addons.add(self.codeRewriter)
        m.addons.add(self.pathTracer)

        m.run()

    def screenshotSize(self):
        return self.sessions[0].screenshotSize()


    def branchFeatureSize(self):
        return self.sessions[0].branchFeatureSize()


    def getImages(self):
        imageFutures = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for session in self.sessions:
                resultFuture = executor.submit(session.getImage)
                imageFutures.append(resultFuture)

        images = [
            imageFuture.result() for imageFuture in imageFutures
        ]
        return images


    def getActionMaps(self):
        actionMapFutures = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for session in self.sessions:
                resultFuture = executor.submit(session.getActionMaps)
                actionMapFutures.append(resultFuture)

        actionMaps = [
            imageFuture.result() for imageFuture in actionMapFutures
        ]
        return actionMaps

    def getBranchFeatures(self):
        tabFeatures = [
            tab.getBranchFeature()
            for tab in self.sessions
        ]

        return np.array(tabFeatures)


    def getExecutionTraceFeatures(self):
        tabFeatures = [
            tab.getExecutionTraceFeature()
            for tab in self.sessions
        ]

        return np.array(tabFeatures)


    def numberParallelSessions(self):
        return len(self.sessions)


    def runActions(self, actions, executionSessionIds):
        """
            Run a single action on each of the browser tabs within this environment.

            :param actions:
            :return:
        """

        resultFutures = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for tab, action, executionSessionId in zip(self.sessions, actions, executionSessionIds):
                resultFuture = executor.submit(tab.runAction, action, executionSessionId)
                resultFutures.append(resultFuture)

        results = [
            resultFuture.result() for resultFuture in resultFutures
        ]
        return results


    def createMovies(self):
        moviePaths = [
            tab.createMovie()
            for tab in self.sessions
        ]

        return np.array(moviePaths)