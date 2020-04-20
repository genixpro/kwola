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
from .WebEnvironmentSession import WebEnvironmentSession
from contextlib import closing
from mitmproxy.tools.dump import DumpMaster
from threading import Thread
import asyncio
import concurrent.futures
from datetime import datetime
import numpy as np
import socket
import time
import os


class WebEnvironment:
    """
        This class represents web / browser based environments. It will boot up a headless browser and use it to communicate
        with the software.
    """

    def __init__(self, config, sessionLimit=None):
        self.config = config

        def createSession(number):
            return WebEnvironmentSession(config, number)

        with concurrent.futures.ThreadPoolExecutor(max_workers=config['web_session_max_startup_workers']) as executor:
            sessionCount = config['web_session_parallel_execution_sessions']
            if sessionLimit is not None:
                sessionCount = min(sessionLimit, sessionCount)

            print(datetime.now(), f"[{os.getpid()}]", f"Starting up {sessionCount} parallel browser sessions.")

            sessionFutures = [
                executor.submit(createSession, sessionNumber) for sessionNumber in range(sessionCount)
            ]

            self.sessions = [
                future.result() for future in sessionFutures
            ]

    def shutdown(self):
        for session in self.sessions:
            session.shutdown()


    def screenshotSize(self):
        return self.sessions[0].screenshotSize()

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
