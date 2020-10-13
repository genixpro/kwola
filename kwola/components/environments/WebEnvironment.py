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


from ...config.logger import getLogger
from ...components.proxy.RewriteProxy import RewriteProxy
from ...components.proxy.PathTracer import PathTracer
from .WebEnvironmentSession import WebEnvironmentSession
from contextlib import closing
from mitmproxy.tools.dump import DumpMaster
from threading import Thread
from pprint import pformat
import asyncio
import concurrent.futures
from datetime import datetime
import numpy
import socket
import time
import os
import psutil
from ..utils.retry import autoretry
from ..plugins.core.RecordAllPaths import RecordAllPaths
from ..plugins.core.RecordBranchTrace import RecordBranchTrace
from ..plugins.core.RecordCursorAtAction import RecordCursorAtAction
from ..plugins.core.RecordExceptions import RecordExceptions
from ..plugins.core.RecordLogEntriesAndLogErrors import RecordLogEntriesAndLogErrors
from ..plugins.core.RecordNetworkErrors import RecordNetworkErrors
from ..plugins.core.RecordPageURLs import RecordPageURLs
from ..plugins.core.RecordScreenshots import RecordScreenshots


class WebEnvironment:
    """
        This class represents web / browser based environments. It will boot up a headless browser and use it to communicate
        with the software.
    """

    def __init__(self, config, sessionLimit=None, plugins=None, executionSessions=None):
        self.config = config

        defaultPlugins = [
            RecordCursorAtAction(),
            RecordExceptions(),
            RecordLogEntriesAndLogErrors(config),
            RecordNetworkErrors(),
            RecordPageURLs(),
            RecordAllPaths(),
            RecordBranchTrace(),
            RecordScreenshots()
        ]

        if plugins is None:
            # Put in the default set up plugins
            self.plugins = defaultPlugins
        else:
            self.plugins = defaultPlugins + plugins

        @autoretry()
        def createSession(sessionNumber):
            session = WebEnvironmentSession(config, sessionNumber, self.plugins, self.executionSessions[sessionNumber])
            return session

        @autoretry()
        def initializeSession(session):
            session.initialize()

        with concurrent.futures.ThreadPoolExecutor(max_workers=config['web_session_max_startup_workers']) as executor:
            sessionCount = config['web_session_parallel_execution_sessions']
            if sessionLimit is not None:
                sessionCount = min(sessionLimit, sessionCount)

            if executionSessions is None:
                self.executionSessions = [None] * sessionCount
            else:
                self.executionSessions = executionSessions

            getLogger().info(f"[{os.getpid()}] Starting up {sessionCount} parallel browser sessions.")

            self.sessions = [
                createSession(sessionNumber)
                for sessionNumber in range(sessionCount)
            ]

            for sessionNumber in range(sessionCount):
                executor.submit(initializeSession, self.sessions[sessionNumber])

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

    def runActions(self, actions):
        """
            Run a single action on each of the browser tabs within this environment.

            :param actions:
            :return:
        """

        startTime = datetime.now()
        resultFutures = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for tab, action in zip(self.sessions, actions):
                resultFuture = executor.submit(tab.runAction, action)
                resultFutures.append(resultFuture)

        traces = [
            resultFuture.result() for resultFuture in resultFutures
        ]

        self.synchronizeNoActivityTimeouts()

        timeTaken = (datetime.now() - startTime).total_seconds()
        subTimes = {}
        if timeTaken > 10:
            # Log data for all of the sub times
            validTraces = [trace for trace in traces if trace is not None]
            if len(validTraces) > 0:
                for key in validTraces[0].actionExecutionTimes:
                    subTimes[key] = {
                        "min": numpy.min([trace.actionExecutionTimes[key] for trace in validTraces]),
                        "max": numpy.max([trace.actionExecutionTimes[key] for trace in validTraces]),
                        "mean": numpy.mean([trace.actionExecutionTimes[key] for trace in validTraces]),
                        "median": numpy.median([trace.actionExecutionTimes[key] for trace in validTraces]),
                        "std": numpy.std([trace.actionExecutionTimes[key] for trace in validTraces])
                    }
                getLogger().warning(f"Time taken to execute the actions in the browser was unusually long: {timeTaken} seconds. Here are the subtimes: {pformat(subTimes)}")

        return traces

    def removeBadSessionIfNeeded(self):
        """
            This method checks all the browser sessions to see if there are any bad ones. If so, it will remove
            the first bad one it finds and return the index of that session

            :return: None if all sessions are good, integer of the first bad session removed if there was a bad session
                    removed.
        """

        for sessionN, session in enumerate(self.sessions):
            if session.hasBrowserDied:
                del self.sessions[sessionN]
                return sessionN

        stats = psutil.virtual_memory()
        if stats.percent > 80 and len(self.sessions) > 1:
            # If we are using more then 90% memory, then cleave off one session from the pack to try and cut back on memory usage
            getLogger().warning(f"Had to kill one of the web browser sessions because the system was running out of available memory. Cleaving one to save the herd. The session being killed is {self.sessions[0].executionSession.id}")
            sessionToDestroy = self.sessions.pop(0)
            sessionToDestroy.shutdown()
            time.sleep(3)
            return 0

        return None


    def runSessionCompletedHooks(self):
        for tab in self.sessions:
            tab.runSessionCompletedHooks()


    def synchronizeNoActivityTimeouts(self):
        # In this section we synchronize the no-activity timeouts of all the sessions. The session adjusts the no activity
        # If we observe timeouts in one browser, we
        # can assume it will likely show up in other browsers, so lets not hold thm all up.
        minNoActivityTimeout = None
        for session in self.sessions:
            if minNoActivityTimeout is None:
                minNoActivityTimeout = session.noActivityTimeout
            else:
                minNoActivityTimeout = min(minNoActivityTimeout, session.noActivityTimeout)
        for session in self.sessions:
            session.noActivityTimeout = minNoActivityTimeout
