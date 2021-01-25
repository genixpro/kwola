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
import traceback
import psutil
from kwola.components.utils.asyncthreadfuture import AsyncThreadFuture
from ..utils.retry import autoretry
from ..plugins.core.RecordAllPaths import RecordAllPaths
from ..plugins.core.RecordBranchTrace import RecordBranchTrace
from ..plugins.core.RecordCursorAtAction import RecordCursorAtAction
from ..plugins.core.RecordExceptions import RecordExceptions
from ..plugins.core.RecordLogEntriesAndLogErrors import RecordLogEntriesAndLogErrors
from ..plugins.core.RecordNetworkErrors import RecordNetworkErrors
from ..plugins.core.RecordDotNetRPCErrors import RecordDotNetRPCErrors
from ..plugins.core.RecordPageURLs import RecordPageURLs
from ..plugins.core.RecordScreenshots import RecordScreenshots
from ..plugins.core.RecordPageHTML import RecordPageHTML


class WebEnvironment:
    """
        This class represents web / browser based environments. It will boot up a headless browser and use it to communicate
        with the software.
    """

    def __init__(self, config, sessionLimit=None, plugins=None, executionSessions=None, browser=None, windowSize=None):
        self.config = config

        defaultPlugins = [
            RecordCursorAtAction(),
            RecordPageURLs(),
            RecordExceptions(),
            RecordLogEntriesAndLogErrors(config),
            RecordNetworkErrors(),
            RecordDotNetRPCErrors(),
            RecordAllPaths(),
            RecordBranchTrace(),
            RecordScreenshots(config)
        ]

        if config['enable_record_page_html']:
            defaultPlugins.append(RecordPageHTML(config))

        if plugins is None:
            # Put in the default set up plugins
            self.plugins = defaultPlugins
        else:
            self.plugins = defaultPlugins + plugins

        self.windowSize = windowSize

        @autoretry()
        def createSession(sessionNumber):
            session = WebEnvironmentSession(config, sessionNumber, self.plugins, self.executionSessions[sessionNumber], browser=browser, windowSize=windowSize)
            return session

        def onInitializeFailure(session):
            session.hasBrowserDied = True
            session.browserDeathReason = f"A fatal error occurred during session initialization: {traceback.format_exc()}"

        @autoretry(ignoreFailure=True, onFinalFailure=onInitializeFailure)
        def initializeSession(session):
            session.initialize()

        sessionCount = config['web_session_parallel_execution_sessions']
        if sessionLimit is not None:
            sessionCount = min(sessionLimit, sessionCount)

        if executionSessions is None:
            self.executionSessions = [None] * sessionCount
        else:
            self.executionSessions = executionSessions

        getLogger().info(f"Starting up {sessionCount} parallel browser sessions.")

        self.sessions = []
        for sessionNumber in range(sessionCount):
            self.sessions.append(createSession(sessionNumber))

        futures = []
        for sessionNumber in range(sessionCount):
            future = AsyncThreadFuture(initializeSession, [self.sessions[sessionNumber]], timeout=self.config['web_session_initialization_timeout'])
            futures.append((future, self.sessions[sessionNumber]))

        for future, session in futures:
            try:
                result = future.result()
            except TimeoutError:
                session.hasBrowserDied = True
                session.browserDeathReason = f"A fatal error occurred during session initialization: {traceback.format_exc()}"


    def shutdown(self):
        for session in self.sessions:
            session.shutdown()


    def screenshotSize(self):
        return self.sessions[0].screenshotSize()

    def getImages(self):
        results = []
        imageFutures = []
        for session in self.sessions:
            future = AsyncThreadFuture(session.getImage, [], timeout=self.config['testing_get_image_timeout'])
            imageFutures.append(future)

        for future, session in zip(imageFutures, self.sessions):
            try:
                result = future.result()
            except TimeoutError:
                getLogger().warning("Warning: timeout exceeded in WebEnvironment.getImages")
                result = numpy.zeros(shape=[self.config['web_session_height'][self.windowSize], self.config['web_session_width'][self.windowSize], 3])
                session.hasBrowserDied = True
                session.browserDeathReason = f"The browser timed out while inside WebEnvironment.getImages"
            results.append(result)

        return results

    def getActionMaps(self):
        results = []

        actionMapFutures = []
        for session in self.sessions:
            future = AsyncThreadFuture(session.getActionMaps, [], timeout=self.config['testing_fetch_action_map_timeout'])
            actionMapFutures.append(future)

        for future, session in zip(actionMapFutures, self.sessions):
            try:
                result = future.result()
            except TimeoutError:
                getLogger().warning("Warning: timeout exceeded in WebEnvironment.getActionMaps")
                result = []
                session.hasBrowserDied = True
                session.browserDeathReason = f"The browser timed out while inside WebEnvironment.getActionMaps"
            results.append(result)

        return results

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

        results = []
        for session, action in zip(self.sessions, actions):
            if action is not None:
                future = AsyncThreadFuture(session.runAction, [action], timeout=self.config['testing_run_action_timeout'])
                resultFutures.append(future)
            else:
                resultFutures.append(None)

        for session, future in zip(self.sessions, resultFutures):
            try:
                if future is not None:
                    result = future.result()
                else:
                    result = (None, {})
            except TimeoutError:
                getLogger().warning("Warning: timeout exceeded in WebEnvironment.runActions")
                result = (None, {})
                session.hasBrowserDied = True
                session.browserDeathReason = f"The browser timed out while inside WebEnvironment.runActions"

            results.append(result)

        traces = [
            result[0] for result in results
        ]

        actionExecutionTimes = [
            result[1] for result in results
        ]

        self.synchronizeNoActivityTimeouts()

        timeTaken = (datetime.now() - startTime).total_seconds()
        if timeTaken > 10:
            timeList = []

            # Log data for all of the sub times
            for key in actionExecutionTimes[0]:
                timeList.append({
                    "key": key,
                    "min": numpy.min([times[key] for times in actionExecutionTimes if key in times]),
                    "max": numpy.max([times[key] for times in actionExecutionTimes if key in times]),
                    "mean": numpy.mean([times[key] for times in actionExecutionTimes if key in times]),
                    "median": numpy.median([times[key] for times in actionExecutionTimes if key in times]),
                    "std": numpy.std([times[key] for times in actionExecutionTimes if key in times])
                })

            timeList = sorted(timeList, key=lambda x: x['max'], reverse=True)
            maxTimes = [(t['key'], t['max']) for t in timeList if t['max'] > 0.5]
            stdTimes = [(t['key'], t['std']) for t in timeList if t['max'] > 0.5]

            getLogger().warning(f"Time taken to execute the actions in the browser was unusually long: {timeTaken} seconds. Here are the subtimes: {pformat(maxTimes)} and standard deviations: {pformat(stdTimes)}")

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
                getLogger().warning(
                    f"Removing web browser session at index {sessionN} because the browser has failed. Reason: {self.sessions[sessionN].browserDeathReason}")
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
