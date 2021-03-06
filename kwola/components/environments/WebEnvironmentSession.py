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


from ...datamodels.ActionMapModel import ActionMap
from ...datamodels.actions.ClearFieldAction import ClearFieldAction
from ...datamodels.actions.ClickTapAction import ClickTapAction
from ...datamodels.actions.RightClickAction import RightClickAction
from ...datamodels.actions.TypeAction import TypeAction
from ...datamodels.actions.WaitAction import WaitAction
from ...datamodels.errors.ExceptionError import ExceptionError
from ...datamodels.errors.LogError import LogError
from ...datamodels.ExecutionTraceModel import ExecutionTrace
from ..utils.video import chooseBestFfmpegVideoCodec
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.proxy import Proxy, ProxyType
from ..proxy.ProxyProcess import ProxyProcess
import cv2
import hashlib
import numpy
import numpy as np
import os
import os.path
import selenium.common.exceptions
import subprocess
import tempfile
import time
import pickle


class WebEnvironmentSession:
    """
        This class represents a single tab in the web environment.
    """

    def __init__(self, config, tabNumber):
        self.config = config
        self.targetURL = config['url']
        self.proxy = ProxyProcess(config)

        chrome_options = Options()
        chrome_options.headless = config['web_session_headless']

        capabilities = webdriver.DesiredCapabilities.CHROME
        capabilities['loggingPrefs'] = {'browser': 'ALL'}
        proxyConfig = Proxy()
        proxyConfig.proxy_type = ProxyType.MANUAL
        proxyConfig.http_proxy = f"localhost:{self.proxy.port}"
        proxyConfig.ssl_proxy = f"localhost:{self.proxy.port}"
        proxyConfig.add_to_capabilities(capabilities)

        self.driver = webdriver.Chrome(desired_capabilities=capabilities, chrome_options=chrome_options)

        window_size = self.driver.execute_script("""
            return [window.outerWidth - window.innerWidth + arguments[0],
              window.outerHeight - window.innerHeight + arguments[1]];
            """, 800, 600)
        self.driver.set_window_size(*window_size)
        self.driver.get(self.targetURL)

        # HACK! give time for page to load before proceeding.
        time.sleep(2)

        if self.config.autologin:
            self.runAutoLogin()

        # Inject bug detection script
        self.driver.execute_script("""
            window.kwolaExceptions = [];
            var currentOnError = window.onerror;
            window.onerror=function(msg, source, lineno, colno, error) {
                let stack = null;
                if (error)
                {
                    stack = error.stack;
                }
                
                window.kwolaExceptions.push([msg, source, lineno, colno, stack]);
                if (currentOnError)
                {
                    currentOnError(msg, source, lineno, colno, error);
                }
            }
        """)

        self.lastScreenshotHash = None
        self.lastProxyPaths = set()

        self.allUrls = set()

        self.tabNumber = tabNumber

        self.frameNumber = 0

        self.screenshotDirectory = tempfile.mkdtemp()
        self.screenshotPaths = []
        self.screenshotHashes = set()

        self.allUrls.add(self.targetURL)
        self.allUrls.add(self.driver.current_url)

        screenHash = self.addScreenshot()
        self.frameNumber = 1
        self.screenshotHashes.add(screenHash)
        self.lastScreenshotHash = screenHash

        self.cumulativeBranchTrace = self.extractBranchTrace()

        self.errorHashes = set()

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        # Cleanup the screenshot files
        if hasattr(self, "screenshotPaths"):
            for filePath in self.screenshotPaths:
                os.unlink(filePath)

            self.screenshotPaths = []
            if hasattr(self, "tabNumber") and os.path.exists(self.movieFilePath()):
                os.unlink(self.movieFilePath())

        if hasattr(self, "screenshotDirectory"):
            if os.path.exists(self.screenshotDirectory):
                os.rmdir(self.screenshotDirectory)

        if hasattr(self, "driver"):
            if self.driver:
                self.driver.quit()

        self.driver = None

    def addScreenshot(self):
        fileName = f"kwola-screenshot-{self.frameNumber:05d}.png"

        filePath = os.path.join(self.screenshotDirectory, fileName)

        self.driver.save_screenshot(filePath)

        hasher = hashlib.md5()
        with open(filePath, 'rb') as imageFile:
            buf = imageFile.read()
            hasher.update(buf)

        screenshotHash = hasher.hexdigest()

        self.screenshotPaths.append(filePath)

        return screenshotHash

    def movieFileName(self):
        return f"kwola-video-{self.tabNumber}.mp4"

    def movieFilePath(self):
        return os.path.join(self.screenshotDirectory, self.movieFileName())

    def createMovie(self):
        result = subprocess.run(['ffmpeg', '-f', 'image2', "-r", "3", '-i', 'kwola-screenshot-%05d.png', '-vcodec', chooseBestFfmpegVideoCodec(), '-crf', '15', self.movieFileName()], cwd=self.screenshotDirectory, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            print(f"Error! Attempted to create a movie using ffmpeg and the process exited with exit-code {result.returncode}. The following output was observed:")
            print(str(result.stdout, 'utf8'))
            print(str(result.stderr, 'utf8'))

        return self.movieFilePath()

    def runAutoLogin(self):
        """
            This method is used to perform the automatic heuristic based login.
        """
        actionMaps = self.getActionMaps()

        # First, try to find the email, password, and submit inputs
        emailInput = None
        passwordInput = None
        loginButton = None

        emailKeywords = ['use', 'mail', 'name']
        passwordKeywords = ['pass']
        loginKeywords = ['log', 'sub']

        for map in actionMaps:
            for matchKeyword in emailKeywords:
                if matchKeyword in map.keywords and map.elementType == "input":
                    emailInput = map
                    break
            for matchKeyword in passwordKeywords:
                if matchKeyword in map.keywords and map.elementType == "input":
                    passwordInput = map
                    break
            for matchKeyword in loginKeywords:
                if matchKeyword in map.keywords and map.elementType in ['input', 'button', 'div']:
                    loginButton = map
                    break

        if emailInput is None or passwordInput is None or loginButton is None:
            print(datetime.now(), f"[{os.getpid()}]", "Error! Did not detect the all of the necessary HTML elements to perform an autologin. Kwola will be proceeding without automatically logging in.", flush=True)
            return

        emailTypeAction = TypeAction(x=emailInput.left + 1,
                                     y=emailInput.top + 1,
                                     source="autologin",
                                     label="email",
                                     text=self.config.email,
                                     type="typeEmail")

        passwordTypeAction = TypeAction(x=passwordInput.left + 1,
                                        y=passwordInput.top + 1,
                                        source="autologin",
                                        label="password",
                                        text=self.config.password,
                                        type="typePassword")

        loginClickAction = ClickTapAction(x=loginButton.left + 1,
                                          y=loginButton.top + 1,
                                          source="autologin",
                                          times=1,
                                          type="click")

        success1 = self.performActionInBrowser(emailTypeAction)
        success2 = self.performActionInBrowser(passwordTypeAction)
        success3 = self.performActionInBrowser(loginClickAction)

        if success1 and success2 and success3:
            print(datetime.now(), f"[{os.getpid()}]",
                  "Heuristic autologin appears to have worked!",
                  flush=True)
        else:
            print(datetime.now(), f"[{os.getpid()}]",
                  "There was an error running one of the actions required for the heuristic auto login.",
                  flush=True)


    def extractBranchTrace(self):
        # The JavaScript that we want to inject. This will extract out the Kwola debug information.
        injected_javascript = (
            'return window.kwolaCounters;'
        )

        result = self.driver.execute_script(injected_javascript)

        # The JavaScript that we want to inject. This will extract out the Kwola debug information.
        injected_javascript = (
            'if (!window.kwolaCounters)'
            '{'
            '   window.kwolaCounters = {};'
            '}'
            'Object.keys(window.kwolaCounters).forEach((fileName) => {'
            '   window.kwolaCounters[fileName].fill(0);'
            '});'
        )

        self.driver.execute_script(injected_javascript)

        if result is not None:
            # Cast everything to a numpy array so we don't have to do it later
            for fileName, vector in result.items():
                result[fileName] = numpy.array(vector)
        else:
            print(datetime.now(), f"[{os.getpid()}]", "Warning, did not find the kwola line counter object in the browser. This usually "
                  "indicates that there was an error either in translating the javascript, an error "
                  "in loading the page, or that the page has absolutely no javascript. "
                  f"On page: {self.driver.current_url}")
            result = {}

        return result

    def extractExceptions(self):
        # The JavaScript that we want to inject. This will extract out the exceptions
        # that the Kwola error handler was able to pick up
        injected_javascript = (
            'const exceptions = window.kwolaExceptions; window.kwolaExceptions = []; return exceptions;'
        )

        result = self.driver.execute_script(injected_javascript)

        if result is None:
            return []

        return result

    def getActionMaps(self):
        elementActionMaps = self.driver.execute_script("""
            function isFunction(functionToCheck) {
             return functionToCheck && {}.toString.call(functionToCheck) === '[object Function]';
            }
        
            const actionMaps = [];
            const domElements = document.querySelectorAll("*");
            
            for(let element of domElements)
            {
                const bounds = element.getBoundingClientRect();
                
                const paddingLeft = Number(window.getComputedStyle(element, null).getPropertyValue('padding-left').replace("px", ""));
                const paddingRight = Number(window.getComputedStyle(element, null).getPropertyValue('padding-right').replace("px", ""));
                const paddingTop = Number(window.getComputedStyle(element, null).getPropertyValue('padding-top').replace("px", ""));
                const paddingBottom = Number(window.getComputedStyle(element, null).getPropertyValue('padding-bottom').replace("px", ""));
                
                const data = {
                    canClick: false,
                    canRightClick: false,
                    canType: false,
                    left: bounds.left + paddingLeft + 3,
                    right: bounds.right - paddingRight - 3,
                    top: bounds.top + paddingTop + 3,
                    bottom: bounds.bottom - paddingBottom - 3,
                    width: bounds.width - paddingLeft - paddingRight - 6,
                    height: bounds.height - paddingTop - paddingBottom - 6,
                    elementType: element.tagName.toLowerCase(),
                    keywords: (element.textContent + " " + element.getAttribute("class") + " " + element.getAttribute("name") + " " + element.getAttribute("id")).toLowerCase() 
                };
                
                if (element.tagName === "A"
                        || element.tagName === "BUTTON"
                        || element.tagName === "AREA"
                        || element.tagName === "AUDIO"
                        || element.tagName === "VIDEO"
                        || element.tagName === "SELECT")
                    data.canClick = true;
                
                if (element.tagName === "INPUT" && !(element.getAttribute("type") === "text" 
                                                     || element.getAttribute("type") === "" 
                                                     || element.getAttribute("type") === "password"
                                                     || element.getAttribute("type") === "email"
                                                     || element.getAttribute("type") === null 
                                                     || element.getAttribute("type") === undefined 
                                                     || !element.getAttribute("type")
                                                  ))
                    data.canClick = true;
                
                if (element.tagName === "INPUT" || element.tagName === "TEXTAREA")
                    data.canType = true;
                
                if (isFunction(element.onclick) 
                    || isFunction(element.onauxclick) 
                    || isFunction(element.onmousedown)
                    || isFunction(element.onmouseup)
                    || isFunction(element.ontouchend)
                    || isFunction(element.ontouchstart))
                    data.canClick = true;
                
                if (isFunction(element.oncontextmenu))
                    data.canRightClick = true;
                
                if (isFunction(element.onkeydown) 
                    || isFunction(element.onkeypress) 
                    || isFunction(element.onkeyup))
                    data.canType = true;
                    
                if (window.kwolaEvents && window.kwolaEvents.has(element))
                {
                    const knownEvents = window.kwolaEvents.get(element);
                    if (knownEvents.indexOf("click") != -1)
                    {
                        data.canClick = true;
                    }
                    if (knownEvents.indexOf("contextmenu") != -1)
                    {
                        data.canRightClick = true;
                    }
                    if (knownEvents.indexOf("dblclick") != -1)
                    {
                        data.canClick = true;
                    }
                    
                    if (knownEvents.indexOf("mousedown") != -1)
                    {
                        data.canClick = true;
                        data.canRightClick = true;
                    }
                    if (knownEvents.indexOf("mouseup") != -1)
                    {
                        data.canClick = true;
                        data.canRightClick = true;
                    }
                    
                    if (knownEvents.indexOf("keydown") != -1)
                    {
                        data.canType = true;
                    }
                    if (knownEvents.indexOf("keypress") != -1)
                    {
                        data.canType = true;
                    }
                    if (knownEvents.indexOf("keyup") != -1)
                    {
                        data.canType = true;
                    }
                }
                
                if (data.canType || data.canClick || data.canRightClick)
                    if (data.width > 0 && data.height > 0)
                        actionMaps.push(data);
            }
            
            return actionMaps;
        """)

        actionMaps = [ActionMap(**actionMapData) for actionMapData in elementActionMaps]
        return actionMaps

    def performActionInBrowser(self, action):
        uiReactionWaitTime = 0.50

        success = True

        try:
            element = self.driver.execute_script("""
            return document.elementFromPoint(arguments[0], arguments[1]);
            """, action.x, action.y)

            if isinstance(action, ClickTapAction):
                actionChain = webdriver.common.action_chains.ActionChains(self.driver)
                actionChain.move_to_element_with_offset(element, 0, 0)
                if action.times == 1:
                    if self.config['web_session_print_every_action']:
                        print(datetime.now(), f"[{os.getpid()}]", "Clicking", action.x, action.y, action.source, flush=True)
                    actionChain.click(on_element=element)
                    actionChain.pause(uiReactionWaitTime)
                elif action.times == 2:
                    if self.config['web_session_print_every_action']:
                        print(datetime.now(), f"[{os.getpid()}]", "Double Clicking", action.x, action.y, action.source, flush=True)
                    actionChain.double_click(on_element=element)
                    actionChain.pause(uiReactionWaitTime)

                actionChain.perform()

            if isinstance(action, RightClickAction):
                if self.config['web_session_print_every_action']:
                    print(datetime.now(), f"[{os.getpid()}]", "Right Clicking", action.x, action.y, action.source, flush=True)
                actionChain = webdriver.common.action_chains.ActionChains(self.driver)
                actionChain.move_to_element_with_offset(element, 0, 0)
                actionChain.context_click(on_element=element)
                actionChain.pause(uiReactionWaitTime)
                actionChain.perform()

            if isinstance(action, TypeAction):
                if self.config['web_session_print_every_action']:
                    print(datetime.now(), f"[{os.getpid()}]", "Typing", action.text, "at", action.x, action.y, action.source, flush=True)
                actionChain = webdriver.common.action_chains.ActionChains(self.driver)
                actionChain.move_to_element_with_offset(element, 0, 0)
                actionChain.click(on_element=element)
                actionChain.pause(uiReactionWaitTime)
                actionChain.send_keys_to_element(element, action.text)
                actionChain.pause(uiReactionWaitTime)
                actionChain.perform()

            if isinstance(action, ClearFieldAction):
                if self.config['web_session_print_every_action']:
                    print(datetime.now(), f"[{os.getpid()}]", "Clearing field at", action.x, action.y, action.source, flush=True)
                element.clear()

            if isinstance(action, WaitAction):
                print(datetime.now(), f"[{os.getpid()}]", "Waiting for ", action.time, "at", action.x, action.y, action.source)
                time.sleep(action.time)

        except selenium.common.exceptions.MoveTargetOutOfBoundsException as e:
            if self.config['web_session_print_every_action_failure']:
                print(datetime.now(), f"[{os.getpid()}]", f"Running action {action.type} {action.source} at {action.x},{action.y} failed!", flush=True)

            success = False
        except selenium.common.exceptions.StaleElementReferenceException as e:
            if self.config['web_session_print_every_action_failure']:
                print(datetime.now(), f"[{os.getpid()}]", f"Running action {action.type} {action.source} at {action.x},{action.y} failed!", flush=True)
            success = False
        except selenium.common.exceptions.InvalidElementStateException as e:
            if self.config['web_session_print_every_action_failure']:
                print(datetime.now(), f"[{os.getpid()}]", f"Running action {action.type} {action.source} at {action.x},{action.y} failed!", flush=True)

            success = False
        except AttributeError as e:
            if self.config['web_session_print_every_action_failure']:
                print(datetime.now(), f"[{os.getpid()}]", f"Running action {action.type} {action.source} at {action.x},{action.y} failed!", flush=True)

            success = False

        # If there was an alert generated as a result of the action, then try to close it.
        try:
            self.driver.switch_to.alert.accept()
        except selenium.common.exceptions.NoAlertPresentException:
            pass

        return success

    def runAction(self, action, executionSessionId):
        executionTrace = ExecutionTrace(id=str(executionSessionId) + "_trace_" + str(self.frameNumber))
        executionTrace.time = datetime.now()
        executionTrace.actionPerformed = action
        executionTrace.errorsDetected = []
        executionTrace.startURL = self.driver.current_url
        executionTrace.actionMaps = self.getActionMaps()

        startLogCount = len(self.driver.get_log('browser'))

        self.proxy.resetPathTrace()

        element = self.driver.execute_script("""
        return document.elementFromPoint(arguments[0], arguments[1]);
        """, action.x, action.y)

        if element is not None:
            executionTrace.cursor = element.value_of_css_property("cursor")
        else:
            executionTrace.cursor = None

        success = self.performActionInBrowser(action)

        executionTrace.didActionSucceed = success

        hadNewError = False
        exceptions = self.extractExceptions()
        for exception in exceptions:
            msg, source, lineno, colno, stack = tuple(exception)
            error = ExceptionError(stacktrace=stack, message=msg, source=source, lineNumber=lineno, columnNumber=colno)
            executionTrace.errorsDetected.append(error)
            errorHash = error.computeHash()

            if errorHash not in self.errorHashes:
                print(datetime.now(), f"[{os.getpid()}]", "An unhandled exception was detected in client application:")
                print(datetime.now(), f"[{os.getpid()}]", f"{msg} at line {lineno} column {colno} in {source}")
                print(datetime.now(), f"[{os.getpid()}]", str(stack), flush=True)

                self.errorHashes.add(errorHash)
                hadNewError = True

        logEntries = self.driver.get_log('browser')[startLogCount:]
        for log in logEntries:
            if log['level'] == 'SEVERE':
                message = str(log['message'])
                message = message.replace("\\n", "\n")
                error = LogError(message=message, logLevel=log['level'])
                executionTrace.errorsDetected.append(error)
                errorHash = error.computeHash()

                if errorHash not in self.errorHashes:
                    print(datetime.now(), f"[{os.getpid()}]", "A log error was detected in client application:")
                    print(datetime.now(), f"[{os.getpid()}]", message, flush=True)

                    self.errorHashes.add(errorHash)
                    hadNewError = True

        screenHash = self.addScreenshot()

        branchTrace = self.extractBranchTrace()

        urlPathTrace = self.proxy.getPathTrace()

        cumulativeProxyPaths = urlPathTrace['seen']
        newProxyPaths = cumulativeProxyPaths.difference(self.lastProxyPaths)
        newBranches = False
        filteredBranchTrace = {}

        for fileName in branchTrace.keys():
            traceVector = branchTrace[fileName]
            didExecuteFile = bool(numpy.sum(traceVector) > 0)

            if didExecuteFile:
                filteredBranchTrace[fileName] = traceVector

            if fileName in self.cumulativeBranchTrace:
                cumulativeTraceVector = self.cumulativeBranchTrace[fileName]

                if len(traceVector) == len(cumulativeTraceVector):
                    newBranchCount = np.sum(traceVector[cumulativeTraceVector == 0])
                    if newBranchCount > 0:
                        newBranches = True
                else:
                    if didExecuteFile:
                        newBranches = True
            else:
                if didExecuteFile:
                    newBranches = True

        executionTrace.branchTrace = {k:v.tolist() for k,v in filteredBranchTrace.items()}

        executionTrace.networkTrafficTrace = list(urlPathTrace['recent'])

        executionTrace.startScreenshotHash = self.lastScreenshotHash
        executionTrace.finishScreenshotHash = screenHash
        executionTrace.tabNumber = self.tabNumber
        executionTrace.frameNumber = self.frameNumber

        executionTrace.logOutput = "\n".join([str(log) for log in logEntries])
        executionTrace.finishURL = self.driver.current_url

        executionTrace.didErrorOccur = len(executionTrace.errorsDetected) > 0
        executionTrace.didNewErrorOccur = hadNewError
        executionTrace.didCodeExecute = bool(len(filteredBranchTrace) > 0)
        executionTrace.didNewBranchesExecute = bool(newBranches)

        executionTrace.hadNetworkTraffic = len(urlPathTrace['recent']) > 0
        executionTrace.hadNewNetworkTraffic = len(newProxyPaths) > 0
        executionTrace.didScreenshotChange = screenHash != self.lastScreenshotHash
        executionTrace.isScreenshotNew = screenHash not in self.screenshotHashes
        executionTrace.didURLChange = executionTrace.startURL != executionTrace.finishURL
        executionTrace.isURLNew = executionTrace.finishURL not in self.allUrls

        executionTrace.hadLogOutput = bool(executionTrace.logOutput)

        total = 0
        executedAtleastOnce = 0
        for fileName in self.cumulativeBranchTrace:
            total += len(self.cumulativeBranchTrace[fileName])
            executedAtleastOnce += np.count_nonzero(self.cumulativeBranchTrace[fileName])

        # Just an extra check here to cover our ass in case of division by zero
        if total == 0:
            total += 1

        executionTrace.cumulativeBranchCoverage = float(executedAtleastOnce) / float(total)

        for fileName in filteredBranchTrace.keys():
            if fileName in self.cumulativeBranchTrace:
                if len(branchTrace[fileName]) == len(self.cumulativeBranchTrace[fileName]):
                    self.cumulativeBranchTrace[fileName] += branchTrace[fileName]
                else:
                    print(f"Warning! The file with fileName {fileName} has changed the size of its trace vector. This "
                          f"is very unusual and could indicate some strange situation with dynamically loaded javascript")
            else:
                self.cumulativeBranchTrace[fileName] = branchTrace[fileName]

        self.allUrls.add(self.driver.current_url)

        self.lastScreenshotHash = screenHash
        self.screenshotHashes.add(screenHash)

        self.lastProxyPaths = set(urlPathTrace['seen'])

        self.frameNumber += 1

        return executionTrace

    def screenshotSize(self):
        rect = self.driver.get_window_rect()
        return rect

    def getImage(self):
        image = cv2.imdecode(numpy.frombuffer(self.driver.get_screenshot_as_png(), numpy.uint8), -1)

        image = numpy.flip(image[:, :, :3], axis=2)  # OpenCV always reads things in BGR for some reason, so we have to flip into RGB ordering

        return image
