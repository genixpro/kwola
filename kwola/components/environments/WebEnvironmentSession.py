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
import urllib.parse
import urllib3

class WebEnvironmentSession:
    """
        This class represents a single tab in the web environment.
    """

    def __init__(self, config, tabNumber):
        self.config = config
        self.targetURL = config['url']

        self.targetHostRoot = self.getHostRoot(self.targetURL)

        self.proxy = ProxyProcess(config)

        chrome_options = Options()
        chrome_options.headless = config['web_session_headless']
        if config['web_session_enable_shared_chrome_cache']:
            chrome_options.add_argument(f"--disk-cache-dir={self.config.getKwolaUserDataDirectory('chrome_cache')}")
            chrome_options.add_argument(f"--disk-cache-size={1024*1024*1024}")

        chrome_options.add_argument(f"--no-sandbox")

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
            """, self.config['web_session_width'], self.config['web_session_height'])
        self.driver.set_window_size(*window_size)
        try:
            self.driver.get(self.targetURL)
        except selenium.common.exceptions.TimeoutException:
            raise RuntimeError(f"The web-browser timed out while attempting to load the target URL {self.targetURL}")

        time.sleep(2)

        self.waitUntilNoNetworkActivity()

        if self.config.autologin:
            self.runAutoLogin()

        # Inject bug detection script
        self.driver.execute_script("""
            window.kwolaExceptions = [];
            var kwolaCurrentOnError = window.onerror;
            window.onerror=function(msg, source, lineno, colno, error) {
                let stack = null;
                if (error)
                {
                    stack = error.stack;
                }
                
                window.kwolaExceptions.push([msg, source, lineno, colno, stack]);
                if (kwolaCurrentOnError)
                {
                    kwolaCurrentOnError(msg, source, lineno, colno, error);
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

        self.kwolaJSRewriteErrorDetectionStrings = [
            "globalKwola",
            "kwolaError",
            "global_removeEventListener",
            "global_addEventListener",
        ]

    def __del__(self):
        self.shutdown()

    def getHostRoot(self, url):
        host = str(urllib.parse.urlparse(url).hostname)

        hostParts = host.split(".")

        if ".com." in host or ".co." in host:
            hostRoot = ".".join(hostParts[-3:])
        else:
            hostRoot = ".".join(hostParts[-2:])

        return hostRoot

    def shutdown(self):
        # Cleanup the screenshot files
        if hasattr(self, "screenshotPaths"):
            for filePath in self.screenshotPaths:
                if os.path.exists(filePath):
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

        hasher = hashlib.sha256()
        with open(filePath, 'rb') as imageFile:
            buf = imageFile.read()
            hasher.update(buf)

        screenshotHash = hasher.hexdigest()

        self.screenshotPaths.append(filePath)

        return screenshotHash

    def waitUntilNoNetworkActivity(self):
        startTime = datetime.now()
        elapsedTime = 0
        while abs((self.proxy.getMostRecentNetworkActivityTime() - datetime.now()).total_seconds()) < self.config['web_session_no_network_activity_wait_time']:
            time.sleep(0.10)
            elapsedTime = abs((datetime.now() - startTime).total_seconds())
            if elapsedTime > self.config['web_session_no_network_activity_timeout']:
                print(datetime.now(), f"[{os.getpid()}]",
                      "Warning! There was a timeout while waiting for network activity from the browser to die down. Maybe it is causing non"
                      " stop network activity all on its own? Try the config variable tweaking web_session_no_network_activity_wait_time down"
                      " if constant network activity is the expected behaviour.",
                      flush=True)
                break

    def movieFileName(self):
        return f"kwola-video-{self.tabNumber}.mp4"

    def movieFilePath(self):
        return os.path.join(self.screenshotDirectory, self.movieFileName())

    def createMovie(self):
        result = subprocess.run(['ffmpeg', '-f', 'image2', "-r", "3", '-i', 'kwola-screenshot-%05d.png', '-vcodec', chooseBestFfmpegVideoCodec(), '-pix_fmt', 'yuv420p', '-crf', '15', self.movieFileName()], cwd=self.screenshotDirectory, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            print(f"Error! Attempted to create a movie using ffmpeg and the process exited with exit-code {result.returncode}. The following output was observed:")
            print(str(result.stdout, 'utf8'))
            print(str(result.stderr, 'utf8'))

        return self.movieFilePath()

    def findElementsForAutoLogin(self):
        actionMaps = self.getActionMaps()

        # First, try to find the email, password, and submit inputs
        emailInputs = []
        passwordInputs = []
        loginButtons = []

        emailKeywords = ['use', 'mail', 'name']
        passwordKeywords = ['pass']
        loginKeywords = ['log', 'sub', 'sign']

        for map in actionMaps:
            for matchKeyword in emailKeywords:
                if matchKeyword in map.keywords and map.elementType == "input":
                    emailInputs.append(map)
                    break
            for matchKeyword in passwordKeywords:
                if matchKeyword in map.keywords and map.elementType == "input":
                    passwordInputs.append(map)
                    break
            for matchKeyword in loginKeywords:
                if matchKeyword in map.keywords and map.elementType in ['input', 'button', 'div']:
                    loginButtons.append(map)
                    break

        return emailInputs, passwordInputs, loginButtons

    def runAutoLogin(self):
        """
            This method is used to perform the automatic heuristic based login.
        """
        time.sleep(2)
        self.waitUntilNoNetworkActivity()

        emailInputs, passwordInputs, loginButtons = self.findElementsForAutoLogin()

        # check to see if there is a "login" button that we need to click first to expose
        # the login form
        if (len(emailInputs) == 0) and (len(passwordInputs) == 0) and len(loginButtons) > 0:
            loginTriggerButton = loginButtons[0]

            loginClickAction = ClickTapAction(x=loginTriggerButton.left + 1,
                                              y=loginTriggerButton.top + 1,
                                              source="autologin",
                                              times=1,
                                              type="click")
            success = self.performActionInBrowser(loginClickAction)

            emailInputs, passwordInputs, loginButtons = self.findElementsForAutoLogin()

            loginButtons = list(filter(lambda button: button.keywords != loginTriggerButton.keywords, loginButtons))


        if len(emailInputs) == 0 or len(passwordInputs) == 0 or len(loginButtons) == 0:
            print(datetime.now(), f"[{os.getpid()}]", "Error! Did not detect the all of the necessary HTML elements to perform an autologin. Kwola will be proceeding without automatically logging in.", flush=True)
            return

        startURL = self.driver.current_url

        emailTypeAction = TypeAction(x=emailInputs[0].left + 1,
                                     y=emailInputs[0].top + 1,
                                     source="autologin",
                                     label="email",
                                     text=self.config.email,
                                     type="typeEmail")

        passwordTypeAction = TypeAction(x=passwordInputs[0].left + 1,
                                        y=passwordInputs[0].top + 1,
                                        source="autologin",
                                        label="password",
                                        text=self.config.password,
                                        type="typePassword")

        loginClickAction = ClickTapAction(x=loginButtons[0].left + 1,
                                          y=loginButtons[0].top + 1,
                                          source="autologin",
                                          times=1,
                                          type="click")

        success1 = self.performActionInBrowser(emailTypeAction)
        success2 = self.performActionInBrowser(passwordTypeAction)
        success3 = self.performActionInBrowser(loginClickAction)

        time.sleep(2)
        self.waitUntilNoNetworkActivity()

        didURLChange = bool(startURL != self.driver.current_url)

        if success1 and success2 and success3:
            if didURLChange:
                print(datetime.now(), f"[{os.getpid()}]",
                      "Heuristic autologin appears to have worked!",
                      flush=True)
            else:
                print(datetime.now(), f"[{os.getpid()}]",
                      "Warning! Unable to verify that the heuristic login worked. The login actions were performed but the URL did not change.",
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

        try:
            self.driver.execute_script(injected_javascript)
        except selenium.common.exceptions.TimeoutException:
            print(datetime.now(), f"[{os.getpid()}]", "Warning, timeout while running the script to reset the kwola line counters.")

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
               
                if (bounds.bottom < 0 || bounds.right < 0)
                    continue;
                    
                if (bounds.top > window.innerHeight || bounds.right > window.innerWidth)
                    continue;
                
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
                
                // Skip this element if it covers basically the whole screen.
                if (data.width > (window.innerWidth * 0.80) && data.height > (window.innerHeight * 0.80))
                {
                    continue;
                }
                
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

        actionMaps = []

        for actionMapData in elementActionMaps:
            actionMap = ActionMap(**actionMapData)
            # Cut the keywords off at 1024 characters to prevent too much memory / storage usage
            actionMap.keywords = actionMap.keywords[:self.config['web_session_action_map_max_keyword_size']]
            actionMaps.append(actionMap)

        return actionMaps

    def performActionInBrowser(self, action):
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
                    actionChain.pause(self.config.web_session_perform_action_wait_time)
                elif action.times == 2:
                    if self.config['web_session_print_every_action']:
                        print(datetime.now(), f"[{os.getpid()}]", "Double Clicking", action.x, action.y, action.source, flush=True)
                    actionChain.double_click(on_element=element)
                    actionChain.pause(self.config.web_session_perform_action_wait_time)

                actionChain.perform()

            if isinstance(action, RightClickAction):
                if self.config['web_session_print_every_action']:
                    print(datetime.now(), f"[{os.getpid()}]", "Right Clicking", action.x, action.y, action.source, flush=True)
                actionChain = webdriver.common.action_chains.ActionChains(self.driver)
                actionChain.move_to_element_with_offset(element, 0, 0)
                actionChain.context_click(on_element=element)
                actionChain.pause(self.config.web_session_perform_action_wait_time)
                actionChain.perform()

            if isinstance(action, TypeAction):
                if self.config['web_session_print_every_action']:
                    print(datetime.now(), f"[{os.getpid()}]", "Typing", action.text, "at", action.x, action.y, action.source, flush=True)
                actionChain = webdriver.common.action_chains.ActionChains(self.driver)
                actionChain.move_to_element_with_offset(element, 0, 0)
                actionChain.click(on_element=element)
                actionChain.pause(self.config.web_session_perform_action_wait_time)
                actionChain.send_keys_to_element(element, action.text)
                actionChain.pause(self.config.web_session_perform_action_wait_time)
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
                print(datetime.now(), f"[{os.getpid()}]", f"Running {action.source} action {action.type} at {action.x},{action.y} failed due to a MoveTargetOutOfBoundsException exception!", flush=True)

            success = False
        except selenium.common.exceptions.StaleElementReferenceException as e:
            if self.config['web_session_print_every_action_failure']:
                print(datetime.now(), f"[{os.getpid()}]", f"Running {action.source} action {action.type} at {action.x},{action.y} failed due to a StaleElementReferenceException!", flush=True)
            success = False
        except selenium.common.exceptions.InvalidElementStateException as e:
            if self.config['web_session_print_every_action_failure']:
                print(datetime.now(), f"[{os.getpid()}]", f"Running {action.source} action {action.type} at {action.x},{action.y} failed due to a InvalidElementStateException!", flush=True)

            success = False
        except selenium.common.exceptions.TimeoutException as e:
            if self.config['web_session_print_every_action_failure']:
                print(datetime.now(), f"[{os.getpid()}]", f"Running {action.source} action {action.type} at {action.x},{action.y} failed due to a TimeoutException!", flush=True)

            success = False
        except selenium.common.exceptions.JavascriptException as e:
            if self.config['web_session_print_every_action_failure']:
                print(datetime.now(), f"[{os.getpid()}]", f"Running {action.source} action {action.type} at {action.x},{action.y} failed due to a JavascriptException!", flush=True)

            success = False
        except urllib3.exceptions.MaxRetryError as e:
            if self.config['web_session_print_every_action_failure']:
                print(datetime.now(), f"[{os.getpid()}]", f"Running {action.source} action {action.type} at {action.x},{action.y} failed due to a MaxRetryError!", flush=True)

            success = False
        except AttributeError as e:
            if self.config['web_session_print_every_action_failure']:
                print(datetime.now(), f"[{os.getpid()}]", f"Running {action.source} action {action.type} at {action.x},{action.y} failed due to an AttributeError!", flush=True)

            success = False

        # If there was an alert generated as a result of the action, then try to close it.
        try:
            self.driver.switch_to.alert.accept()
        except selenium.common.exceptions.NoAlertPresentException:
            pass

        self.waitUntilNoNetworkActivity()

        return success

    def checkOffsite(self, priorURL):
        self.waitUntilNoNetworkActivity()

        try:
            # If the browser went off site and off site links are disabled, then we send it back to the url it started from
            if self.config['prevent_offsite_links']:
                if self.driver.current_url != "data:," and self.getHostRoot(self.driver.current_url) != self.targetHostRoot:
                    print(datetime.now(), f"[{os.getpid()}]", f"The browser session went offsite (to {self.driver.current_url}) and going offsite is disabled. The browser is being reset back to the URL it was at prior to this action: {priorURL}")
                    self.driver.get(priorURL)
                    self.waitUntilNoNetworkActivity()
        except selenium.common.exceptions.TimeoutException:
            pass

    def checkLoadFailure(self):
        try:
            if self.driver.current_url == "data:,":
                print(datetime.now(), f"[{os.getpid()}]", f"The browser session needed to be reset back to the origin url {self.targetURL}")
                self.driver.get(self.targetURL)
                self.waitUntilNoNetworkActivity()
        except selenium.common.exceptions.TimeoutException:
            pass

    def runAction(self, action, executionSessionId):
        self.checkOffsite(priorURL=self.targetURL)

        executionTrace = ExecutionTrace(id=str(executionSessionId) + "_trace_" + str(self.frameNumber))
        executionTrace.time = datetime.now()
        executionTrace.actionPerformed = action
        executionTrace.errorsDetected = []
        executionTrace.startURL = self.driver.current_url
        executionTrace.actionMaps = self.getActionMaps()

        startLogCount = len(self.driver.get_log('browser'))

        self.proxy.resetPathTrace()

        try:
            element = self.driver.execute_script("""
            return document.elementFromPoint(arguments[0], arguments[1]);
            """, action.x, action.y)

            if element is not None:
                executionTrace.cursor = element.value_of_css_property("cursor")
            else:
                executionTrace.cursor = None

        except selenium.common.exceptions.StaleElementReferenceException as e:
            executionTrace.cursor = None

        success = self.performActionInBrowser(action)

        executionTrace.didActionSucceed = success

        self.checkOffsite(priorURL=executionTrace.startURL)
        self.checkLoadFailure()

        hadNewError = False
        exceptions = self.extractExceptions()
        for exception in exceptions:
            msg, source, lineno, colno, stack = tuple(exception)

            msg = str(msg)
            source = str(source)
            stack = str(stack)

            combinedMessage = msg + source + stack

            kwolaJSRewriteErrorFound = False
            for detectionString in self.kwolaJSRewriteErrorDetectionStrings:
                if detectionString in combinedMessage:
                    kwolaJSRewriteErrorFound = True

            if kwolaJSRewriteErrorFound:
                print(datetime.now(), f"[{os.getpid()}]", f"Error. There was a bug generated by the underlying javascript application, "
                                                          f"but it appears to be a bug in Kwola's JS rewriting. Please notify the Kwola "
                                                          f"developers that this url: {self.driver.current_url} gave you a js-code-rewriting "
                                                          f"issue.")
                print(datetime.now(), f"[{os.getpid()}]", f"{msg} at line {lineno} column {colno} in {source}")
                print(datetime.now(), f"[{os.getpid()}]", str(stack), flush=True)
            else:
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

                kwolaJSRewriteErrorFound = False
                for detectionString in self.kwolaJSRewriteErrorDetectionStrings:
                    if detectionString in message:
                        kwolaJSRewriteErrorFound = True

                if kwolaJSRewriteErrorFound:
                    print(datetime.now(), f"[{os.getpid()}]", f"Error. There was a bug generated by the underlying javascript application, "
                                                              f"but it appears to be a bug in Kwola's JS rewriting. Please notify the Kwola "
                                                              f"developers that this url: {self.driver.current_url} gave you a js-code-rewriting "
                                                              f"issue.")
                    print(datetime.now(), f"[{os.getpid()}]", message, flush=True)
                else:
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
