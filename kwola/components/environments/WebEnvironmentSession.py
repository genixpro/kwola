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
from ...datamodels.ActionMapModel import ActionMap
from ...datamodels.actions.ClearFieldAction import ClearFieldAction
from ...datamodels.actions.ClickTapAction import ClickTapAction
from ...datamodels.actions.RightClickAction import RightClickAction
from ...datamodels.actions.TypeAction import TypeAction
from ...datamodels.actions.WaitAction import WaitAction
from ...datamodels.errors.ExceptionError import ExceptionError
from ...datamodels.errors.LogError import LogError
from ...datamodels.ExecutionTraceModel import ExecutionTrace
from ..plugins.base.ProxyPluginBase import ProxyPluginBase
from ..plugins.base.WebEnvironmentPluginBase import WebEnvironmentPluginBase
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
import re
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

    def __init__(self, config, tabNumber, plugins=None, executionSession=None):
        self.config = config
        self.targetURL = config['url']
        self.hasBrowserDied = False

        if plugins is None:
            self.plugins = []
        else:
            self.plugins = plugins


        proxyPlugins = [plugin for plugin in self.plugins if isinstance(plugin, ProxyPluginBase)]
        self.plugins = [plugin for plugin in self.plugins if isinstance(plugin, WebEnvironmentPluginBase)]

        self.executionSession = executionSession

        self.targetHostRoot = self.getHostRoot(self.targetURL)

        self.proxy = ProxyProcess(config, plugins=proxyPlugins)
        self.urlWhitelistRegexes = [re.compile(pattern) for pattern in self.config['web_session_restrict_url_to_regexes']]

        chrome_options = Options()
        chrome_options.headless = config['web_session_headless']
        if config['web_session_enable_shared_chrome_cache']:
            chrome_options.add_argument(f"--disk-cache-dir={self.config.getKwolaUserDataDirectory('chrome_cache')}")
            chrome_options.add_argument(f"--disk-cache-size={1024*1024*1024}")

        chrome_options.add_argument(f"--no-sandbox")
        chrome_options.add_argument(f"--disable-gpu")
        chrome_options.add_argument(f"--disable-features=VizDisplayCompositor")

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

        time.sleep(3)

        if self.config.autologin:
            self.runAutoLogin()

        self.tabNumber = tabNumber
        self.traceNumber = 0

        for plugin in self.plugins:
            if self.executionSession is not None:
                plugin.browserSessionStarted(self.driver, self.proxy, self.executionSession)

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
        if hasattr(self, 'plugins'):
            for plugin in self.plugins:
                if self.executionSession is not None:
                    plugin.cleanup(self.driver, self.proxy, self.executionSession)

        if hasattr(self, "driver"):
            if self.driver:
                self.driver.quit()

        self.driver = None

    def waitUntilNoNetworkActivity(self):
        startTime = datetime.now()
        elapsedTime = 0
        startPaths = set(self.proxy.getPathTrace()['recent'])
        while abs((self.proxy.getMostRecentNetworkActivityTime() - datetime.now()).total_seconds()) < self.config['web_session_no_network_activity_wait_time']:
            time.sleep(0.10)
            elapsedTime = abs((datetime.now() - startTime).total_seconds())
            if elapsedTime > self.config['web_session_no_network_activity_timeout']:
                if self.config['web_session_no_network_activity_timeout'] > 1:
                    getLogger().warning(f"[{os.getpid()}] Warning! There was a timeout while waiting for network activity from the browser to die down. Maybe it is causing non"
                          f" stop network activity all on its own? Try the config variable tweaking web_session_no_network_activity_wait_time down"
                          f" if constant network activity is the expected behaviour. List of suspect paths: {set(self.proxy.getPathTrace()['recent']).difference(startPaths)}")

                    # We adjust the configuration value for this downwards so that if these timeouts are occuring, they're impact on the rest
                    # of the operations are gradually reduced so that the run can proceed
                    self.config['web_session_no_network_activity_timeout'] = self.config['web_session_no_network_activity_timeout'] * 0.50

                break

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
            found = False
            for matchKeyword in emailKeywords:
                if matchKeyword in map.keywords and map.elementType == "input":
                    emailInputs.append(map)
                    found = True
                    break
            if found:
                continue
            for matchKeyword in passwordKeywords:
                if matchKeyword in map.keywords and map.elementType == "input":
                    passwordInputs.append(map)
                    found = True
                    break
            if found:
                continue
            for matchKeyword in loginKeywords:
                if matchKeyword in map.keywords and map.elementType in ['input', 'button', 'div']:
                    loginButtons.append(map)
                    found = True
                    break
            if found:
                continue

        return emailInputs, passwordInputs, loginButtons

    def runAutoLogin(self):
        """
            This method is used to perform the automatic heuristic based login.
        """
        try:
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
                getLogger().warning(f"[{os.getpid()}] Error! Did not detect the all of the necessary HTML elements to perform an autologin. Found: {len(emailInputs)} email looking elements, {len(passwordInputs)} password looking elements, and {len(loginButtons)} submit looking elements. Kwola will be proceeding without automatically logging in.")
                return

            if len(emailInputs) == 1:
                # Find the login button that is closest to the email input while being below it
                loginButtons = sorted(
                    filter(lambda button: bool(button.top > emailInputs[0].bottom), loginButtons),
                    key=lambda button: abs(emailInputs[0].top - button.top)
                )
            elif len(passwordInputs) == 1:
                # Find the login button that is closest to the password input while being below it
                loginButtons = sorted(
                    filter(lambda button: bool(button.top > passwordInputs[0].bottom), loginButtons),
                    key=lambda button: abs(passwordInputs[0].top - button.top)
                )
            else:
                # Find the login button that is lowest down on the page
                loginButtons = sorted(loginButtons, key=lambda button: button.top, reverse=True)

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
                    getLogger().info(f"[{os.getpid()}] Heuristic autologin appears to have worked!")
                else:
                    getLogger().warning(f"[{os.getpid()}] Warning! Unable to verify that the heuristic login worked. The login actions were performed but the URL did not change.")
            else:
                getLogger().warning(f"[{os.getpid()}] There was an error running one of the actions required for the heuristic auto login.")
        except urllib3.exceptions.MaxRetryError:
            self.hasBrowserDied = True
            return None
        except selenium.common.exceptions.WebDriverException:
            self.hasBrowserDied = True
            return None
        except urllib3.exceptions.ProtocolError:
            self.hasBrowserDied = True
            return None

    def getActionMaps(self):
        try:
            if self.hasBrowserDied:
                return []

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
                        
                    if (bounds.top > window.innerHeight || bounds.left > window.innerWidth)
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
                        keywords: (element.textContent + " " + element.getAttribute("class") + " " + element.getAttribute("name") + " " + element.getAttribute("id") + " " + element.getAttribute("type")).toLowerCase() 
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
                    
                    const elemStyle = window.getComputedStyle(element);
                    if (elemStyle.getPropertyValue("cursor") === "pointer")
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
                        if (data.width >= 1 && data.height >= 1)
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
        except urllib3.exceptions.MaxRetryError:
            self.hasBrowserDied = True
            return []
        except selenium.common.exceptions.WebDriverException:
            self.hasBrowserDied = True
            return []
        except urllib3.exceptions.ProtocolError:
            self.hasBrowserDied = True
            return []

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
                        getLogger().info(f"[{os.getpid()}] Clicking {action.x} {action.y} from {action.source}")
                    actionChain.click(on_element=element)
                    actionChain.pause(self.config.web_session_perform_action_wait_time)
                elif action.times == 2:
                    if self.config['web_session_print_every_action']:
                        getLogger().info(f"[{os.getpid()}] Double Clicking {action.x} {action.y} from {action.source}")
                    actionChain.double_click(on_element=element)
                    actionChain.pause(self.config.web_session_perform_action_wait_time)

                actionChain.perform()

            if isinstance(action, RightClickAction):
                if self.config['web_session_print_every_action']:
                    getLogger().info(f"[{os.getpid()}] Right Clicking {action.x} {action.y} from {action.source}")
                actionChain = webdriver.common.action_chains.ActionChains(self.driver)
                actionChain.move_to_element_with_offset(element, 0, 0)
                actionChain.context_click(on_element=element)
                actionChain.pause(self.config.web_session_perform_action_wait_time)
                actionChain.perform()

            if isinstance(action, TypeAction):
                if self.config['web_session_print_every_action']:
                    getLogger().info(f"[{os.getpid()}] Typing {action.text} at {action.x} {action.y} from {action.source}")
                actionChain = webdriver.common.action_chains.ActionChains(self.driver)
                actionChain.move_to_element_with_offset(element, 0, 0)
                actionChain.click(on_element=element)
                actionChain.pause(self.config.web_session_perform_action_wait_time)
                actionChain.send_keys_to_element(element, action.text)
                actionChain.pause(self.config.web_session_perform_action_wait_time)
                actionChain.perform()

            if isinstance(action, ClearFieldAction):
                if self.config['web_session_print_every_action']:
                    getLogger().info(f"[{os.getpid()}] Clearing field at {action.x} {action.y} from {action.source}")
                element.clear()

            if isinstance(action, WaitAction):
                getLogger().info(f"[{os.getpid()}] Waiting for {action.time} at {action.x} {action.y} from {action.source}")
                time.sleep(action.time)

        except selenium.common.exceptions.MoveTargetOutOfBoundsException as e:
            if self.config['web_session_print_every_action_failure']:
                getLogger().warning(f"[{os.getpid()}] Running {action.source} action {action.type} at {action.x},{action.y} failed due to a MoveTargetOutOfBoundsException exception!")

            success = False
        except selenium.common.exceptions.StaleElementReferenceException as e:
            if self.config['web_session_print_every_action_failure']:
                getLogger().warning(f"[{os.getpid()}] Running {action.source} action {action.type} at {action.x},{action.y} failed due to a StaleElementReferenceException!")
            success = False
        except selenium.common.exceptions.InvalidElementStateException as e:
            if self.config['web_session_print_every_action_failure']:
                getLogger().warning(f"[{os.getpid()}] Running {action.source} action {action.type} at {action.x},{action.y} failed due to a InvalidElementStateException!")

            success = False
        except selenium.common.exceptions.TimeoutException as e:
            if self.config['web_session_print_every_action_failure']:
                getLogger().warning(f"[{os.getpid()}] Running {action.source} action {action.type} at {action.x},{action.y} failed due to a TimeoutException!")

            success = False
        except selenium.common.exceptions.JavascriptException as e:
            if self.config['web_session_print_every_action_failure']:
                getLogger().warning(f"[{os.getpid()}] Running {action.source} action {action.type} at {action.x},{action.y} failed due to a JavascriptException!")

            success = False
        except urllib3.exceptions.MaxRetryError as e:
            if self.config['web_session_print_every_action_failure']:
                getLogger().warning(f"[{os.getpid()}] Running {action.source} action {action.type} at {action.x},{action.y} failed due to a MaxRetryError!")

            success = False
        except AttributeError as e:
            if self.config['web_session_print_every_action_failure']:
                getLogger().warning(f"[{os.getpid()}] Running {action.source} action {action.type} at {action.x},{action.y} failed due to an AttributeError!")

            success = False

        # If there was an alert generated as a result of the action, then try to close it.
        try:
            self.driver.switch_to.alert.accept()
        except selenium.common.exceptions.NoAlertPresentException:
            pass

        self.waitUntilNoNetworkActivity()

        return success

    def checkOffsite(self, priorURL):
        try:
            # If the browser went off site and off site links are disabled, then we send it back to the url it started from
            if self.config['prevent_offsite_links']:
                self.waitUntilNoNetworkActivity()

                offsite = False
                if self.driver.current_url != "data:," and self.getHostRoot(self.driver.current_url) != self.targetHostRoot:
                    offsite = True

                whitelistMatched = False
                if len(self.urlWhitelistRegexes) == 0:
                    whitelistMatched = True

                for regex in self.urlWhitelistRegexes:
                    if regex.search(self.driver.current_url) is not None:
                        whitelistMatched = True

                if not whitelistMatched:
                    offsite = True

                if offsite:
                    getLogger().info(f"[{os.getpid()}] The browser session went offsite (to {self.driver.current_url}) and going offsite is disabled. The browser is being reset back to the URL it was at prior to this action: {priorURL}")
                    self.driver.get(priorURL)
                    self.waitUntilNoNetworkActivity()
        except selenium.common.exceptions.TimeoutException:
            pass

    def checkLoadFailure(self):
        try:
            if self.driver.current_url == "data:,":
                getLogger().warning(f"[{os.getpid()}] The browser session needed to be reset back to the origin url {self.targetURL}")
                self.driver.get(self.targetURL)
                self.waitUntilNoNetworkActivity()
        except selenium.common.exceptions.TimeoutException:
            pass

    def runAction(self, action):
        try:
            if self.hasBrowserDied:
                return None

            self.checkOffsite(priorURL=self.targetURL)

            executionTrace = ExecutionTrace(id=str(self.executionSession.id) + "_trace_" + str(self.traceNumber))
            executionTrace.time = datetime.now()
            executionTrace.actionPerformed = action
            executionTrace.errorsDetected = []
            executionTrace.actionMaps = self.getActionMaps()
            executionTrace.didErrorOccur = False
            executionTrace.didNewErrorOccur = False
            executionTrace.hadNetworkTraffic = False
            executionTrace.hadNewNetworkTraffic = False
            executionTrace.didCodeExecute = False
            executionTrace.didNewBranchesExecute = False
            executionTrace.didScreenshotChange = False
            executionTrace.isScreenshotNew = False
            executionTrace.tabNumber = self.tabNumber
            executionTrace.traceNumber = self.traceNumber

            for plugin in self.plugins:
                plugin.beforeActionRuns(self.driver, self.proxy, self.executionSession, executionTrace, action)

            success = self.performActionInBrowser(action)

            executionTrace.didActionSucceed = success

            self.checkOffsite(priorURL=executionTrace.startURL)
            self.checkLoadFailure()

            for plugin in self.plugins:
                plugin.afterActionRuns(self.driver, self.proxy, self.executionSession, executionTrace, action)

            self.traceNumber += 1

            return executionTrace
        except urllib3.exceptions.MaxRetryError:
            self.hasBrowserDied = True
            return None
        except selenium.common.exceptions.WebDriverException:
            self.hasBrowserDied = True
            return None
        except urllib3.exceptions.ProtocolError:
            self.hasBrowserDied = True
            return None

    def screenshotSize(self):
        rect = self.driver.get_window_rect()
        return rect

    def getImage(self):
        try:
            if self.hasBrowserDied:
                return numpy.zeros(shape=[self.config['web_session_height'], self.config['web_session_width'], 3])

            image = cv2.imdecode(numpy.frombuffer(self.driver.get_screenshot_as_png(), numpy.uint8), -1)

            image = numpy.flip(image[:, :, :3], axis=2)  # OpenCV always reads things in BGR for some reason, so we have to flip into RGB ordering

            return image
        except urllib3.exceptions.MaxRetryError:
            self.hasBrowserDied = True
            return numpy.zeros(shape=[self.config['web_session_height'], self.config['web_session_width'], 3])
        except selenium.common.exceptions.WebDriverException:
            self.hasBrowserDied = True
            return numpy.zeros(shape=[self.config['web_session_height'], self.config['web_session_width'], 3])
        except urllib3.exceptions.ProtocolError:
            self.hasBrowserDied = True
            return numpy.zeros(shape=[self.config['web_session_height'], self.config['web_session_width'], 3])

    def runSessionCompletedHooks(self):
        if self.hasBrowserDied:
            return

        for plugin in self.plugins:
            plugin.browserSessionFinished(self.driver, self.proxy, self.executionSession)

