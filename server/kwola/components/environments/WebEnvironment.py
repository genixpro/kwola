from .BaseEnvironment import BaseEnvironment
from selenium import webdriver
from selenium.webdriver.common.proxy import Proxy, ProxyType
from selenium.webdriver.chrome.options import Options
import time
import numpy as np
from mitmproxy.tools.dump import DumpMaster
from kwola.components.proxy.JSRewriteProxy import JSRewriteProxy
from kwola.components.proxy.PathTracer import PathTracer
from kwola.models.actions.ClickTapAction import ClickTapAction
from kwola.models.actions.RightClickAction import RightClickAction
from kwola.models.actions.TypeAction import TypeAction
from kwola.models.actions.WaitAction import WaitAction
from kwola.models.ExecutionTraceModel import ExecutionTrace
import selenium.common.exceptions
import tempfile
import subprocess
import os
import os.path
import hashlib
from threading import Thread
import asyncio
import socket
from contextlib import closing



class WebEnvironment(BaseEnvironment):
    """
        This class represents web / browser based environments. It will boot up a headless browser and use it to communicate
        with the software.
    """
    def __init__(self, targetURL="http://172.17.0.2:3000/"):
        self.targetURL = targetURL

        self.lastScreenshotHash = None
        self.lastProxyPaths = set()

        self.allUrls = set()

        self.frameNumber = 0

        self.screenshotDirectory = tempfile.mkdtemp()
        self.screenshotPaths = []
        self.screenshotHashes = set()

        self.proxyPort = self.find_free_port()

        self.proxyThread = Thread(target=lambda: self.runProxyServer())
        self.proxyThread.start()

        chrome_options = Options()
        chrome_options.headless = True

        capabilities = webdriver.DesiredCapabilities.CHROME
        capabilities['loggingPrefs'] = {'browser': 'ALL'}
        proxyConfig = Proxy()
        proxyConfig.proxy_type = ProxyType.MANUAL
        proxyConfig.http_proxy = f"localhost:{self.proxyPort}"
        proxyConfig.add_to_capabilities(capabilities)

        self.driver = webdriver.Chrome(desired_capabilities=capabilities, chrome_options=chrome_options)

        self.driver.get(self.targetURL)

        self.lastBranchTrace = self.extractBranchTrace()

        self.allUrls.add(self.targetURL)
        self.allUrls.add(self.driver.current_url)

        screenHash = self.addScreenshot()
        self.screenshotHashes.add(screenHash)
        self.lastScreenshotHash = screenHash

        self.lastCumulativeBranchExecutionVector = self.computeCumulativeBranchExecutionVector(self.extractBranchTrace())
        self.decayingExecutionTrace = np.zeros_like(self.lastCumulativeBranchExecutionVector)


    def __del__(self):
        # Cleanup the screenshot files
        for path in self.screenshotPaths:
            os.unlink(path)

        self.driver.quit()
        # self.proxyThread.kill

    def find_free_port(self):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    def runProxyServer(self):
        from mitmproxy import proxy, options

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        self.codeRewriter = JSRewriteProxy()
        self.pathTracer = PathTracer()

        opts = options.Options(listen_port=self.proxyPort)
        pconf = proxy.config.ProxyConfig(opts)


        m = DumpMaster(opts)
        m.server = proxy.server.ProxyServer(pconf)
        m.addons.add(self.codeRewriter)
        m.addons.add(self.pathTracer)

        m.run()


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

    def createMovie(self):
        subprocess.run(['ffmpeg', '-r', '60', '-f', 'image2', "-r", "2", '-i', 'kwola-screenshot-%05d.png', '-vcodec', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', 'kwola-video.mp4'], cwd=self.screenshotDirectory)

        return os.path.join(self.screenshotDirectory, "kwola-video.mp4")


    def extractBranchTrace(self):
        # The JavaScript that we want to inject. This will extract out the Kwola debug information.
        injected_javascript = (
            'return window.kwolaCounters;'
        )

        result = self.driver.execute_script(injected_javascript)

        return result

    def branchFeatureSize(self):
        return len(self.lastCumulativeBranchExecutionVector)

    def computeCumulativeBranchExecutionVector(self, branchTrace):
        cumulativeBranchExecutionVector = np.array([])

        for fileName in sorted(branchTrace.keys()):
            if fileName in self.lastBranchTrace:
                counterVector = branchTrace[fileName]
                cumulativeBranchExecutionVector = np.concatenate([cumulativeBranchExecutionVector, np.array(counterVector)])

        return cumulativeBranchExecutionVector


    def runAction(self, action):
        executionTrace = ExecutionTrace()
        executionTrace.actionPerformed = action
        executionTrace.errorsDetected = []
        executionTrace.startURL = self.driver.current_url

        startLogCount = len(self.driver.get_log('browser'))

        self.pathTracer.recentPaths = set()

        success = True

        try:
            if isinstance(action, ClickTapAction):
                    actionChain = webdriver.common.action_chains.ActionChains(self.driver)
                    actionChain.move_to_element_with_offset(self.driver.find_element_by_tag_name('body'), 0, 0)
                    actionChain.move_by_offset(action.x, action.y)
                    if action.times == 1:
                        print("Clicking", action.x, action.y)
                        actionChain.click()
                    elif action.times == 2:
                        print("Double Clicking", action.x, action.y)
                        actionChain.double_click()

                    actionChain.perform()

            if isinstance(action, RightClickAction):
                    print("Right Clicking", action.x, action.y)
                    actionChain = webdriver.common.action_chains.ActionChains(self.driver)
                    actionChain.move_to_element_with_offset(self.driver.find_element_by_tag_name('body'), 0, 0)
                    actionChain.move_by_offset(action.x, action.y)
                    actionChain.context_click()
                    actionChain.perform()

            if isinstance(action, TypeAction):
                    print("Typing", action.text, "at", action.x, action.y)
                    actionChain = webdriver.common.action_chains.ActionChains(self.driver)
                    actionChain.move_to_element_with_offset(self.driver.find_element_by_tag_name('body'), 0, 0)
                    actionChain.move_by_offset(action.x, action.y)
                    actionChain.send_keys(action.text)
                    actionChain.perform()

            if isinstance(action, WaitAction):
                print("Waiting for ", action.time)
                time.sleep(action.time)

        except selenium.common.exceptions.MoveTargetOutOfBoundsException:
            print("Running action failed!")

            success = False

        executionTrace.didActionSucceed = success

        screenHash = self.addScreenshot()

        branchTrace = self.extractBranchTrace()

        cumulativeProxyPaths = self.pathTracer.seenPaths
        newProxyPaths = cumulativeProxyPaths.difference(self.lastProxyPaths)

        cumulativeBranchExecutionVector = self.computeCumulativeBranchExecutionVector(branchTrace)

        if self.lastCumulativeBranchExecutionVector is not None:
            branchExecutionVector = cumulativeBranchExecutionVector - self.lastCumulativeBranchExecutionVector
        else:
            branchExecutionVector = cumulativeBranchExecutionVector

        executionTrace.branchExecutionTrace = branchExecutionVector.tolist()
        executionTrace.startCumulativeBranchExecutionTrace = self.lastCumulativeBranchExecutionVector.tolist()
        executionTrace.startDecayingExecutionTrace = self.decayingExecutionTrace.tolist()

        executionTrace.networkTrafficTrace = list(self.pathTracer.recentPaths)

        executionTrace.startScreenshotHash = self.lastScreenshotHash
        executionTrace.finishScreenshotHash = screenHash
        executionTrace.frameNumber = self.frameNumber

        executionTrace.logOutput = "\n".join(list(self.driver.get_log('browser'))[startLogCount:])
        executionTrace.finishURL = self.driver.current_url

        executionTrace.didErrorOccur = len(executionTrace.errorsDetected) > 0
        executionTrace.didNewErrorOccur = False
        executionTrace.didCodeExecute = bool(np.sum(branchExecutionVector) > 0)

        if self.lastCumulativeBranchExecutionVector is not None:
            executionTrace.didNewBranchesExecute = bool(np.sum(branchExecutionVector[self.lastCumulativeBranchExecutionVector == 0]) > 0)
        else:
            executionTrace.didNewBranchesExecute = True

        executionTrace.hadNetworkTraffic = len(self.pathTracer.recentPaths) > 0
        executionTrace.hadNewNetworkTraffic = len(newProxyPaths) > 0
        executionTrace.didScreenshotChange = screenHash != self.lastScreenshotHash
        executionTrace.isScreenshotNew = screenHash not in self.screenshotHashes
        executionTrace.didURLChange = executionTrace.startURL != executionTrace.finishURL
        executionTrace.isURLNew = executionTrace.finishURL not in self.allUrls

        executionTrace.hadLogOutput = bool(executionTrace.logOutput)
        executionTrace.cumulativeBranchCoverage = float(np.count_nonzero(cumulativeBranchExecutionVector)) / len(cumulativeBranchExecutionVector)

        # Update the decaying execution trace to use as input for the next round
        self.decayingExecutionTrace = np.maximum(self.decayingExecutionTrace * 0.95, np.minimum(np.ones_like(branchExecutionVector), branchExecutionVector))

        self.lastBranchTrace = branchTrace
        self.lastCumulativeBranchExecutionVector = cumulativeBranchExecutionVector
        self.allUrls.add(self.driver.current_url)

        self.lastScreenshotHash = screenHash
        self.screenshotHashes.add(screenHash)

        self.lastProxyPaths = set(self.pathTracer.seenPaths)

        self.frameNumber += 1

        return executionTrace

    def screenshotSize(self):
        rect = self.driver.get_window_rect()
        return rect
