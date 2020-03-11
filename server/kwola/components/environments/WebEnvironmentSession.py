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
from selenium.webdriver.common.keys import Keys
import tempfile
import subprocess
import os
import os.path
import hashlib
import cv2
import numpy
from threading import Thread
import asyncio
import socket
from contextlib import closing



class WebEnvironmentSession(BaseEnvironment):
    """
        This class represents a single tab in the web environment.
    """
    def __init__(self, targetURL, tabNumber, proxyPort, pathTracer):
        self.targetURL = targetURL


        chrome_options = Options()
        chrome_options.headless = True
        # chrome_options.headless = False

        capabilities = webdriver.DesiredCapabilities.CHROME
        capabilities['loggingPrefs'] = {'browser': 'ALL'}
        proxyConfig = Proxy()
        proxyConfig.proxy_type = ProxyType.MANUAL
        proxyConfig.http_proxy = f"localhost:{proxyPort}"
        proxyConfig.add_to_capabilities(capabilities)

        self.driver = webdriver.Chrome(desired_capabilities=capabilities, chrome_options=chrome_options)

        window_size = self.driver.execute_script("""
            return [window.outerWidth - window.innerWidth + arguments[0],
              window.outerHeight - window.innerHeight + arguments[1]];
            """, 800, 600)
        self.driver.set_window_size(*window_size)

        self.driver.get(targetURL)

        # HACK! give time for page to load before proceeding.
        time.sleep(1)

        self.lastScreenshotHash = None
        self.lastProxyPaths = set()

        self.allUrls = set()

        self.tabNumber = tabNumber

        self.frameNumber = 0

        self.screenshotDirectory = tempfile.mkdtemp()
        self.screenshotPaths = []
        self.screenshotHashes = set()

        # TODO: NEED TO FIND A BETTER WAY FOR THE PATH TRACER TO SEPARATE OUT FLOWS FROM DIFFERENT TABS
        self.pathTracer = pathTracer

        self.lastBranchTrace = self.extractBranchTrace()

        self.allUrls.add(self.targetURL)
        self.allUrls.add(self.driver.current_url)

        screenHash = self.addScreenshot()
        self.frameNumber += 1
        self.screenshotHashes.add(screenHash)
        self.lastScreenshotHash = screenHash

        self.lastCumulativeBranchExecutionVector = self.computeCumulativeBranchExecutionVector(self.extractBranchTrace())
        self.decayingExecutionTrace = np.zeros_like(self.lastCumulativeBranchExecutionVector)


    def __del__(self):
        self.shutdown()

    def shutdown(self):
        # Cleanup the screenshot files
        for filePath in self.screenshotPaths:
            os.unlink(filePath)

        self.screenshotPaths = []
        if os.path.exists(self.movieFilePath()):
            os.unlink(self.movieFilePath())

        if os.path.exists(self.screenshotDirectory):
            os.rmdir(self.screenshotDirectory)

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
        subprocess.run(['ffmpeg', '-r', '60', '-f', 'image2', "-r", "3", '-i', 'kwola-screenshot-%05d.png', '-vcodec', 'libx264', '-crf', '15', '-pix_fmt', 'yuv420p', self.movieFileName()], cwd=self.screenshotDirectory)

        return self.movieFilePath()


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
                print("Waiting for ", action.time, "at", action.x, action.y)
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
        executionTrace.tabNumber = self.tabNumber
        executionTrace.frameNumber = self.frameNumber

        executionTrace.logOutput = "\n".join([str(log) for log in self.driver.get_log('browser')][startLogCount:])
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


    def getImage(self):
        image = cv2.imdecode(numpy.frombuffer(self.driver.get_screenshot_as_png(), numpy.uint8), -1)

        return image


    def getBranchFeature(self):
        return numpy.minimum(self.lastCumulativeBranchExecutionVector, numpy.ones_like(self.lastCumulativeBranchExecutionVector))


    def getExecutionTraceFeature(self):
        return self.decayingExecutionTrace
