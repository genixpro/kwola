from kwola.components.plugins.base.WebEnvironmentPluginBase import WebEnvironmentPluginBase
import tempfile
import os
import hashlib
import subprocess
from kwola.config.logger import getLogger
from ...utils.video import chooseBestFfmpegVideoCodec
from ...utils.retry import autoretry

class RecordScreenshots(WebEnvironmentPluginBase):
    def __init__(self, config):
        self.config = config
        self.lastScreenshotHash = {}
        self.frameNumber = {}
        self.screenshotDirectory = {}
        self.screenshotPaths = {}
        self.screenshotHashes = {}

    def browserSessionStarted(self, webDriver, proxy, executionSession):
        self.lastScreenshotHash[executionSession.id] = None

        self.frameNumber[executionSession.id] = 0

        self.screenshotDirectory[executionSession.id] = tempfile.mkdtemp()
        self.screenshotPaths[executionSession.id] = []
        self.screenshotHashes[executionSession.id] = set()

        screenHash = self.addScreenshot(webDriver, executionSession)
        self.frameNumber[executionSession.id] = 1
        self.screenshotHashes[executionSession.id].add(screenHash)
        self.lastScreenshotHash[executionSession.id] = screenHash


    def beforeActionRuns(self, webDriver, proxy, executionSession, executionTrace, actionToExecute):
        pass


    def afterActionRuns(self, webDriver, proxy, executionSession, executionTrace, actionExecuted):
        screenHash = self.addScreenshot(webDriver, executionSession)

        executionTrace.startScreenshotHash = self.lastScreenshotHash[executionSession.id]
        executionTrace.finishScreenshotHash = screenHash

        executionTrace.frameNumber = self.frameNumber[executionSession.id]

        executionTrace.didScreenshotChange = screenHash != self.lastScreenshotHash[executionSession.id]
        executionTrace.isScreenshotNew = screenHash not in self.screenshotHashes[executionSession.id]

        self.lastScreenshotHash[executionSession.id] = screenHash
        self.screenshotHashes[executionSession.id].add(screenHash)

        self.frameNumber[executionSession.id] += 1

    @autoretry()
    def browserSessionFinished(self, webDriver, proxy, executionSession):
        getLogger().info(f"Creating movie file for the execution session {executionSession.id}")
        self.encodeAndSaveVideo(executionSession, "videos", False)
        self.encodeAndSaveVideo(executionSession, "videos_lossless", True)


    def encodeAndSaveVideo(self, executionSession, folder, lossless):
        codec = chooseBestFfmpegVideoCodec(losslessPreferred=lossless)
        crfRating = 25
        if lossless:
            crfRating = 0
        result = subprocess.run(['ffmpeg', '-f', 'image2', "-r", "1", '-i', 'kwola-screenshot-%05d.png', '-vcodec',
                                 codec, '-crf', str(crfRating), '-preset',
                                 'veryslow', self.movieFileName(executionSession)],
                                cwd=self.screenshotDirectory[executionSession.id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            errorMsg = f"Error! Attempted to create a movie using ffmpeg and the process exited with exit-code {result.returncode}. The following output was observed:\n"
            errorMsg += str(result.stdout, 'utf8') + "\n"
            errorMsg += str(result.stderr, 'utf8') + "\n"
            getLogger().error(errorMsg)
            raise RuntimeError(errorMsg)
        else:
            localVideoPath = self.movieFilePath(executionSession)

            with open(localVideoPath, 'rb') as f:
                videoData = f.read()

            fileName = f'{str(executionSession.id)}.mp4'
            self.config.saveKwolaFileData(folder, fileName, videoData)

            os.unlink(localVideoPath)

    @autoretry()
    def addScreenshot(self, webDriver, executionSession):
        fileName = f"kwola-screenshot-{self.frameNumber[executionSession.id]:05d}.png"

        filePath = os.path.join(self.screenshotDirectory[executionSession.id], fileName)

        webDriver.save_screenshot(filePath)

        hasher = hashlib.sha256()
        with open(filePath, 'rb') as imageFile:
            buf = imageFile.read()
            hasher.update(buf)

        screenshotHash = hasher.hexdigest()

        self.screenshotPaths[executionSession.id].append(filePath)

        return screenshotHash

    def cleanup(self, webDriver, proxy, executionSession):
        if executionSession.id not in self.screenshotPaths:
            return

        # Cleanup the screenshot files
        for filePath in self.screenshotPaths[executionSession.id]:
            if os.path.exists(filePath):
                os.unlink(filePath)

        self.screenshotPaths[executionSession.id] = {}
        if os.path.exists(self.movieFilePath(executionSession)):
            os.unlink(self.movieFilePath(executionSession))

        if os.path.exists(self.screenshotDirectory[executionSession.id]):
            try:
                os.rmdir(self.screenshotDirectory[executionSession.id])
            except OSError:
                pass # Sometimes get "Directory not empty" here, not sure why,
                     # since above commands should have removed all data in the directory.
                     # Need to investigate more at somepoint

        del self.screenshotDirectory[executionSession.id]
        del self.screenshotPaths[executionSession.id]
        del self.screenshotHashes[executionSession.id]
        del self.frameNumber[executionSession.id]
        del self.lastScreenshotHash[executionSession.id]


    def movieFilePath(self, executionSession):
        return os.path.join(self.screenshotDirectory[executionSession.id], self.movieFileName(executionSession))


    def movieFileName(self, executionSession):
        return f"kwola-video-{executionSession.tabNumber}.mp4"
