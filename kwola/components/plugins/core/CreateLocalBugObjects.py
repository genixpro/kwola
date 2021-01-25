from kwola.config.logger import getLogger
from kwola.datamodels.BugModel import BugModel
from kwola.datamodels.CustomIDField import CustomIDField
from ...utils.debug_video import createDebugVideoSubProcess
from ..base.TestingStepPluginBase import TestingStepPluginBase
from kwola.components.utils.retry import autoretry
from datetime import datetime
import atexit
import concurrent.futures
import billiard as multiprocessing
import os
import numpy
import billiard.exceptions
from kwola.components.utils.deunique import deuniqueString




class CreateLocalBugObjects(TestingStepPluginBase):
    """
        This plugin creates bug objects for all of the errors discovered during this testing step
    """
    def __init__(self, config):
        self.config = config

        self.allKnownErrorHashes = {}
        self.newErrorsThisTestingStep = {}
        self.newErrorOriginalExecutionSessionIds = {}
        self.newErrorOriginalStepNumbers = {}
        self.executionSessionTraces = {}


    def testingStepStarted(self, testingStep, executionSessions):
        self.allKnownErrorHashes[testingStep.id] = set()
        self.newErrorsThisTestingStep[testingStep.id] = []
        self.newErrorOriginalExecutionSessionIds[testingStep.id] = []
        self.newErrorOriginalStepNumbers[testingStep.id] = []

        self.loadKnownErrorHashes(testingStep)

        for session in executionSessions:
            self.executionSessionTraces[session.id] = []


    def beforeActionsRun(self, testingStep, executionSessions, actions):
        pass


    def afterActionsRun(self, testingStep, executionSessions, traces):
        for sessionN, executionSession, trace in zip(range(len(traces)), executionSessions, traces):
            if trace is None:
                continue

            for error in trace.errorsDetected:
                hash = error.computeHash()

                if hash not in self.allKnownErrorHashes:
                    self.allKnownErrorHashes[testingStep.id].add(hash)
                    self.newErrorsThisTestingStep[testingStep.id].append(error)
                    self.newErrorOriginalExecutionSessionIds[testingStep.id].append(str(executionSession.id))
                    self.newErrorOriginalStepNumbers[testingStep.id].append(trace.traceNumber)

            self.executionSessionTraces[executionSession.id].append(trace)

    def testingStepFinished(self, testingStep, executionSessions):
        existingBugs = self.loadAllBugs(testingStep)

        bugObjects = []

        executionSessionsById = {}
        for session in executionSessions:
            executionSessionsById[session.id] = session

        for errorIndex, error, executionSessionId, stepNumber in zip(range(len(self.newErrorsThisTestingStep[testingStep.id])),
                                                                     self.newErrorsThisTestingStep[testingStep.id],
                                                                     self.newErrorOriginalExecutionSessionIds[testingStep.id],
                                                                     self.newErrorOriginalStepNumbers[testingStep.id]):
            if error.type == "http":
                if error.statusCode == 400 and not self.config['enable_400_error']:
                    continue # Skip this error
                if error.statusCode == 401 and not self.config['enable_401_error']:
                    continue # Skip this error
                if error.statusCode == 403 and not self.config['enable_403_error']:
                    continue # Skip this error
                if error.statusCode == 404 and not self.config['enable_404_error']:
                    continue # Skip this error
                if error.statusCode >= 500 and not self.config['enable_5xx_error']:
                    continue # Skip this error
            elif error.type == "log":
                if not self.config['enable_javascript_console_error']:
                    continue # Skip this error
            elif error.type == "exception":
                if not self.config['enable_unhandled_exception_error']:
                    continue # Skip this error

            bug = BugModel()
            bug.owner = testingStep.owner
            bug.applicationId = testingStep.applicationId
            bug.testingStepId = testingStep.id
            bug.executionSessionId = executionSessionId
            bug.creationDate = datetime.now()
            bug.stepNumber = stepNumber
            bug.error = error
            bug.testingRunId = testingStep.testingRunId
            bug.actionsPerformed = [
                trace.actionPerformed for trace in self.executionSessionTraces[executionSessionId]
            ][:(bug.stepNumber + 2)]
            bug.browser = executionSessionsById[executionSessionId].browser
            bug.userAgent = executionSessionsById[executionSessionId].userAgent
            bug.windowSize = executionSessionsById[executionSessionId].windowSize
            bug.recomputeCanonicalPageUrl()
            tracesForScore = [
                trace for trace in self.executionSessionTraces[executionSessionId][max(0, stepNumber-5):(stepNumber + 1)]
                if trace.codePrevalenceScore is not None
            ]
            if len(tracesForScore) > 0:
                bug.codePrevalenceScore = numpy.mean([trace.codePrevalenceScore for trace in tracesForScore])
            else:
                bug.codePrevalenceScore = None
            bug.isBugNew = True
            bug.recomputeBugQualitativeFeatures()

            duplicate = False
            for existingBug in existingBugs:
                if bug.isDuplicateOf(existingBug):
                    duplicate = True
                    break

            if not duplicate:
                bug.id = CustomIDField.generateNewUUID(BugModel, self.config)
                bug.saveToDisk(self.config, overrideSaveFormat="json", overrideCompression=0)
                bug.saveToDisk(self.config)

                bugTextFile = bug.id + ".txt"

                self.config.saveKwolaFileData("bugs", bugTextFile, bytes(bug.generateBugText(), "utf8"))

                bugVideoFileName = bug.id + ".mp4"
                origVideoFileName = f'{str(executionSessionId)}.mp4'
                origVideoFileData = self.config.loadKwolaFileData("videos", origVideoFileName)

                self.config.saveKwolaFileData("bugs", bugVideoFileName, origVideoFileData)

                existingBugs.append(bug)
                bugObjects.append(bug)

                getLogger().info(f"\n\nBug #{len(bugObjects)}:\n{bug.generateBugText()}\n")

        getLogger().info(f"Found {len(self.newErrorsThisTestingStep[testingStep.id])} new unique errors this session.")

        testingStep.bugsFound = len(self.newErrorsThisTestingStep[testingStep.id])
        testingStep.errors = self.newErrorsThisTestingStep[testingStep.id]

        self.generateVideoFilesForBugs(testingStep, bugObjects)

    def sessionFailed(self, testingStep, executionSession):
        n = 0
        while n < len(self.newErrorsThisTestingStep[testingStep.id]):
            if self.newErrorOriginalExecutionSessionIds[testingStep.id][n] == executionSession.id:
                del self.newErrorsThisTestingStep[testingStep.id][n]
                del self.newErrorOriginalStepNumbers[testingStep.id][n]
                del self.newErrorOriginalExecutionSessionIds[testingStep.id][n]
            else:
                n += 1

        del self.executionSessionTraces[executionSession.id]

    def loadKnownErrorHashes(self, testingStep):
        for bug in self.loadAllBugs(testingStep):
            hash = bug.error.computeHash()
            self.allKnownErrorHashes[testingStep.id].add(hash)

    def loadAllBugs(self, testingStep):
        bugsDir = self.config.getKwolaUserDataDirectory("bugs")

        bugs = []
        bugIds = set()

        for fileName in os.listdir(bugsDir):
            if ".lock" not in fileName and ".txt" not in fileName and ".mp4" not in fileName:
                bugId = fileName
                bugId = bugId.replace(".json", "")
                bugId = bugId.replace(".gz", "")
                bugId = bugId.replace(".pickle", "")

                if bugId not in bugIds:
                    bugIds.add(bugId)

                    bug = BugModel.loadFromDisk(bugId, self.config)

                    if bug is not None:
                        bugs.append(bug)

        return bugs

    @autoretry(exponentialBackOffBase=3)
    def generateVideoFilesForBugs(self, testingStep, bugObjects):
        pool = multiprocessing.Pool(self.config['video_generation_processes'], maxtasksperchild=1)
        futures = []
        for bugIndex, bug in enumerate(bugObjects):
            future = pool.apply_async(func=createDebugVideoSubProcess, args=(
                self.config.serialize(), str(bug.executionSessionId), f"{bug.id}_bug", False, False, bug.stepNumber,
                bug.stepNumber + 3, "bugs"))
            futures.append((bugIndex, bug, future))

        for bugIndex, bug, future in futures:
            localFuture = future
            # for retry in range(5):
            # try:
            value = localFuture.get(timeout=self.config['debug_video_generation_timeout'])
            if value:
                getLogger().error(value)
            # break
            # except billiard.exceptions.WorkerLostError:
            #     if retry == 4:
            #         raise
            #     localFuture = pool.apply_async(func=createDebugVideoSubProcess, args=(
            #         self.config.serialize(), str(bug.executionSessionId), f"{bug.id}_bug", False, False, bug.stepNumber,
            #         bug.stepNumber + 3, "bugs"))
            # except BrokenPipeError:
            #     if retry == 4:
            #         raise
            #     localFuture = pool.apply_async(func=createDebugVideoSubProcess, args=(
            #         self.config.serialize(), str(bug.executionSessionId), f"{bug.id}_bug", False, False, bug.stepNumber,
            #         bug.stepNumber + 3, "bugs"))

        pool.close()
        pool.join()



