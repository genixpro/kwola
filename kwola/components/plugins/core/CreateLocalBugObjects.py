from kwola.config.logger import getLogger
from kwola.datamodels.BugModel import BugModel
from kwola.datamodels.CustomIDField import CustomIDField
from ...utils.debug_video import createDebugVideoSubProcess
from ..base.TestingStepPluginBase import TestingStepPluginBase
from datetime import datetime
import atexit
import concurrent.futures
import multiprocessing
import os




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


    def testingStepStarted(self, testingStep, executionSessions):
        self.allKnownErrorHashes[testingStep.id] = set()
        self.newErrorsThisTestingStep[testingStep.id] = []
        self.newErrorOriginalExecutionSessionIds[testingStep.id] = []
        self.newErrorOriginalStepNumbers[testingStep.id] = []

        self.loadKnownErrorHashes(testingStep)


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

    def testingStepFinished(self, testingStep, executionSessions):
        kwolaVideoDirectory = self.config.getKwolaUserDataDirectory("videos")

        existingBugs = self.loadAllBugs(testingStep)

        bugObjects = []

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
            bug.id = CustomIDField.generateNewUUID(BugModel, self.config)
            bug.owner = testingStep.owner
            bug.applicationId = testingStep.applicationId
            bug.testingStepId = testingStep.id
            bug.executionSessionId = executionSessionId
            bug.creationDate = datetime.now()
            bug.stepNumber = stepNumber
            bug.error = error
            bug.testingRunId = testingStep.testingRunId

            duplicate = False
            for existingBug in existingBugs:
                if bug.isDuplicateOf(existingBug):
                    duplicate = True
                    break

            if not duplicate:
                bug.saveToDisk(self.config, overrideSaveFormat="json", overrideCompression=0)
                bug.saveToDisk(self.config)

                bugTextFile = os.path.join(self.config.getKwolaUserDataDirectory("bugs"), bug.id + ".txt")
                with open(bugTextFile, "wb") as file:
                    file.write(bytes(bug.generateBugText(), "utf8"))

                bugVideoFilePath = os.path.join(self.config.getKwolaUserDataDirectory("bugs"), bug.id + ".mp4")
                with open(os.path.join(kwolaVideoDirectory, f'{str(executionSessionId)}.mp4'), "rb") as origFile:
                    with open(bugVideoFilePath, 'wb') as cloneFile:
                        cloneFile.write(origFile.read())

                getLogger().info(f"\n\n[{os.getpid()}] Bug #{errorIndex + 1}:\n{bug.generateBugText()}\n")

                existingBugs.append(bug)
                bugObjects.append(bug)

                getLogger().info(f"\n\n[{os.getpid()}] Bug #{errorIndex + 1}:\n{bug.generateBugText()}\n")

        getLogger().info(f"[{os.getpid()}] Found {len(self.newErrorsThisTestingStep[testingStep.id])} new unique errors this session.")

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


    def loadKnownErrorHashes(self, testingStep):
        for bug in self.loadAllBugs(testingStep):
            hash = bug.error.computeHash()
            self.allKnownErrorHashes[testingStep.id].add(hash)

    def loadAllBugs(self, testingStep):
        bugsDir = self.config.getKwolaUserDataDirectory("bugs")

        bugs = []

        for fileName in os.listdir(bugsDir):
            if ".lock" not in fileName and ".txt" not in fileName and ".mp4" not in fileName:
                bugId = fileName
                bugId = bugId.replace(".json", "")
                bugId = bugId.replace(".gz", "")
                bugId = bugId.replace(".pickle", "")

                bug = BugModel.loadFromDisk(bugId, self.config)

                if bug is not None:
                    bugs.append(bug)

        return bugs

    def generateVideoFilesForBugs(self, testingStep, bugObjects):
        debugVideoSubprocesses = []

        for bugIndex, bug in enumerate(bugObjects):
            debugVideoSubprocess = multiprocessing.Process(target=createDebugVideoSubProcess, args=(
                self.config.configurationDirectory, str(bug.executionSessionId), f"{bug.id}_bug", False, False, bug.stepNumber,
                bug.stepNumber + 3, "bugs"))
            atexit.register(lambda: debugVideoSubprocess.terminate())
            debugVideoSubprocesses.append(debugVideoSubprocess)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config['video_generation_processes']) as executor:
            futures = []
            for debugVideoSubprocess in debugVideoSubprocesses:
                futures.append(executor.submit(CreateLocalBugObjects.runAndJoinSubprocess, debugVideoSubprocess))
            for future in futures:
                future.result()

    @staticmethod
    def runAndJoinSubprocess(debugVideoSubprocess):
        debugVideoSubprocess.start()
        debugVideoSubprocess.join()

