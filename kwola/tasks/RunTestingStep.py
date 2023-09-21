#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#


from ..tasks.TaskProcess import TaskProcess
from ..components.managers.TestingStepManager import TestingStepManager


def runTestingStep(config, testingStepId, shouldBeRandom=False, generateDebugVideo=False, plugins=None, browser=None, windowSize=None):
    manager = TestingStepManager(config, testingStepId, shouldBeRandom, generateDebugVideo, plugins, browser=browser, windowSize=windowSize)
    return manager.runTesting()


if __name__ == "__main__":
    task = TaskProcess(runTestingStep)
    task.run()
