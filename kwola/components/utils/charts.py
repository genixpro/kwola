import matplotlib.pyplot as plt
import billiard as multiprocessing
from ...datamodels.TestingStepModel import TestingStep
from ...datamodels.ExecutionSessionModel import ExecutionSession
from ...components.managers.TrainingManager import TrainingManager
from ...datamodels.ExecutionTraceModel import ExecutionTrace
from ...datamodels.TrainingStepModel import TrainingStep
from ...datamodels.BugModel import BugModel
from ...config.logger import getLogger, setupLocalLogging
from ...config.config import KwolaCoreConfiguration
import matplotlib
import numpy
import os
import scipy.signal
import tempfile
matplotlib.use("Agg")

def averageRewardForTestingStep(config, testingStepId):
    testingStep = TestingStep.loadFromDisk(testingStepId, config)

    stepRewards = []
    for sessionId in testingStep.executionSessions:
        session = ExecutionSession.loadFromDisk(sessionId, config)
        if session.status == "completed":
            stepRewards.append(session.totalReward)

    if len(stepRewards) > 0:
        return numpy.mean(stepRewards)
    else:
        return None


def generateRewardChart(config, applicationId):
    getLogger().info(f"Generating the reward chart")

    config = KwolaCoreConfiguration(config)

    testingSteps = sorted(
        [step for step in TrainingManager.loadAllTestingSteps(config, applicationId=applicationId) if step.status == "completed"],
        key=lambda step: step.startTime, reverse=False)

    rewardValueFutures = []

    pool = multiprocessing.Pool(config['chart_generation_dataload_workers'])

    for step in testingSteps:
        rewardValueFutures.append(pool.apply_async(averageRewardForTestingStep, [config, step.id]))

    rewardValues = [future.get() for future in rewardValueFutures if future.get() is not None]

    fig, ax = plt.subplots()

    rewardValues = scipy.signal.medfilt(rewardValues, kernel_size=9)

    ax.plot(range(len(rewardValues)), rewardValues, color='green')

    ax.set_ylim(0, 15)

    ax.set(xlabel='Testing Step #', ylabel='Reward',
           title='Reward per session')
    ax.grid()

    _, localFilePath = tempfile.mkstemp(suffix=".png")
    fig.savefig(localFilePath)
    with open(localFilePath, 'rb') as f:
        config.saveKwolaFileData("charts", "reward_chart.png", f.read())
    os.unlink(localFilePath)


    pool.close()
    pool.join()


def generateCoverageChart(config, applicationId):
    getLogger().info(f"Generating the coverage chart")

    config = KwolaCoreConfiguration(config)

    testingSteps = sorted(
        [step for step in TrainingManager.loadAllTestingSteps(config, applicationId=applicationId) if step.status == "completed"],
        key=lambda step: step.startTime, reverse=False)

    coverageValueFutures = []

    pool = multiprocessing.Pool(config['chart_generation_dataload_workers'])

    for step in testingSteps:
        coverageValueFutures.append(pool.apply_async(computeCumulativeCoverageForTestingSteps, [[step.id], config]))

    coverageValues = [future.get()[0] for future in coverageValueFutures]
    executedLinesValues = [future.get()[1] for future in coverageValueFutures]
    totalLinesValues = [future.get()[2] for future in coverageValueFutures]

    coverageValues = scipy.signal.medfilt(coverageValues, kernel_size=9)
    executedLinesValues = scipy.signal.medfilt(executedLinesValues, kernel_size=9)
    totalLinesValues = scipy.signal.medfilt(totalLinesValues, kernel_size=9)

    fig, ax = plt.subplots()
    ax.plot(range(len(coverageValues)), coverageValues, color='green')
    ax.set(xlabel='Testing Step #', ylabel='Coverage',
           title='Code Coverage')
    ax.grid()

    _, localFilePath = tempfile.mkstemp(suffix=".png")
    fig.savefig(localFilePath)
    with open(localFilePath, 'rb') as f:
        config.saveKwolaFileData("charts", "coverage_chart.png", f.read())
    os.unlink(localFilePath)

    fig, ax = plt.subplots()
    ax.plot(range(len(executedLinesValues)), executedLinesValues, color='green')
    ax2 = ax.twinx()
    ax2.plot(range(len(totalLinesValues)), totalLinesValues, color='red')
    ax.set(xlabel='Testing Step #', ylabel='Lines Executed (green)',
           title='Lines Available / Lines Triggered')
    ax2.set(ylabel="Lines Available (red)")
    ax.grid()
    ax2.grid()

    _, localFilePath = tempfile.mkstemp(suffix=".png")
    fig.savefig(localFilePath)
    with open(localFilePath, 'rb') as f:
        config.saveKwolaFileData("charts", "lines_triggered.png", f.read())
    os.unlink(localFilePath)

    pool.close()
    pool.join()

def findAllTrainingStepIds(config, applicationId=None):
    trainStepsDir = config.getKwolaUserDataDirectory("training_steps")

    if config['data_serialization_method']['default'] == 'mongo':
        return [step.id for step in TrainingStep.objects(applicationId=applicationId).no_dereference().only("id")]
    else:
        trainingStepIds = []

        for fileName in os.listdir(trainStepsDir):
            if ".lock" not in fileName:
                stepId = fileName
                stepId = stepId.replace(".json", "")
                stepId = stepId.replace(".gz", "")
                stepId = stepId.replace(".pickle", "")

                trainingStepIds.append(stepId)

        return trainingStepIds

def loadTrainingStepLossData(config, trainingStepId, attribute):
    step = TrainingStep.loadFromDisk(trainingStepId, config)
    losses = getattr(step, attribute)
    if len(losses) > 0:
        return numpy.mean(losses), step.startTime, step.status
    else:
        return 0, step.startTime, step.status

def generateLossChart(config, applicationId, attribute, title, fileName):
    getLogger().info(f"Generating the loss chart for {attribute}")

    config = KwolaCoreConfiguration(config)

    trainingStepIds = findAllTrainingStepIds(config, applicationId=applicationId)

    pool = multiprocessing.Pool(config['chart_generation_dataload_workers'])

    lossValueFutures = []
    for id in trainingStepIds:
        lossValueFutures.append(pool.apply_async(loadTrainingStepLossData, [config, id, attribute]))

    lossValuesSorted = sorted(
        [future.get() for future in lossValueFutures if future.get()[2] == "completed"],
        key=lambda result: result[1], reverse=False)

    lossValues = [result[0] for result in lossValuesSorted]

    if len(lossValues) == 0:
        return

    fig, ax = plt.subplots()

    lossValues = scipy.signal.medfilt(lossValues, kernel_size=9)

    ax.plot(range(len(lossValues)), lossValues, color='green')

    ax.set_ylim(0, numpy.percentile(lossValues, 99))

    ax.set(xlabel='Training Step #', ylabel='Reward', title=title)
    ax.grid()

    _, localFilePath = tempfile.mkstemp(suffix=".png")
    fig.savefig(localFilePath)
    with open(localFilePath, 'rb') as f:
        config.saveKwolaFileData("charts", fileName, f.read())
    os.unlink(localFilePath)

    pool.close()
    pool.join()

def computeCumulativeBranchTraceForTestingSteps(testingStepId, config):
    testingStep = TestingStep.loadFromDisk(testingStepId, config)

    cumulativeBranchTrace = {}
    for sessionId in testingStep.executionSessions:
        session = ExecutionSession.loadFromDisk(sessionId, config)
        if session.status == "completed":
            for traceId in session.executionTraces:
                trace = ExecutionTrace.loadFromDisk(traceId, config)
                for fileName in trace.branchTrace:
                    if fileName not in cumulativeBranchTrace:
                        cumulativeBranchTrace[fileName] = trace.branchTrace[fileName]
                    else:
                        cumulativeBranchTrace[fileName] = trace.branchTrace[fileName].maximum(cumulativeBranchTrace[fileName])

    return cumulativeBranchTrace

def computeCumulativeCoverageForTestingSteps(testingStepIds, config):
    futures = []

    pool = multiprocessing.Pool(config['chart_generation_dataload_workers'])

    for stepId in testingStepIds:
        futures.append(pool.apply_async(computeCumulativeBranchTraceForTestingSteps, [stepId, config]))

    cumulativeBranchTrace = {}

    for future in futures:
        branchTrace = future.get()
        for fileName in branchTrace:
            if fileName not in cumulativeBranchTrace:
                cumulativeBranchTrace[fileName] = branchTrace[fileName]
            else:
                cumulativeBranchTrace[fileName] = cumulativeBranchTrace[fileName].maximum(branchTrace[fileName])

    total = 0
    executedAtleastOnce = 0
    for fileName in cumulativeBranchTrace:
        total += cumulativeBranchTrace[fileName].shape[0]
        executedAtleastOnce += len(numpy.nonzero(cumulativeBranchTrace[fileName])[0])

    # Just an extra check here to cover our ass in case of division by zero
    if total == 0:
        total += 1

    pool.close()
    pool.join()

    return float(executedAtleastOnce) / float(total), executedAtleastOnce, total


def generateCumulativeCoverageChart(config, applicationId=None, numberOfTestingStepsPerValue=100):
    getLogger().info(f"Generating the cumulative coverage chart using {numberOfTestingStepsPerValue} testing steps per value")

    config = KwolaCoreConfiguration(config)

    testingSteps = sorted(
        [step for step in TrainingManager.loadAllTestingSteps(config, applicationId=applicationId) if step.status == "completed"],
        key=lambda step: step.startTime, reverse=False)

    cumulativeLinesExecutedValues = []
    cumulativeTotalLinesValues = []
    cumulativeCoverageValues = []
    for n in range(int(len(testingSteps) / numberOfTestingStepsPerValue) + 1):
        testingStepsForValue = testingSteps[n * numberOfTestingStepsPerValue:(n+1)*numberOfTestingStepsPerValue]

        coverage, linesExecuted, totalLines = computeCumulativeCoverageForTestingSteps([step.id for step in testingStepsForValue], config)

        cumulativeCoverageValues.append(coverage)
        cumulativeLinesExecutedValues.append(linesExecuted)
        cumulativeTotalLinesValues.append(totalLines)

    fig, ax = plt.subplots()
    ax.plot(numpy.array(range(len(cumulativeLinesExecutedValues))) * numberOfTestingStepsPerValue, cumulativeCoverageValues, color='green')
    ax.set(xlabel='Testing Steps Completed (x1000)', ylabel='Cumulative Coverage', title=f"Cumulative Coverage Chart, Group Size: {numberOfTestingStepsPerValue}")
    ax.grid()

    _, localFilePath = tempfile.mkstemp(suffix=".png")
    fig.savefig(localFilePath)
    with open(localFilePath, 'rb') as f:
        config.saveKwolaFileData("charts", f"cumulative_coverage_chart_groupsize_{numberOfTestingStepsPerValue}.png", f.read())
    os.unlink(localFilePath)


    fig, ax = plt.subplots()
    ax.plot(numpy.array(range(len(cumulativeLinesExecutedValues))) * numberOfTestingStepsPerValue, cumulativeLinesExecutedValues, color='green')
    ax.set_ylim(0, 600)
    ax.set(xlabel='Testing Steps Completed', ylabel='Cumulative Total Lines Triggered (green)', title=f"Cumulative Lines Triggered Chart, Group Size: {numberOfTestingStepsPerValue}")
    # ax2 = ax.twinx()
    # ax2.plot(numpy.array(range(len(cumulativeTotalLinesValues))) * numberOfTestingStepsPerValue, cumulativeTotalLinesValues, color='red')
    # ax2.set(ylabel="Cumulative Lines Available (red)")
    ax.grid()
    # ax2.grid()

    _, localFilePath = tempfile.mkstemp(suffix=".png")
    fig.savefig(localFilePath)
    with open(localFilePath, 'rb') as f:
        config.saveKwolaFileData("charts", f"cumulative_lines_triggered_groupsize_{numberOfTestingStepsPerValue}.png", f.read())
    os.unlink(localFilePath)

    getLogger().info(f"Best Cumulative Coverage: {numpy.max(cumulativeLinesExecutedValues)} / {numpy.max(cumulativeTotalLinesValues)} = {numpy.max(cumulativeCoverageValues)}")

def loadAllBugs(config, applicationId=None):
    if config['data_serialization_method']['default'] == 'mongo':
        return [bug for bug in BugModel.objects(applicationId=applicationId).no_dereference()]
    else:
        bugsDir = config.getKwolaUserDataDirectory("bugs")

        bugIds = set()
        bugs = []

        for fileName in os.listdir(bugsDir):
            if ".lock" not in fileName and ".txt" not in fileName and ".mp4" not in fileName:
                bugId = fileName
                bugId = bugId.replace(".json", "")
                bugId = bugId.replace(".gz", "")
                bugId = bugId.replace(".pickle", "")

                if bugId not in bugIds:
                    bugIds.add(bugId)

                    bug = BugModel.loadFromDisk(bugId, config)

                    if bug is not None:
                        bugs.append(bug)

        return bugs

def generateCumulativeErrorsFoundChart(config, applicationId):
    getLogger().info(f"Generating the cumulative errors chart")

    config = KwolaCoreConfiguration(config)

    testingSteps = sorted(
        [step for step in TrainingManager.loadAllTestingSteps(config, applicationId=applicationId) if step.status == "completed"],
        key=lambda step: step.startTime, reverse=False)

    bugsByTestingStepId = {
        step.id: 0
        for step in testingSteps
    }

    for bug in loadAllBugs(config, applicationId):
        if bug.testingStepId in bugsByTestingStepId:
            bugsByTestingStepId[bug.testingStepId] += 1

    cumulativeErrorsFound = []

    pool = multiprocessing.Pool(config['chart_generation_dataload_workers'])

    currentTotal = 0
    for step in testingSteps:
        currentTotal += bugsByTestingStepId[step.id]
        cumulativeErrorsFound.append(currentTotal)

    fig, ax = plt.subplots()

    ax.plot(range(len(cumulativeErrorsFound)), cumulativeErrorsFound, color='green')

    ax.set(xlabel='Testing Step #', ylabel='Total Errors Found', title='Cumulative Errors Found')
    ax.grid()


    _, localFilePath = tempfile.mkstemp(suffix=".png")
    fig.savefig(localFilePath)
    with open(localFilePath, 'rb') as f:
        config.saveKwolaFileData("charts", "errors_found.png", f.read())

    os.unlink(localFilePath)

    pool.close()
    pool.join()


def generateAllCharts(config, applicationId=None, enableCumulativeCoverage=False):
    getLogger().info(f"Generating charts based on results.")

    pool = multiprocessing.Pool(config['chart_generation_workers'], initializer=setupLocalLogging)

    futures = []

    if config['chart_enable_cumulative_coverage_chart'] and enableCumulativeCoverage:
        futures.append(pool.apply_async(generateCumulativeCoverageChart, [config.serialize(), applicationId, 100]))
        futures.append(pool.apply_async(generateCumulativeCoverageChart, [config.serialize(), applicationId, 25]))
        futures.append(pool.apply_async(generateCumulativeCoverageChart, [config.serialize(), applicationId, 10]))
        futures.append(pool.apply_async(generateCumulativeCoverageChart, [config.serialize(), applicationId, 5]))

    futures.append(pool.apply_async(generateRewardChart, [config.serialize(), applicationId]))
    futures.append(pool.apply_async(generateCoverageChart, [config.serialize(), applicationId]))

    if config['chart_enable_cumulative_errors_chart']:
        futures.append(pool.apply_async(generateCumulativeErrorsFoundChart, [config.serialize(), applicationId]))

    futures.append(pool.apply_async(generateLossChart, [config.serialize(), applicationId, 'totalLosses', "Total Loss", 'total_loss_chart.png']))
    futures.append(pool.apply_async(generateLossChart, [config.serialize(), applicationId, 'presentRewardLosses', "Present Reward Loss", 'present_reward_loss_chart.png']))
    futures.append(pool.apply_async(generateLossChart, [config.serialize(), applicationId, 'discountedFutureRewardLosses', "Discounted Future Reward Loss", 'discounted_future_reward_loss_chart.png']))
    futures.append(pool.apply_async(generateLossChart, [config.serialize(), applicationId, 'stateValueLosses', "State Value Loss", 'state_value_loss_chart.png']))
    futures.append(pool.apply_async(generateLossChart, [config.serialize(), applicationId, 'advantageLosses', "Advantage Loss", 'advantage_loss_chart.png']))
    futures.append(pool.apply_async(generateLossChart, [config.serialize(), applicationId, 'actionProbabilityLosses', "Action Probability Loss", 'action_probability_loss_chart.png']))

    for future in futures:
        future.get()

    pool.close()
    pool.join()
    getLogger().info(f"Completed generating all the charts.")
