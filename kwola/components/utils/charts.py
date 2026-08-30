import matplotlib.pyplot as plt
from ...datamodels.TestingStepModel import TestingStep
from ...datamodels.ExecutionSessionModel import ExecutionSession
from ...components.managers.TrainingManager import TrainingManager
from ...datamodels.ExecutionTraceModel import ExecutionTrace
from ...datamodels.TrainingStepModel import TrainingStep
from ...datamodels.BugModel import BugModel
from ...config.logger import getLogger
from ...config.config import KwolaCoreConfiguration
import matplotlib
import numpy
import os
import scipy.signal
import tempfile
matplotlib.use("Agg")


def medianSmooth(values, maximumKernelSize=9):
    """Apply a median filter only when the sample set is large enough.

    SciPy intentionally zero-pads an oversized median kernel.  Besides making
    one-step experiment charts misleading, that path crashes the current
    manylinux NumPy/Matplotlib combination during Agg rendering.  A one-point
    filter has the same no-op semantics without entering that unsafe path.
    """
    kernelSize = min(maximumKernelSize, len(values))
    if kernelSize % 2 == 0:
        kernelSize -= 1
    if kernelSize < 3:
        return numpy.asarray(values)
    return scipy.signal.medfilt(values, kernel_size=kernelSize)


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

    rewardValues = [averageRewardForTestingStep(config, step.id) for step in testingSteps]
    rewardValues = [value for value in rewardValues if value is not None]

    fig, ax = plt.subplots()

    rewardValues = medianSmooth(rewardValues)

    ax.plot(range(len(rewardValues)), rewardValues, color='green')

    ax.set_ylim(0, 25)

    ax.set(xlabel='Testing Step #', ylabel='Reward',
           title='Reward per session')
    ax.grid()

    _, localFilePath = tempfile.mkstemp(suffix=".png")
    fig.savefig(localFilePath)
    with open(localFilePath, 'rb') as f:
        config.saveKwolaFileData("charts", "reward_chart.png", f.read())
    os.unlink(localFilePath)
    plt.close(fig)

def averageFitnessForTestingStep(config, testingStepId):
    testingStep = TestingStep.loadFromDisk(testingStepId, config)

    stepFitnessValues = []
    for sessionId in testingStep.executionSessions:
        session = ExecutionSession.loadFromDisk(sessionId, config)
        if session.status == "completed" and session.bestApplicationProvidedCumulativeFitness is not None:
            stepFitnessValues.append(session.bestApplicationProvidedCumulativeFitness)

    if len(stepFitnessValues) > 0:
        return numpy.mean(stepFitnessValues)
    else:
        return None


def generateFitnessChart(config, applicationId):
    getLogger().info(f"Generating the fitness chart")

    config = KwolaCoreConfiguration(config)

    testingSteps = sorted(
        [step for step in TrainingManager.loadAllTestingSteps(config, applicationId=applicationId) if step.status == "completed"],
        key=lambda step: step.startTime, reverse=False)

    fitnessValues = [averageFitnessForTestingStep(config, step.id) for step in testingSteps]
    fitnessValues = [value for value in fitnessValues if value is not None]

    if len(fitnessValues) > 0:
        bestFitness = numpy.max(fitnessValues)

        fig, ax = plt.subplots()

        fitnessValues = medianSmooth(fitnessValues)

        ax.plot(range(len(fitnessValues)), fitnessValues, color='green')

        ax.set_ylim(0, 100)

        ax.set(xlabel='Testing Step #', ylabel='Fitness',
               title='Fitness per session')
        ax.grid()

        _, localFilePath = tempfile.mkstemp(suffix=".png")
        fig.savefig(localFilePath)
        with open(localFilePath, 'rb') as f:
            config.saveKwolaFileData("charts", "fitness_chart.png", f.read())
        os.unlink(localFilePath)
        plt.close(fig)

        getLogger().info(f"Best Fitness Value: {bestFitness}")


def averageTracesWithNewBranchesForTestingStep(config, testingStepId):
    testingStep = TestingStep.loadFromDisk(testingStepId, config)

    stepTraceWithNewBranchCounts = []
    for sessionId in testingStep.executionSessions:
        session = ExecutionSession.loadFromDisk(sessionId, config)
        if session.status == "completed" and session.countTracesWithNewBranches is not None:
            stepTraceWithNewBranchCounts.append(session.countTracesWithNewBranches)

    if len(stepTraceWithNewBranchCounts) > 0:
        return numpy.mean(stepTraceWithNewBranchCounts)
    else:
        return None


def generateTracesWithNewBranchesChart(config, applicationId):
    getLogger().info(f"Generating the traces with new branches chart")

    config = KwolaCoreConfiguration(config)

    testingSteps = sorted(
        [step for step in TrainingManager.loadAllTestingSteps(config, applicationId=applicationId) if step.status == "completed"],
        key=lambda step: step.startTime, reverse=False)

    countTracesWithNewBranchesValues = [averageTracesWithNewBranchesForTestingStep(config, step.id) for step in testingSteps]
    countTracesWithNewBranchesValues = [value for value in countTracesWithNewBranchesValues if value is not None]

    if len(countTracesWithNewBranchesValues) > 0:
        fig, ax = plt.subplots()

        countTracesWithNewBranchesValues = medianSmooth(countTracesWithNewBranchesValues)

        ax.plot(range(len(countTracesWithNewBranchesValues)), countTracesWithNewBranchesValues, color='green')

        ax.set_ylim(0, config['testing_sequence_length'])

        ax.set(xlabel='Testing Step #', ylabel='Traces with new branches',
               title='# of testing traces that have new branches')
        ax.grid()

        _, localFilePath = tempfile.mkstemp(suffix=".png")
        fig.savefig(localFilePath)
        with open(localFilePath, 'rb') as f:
            config.saveKwolaFileData("charts", "traces_with_new_branches.png", f.read())
        os.unlink(localFilePath)
        plt.close(fig)

def generateCoverageChart(config, applicationId):
    getLogger().info(f"Generating the coverage chart")

    config = KwolaCoreConfiguration(config)

    testingSteps = sorted(
        [step for step in TrainingManager.loadAllTestingSteps(config, applicationId=applicationId) if step.status == "completed"],
        key=lambda step: step.startTime, reverse=False)

    coverageData = [computeCumulativeCoverageForTestingSteps([step.id], config) for step in testingSteps]
    coverageValues = [result[0] for result in coverageData]
    executedLinesValues = [result[1] for result in coverageData]
    totalLinesValues = [result[2] for result in coverageData]

    coverageValues = medianSmooth(coverageValues)
    executedLinesValues = medianSmooth(executedLinesValues)
    totalLinesValues = medianSmooth(totalLinesValues)

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
    plt.close(fig)

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
    plt.close(fig)

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
                stepId = stepId.replace(".enc", "")

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

    lossValuesSorted = sorted(
        [value for value in (loadTrainingStepLossData(config, id, attribute) for id in trainingStepIds) if value[2] == "completed"],
        key=lambda result: result[1], reverse=False)

    lossValues = [result[0] for result in lossValuesSorted]

    if len(lossValues) == 0:
        return

    fig, ax = plt.subplots()

    lossValues = medianSmooth(lossValues)

    ax.plot(range(len(lossValues)), lossValues, color='green')

    ax.set_ylim(0, max(float(numpy.percentile(lossValues, 99)), 1e-12))

    ax.set(xlabel='Training Step #', ylabel='Reward', title=title)
    ax.grid()

    _, localFilePath = tempfile.mkstemp(suffix=".png")
    fig.savefig(localFilePath)
    with open(localFilePath, 'rb') as f:
        config.saveKwolaFileData("charts", fileName, f.read())
    os.unlink(localFilePath)
    plt.close(fig)

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
    cumulativeBranchTrace = {}

    for stepId in testingStepIds:
        branchTrace = computeCumulativeBranchTraceForTestingSteps(stepId, config)
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
    plt.close(fig)


    fig, ax = plt.subplots()
    ax.plot(numpy.array(range(len(cumulativeLinesExecutedValues))) * numberOfTestingStepsPerValue, cumulativeLinesExecutedValues, color='green')
    ax.set_ylim(0, 600)
    ax.set(xlabel='Testing Steps Completed', ylabel='Cumulative Total Lines Triggered (green)', title=f"Cumulative Lines Triggered Chart, Group Size: {numberOfTestingStepsPerValue}")
    ax.set_ylim(650, 750)
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
    plt.close(fig)

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
                bugId = bugId.replace(".enc", "")

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
    plt.close(fig)


def generateAllCharts(config, applicationId=None, enableCumulativeCoverage=False):
    getLogger().info(f"Generating charts based on results.")

    # Matplotlib is not reliably safe to use from processes which are spawned
    # after CUDA and browser-native libraries.  Render sequentially in this
    # coordinator and close each figure eagerly; charting runs infrequently
    # and deterministic teardown is more valuable than parallelism here.
    serializedConfig = config.serialize()
    chartFunctions = [
        (generateRewardChart, [serializedConfig, applicationId]),
        (generateFitnessChart, [serializedConfig, applicationId]),
        (generateTracesWithNewBranchesChart, [serializedConfig, applicationId]),
    ]
    if enableCumulativeCoverage:
        chartFunctions.append((generateCoverageChart, [serializedConfig, applicationId]))

    chartFunctions.extend([
        (generateLossChart, [serializedConfig, applicationId, 'totalLosses', "Total Loss", 'total_loss_chart.png']),
        (generateLossChart, [serializedConfig, applicationId, 'presentRewardLosses', "Present Reward Loss", 'present_reward_loss_chart.png']),
        (generateLossChart, [serializedConfig, applicationId, 'discountedFutureRewardLosses', "Discounted Future Reward Loss", 'discounted_future_reward_loss_chart.png']),
        (generateLossChart, [serializedConfig, applicationId, 'stateValueLosses', "State Value Loss", 'state_value_loss_chart.png']),
        (generateLossChart, [serializedConfig, applicationId, 'advantageLosses', "Advantage Loss", 'advantage_loss_chart.png']),
        (generateLossChart, [serializedConfig, applicationId, 'actionProbabilityLosses', "Action Probability Loss", 'action_probability_loss_chart.png']),
    ])

    if config['chart_enable_cumulative_coverage_chart'] and enableCumulativeCoverage:
        chartFunctions.extend((generateCumulativeCoverageChart, [serializedConfig, applicationId, size]) for size in (100, 25, 10, 5))

    if config['chart_enable_cumulative_errors_chart']:
        chartFunctions.append((generateCumulativeErrorsFoundChart, [serializedConfig, applicationId]))

    for function, arguments in chartFunctions:
        function(*arguments)

    getLogger().info(f"Completed generating all the charts.")
