#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#


from ..tasks.TaskProcess import TaskProcess
import torch.distributed
from ..components.managers.TrainingManager import TrainingManager

def runTrainingStep(config, trainingSequenceId, trainingStepIndex, gpu=None, coordinatorTempFileName="kwola_distributed_coordinator", testingRunId=None, applicationId=None, gpuWorldSize=torch.cuda.device_count()):
    manager = TrainingManager(config, trainingSequenceId, trainingStepIndex, gpu, coordinatorTempFileName, testingRunId, applicationId, gpuWorldSize)
    return manager.runTraining()


if __name__ == "__main__":
    task = TaskProcess(runTrainingStep)
    task.run()
