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


from ..tasks.TaskProcess import TaskProcess
import torch.distributed
from ..components.managers.TrainingManager import TrainingManager

def runTrainingStep(config, trainingSequenceId, trainingStepIndex, gpu=None, coordinatorTempFileName="kwola_distributed_coordinator", testingRunId=None, applicationId=None, gpuWorldSize=torch.cuda.device_count()):
    manager = TrainingManager(config, trainingSequenceId, trainingStepIndex, gpu, coordinatorTempFileName, testingRunId, applicationId, gpuWorldSize)
    return manager.runTraining()


if __name__ == "__main__":
    task = TaskProcess(runTrainingStep)
    task.run()
