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

from ..config.config import Configuration
from ..components.agents.DeepLearningAgent import DeepLearningAgent
import torch
import traceback


def runNeuralNetworkTestOnGPU(gpu, config):
    try:
        branchSize = 50

        agent = DeepLearningAgent(config=config, whichGpu=gpu)

        agent.initialize(branchSize, enableTraining=True)

        print("Saving and loading the network to disk")
        agent.save()
        agent.load()

        print("Starting training.")

        for i in range(3):
            print("Running iteration", i+1)
            batches = [agent.prepareEmptyBatch() for n in range(2)]
            agent.learnFromBatches(batches)

        return True
    except Exception:
        traceback.print_exc()
        return False


def main():
    """
        This is the entry for the neural network testing command.
    """


    configDir = Configuration.createNewLocalKwolaConfigDir("testing",
                                                           url="http://demo.kwolatesting.com/",
                                                           email="",
                                                           password="",
                                                           name="",
                                                           paragraph="",
                                                           enableRandomNumberCommand=False,
                                                           enableRandomBracketCommand=False,
                                                           enableRandomMathCommand=False,
                                                           enableRandomOtherSymbolCommand=False,
                                                           enableDoubleClickCommand=False,
                                                           enableRightClickCommand=False
                                                           )

    config = Configuration(configDir)

    allSuccess = True

    print("Initializing the deep neural network on the CPU.")
    success = runNeuralNetworkTestOnGPU(gpu=None, config=config)
    if success:
        print("We have successfully initialized a neural network on the CPU and run a few a training batches through it.")
    else:
        print("Neural network training appears to have failed on the CPU.")
        allSuccess = False

    gpus = torch.cuda.device_count()
    for gpu in range(gpus):
        print(f"Initializing the deep neural network on your CUDA GPU #{gpu}")
        success = runNeuralNetworkTestOnGPU(gpu=gpu, config=config)
        if success:
            print(f"We have successfully initialized a neural network on GPU #{gpu} and run a few a training batches through it.")
        else:
            print(f"Neural network training appears to have failed on GPU #{gpu}")
            allSuccess = False

    if allSuccess:
        print("Everything worked! Kwola deep learning appears to be fully working.")
