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
import torch.distributed
import traceback
import shutil

def runNeuralNetworkTestOnGPU(gpu, config, verbose=True):
    try:
        agent = DeepLearningAgent(config=config, whichGpu=gpu)

        agent.initialize(enableTraining=True)

        if verbose:
            print("Saving and loading the network to disk")
        agent.save()
        agent.load()

        if verbose:
            print("Running a test training iteration")

        batches = [agent.prepareEmptyBatch()]
        agent.learnFromBatches(batches)

        return True
    except Exception:
        traceback.print_exc()
        return False


def testNeuralNetworkAllGPUs(verbose=True):
    """
        This method is used to test whether or not the pytorch and the neural network is installed.
        If GPU's are detected, it will try to train using them, ensuring that everything is working
        there.
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

    if verbose:
        print("Initializing the deep neural network on the CPU.")
    success = runNeuralNetworkTestOnGPU(gpu=None, config=config, verbose=verbose)
    if success:
        if verbose:
            print("We have successfully initialized a neural network on the CPU and run a few a training batches through it.")
    else:
        if verbose:
            print("Neural network training appears to have failed on the CPU.")
        allSuccess = False

    gpus = torch.cuda.device_count()
    if gpus > 0:
        torch.distributed.init_process_group(backend="gloo",
                                             world_size=1,
                                             rank=0,
                                             init_method="file:///tmp/kwola_distributed_coordinator", )

        for gpu in range(gpus):
            if verbose:
                print(f"Initializing the deep neural network on your CUDA GPU #{gpu}")
            success = runNeuralNetworkTestOnGPU(gpu=gpu, config=config, verbose=verbose)
            if success:
                if verbose:
                    print(f"We have successfully initialized a neural network on GPU #{gpu} and run a few a training batches through it.")
            else:
                if verbose:
                    print(f"Neural network training appears to have failed on GPU #{gpu}")
                allSuccess = False

    if allSuccess:
        if verbose:
            print("Everything worked! Kwola deep learning appears to be fully working.")

    shutil.rmtree(configDir)

    return allSuccess

