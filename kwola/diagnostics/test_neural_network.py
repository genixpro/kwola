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

from ..config.config import KwolaCoreConfiguration
from ..components.agents.DeepLearningAgent import DeepLearningAgent
import torch
import torch.distributed
import traceback
import shutil
import os
import tempfile
import sys

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


    configDir = KwolaCoreConfiguration.createNewLocalKwolaConfigDir("testing",
                                                                    url="http://demo.kwolatesting.com/",
                                                                    email="",
                                                                    password="",
                                                                    name="",
                                                                    paragraph="",
                                                                    enableTypeEmail=True,
                                                                    enableTypePassword=True,
                                                                    enableRandomNumberCommand=False,
                                                                    enableRandomBracketCommand=False,
                                                                    enableRandomMathCommand=False,
                                                                    enableRandomOtherSymbolCommand=False,
                                                                    enableDoubleClickCommand=False,
                                                                    enableRightClickCommand=False
                                                                    )
    try:
        config = KwolaCoreConfiguration.loadConfigurationFromDirectory(configDir)

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
            init_method = f"file://{os.path.join(tempfile.gettempdir(), 'kwola_distributed_coordinator')}"

            if sys.platform == "win32" or sys.platform == "win64":
                init_method = f"file:///{os.path.join(tempfile.gettempdir(), 'kwola_distributed_coordinator')}"

            torch.distributed.init_process_group(backend="gloo",
                                                 world_size=1,
                                                 rank=0,
                                                 init_method=init_method)

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

        return allSuccess
    finally:
        shutil.rmtree(configDir)
