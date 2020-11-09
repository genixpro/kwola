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



from ...components.agents.DeepLearningAgent import DeepLearningAgent
from ...config.config import KwolaCoreConfiguration
from ...config.logger import getLogger, setupLocalLogging
from ...datamodels.ExecutionSessionModel import ExecutionSession
from ..utils.file import loadKwolaFileData, saveKwolaFileData
import os
from .retry import autoretry


def createDebugVideoSubProcess(configDir, executionSessionId, name="", includeNeuralNetworkCharts=True, includeNetPresentRewardChart=True, hilightStepNumber=None, cutoffStepNumber=None, folder="debug_videos"):
    setupLocalLogging()

    getLogger().info(f"Creating debug video for session {executionSessionId} with options includeNeuralNetworkCharts={includeNeuralNetworkCharts}, includeNetPresentRewardChart={includeNetPresentRewardChart}, hilightStepNumber={hilightStepNumber}, cutoffStepNumber={cutoffStepNumber}")

    config = KwolaCoreConfiguration(configDir)

    agent = DeepLearningAgent(config, whichGpu=None)
    agent.initialize(enableTraining=False)
    agent.load()

    kwolaDebugVideoDirectory = config.getKwolaUserDataDirectory(folder)

    executionSession = ExecutionSession.loadFromDisk(executionSessionId, config)

    videoData = agent.createDebugVideoForExecutionSession(executionSession, includeNeuralNetworkCharts=includeNeuralNetworkCharts, includeNetPresentRewardChart=includeNetPresentRewardChart, hilightStepNumber=hilightStepNumber, cutoffStepNumber=cutoffStepNumber)

    filePath = os.path.join(kwolaDebugVideoDirectory, f'{name + "_" if name else ""}{str(executionSession.id)}.mp4')
    saveKwolaFileData(filePath, videoData, config)

    del agent
