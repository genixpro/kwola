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
import os
import traceback
from .retry import autoretry
import skimage.io
import pkg_resources
import numpy


def createDebugVideoSubProcess(config, executionSessionId, name="", includeNeuralNetworkCharts=True, includeNetPresentRewardChart=True, hilightStepNumber=None, cutoffStepNumber=None, folder="debug_videos"):
    try:
        setupLocalLogging(config)

        getLogger().info(f"Creating debug video for session {executionSessionId} with options includeNeuralNetworkCharts={includeNeuralNetworkCharts}, includeNetPresentRewardChart={includeNetPresentRewardChart}, hilightStepNumber={hilightStepNumber}, cutoffStepNumber={cutoffStepNumber}")

        config = KwolaCoreConfiguration(config)

        agent = DeepLearningAgent(config, whichGpu=None)
        agent.initialize(enableTraining=False)
        agent.load()

        executionSession = ExecutionSession.loadFromDisk(executionSessionId, config)

        videoData = agent.createDebugVideoForExecutionSession(executionSession, includeNeuralNetworkCharts=includeNeuralNetworkCharts, includeNetPresentRewardChart=includeNetPresentRewardChart, hilightStepNumber=hilightStepNumber, cutoffStepNumber=cutoffStepNumber)

        fileName = f'{name + "_" if name else ""}{str(executionSession.id)}.mp4'
        config.saveKwolaFileData(folder, fileName, videoData)

        del agent

        getLogger().info(f"Finished creating debug video for session {executionSessionId}")
    except Exception as e:
        getLogger().error(f"An error was triggered while generating a debug video: {traceback.format_exc()}")
        return traceback.format_exc()



def addDebugActionCursorToImage(image, position, actionType):
    if actionType.startswith("type"):
        cursorImage = skimage.io.imread(pkg_resources.resource_filename("kwola", "images/type.png"))
        cursorImageOriginPos = (16, 16)
    elif actionType.startswith("scroll"):
        cursorImage = skimage.io.imread(pkg_resources.resource_filename("kwola", "images/scroll.png"))
        cursorImageOriginPos = (16, 16)
    elif actionType.startswith("clear"):
        cursorImage = skimage.io.imread(pkg_resources.resource_filename("kwola", "images/clear.png"))
        cursorImageOriginPos = (16, 19)
    else:
        cursorImage = skimage.io.imread(pkg_resources.resource_filename("kwola", "images/click.png"))
        cursorImageOriginPos = (9, 9)

    pointerTop = (position[0] - cursorImageOriginPos[0])
    pointerLeft = (position[1] - cursorImageOriginPos[1])

    pointerWidth = cursorImage.shape[1]
    pointerHeight = cursorImage.shape[0]

    cutOffTop = 0 if pointerTop >= 0 else -pointerTop
    cutOffLeft = 0 if pointerLeft >= 0 else -pointerLeft

    cutOffRight = 0 if (pointerLeft + pointerWidth) < image.shape[1] else (pointerLeft + pointerWidth) - image.shape[1]
    cutOffBottom = 0 if (pointerTop + pointerHeight) < image.shape[0] else (pointerTop + pointerHeight) - image.shape[0]

    imageSlice = image[
                 int(max(0, pointerTop)): int(pointerTop + pointerHeight - cutOffBottom),
                 int(max(0, pointerLeft)): int(pointerLeft + pointerWidth - cutOffRight)
                 ]

    cursorImageSlice = cursorImage[
                       int(cutOffTop):pointerHeight - int(cutOffBottom),
                       int(cutOffLeft):pointerWidth - int(cutOffRight)
                       ]

    imageSlice[:, :, :] = cursorImageSlice[:, :, 0:3] * (cursorImageSlice[:, :, 3:4] / 255.0) + \
                          imageSlice[:, :, :] * (1.0 - (cursorImageSlice[:, :, 3:4] / 255.0))
