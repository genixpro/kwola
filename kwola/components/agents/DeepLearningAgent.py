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


from ...config.config import Configuration
from ...datamodels.actions.ClickTapAction import ClickTapAction
from ...datamodels.actions.ClearFieldAction import ClearFieldAction
from ...datamodels.actions.RightClickAction import RightClickAction
from ...datamodels.actions.TypeAction import TypeAction
from ...datamodels.ExecutionSessionModel import ExecutionSession
from ...datamodels.ExecutionTraceModel import ExecutionTrace
from .TraceNet import TraceNet
from datetime import datetime
import concurrent.futures
import copy
import cv2
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy
import os
import os.path
import os.path
import scipy.signal
import scipy.special
import shutil
import skimage
import skimage.color
import skimage.draw
import skimage.io
import skimage.measure
import skimage.segmentation
import skimage.transform
import sklearn.preprocessing
import subprocess
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
import traceback


class DeepLearningAgent:
    """
        This class represents a deep learning agent, which is an agent that uses deep learning in order to learn how
        to use an application given to it. This class is tightly coupled with the WebEnvironment and
        WebEnvironmentSession classes.
    """

    def __init__(self, config, whichGpu="all"):
        self.config = config

        self.whichGpu = whichGpu

        # We create a method that will convert torch CPU tensors into
        # torch CUDA tensors if this model is set in GPU mode.
        if self.whichGpu == "all":
            self.variableWrapperFunc = lambda t, x: t(x).cuda()
        elif self.whichGpu is None:
            self.variableWrapperFunc = lambda t, x: t(x)
        else:
            self.variableWrapperFunc = lambda t, x: t(x).cuda(device=f"cuda:{self.whichGpu}")

        # Fetch the folder that we will store the model parameters in
        self.modelPath = os.path.join(config.getKwolaUserDataDirectory("models"), "deep_learning_model")

        # This is just a generic list of known HTML cursors. Its pretty comprehensive,
        # and is used for the cursor prediction, which is an additional loss that
        # helps stabilize the neural network predictions
        self.cursors = [
            "alias",
            "all-scroll",
            "auto",
            "cell",
            "context-menu",
            "col-resize",
            "copy",
            "crosshair",
            "default",
            "e-resize",
            "ew-resize",
            "grab",
            "grabbing",
            "help",
            "move",
            "n-resize",
            "ne-resize",
            "nesw-resize",
            "ns-resize",
            "nw-resize",
            "nwse-resize",
            "no-drop",
            "none",
            "not-allowed",
            "pointer",
            "progress",
            "row-resize",
            "s-resize",
            "se-resize",
            "sw-resize",
            "text",
            "url",
            "w-resize",
            "wait",
            "zoom-in",
            "zoom-out",
            "none"
        ]

        # This is a variable that maps the action name to a lambda which creates
        # the actual action object which can be stored in the database.
        # By default, we always have the click action enabled. It can never be
        # disabled.
        self.actions = {
            "click": lambda x, y: ClickTapAction(type="click", x=x, y=y, times=1)
        }

        # This variable stores the base weighting for each of the actions.
        # These weights are used to pick which action to perform when
        # we are using a randomly selected action
        self.actionBaseWeights = [
            config['random_weight_click']
        ]

        # This variable is used to store a list of keywords that are associated with each action.
        # This is used to help make sure that, on average, the randomly selected actions will put
        # emails in email inputs and passwords in password inputs. We will analyze all the available
        # data on the html element, e.g. its classes, attributes, id, name, etc.. to see if there is
        # anything that matches one of these keywords. And if so, we give the probability of choosing
        # that action on that element a significant boost
        self.actionProbabilityBoostKeywords = [
            []
        ]

        # Keep track of whether there is a typing action, since if there is a typing
        # action we also need the clear action to be able to clear anything already typed.
        hasTypingAction = False

        # Only add in the email action if the user configured it
        if config['email']:
            self.actions['typeEmail'] = lambda x, y: TypeAction(type="typeEmail", x=x, y=y, label="email", text=config['email'])
            self.actionBaseWeights.append(config['random_weight_type_email'])
            self.actionProbabilityBoostKeywords.append(["email", "user"])
            hasTypingAction = True

        # Only add in the password action if the user configured it
        if config['password']:
            self.actions['typePassword'] = lambda x, y: TypeAction(type="typePassword", x=x, y=y, label="password", text=config['password'])
            self.actionBaseWeights.append(config['random_weight_type_password'])
            self.actionProbabilityBoostKeywords.append(["pass"])
            hasTypingAction = True

        # Only add in the name action if the user configured it
        if config['name']:
            self.actions['typeName'] = lambda x, y: TypeAction(type="typeName", x=x, y=y, label="name", text=config['name'])
            self.actionBaseWeights.append(config['random_weight_type_name'])
            self.actionProbabilityBoostKeywords.append(["email", "user"])
            hasTypingAction = True

        # Only add in the paragraph action if the user configured it
        if config['paragraph']:
            self.actions['typeParagraph'] = lambda x, y: TypeAction(type="typeParagraph", x=x, y=y, label="paragraph", text=config['paragraph'])
            self.actionBaseWeights.append(config['random_weight_type_paragraph'])
            self.actionProbabilityBoostKeywords.append(["pass"])
            hasTypingAction = True

        # Only add in the random number action if the user configured it
        if config['enableRandomNumberCommand']:
            self.actions['typeNumber'] = lambda x, y: TypeAction(type="typeNumber", x=x, y=y, label="number", text=self.randomString('-.0123456789$%', random.randint(1, 5)))
            self.actionBaseWeights.append(config['random_weight_type_number'])
            self.actionProbabilityBoostKeywords.append(["num", "count", "int", "float"])
            hasTypingAction = True

        # Only add in the double click action if the user configured it
        if config['enableDoubleClickCommand']:
            self.actions['doubleClick'] = lambda x, y: ClickTapAction(type="double_click", x=x, y=y, times=2)
            self.actionBaseWeights.append(config['random_weight_double_click'])
            self.actionProbabilityBoostKeywords.append([])

        # Only add in the right click action if the user configured it
        if config['enableRightClickCommand']:
            self.actions['rightClick'] = lambda x, y: RightClickAction(type="right_click", x=x, y=y)
            self.actionBaseWeights.append(config['random_weight_right_click'])
            self.actionProbabilityBoostKeywords.append([])

        # Only add in the right click action if the user configured it
        # This command basically types in a randomly chosen bracket of
        # any type, curly or straight
        if config['enableRandomBracketCommand']:
            self.actions['typeBrackets'] = lambda x, y: TypeAction(type="typeBrackets", x=x, y=y, label="brackets", text=self.randomString('{}[]()', random.randint(1, 3)))
            self.actionBaseWeights.append(config['random_weight_type_brackets'])
            self.actionProbabilityBoostKeywords.append([])
            hasTypingAction = True

        # Only add in the random math action if the user configured it
        # This action basically types in equation related letters, like
        # plus and minus
        if config['enableRandomMathCommand']:
            self.actions['typeMath'] = lambda x, y: TypeAction(type="typeMath", x=x, y=y, label="symbol", text=self.randomString('*=+<>-', random.randint(1, 3)))
            self.actionBaseWeights.append(config['random_weight_type_math'])
            self.actionProbabilityBoostKeywords.append([])
            hasTypingAction = True

        # Only add in the random other symbol action if the user configured it
        # This action will basically type in random symbols sampled from the
        # list of symbols not already covered by the random number, brackets
        # and math commands
        if config['enableRandomOtherSymbolCommand']:
            self.actions['typeOtherSymbol'] = lambda x, y: TypeAction(type="typeOtherSymbol", x=x, y=y, label="symbol", text=self.randomString('"\';:/?,!^&#@', random.randint(1, 3)))
            self.actionBaseWeights.append(config['random_weight_type_other_symbol'])
            self.actionProbabilityBoostKeywords.append([])
            hasTypingAction = True

        # If there was at least one typing action, then we add in the clear
        # action as well. The clear action basically operates like the delete
        # key but is slightly more useful for the AI since it just deletes
        # everything in the input box all at once.
        if hasTypingAction:
            self.actions['clear'] = lambda x, y: ClearFieldAction(type="clear", x=x, y=y)
            self.actionBaseWeights.append(config['random_weight_clear'])
            self.actionProbabilityBoostKeywords.append([])

        # This dictionary provides random weightings for each HTML element.
        # We use this to focus the random action selection on interacting
        # with input elements and buttons, and away from clicking on links
        # which go to new pages, since that would erase any progress the
        # algo has made on the current page.
        self.elementBaseWeights = {
            "a": config['random_html_element_a_weight'],
            "input": config['random_html_element_input_weight'],
            "button": config['random_html_element_button_weight'],
            "p": config['random_html_element_p_weight'],
            "span": config['random_html_element_span_weight'],
            "div": config['random_html_element_div_weight'],
            "canvas": config['random_html_element_canvas_weight'],
            "other": config['random_html_element_other_weight']
        }

        # Create the sorted list of action names. This sorted list is used to
        # ensure is a consistent mapping between the action names and indexes
        # Unfortunately keys in dictionary objects are unsorted and so may
        # not be consistent from run to run if code is changed slightly.
        # Therefore its important to maintain the sorted list which is always
        # consistent as long as the action names haven't changed.
        self.actionsSorted = sorted(self.actions.keys())

    def randomString(self, chars, len):
        """
            Generates a random string.

            Just a utility function.

            :param chars: A string containing possible characters to select from.
            :param len: The number of characters to put into the generated string.
            :return:
        """
        base = ""
        for n in range(len):
            base += str(random.choice(chars))
        return base

    def load(self):
        """
            Loads the agent from db / disk. The agent will be loaded from the Kwola configuration
            directory.
        """

        # Only load the model if we actually see the file on disk.
        if os.path.exists(self.modelPath):
            # Depending on whether GPU is turned on, we try load the state dict
            # directly into GPU / CUDA memory.
            device = self.getTorchDevices()[0]
            stateDict = torch.load(self.modelPath, map_location=device)

            # Load the state dictionary into the model itself.
            self.model.load_state_dict(stateDict)

            # Also load a copy of the state dictionary into our target network.
            # The target network is a copy of the primary neural network
            # That is used to predict the future reward values, serving as a target
            # for the updates made to the main model.
            if self.targetNetwork is not None:
                self.targetNetwork.load_state_dict(stateDict)

    def save(self, saveName=""):
        """
            Saves the agent to disk. By default, you will save the agent into the primary slot, which is the
            default one loaded when you call DeepLearningAgent.load(). If you provide a saveName, then the agent
            will be saved with filenames prefixed with that.

            :param saveName: A string containing the prefix for the model file names when the model is saved.
        """

        if saveName:
            saveName = "_" + saveName

        torch.save(self.model.state_dict(), self.modelPath + saveName)

    def getTorchDevices(self):
        """
            This method returns a list of torch device objects indicating which torch devices are
            being used for this run.

            :return: A list containing torch device objects
        """

        if self.whichGpu == "all":
            # Use all available GPUs
            device_ids = [torch.device(f'cuda:{n}') for n in range(torch.cuda.device_count())]
        elif self.whichGpu is None:
            # Only the GPU
            device_ids = [torch.device('cpu')]
        else:
            # Use a specific GPU, ignore the others
            device_ids = [torch.device(f'cuda:{self.whichGpu}')]

        return device_ids

    def initialize(self, branchFeatureSize, enableTraining=True):
        """
            Initializes the deep learning agent. This method is will actually create the neural network in memory.
            Technically you can create a DeepLearningAgent and use some of its methods without initializing it,
            but you will not be able to use the primary methods of nextBestActions or learnFromBatches. You must
            initialize the neural network before calling either of those methods.

            :param branchFeatureSize: This should be the number of code branches that are being traced. It should be the
                                      length of the return value from WebEnvironment.getBranchFeature.

            :param enableTraining: Indicates whether or not the agent should be initialized in training mode. Training mode
                                   costs more RAM, so you should avoid it if you can.

        """
        devices = self.getTorchDevices()
        outputDevice = devices[0]

        # Create the primary torch model object
        self.model = TraceNet(self.config, branchFeatureSize * 2, len(self.actions), branchFeatureSize, 12, len(self.cursors))

        # If training is enabled, we also have to create the target network
        if enableTraining:
            self.targetNetwork = TraceNet(self.config, branchFeatureSize * 2, len(self.actions), branchFeatureSize, 12, len(self.cursors))
        else:
            self.targetNetwork = None

        # If we are setup to use GPUs, then the models need to be moved over to the GPUs.
        # Otherwise they can stay on the CPUs.
        # We also create the DistributedDataParallel wrappers which are used to share the
        # training across multiple separate GPU processes.
        if self.whichGpu == "all":
            self.model = self.model.cuda()
            self.modelParallel = nn.parallel.DistributedDataParallel(module=self.model, device_ids=devices, output_device=outputDevice)
            if enableTraining:
                self.targetNetwork = self.targetNetwork.cuda()
        elif self.whichGpu is None:
            self.model = self.model.cpu()
            self.modelParallel = self.model
            if enableTraining:
                self.targetNetwork = self.targetNetwork.cpu()
        else:
            self.model = self.model.cuda(device=devices[0])
            self.modelParallel = nn.parallel.DistributedDataParallel(module=self.model, device_ids=devices, output_device=outputDevice)
            if enableTraining:
                self.targetNetwork = self.targetNetwork.cuda(device=devices[0])

        # We create the optimizer object if the system is in training mode.
        # The optimizer is what actually updates the neural network parameters
        # based on the calculated gradients.
        if enableTraining:
            self.optimizer = optim.Adamax(
                                          self.model.parameters(),
                                          lr=self.config['training_learning_rate'],
                                          betas=(self.config['training_gradient_exponential_moving_average_decay'],
                                                self.config['training_gradient_squared_exponential_moving_average_decay'])
                                      )
        else:
            self.optimizer = None

    def updateTargetNetwork(self):
        """
            This method is used during training to update the target network. The target network is a second
            copy of the neural network that is used during training to predict future rewards. It is only updated
            every few hundred iterations, in order to provide a more stable target.
        """
        self.targetNetwork.load_state_dict(self.model.state_dict())

    def processImages(self, images):
        """
            This method will process a given numpy array of raw images and prepare them to be processed inside the
            neural network.

            :param images: A numpy array, with dimensions [batch_size, height, width, channels]

            :return: A new numpy array containing the resulting processed images, with dimensions [batch_size, channels, height, width]
        """
        convertedImageFutures = []

        # We try to do each of the images in parallel by using a thread pool executor. In practice only the C++ code
        # within scikit-image will actually multithread due to Python's stupid global interpreter lock flaw.
        # But that still gives us some gains here.
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for image in images:
                convertedImageFuture = executor.submit(DeepLearningAgent.processRawImageParallel, image, self.config)
                convertedImageFutures.append(convertedImageFuture)

        # Fetch  out all the results from each of the threads
        convertedProcessedImages = [
            convertedImageFuture.result() for convertedImageFuture in convertedImageFutures
        ]

        # Merge all the images into a single numpy array.
        return numpy.array(convertedProcessedImages)

    def createPixelActionMap(self, actionMaps, height, width):
        """
            This method generates the pixel action map. The pixel action map is basically an image, containing
            either a 1 or a 0 on a pixel by pixel basis for each action that the agent is able to perform. A 1
            on a given pixel indicates that the action is able to be performed on that pixel, and a 0 indicates
            that the action is not able to be performed.

            :param actionMaps: This must be a list of kwola.datamodels.ActionMapModel instances.
            :param height: The height of the image for which this action map is being generated.
            :param width: The width of the image for which this action map is being generated.

            :return: A numpy array, with the dimensions [# actions, height, width]
        """
        # Start with a blank numpy array with the correct shape
        pixelActionMap = numpy.zeros([len(self.actionsSorted), height, width], dtype=numpy.uint8)

        # Iterate through each of the action maps
        for element in actionMaps:
            actionTypes = []

            # If the element is clickable, add in the relevant actions (single and double click)
            if element['canClick']:
                if "click" in self.actionsSorted:
                    actionTypes.append(self.actionsSorted.index("click"))
                if "doubleClick" in self.actionsSorted:
                    actionTypes.append(self.actionsSorted.index("doubleClick"))

            # Only a handful of elements have right click enabled, so this is treated seperately
            # from the regular left click
            if element['canRightClick']:
                if "rightClick" in self.actionsSorted:
                    actionTypes.append(self.actionsSorted.index("rightClick"))

            # If the element is an input box, then we have to enable all of the typing actions
            if element['canType']:
                if "typeEmail" in self.actionsSorted:
                    actionTypes.append(self.actionsSorted.index("typeEmail"))
                if "typePassword" in self.actionsSorted:
                    actionTypes.append(self.actionsSorted.index("typePassword"))
                if "typeName" in self.actionsSorted:
                    actionTypes.append(self.actionsSorted.index("typeName"))
                if "typeNumber" in self.actionsSorted:
                    actionTypes.append(self.actionsSorted.index("typeNumber"))
                if "typeBrackets" in self.actionsSorted:
                    actionTypes.append(self.actionsSorted.index("typeBrackets"))
                if "typeMath" in self.actionsSorted:
                    actionTypes.append(self.actionsSorted.index("typeMath"))
                if "typeOtherSymbol" in self.actionsSorted:
                    actionTypes.append(self.actionsSorted.index("typeOtherSymbol"))
                if "typeParagraph" in self.actionsSorted:
                    actionTypes.append(self.actionsSorted.index("typeParagraph"))
                if "clear" in self.actionsSorted:
                    actionTypes.append(self.actionsSorted.index("clear"))

            # Here is the essential part. For each of the actions that are supported by this action
            # element, we paint a rectangle of 1's on the pixel action map. Effectively, the pixel
            # action map starts with all 0's for every pixel on the image, and then for every pixel
            # containing within an action map, we set those actions to 1. This allows the model to
            # know on a pixel by pixel basis what actions are possible to be executed on that pixel.
            for actionTypeIndex in actionTypes:
                pixelActionMap[actionTypeIndex, int(element['top'] * self.config['model_image_downscale_ratio'])
                                                :int(element['bottom'] * self.config['model_image_downscale_ratio']),
                int(element['left'] * self.config['model_image_downscale_ratio'])
                :int(element['right'] * self.config['model_image_downscale_ratio'])] = 1

        return pixelActionMap

    def getActionMapsIntersectingWithAction(self, action, actionMaps):
        """
            This method will filter a list of action maps you provide it, returning only the ones that appear
            to intersect with the given action.

            :param action: This should be a kwola.datamodels.actions.BaseAction instance, or one of its derived classes.
            :param actionMaps: This must be a list of kwola.datamodels.ActionMapModel instances.
            :return:
        """
        selected = []
        for actionMap in actionMaps:
            if actionMap.left <= action.x <= actionMap.right and actionMap.top <= action.y <= actionMap.bottom:
                selected.append(actionMap)

        return selected


    def filterActionMapsToPreventRepeatActions(self, sampleActionMaps, sampleRecentActions, width, height):
        """
            This method is used to filter the list of available actions the agent can perform, based on the
            actions its performed recently. Basically, if the agent has been performing the same action
            over and over again without finding new code branches, then we need to take that action off the
            list and force the agent to try something different. This acts as a sort of control mechanism
            for the AI that forces it to explore more.

            :param sampleActionMaps: This is a list of kwola.datamodels.ActionMapModel instances, containing the
                                     list of all possible actions the AI can take at this time
            :param sampleRecentActions: This is a list of kwola.datamodels.actions.BaseAction instances, containing
                                        all of the actions that have been performed since the last time that
                                        new code branches were discovered.
            :param width: The width of the image. This is used to filter out action maps that are outside the bounds.
            :param height: The height of the image. This is used to filter out action maps that are outside the bounds.
            :return: A tuple with (sampleActionMaps, sampleActionRecentActionCounts).

                    The first entry in the tuple is the new, filtered list of action maps. The second entry
                    is the count of actions performed for each of the action maps that have been kept by the
                    filter.
        """
        modelDownscale = self.config['model_image_downscale_ratio']

        # Create variables to build up the filtered list of action maps
        filteredSampleActionMaps = []
        filteredSampleActionRecentActionCounts = []

        # Iterate through each of the action maps
        for map in sampleActionMaps:
            # Check to see if the map is out of the screen
            if (map.top * modelDownscale) > height or (map.bottom * modelDownscale) < 0 or (map.left * modelDownscale) > width or (map.right * modelDownscale) < 0:
                # If so, skip this action map, don't add it to the filtered list
                continue

            # If the action map is on the screen, then now we count how many times we see an action on the recent actions list which
            # overlaps with this action-map. The overlap logic is handled in map.doesOverlapWith, and its a bit of an approximate,
            # heuristic match. Technically there is no way in html to determine if an HTML element now is the same as the HTML element
            # we saw a few frames ago. So instead we do this approximate match that checks to see of the left, right, top and bottom bounds
            # line up, and whether its the same element type with the same properties. If all these things are true, then we assume its the
            # same action map.
            count = 0
            for recentAction in sampleRecentActions:
                for recentActionMap in self.getActionMapsIntersectingWithAction(recentAction, recentAction.actionMapsAvailable):
                    if map.doesOverlapWith(recentActionMap, tolerancePixels=self.config['testing_repeat_action_pixel_overlap_tolerance']):
                        count += 1
                        break

            # We only add the action map to the list of filtered action maps if we have seen few actions on the recent actions list that
            # appear to be the same action. Basically, this whole giant mechanism is here just to prevent he algorithm from repeatedly
            # doing the same thing over and over and over again, which it is prone to do sometimes when it is stuck. This mechanism
            # forces the algorithm to explore.
            if count < self.config['testing_max_repeat_action_maps_without_new_branches']:
                filteredSampleActionMaps.append(map)
                filteredSampleActionRecentActionCounts.append(count)

        # Now we have a special check here. If there are literally no action maps that survived the filtering, then the algorithm
        # is in a pretty dire position. It has completely exhausted all of the actions available to it and nothing has lead to new
        # code being executed. In this instance, we have no choice but to just return the full list of action maps unfiltered, since
        # we can not return an empty list with 0 actions.
        if len(filteredSampleActionMaps) > 0:
            sampleActionMaps = filteredSampleActionMaps
            sampleActionRecentActionCounts = filteredSampleActionRecentActionCounts
        else:
            sampleActionRecentActionCounts = [1] * len(sampleActionMaps)

        # Return a tuple
        return sampleActionMaps, sampleActionRecentActionCounts



    def nextBestActions(self, stepNumber, rawImages, envActionMaps, additionalFeatures, pastExecutionTraces, shouldBeRandom=False):
        """
            This is the main prediction / inference function for the agent. This function will decide what is the next
            best action to take, given a particular state.

            :param stepNumber: This is an integer indicating what step the agent is on from the beginning of the
                               execution session.
            :param rawImages: This is a numpy array containing the raw image objects. There should be a single raw
                              image for each sub environment being predicted on at the same time.
            :param envActionMaps: This is an array of arrays. The outer array contains a list for each of the sub
                                  environments being inferenced for. The inner list should just be a list of
                                  kwola.datamodels.ActionMapModel instances.
            :param additionalFeatures: This should be a numpy array containing the additional features vector,
                                        one for each sub environment.
            :param pastExecutionTraces:  This should be a list of lists. The outer list should contain a list for each sub
                                         environment. The inner list should be a list of kwola.datamodels.ExecutionTrace
                                         instances, providing all of the execution traces leading up to the current state
            :param shouldBeRandom: Whether or not the actions should be selected entirely randomly, or should use them
                                   predictions of the machine learning algorithm.
            :return:
        """
        # Process all of the images
        processedImages = self.processImages(rawImages)
        actions = []

        # Here we have a pretty important mechanism. The name "recentActions" is a bit of a misnomer,
        # and we should probably come up with a new variable name. Basically, what we are doing here
        # is coming up with the list of actions since the last time the algorithm got the big reward -
        # which is triggering new branches of the code to be executed. We can see whether the algorithm
        # is stuck and is performing a lot of actions to find the next reward, or whether it is performing
        # effectively to trigger all of the code to get executed. The "recentActions" is thus a list
        # of all the actions it has tried since the last time it has triggered new code to be executed.
        recentActions = []
        for traceList in pastExecutionTraces:
            sampleRecentActions = []
            traceList.reverse()
            for trace in traceList:
                if not trace.didNewBranchesExecute:
                    sampleRecentActions.append(trace.actionPerformed)
                else:
                    break

            traceList.reverse()
            sampleRecentActions.reverse()
            recentActions.append(sampleRecentActions)

        # Declare function level width and height variables for convenience.
        width = processedImages.shape[3]
        height = processedImages.shape[2]

        # Here we set up a bunch of arrays to store the samples that we want to process.
        # This can get a bit convoluted but basically what we are doing here is separating
        # which of the sub-environments we want to process through the neural network,
        # and which ones can just have a random action generated for them. These lists
        # hold all the data required for the sub-environments we are going to process through
        # the neural network
        batchSampleIndexes = []
        imageBatch = []
        additionalFeatureVectorBatch = []
        pixelActionMapsBatch = []
        epsilonsPerSample = []
        recentActionsBatch = []
        actionMapsBatch = []
        recentActionsCountsBatch = []

        modelDownscale = self.config['model_image_downscale_ratio']

        for sampleIndex, image, additionalFeatureVector, sampleActionMaps, sampleRecentActions in zip(range(len(processedImages)), processedImages, additionalFeatures, envActionMaps, recentActions):
            # Ok here is a very important mechanism. Technically what is being calculated below is the "epsilon" value from classical
            # Q-learning. I'm just giving it a different name which suits my style. The "epsilon" is just the probability that the
            # algorithm will choose a random action instead of the prediction. In our situation, we have two values, because we have
            # both a pure random and a weighted random mechanism.
            # Usually, the epsilon would be given a sliding exponential curve, starting at a value indicating lots of random actions,
            # and steadily sliding to a value indicating mostly using the prediction. This is not suitable for Kwola for a few reasons,
            # one of them being that Kwola is focused on online learning, and so we want Kwola to be able to adapt to unpredictable changes in
            # the environment. So instead, what we do here is we run multiple different probabilities in parallel. Say you are running 12 execution
            # sessions in parallel. What we do is we give a range of epsilon values, from ones close to 0, indicating to use the predictions,
            # to ones close to 1, indicating use mostly random actions. This means we are always running a range of epsilon values at the same time.
            # Additionally, we change the epsilon value along the sequence too. The logic here is that, in our use case, the neural network tends
            # to find lots of new branches of code early on, but then have a harder and harder time as the sequence progresses. Therefore its predictions
            # are more accurate and valuable at the start, and towards the end it needs to explore more to tease out those last few branches of the
            # code.
            # Therefore we make the following two equations which basically combine these two mechanisms to compute the epsilon values for the
            # algorithm.
            randomActionProbability = (float(sampleIndex + 1) / float(len(processedImages))) * 0.50 * (1 + (stepNumber / self.config['testing_sequence_length']))
            weightedRandomActionProbability = (float(sampleIndex + 1) / float(len(processedImages))) * 0.50 * (1 + (stepNumber / self.config['testing_sequence_length']))

            # Filter the action maps to reduce instances where the algorithm is repeating itself over and over again.
            sampleActionMaps, sampleActionRecentActionCounts = self.filterActionMapsToPreventRepeatActions(sampleActionMaps, sampleRecentActions, width, height)

            # Create the pixel action map, which is basically a pixel by pixel representation of what actions are available to the algorithm.
            pixelActionMap = self.createPixelActionMap(sampleActionMaps, height, width)

            # Here is where the algorithm decides whether it will use a random action here
            # or make use of the predictions of the neural network
            if random.random() > randomActionProbability and not shouldBeRandom:
                # We will make use of the predictions of the neural network.
                # Add this sample to all of the lists that will be processed later
                batchSampleIndexes.append(sampleIndex)
                imageBatch.append(image)
                additionalFeatureVectorBatch.append(additionalFeatureVector)
                actionMapsBatch.append(sampleActionMaps)
                pixelActionMapsBatch.append(pixelActionMap)
                recentActionsBatch.append(sampleRecentActions)
                epsilonsPerSample.append(weightedRandomActionProbability)
                recentActionsCountsBatch.append(sampleActionRecentActionCounts)
            else:
                # We will generate a pure random action for this sample. In that case, we just do it, generating the random action
                actionX, actionY, actionType = self.getRandomAction(sampleActionRecentActionCounts, sampleActionMaps, pixelActionMap)

                # Here we make use of the lambda functions that were prepared in the constructor,
                # which map the action string to a function which creates the action object.
                action = self.actions[self.actionsSorted[actionType]](actionX, actionY)
                action.source = "random"
                action.predictedReward = None
                action.wasRepeatOverride = False
                action.actionMapsAvailable = sampleActionMaps

                # Its probably important to recognize what is happening here. Since we are processing the samples bound for the neural network
                # separately from the samples that we generate random actions for, we will need some way of reassembling all of the results
                # back into a list in the same order as the original list. Therefore, when we add an action to this actions list,
                # we add it along with the index of the sample, so that we can later sort the actions list by that sample index and restore
                # the original ordering.
                actions.append((sampleIndex, action))

        # Special catch here on the if statement. We dont need to perform any calculations
        # through the neural network if literally all the actions chosen were random. This
        # doesn't happen very often but in rare circumstances it can and we prepare for that here.
        if len(imageBatch) > 0:
            # Create torch tensors out of the numpy arrays, effectively preparing the data for input into the neural network.
            imageTensor = self.variableWrapperFunc(torch.FloatTensor, numpy.array(imageBatch))
            additionalFeatureTensor = self.variableWrapperFunc(torch.FloatTensor, numpy.array(additionalFeatureVectorBatch))
            pixelActionMapTensor = self.variableWrapperFunc(torch.FloatTensor, pixelActionMapsBatch)
            stepNumberTensor = self.variableWrapperFunc(torch.FloatTensor, numpy.array([stepNumber] * len(imageBatch)))

            # Here we set torch not to calculate any gradients. Technically we don't do the backward step anyhow, but
            # adding this line helps reduce memory usage I think
            with torch.no_grad():
                # Put the model in evaluation model. If you don't do this, the BatchNormalization layers will act in weird
                # ways that alter depending on the size of the batch here, which is inherently random. Putting the model
                # in evaluation mode ensures nice, clean, consistent, reproducible runs.
                self.model.eval()

                # Here we go! Here we are now finally putting data into the neural network and processing it.
                outputs = self.modelParallel({
                    "image": imageTensor,
                    "additionalFeature": additionalFeatureTensor,
                    "pixelActionMaps": pixelActionMapTensor,
                    "stepNumber": stepNumberTensor,
                    "outputStamp": False,
                    "computeExtras": False,
                    "computeRewards": False,
                    "computeActionProbabilities": True,
                    "computeStateValues": False,
                    "computeAdvantageValues": True
                })

                # Here we move the tensors into the CPU. Technically this is only needed if you are doing testing using your
                # GPU, but you can run the same code either way so we do it here to be safe.
                actionProbabilities = outputs['actionProbabilities'].cpu()
                advantageValues = outputs['advantage'].cpu()

            # Now we iterate over all of the data and results for each of the sub environments
            for sampleIndex, sampleEpsilon, sampleActionProbs, sampleAdvantageValues, sampleRecentActions, sampleActionMaps, sampleActionRecentActionCounts, samplePixelActionMap in zip(batchSampleIndexes, epsilonsPerSample, actionProbabilities, advantageValues,
                                                                                                                                                                  recentActionsBatch, actionMapsBatch, recentActionsCountsBatch,
                                                                                                                                                                  pixelActionMapsBatch):
                # Here is where we determine whether the algorithm will use the predicted best action from the neural network,
                # or do a weighted random selection using the outputs
                weighted = bool(random.random() < sampleEpsilon)
                override = False
                source = None
                actionType = None
                actionX = None
                actionY = None
                samplePredictedReward = None
                if not weighted:
                    source = "prediction"

                    # Get the coordinates and action type index of the action which has the highest probability
                    actionX, actionY, actionType = self.getActionInfoTensorsFromRewardMap(sampleActionProbs)
                    actionX = actionX.data.item()
                    actionY = actionY.data.item()
                    actionType = actionType.data.item()

                    # Get the reward that was predicted for this action
                    samplePredictedReward = sampleActionProbs[actionType, actionY, actionX].data.item()

                    # Adjust the x, y coordinates by the downscale ration so that we get x,y coordinates
                    # on the original, unscaled image
                    actionX = int(actionX / self.config['model_image_downscale_ratio'])
                    actionY = int(actionY / self.config['model_image_downscale_ratio'])

                    # Here we create an action object using the lambdas. We aren't creating the final action object,
                    # since we do one last check below to ensure this isn't an exact repeat action.
                    potentialAction = self.actions[self.actionsSorted[actionType]](actionX, actionY)
                    potentialActionMaps = self.getActionMapsIntersectingWithAction(potentialAction, sampleActionMaps)

                    # If the network is predicting the same action as it did within the recent turns list, down to the exact pixels
                    # of the action maps, that usually implies its stuck and its action had no effect on the environment. Switch
                    # to random weighted to try and break out of this stuck condition. The recent action list is reset every
                    # time the algorithm discovers new code branches, e.g. new functionality so this helps ensure the algorithm
                    # stays exploring instead of getting stuck but can learn different behaviours with the same elements
                    for recentAction in sampleRecentActions:
                        recentActionMaps = self.getActionMapsIntersectingWithAction(recentAction, recentAction.actionMapsAvailable)

                        if recentAction.type != potentialAction.type:
                            continue

                        allEqual = True
                        for recentMap in recentActionMaps:
                            found = False
                            for potentialMap in potentialActionMaps:
                                if recentMap.doesOverlapWith(potentialMap, tolerancePixels=self.config['testing_repeat_action_pixel_overlap_tolerance']):
                                    found = True
                                    break
                            if not found:
                                allEqual = False
                                break

                        if allEqual:
                            weighted = True
                            override = True
                            break

                # Here we are doing a weighted random action.
                if weighted:
                    # Here, we prepare the array that will contain the probability for each of the possible actions.
                    # We use advantage values instead of the action probabilities here because the action probabilities
                    # tend to be very narrowly focused - assigning almost all of the probability to the same handful of
                    # pixels. The advantage values give provide a better way of weighting the various pixels.
                    reshaped = numpy.array(sampleAdvantageValues.data).reshape([len(self.actionsSorted) * height * width])
                    # Here we resize the array so that it adds up to 1.
                    reshapedSum = numpy.sum(reshaped)
                    if reshapedSum > 0:
                        reshaped = reshaped / reshapedSum

                    try:
                        # Choose the random action. What we get back is an index for that action in the original array
                        actionIndex = numpy.random.choice(range(len(self.actionsSorted) * height * width), p=reshaped)

                        # Now we do a clever trick here to recover the x,y coordinates of the action and the index
                        # of the action type
                        newProbs = numpy.zeros([len(self.actionsSorted) * height * width])
                        newProbs[actionIndex] = 1
                        newProbs = newProbs.reshape([len(self.actionsSorted), height * width])
                        actionType = newProbs.max(axis=1).argmax(axis=0)
                        newProbs = newProbs.reshape([len(self.actionsSorted), height, width])
                        actionY = newProbs[actionType].max(axis=1).argmax(axis=0)
                        actionX = newProbs[actionType, actionY].argmax(axis=0)

                        source = "weighted_random"
                        samplePredictedReward = sampleActionProbs[actionType, actionY, actionX].data.item()

                        # Adjust the x, y coordinates by the downscale ration so that we get x,y coordinates
                        # on the original, unscaled image
                        actionX = int(actionX / self.config['model_image_downscale_ratio'])
                        actionY = int(actionY / self.config['model_image_downscale_ratio'])

                    except ValueError:
                        # If there is any problem, just override back to a fully random action without predictions involved
                        print(datetime.now(), f"[{os.getpid()}]", "Error in weighted random choice! Probabilities do not all add up to 1. Picking a random action.", flush=True)
                        # This usually occurs when all the probabilities do not add up to 1, due to floating point error.
                        # So instead we just pick an action randomly.
                        actionX, actionY, actionType = self.getRandomAction(sampleActionRecentActionCounts, sampleActionMaps, samplePixelActionMap)
                        source = "random"
                        override = True

                # Here we take advantage of the lambda functions prepared in the constructor. We get the lambda function
                # that is associated with the action type we have selected, and use it to prepare the kwola.datamodels.actions.BaseAction
                # object.
                action = self.actions[self.actionsSorted[actionType]](actionX, actionY)
                action.source = source
                action.predictedReward = samplePredictedReward
                action.actionMapsAvailable = sampleActionMaps
                action.wasRepeatOverride = override
                # Here we append both the action and the original sample index. The original sample index
                # is later used to recover the original ordering of the actions list
                actions.append((sampleIndex, action))

        # The actions list now contained tuples of (sampleIndex, action). Now we just need to use
        # the sampleIndex to sort this list back into the original ordering of the samples.
        # This ensures that the actions we return as a result are in the exact same order as the
        # sub environments we received as input.
        sortedActions = sorted(actions, key=lambda row: row[0])

        # We return a list composed of just the action objects, one for each sub environment.
        return [action[1] for action in sortedActions]

    def getRandomAction(self, sampleActionRecentActionCounts, sampleActionMaps, pixelActionMap):
        """
        This function is used to decide on a totally random action. It uses a somewhat fancy mechanism that involves weighting
        all of the actions it has available to it based on their html element type. It also will down weight any actions that
        have been performed recently. Therefore algorithm is incentivized to explore just like the core neural network, its
        just done using heuristics.

        :param sampleActionRecentActionCounts: This is a list of integers, the same size as sampleActionMaps, that contains the count
                                               of the number of times that action map has been clicked on the recent actions list.
                                               Technically speaking, the recent actions list only contains actions since the last
                                               time the neural network discovered new functionality.
        :param sampleActionMaps: This is the list of action maps that are currently available for this environment.
        :param pixelActionMap: This is the pixel action map as prepared by the createPixelActionMap function.

        :return: A tuple containing three values of types (int, int, string). The first two integers are the x,y coordinates and the
                 string is what action should be performed at those coordinates.
        """
        # Setup the image width and height as function level variables for convenience
        width = pixelActionMap.shape[2]
        height = pixelActionMap.shape[1]

        # Here we are assigning a weight for each of the action maps available to the network to choose from.
        # The weights for the action maps are based on what type of HTML element that action map is representing.
        actionMapWeights = numpy.array([self.elementBaseWeights.get(map.elementType, self.elementBaseWeights['other']) for map in sampleActionMaps]) / (numpy.array(sampleActionRecentActionCounts) + 1)

        # Use the softmax function to rescale all the weights into probabilities, and then use those probabilities to
        # choose a random action map from the list.
        chosenActionMapIndex = numpy.random.choice(range(len(sampleActionMaps)), p=scipy.special.softmax(actionMapWeights))
        chosenActionMap = sampleActionMaps[chosenActionMapIndex]

        # Here we choose a random x,y coordinate from within the bounds of the action map.
        # We also have to compensate for the image downscaling here, since we need to lookup
        # the possible actions on the downscaled pixel action map
        actionX = random.randint(max(0, int(min(width - 1, chosenActionMap.left * self.config['model_image_downscale_ratio']))),
                                 max(0, int(min(chosenActionMap.right * self.config['model_image_downscale_ratio'] - 1, width - 1))))
        actionY = random.randint(max(0, int(min(height - 1, chosenActionMap.top * self.config['model_image_downscale_ratio']))),
                                 max(0, int(min(chosenActionMap.bottom * self.config['model_image_downscale_ratio'] - 1, height - 1))))

        # Get the list of actions that are allowed at the chosen coordinates
        possibleActionsAtPixel = pixelActionMap[:, actionY, actionX]
        possibleActionIndexes = [actionIndex for actionIndex in range(len(self.actionsSorted)) if possibleActionsAtPixel[actionIndex]]
        # Create a list containing a weight for each of the possible actions.
        # The base weights are set in the configuration file and help bias the algroithm towards clicking and away
        # from typing, since there are significantly more typing actions then clicking ones
        possibleActionWeights = [self.actionBaseWeights[actionIndex] for actionIndex in possibleActionIndexes]

        # Now for each of the possible actions, we have to determine if there were any detected keywords
        # on the html element that would warrant this action getting a higher probability for this element.
        # This is used to help the random action selection bias towards performing the correct actions.
        # e.g. this is used to increase the likelihood of typing in the password in a field that looks
        # like it should receive the password, or the likelihood of typing in the email on an email input.
        # Its not that we don't want to try other inputs on those boxes, it just helps if its more common
        # to do it correctly, as it increases the likelihood of finding subsequent code paths
        possibleActionBoosts = []
        for actionIndex in possibleActionIndexes:
            boostKeywords = self.actionProbabilityBoostKeywords[actionIndex]

            boost = False
            for keyword in boostKeywords:
                if keyword in chosenActionMap.keywords:
                    boost = True
                    break

            if boost:
                possibleActionBoosts.append(1.5)
            else:
                possibleActionBoosts.append(1.0)

        # Now we run the weights for each of the actions through a softmax and use numpy to make a final decision on
        # what action we should perform at the given pixel
        try:
            actionType = numpy.random.choice(possibleActionIndexes, p=scipy.special.softmax(numpy.array(possibleActionWeights) * numpy.array(possibleActionBoosts)))
        except ValueError:
            # If something went wrong, well this is very unusual. So we choose a totally random action at large.
            actionType = random.choice(range(len(self.actionsSorted)))

        return int(actionX / self.config['model_image_downscale_ratio']), int(actionY / self.config['model_image_downscale_ratio']), actionType

    @staticmethod
    def computePresentRewards(executionTraces, config):
        """
            This method is used to compute the present reward values for a list of execution traces

            :param executionTraces: This is a list of ExecutionTrace objects. They should all be from the same ExecutionSession and should be in order.
            :param config: An instance of the global configuration object.
            :return: A list of float values. The list will be the same length as the list of executionTraces passed in, and will contain a single reward
                     value for each execution trace.
        """

        # We iterate the execution traces and calculate a present reward value for each one
        presentRewards = []
        for trace in executionTraces:
            # The present reward starts at 0, and depends entirely
            # on the features of the execution trace. There is a reward
            # attached to each of the features on the execution trace
            # object, which all get summed together here.
            tracePresentReward = 0.0

            if trace.didActionSucceed:
                tracePresentReward += config['reward_action_success']
            else:
                tracePresentReward += config['reward_action_failure']

            if trace.didCodeExecute:
                tracePresentReward += config['reward_code_executed']
            else:
                tracePresentReward += config['reward_no_code_executed']

            if trace.didNewBranchesExecute:
                tracePresentReward += config['reward_new_code_executed']
            else:
                tracePresentReward += config['reward_no_new_code_executed']

            if trace.hadNetworkTraffic:
                tracePresentReward += config['reward_network_traffic']
            else:
                tracePresentReward += config['reward_no_network_traffic']

            if trace.hadNewNetworkTraffic:
                tracePresentReward += config['reward_new_network_traffic']
            else:
                tracePresentReward += config['reward_no_new_network_traffic']

            if trace.didScreenshotChange:
                tracePresentReward += config['reward_screenshot_changed']
            else:
                tracePresentReward += config['reward_no_screenshot_change']

            if trace.isScreenshotNew:
                tracePresentReward += config['reward_new_screenshot']
            else:
                tracePresentReward += config['reward_no_new_screenshot']

            if trace.didURLChange:
                tracePresentReward += config['reward_url_changed']
            else:
                tracePresentReward += config['reward_no_url_change']

            if trace.isURLNew:
                tracePresentReward += config['reward_new_url']
            else:
                tracePresentReward += config['reward_no_new_url']

            if trace.hadLogOutput:
                tracePresentReward += config['reward_log_output']
            else:
                tracePresentReward += config['reward_no_log_output']

            presentRewards.append(tracePresentReward)
        return presentRewards

    @staticmethod
    def computeDiscountedFutureRewards(executionTraces, config):
        """
            This method is used to compute the discounted future rewards for a sequence of execution traces.
            These are basically the sum of the present rewards except discounted backwards in time by the
            discount rate given in the configuration file. Although not technically used as labels for training,
            these are used in the debug videos just to illustrate what you would expect the predicted discunted
            reward value to be at each time step.

            :param executionTraces: This is a list of ExecutionTrace objects. They should all be from the same ExecutionSession and should be in order.
            :param config: An instance of the global configuration object.
            :return: A list of float values. The list will be the same length as the list of executionTraces passed in, and will contain a single
                     discounted future reward value for each execution trace.
        """

        # First compute the present reward at each time step
        presentRewards = DeepLearningAgent.computePresentRewards(executionTraces, config)

        # Now compute the discounted reward
        # The easiest way to do this is to just reverse
        # the list, since we are computing values that are
        # based on the future of each trace. On this reversed
        # list, we compute what is effectively an exponential
        # decaying sum - constantly decaying whats already
        # there and adding in the present reward for the frames
        # moving forward.
        discountedFutureRewards = []
        presentRewards.reverse()
        current = 0
        for reward in presentRewards:
            current *= config['reward_discount_rate']
            discountedFutureRewards.append(current)
            current += reward

        # We just have to reverse the list again once we are doing and
        # we get a result in the same order as the original list.
        discountedFutureRewards.reverse()

        return discountedFutureRewards

    @staticmethod
    def readVideoFrames(videoFilePath):
        """
        This method reads a given video file into a numpy array of images that can then be further manipulated

        :param videoFilePath: This is a path to the video file on the local hard drive.
        :return: A list containing numpy arrays, a single numpy array for each frame in the video.
        """

        # Use OpenCV to read the video file. The extra big dependency here is annoying but I haven't dug deeply
        # into other libraries that can load videos into numpy arrays
        cap = cv2.VideoCapture(videoFilePath)

        rawImages = []

        # We basically just keep looping until we have loaded all of the frames.
        while (cap.isOpened()):
            ret, rawImage = cap.read()
            if ret:
                # OpenCV reads everything in BGR format for some reason so flip to RGB
                rawImage = numpy.flip(rawImage, axis=2)
                rawImages.append(rawImage)
            else:
                break

        # Return the list of frames
        return rawImages

    def getActionInfoTensorsFromRewardMap(self, rewardMapTensor):
        """
        This method is used to take a reward image and return back the x,y coordinates and the action type of the highest
        reward position within that reward image.

        :param rewardMapTensor: A pytorch Tensor in the shape [# of actions, height, width]

        :return: A tuple containing three values, (x, y, actionTypeIndex), all of them being pytorch scalar tensors.
        """
        width = rewardMapTensor.shape[2]
        height = rewardMapTensor.shape[1]

        # Basically, what we are doing here is determining the indices of the maximum value one dimension at a time
        # This code is meant to work no matter what ordering the underlying tensors are being stored in, which I'm
        # unsure of. So instead, we just reshape to the first dimension, use argmax to determine the max value there,
        # then home into the subsequent dimensions and repeat the process to determine all of the indices leading to
        # the maximum value in the whole tensor.
        actionType = rewardMapTensor.reshape([len(self.actionsSorted), width * height]).max(dim=1)[0].argmax(0)
        actionY = rewardMapTensor[actionType].max(dim=1)[0].argmax(0)
        actionX = rewardMapTensor[actionType, actionY].argmax(0)

        return actionX, actionY, actionType

    def createDebugVideoForExecutionSession(self, executionSession, includeNeuralNetworkCharts=True, includeNetPresentRewardChart=True, hilightStepNumber=None):
        """
            This method is used to generate a debug video for the given execution session.

            :param executionSession: This is an ExecutionSession object.
            :param includeNeuralNetworkCharts: Whether or not the debug video should contain neural network charts on the right hand side.
                                               These are optional because they require significantly more CPU power to generate.
            :param includeNetPresentRewardChart: Whether or not there should be a net present reward chart at the bottom of the deug videos.
                                                These can also require more CPU and memory to generate so they are strictly optional.
            :param hilightStepNumber: If there is a specific frame within the video that should be hilighted, (e.g. if you are generating this debug video
                                      for a bug), then this should be the frame index. A value of None indicates that no frame should be hilighted.
            :return: Nothing is returned from this function.
        """
        videoPath = self.config.getKwolaUserDataDirectory("videos")

        rawImages = DeepLearningAgent.readVideoFrames(os.path.join(videoPath, f"{str(executionSession.id)}.mp4"))

        executionTraces = [ExecutionTrace.loadFromDisk(traceId, self.config) for traceId in executionSession.executionTraces]

        # Filter out any traces that failed to load. Generally this only happens when you interrupt the process
        # while it is writing a file. So it happens to devs but not in production. Still we protect against
        # this case in several places throughout the code.
        executionTracesFiltered = [trace for trace in executionTraces if trace is not None]

        presentRewards = DeepLearningAgent.computePresentRewards(executionTracesFiltered, self.config)

        discountedFutureRewards = DeepLearningAgent.computeDiscountedFutureRewards(executionTracesFiltered, self.config)

        tempScreenshotDirectory = tempfile.mkdtemp()

        debugImageIndex = 0

        lastRawImage = rawImages.pop(0)

        mpl.use('Agg')
        mpl.rcParams['figure.max_open_warning'] = 1000

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for trace, rawImage in zip(executionTraces, rawImages):
                if trace is not None:
                    hilight = False
                    if hilightStepNumber is not None and hilightStepNumber == (trace.frameNumber - 1):
                        hilight = True

                    future = executor.submit(self.createDebugImagesForExecutionTrace, str(executionSession.id), debugImageIndex, trace.to_json(), rawImage, lastRawImage, presentRewards, discountedFutureRewards, tempScreenshotDirectory, includeNeuralNetworkCharts, includeNetPresentRewardChart, hilight)
                    futures.append(future)

                    debugImageIndex += 2
                    lastRawImage = rawImage

            concurrent.futures.wait(futures)

        subprocess.run(['ffmpeg', '-f', 'image2', "-r", "2", '-i', 'kwola-screenshot-%05d.png', '-vcodec', 'libx264', '-crf', '15', "debug.mp4"], cwd=tempScreenshotDirectory, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        moviePath = os.path.join(tempScreenshotDirectory, "debug.mp4")

        with open(moviePath, "rb") as file:
            videoData = file.read()

        shutil.rmtree(tempScreenshotDirectory)

        return videoData

    def createDebugImagesForExecutionTrace(self, executionSessionId, debugImageIndex, trace, rawImage, lastRawImage, presentRewards, discountedFutureRewards, tempScreenshotDirectory, includeNeuralNetworkCharts=True, includeNetPresentRewardChart=True, hilight=False):
        """
            This method is used to generate a single debug image for a single execution trace. Technically this method actually
            generates two debug images. The first shows what action is being performed, and the second shows what happened after
            that action. Its designed so they can all be strung together into a movie.

            :param executionSessionId: A string containing the ID of the execution session
            :param debugImageIndex: The index of this debug image within the movie
            :param trace: The kwola.datamodels.ExecutionTrace object that this debug image will be generated for
            :param rawImage: A numpy array containing the raw image data for the result image on this trace
            :param lastRawImage: A numpy array containing the raw image data for the input image on this trace, e.g. the image from before the action was performed.
            :param presentRewards: A list containing all of the calculated present reward values for the sequence
            :param discountedFutureRewards: A list containing all of the calculated discounted future reward values for the sequence
            :param tempScreenshotDirectory: A string containing the path of the directory where the images will be saved
            :param includeNeuralNetworkCharts: A boolean indicating whether to include charts showing the neural network predictions in the debug video
            :param includeNetPresentRewardChart: A boolean indicating whether to include the net present reward chart at the bottom of the debug video
            :param hilight: A boolean indicating whether this frame should be hilightted or not. Hilighting a frame will change the background color

            :return: None
        """
        try:
            trace = ExecutionTrace.from_json(trace)

            topSize = self.config.debug_video_top_size
            bottomSize = self.config.debug_video_bottom_size
            leftSize = self.config.debug_video_left_size
            rightSize = self.config.debug_video_right_size
            if includeNeuralNetworkCharts:
                rightSize += self.config.debug_video_neural_network_chart_right_size_addition

                if len(self.actionsSorted) > 4:
                    rightSize += self.config.debug_video_neural_network_chart_right_size_addition_per_four_actions
                if len(self.actionsSorted) > 8:
                    rightSize += self.config.debug_video_neural_network_chart_right_size_addition_per_four_actions
                if len(self.actionsSorted) > 12:
                    rightSize += self.config.debug_video_neural_network_chart_right_size_addition_per_four_actions

            if includeNetPresentRewardChart:
                bottomSize += self.config.debug_video_bottom_reward_chart_height

            chartDPI = self.config.debug_video_chart_dpi

            debugVideoImageChannels = 3

            imageHeight = rawImage.shape[0]
            imageWidth = rawImage.shape[1]

            presentReward = presentRewards[trace.frameNumber - 1]
            discountedFutureReward = discountedFutureRewards[trace.frameNumber - 1]

            def addDebugCircleToImage(image, trace):
                targetCircleCoords1 = skimage.draw.circle_perimeter(int(topSize + trace.actionPerformed.y),
                                                                           int(leftSize + trace.actionPerformed.x), self.config.debug_video_target_circle_1_radius,
                                                                           shape=[int(imageWidth + extraWidth),
                                                                                  int(imageHeight + extraHeight)])
                targetCircleCoords2 = skimage.draw.circle_perimeter(int(topSize + trace.actionPerformed.y),
                                                                           int(leftSize + trace.actionPerformed.x), self.config.debug_video_target_circle_2_radius,
                                                                           shape=[int(imageWidth + extraWidth),
                                                                                  int(imageHeight + extraHeight)])
                targetCircleCoords3 = skimage.draw.circle_perimeter(int(topSize + trace.actionPerformed.y),
                                                                           int(leftSize + trace.actionPerformed.x), self.config.debug_video_target_circle_3_radius,
                                                                           shape=[int(imageWidth + extraWidth),
                                                                                  int(imageHeight + extraHeight)])
                targetCircleCoords4 = skimage.draw.circle_perimeter(int(topSize + trace.actionPerformed.y),
                                                                          int(leftSize + trace.actionPerformed.x), self.config.debug_video_target_circle_4_radius,
                                                                          shape=[int(imageWidth + extraWidth),
                                                                                 int(imageHeight + extraHeight)])
                image[targetCircleCoords1] = [self.config.debug_video_target_circle_color_r, self.config.debug_video_target_circle_color_g, self.config.debug_video_target_circle_color_b]
                image[targetCircleCoords2] = [self.config.debug_video_target_circle_color_r, self.config.debug_video_target_circle_color_g, self.config.debug_video_target_circle_color_b]
                image[targetCircleCoords3] = [self.config.debug_video_target_circle_color_r, self.config.debug_video_target_circle_color_g, self.config.debug_video_target_circle_color_b]
                image[targetCircleCoords4] = [self.config.debug_video_target_circle_color_r, self.config.debug_video_target_circle_color_g, self.config.debug_video_target_circle_color_b]

            def addCropViewToImage(image, trace):
                imageCropWidth = imageWidth * self.config['model_image_downscale_ratio']
                imageCropHeight = imageHeight * self.config['model_image_downscale_ratio']

                actionCropX = trace.actionPerformed.x * self.config['model_image_downscale_ratio']
                actionCropY = trace.actionPerformed.y * self.config['model_image_downscale_ratio']

                cropLeft, cropTop, cropRight, cropBottom = self.calculateTrainingCropPosition(actionCropX, actionCropY, imageCropWidth, imageCropHeight)

                cropLeft = int(cropLeft / self.config['model_image_downscale_ratio'])
                cropTop = int(cropTop / self.config['model_image_downscale_ratio'])
                cropRight = int(cropRight / self.config['model_image_downscale_ratio'])
                cropBottom = int(cropBottom / self.config['model_image_downscale_ratio'])

                cropRectangle = skimage.draw.rectangle_perimeter((int(topSize + cropTop), int(leftSize + cropLeft)), (int(topSize + cropBottom), int(leftSize + cropRight)))
                image[cropRectangle] = [self.config.debug_video_crop_box_color_r, self.config.debug_video_crop_box_color_g, self.config.debug_video_crop_box_color_b]

            def addDebugTextToImage(image, trace):
                fontSize = self.config.debug_video_text_font_size
                fontThickness = int(self.config.debug_video_text_thickness)
                fontColor = (0, 0, 0)

                topMargin = self.config.debug_video_text_top_margin

                columnOneLeft = leftSize
                columnTwoLeft = leftSize + self.config.debug_video_text_column_two_left
                columnThreeLeft = leftSize + self.config.debug_video_text_column_three_left
                lineOneTop = topMargin + self.config.debug_video_text_line_height
                lineTwoTop = topMargin + self.config.debug_video_text_line_height * 2
                lineThreeTop = topMargin + self.config.debug_video_text_line_height * 3
                lineFourTop = topMargin + self.config.debug_video_text_line_height * 4
                lineFiveTop = topMargin + self.config.debug_video_text_line_height * 5
                lineSixTop = topMargin + self.config.debug_video_text_line_height * 6
                lineSevenTop = topMargin + self.config.debug_video_text_line_height * 7
                lineEightTop = topMargin + self.config.debug_video_text_line_height * 8
                lineNineTop = topMargin + self.config.debug_video_text_line_height * 9

                font = cv2.FONT_HERSHEY_SIMPLEX
                antiAliasingMode = cv2.LINE_AA

                cv2.putText(image, f"URL {trace.startURL}", (columnOneLeft, lineOneTop), font, fontSize,
                            fontColor, fontThickness, antiAliasingMode)

                cv2.putText(image, f"{str(executionSessionId)}", (columnOneLeft, lineTwoTop), font,
                            fontSize, fontColor, fontThickness, antiAliasingMode)
                cv2.putText(image, f"Frame {trace.frameNumber}", (columnOneLeft, lineThreeTop), font,
                            fontSize, fontColor, fontThickness, antiAliasingMode)
                cv2.putText(image,
                            f"Action {trace.actionPerformed.type} at {trace.actionPerformed.x},{trace.actionPerformed.y}",
                            (columnOneLeft, lineFourTop), font, fontSize, fontColor, fontThickness,
                            antiAliasingMode)
                cv2.putText(image, f"Source: {str(trace.actionPerformed.source)}", (columnOneLeft, lineFiveTop),
                            font, fontSize, fontColor, fontThickness, antiAliasingMode)

                cv2.putText(image, f"Succeed: {str(trace.didActionSucceed)}", (columnOneLeft, lineSixTop),
                            font, fontSize, fontColor, fontThickness, antiAliasingMode)
                cv2.putText(image, f"Error: {str(trace.didErrorOccur)}", (columnOneLeft, lineSevenTop),
                            font, fontSize, fontColor, fontThickness, antiAliasingMode)
                cv2.putText(image, f"New Error: {str(trace.didNewErrorOccur)}", (columnOneLeft, lineEightTop),
                            font, fontSize, fontColor, fontThickness, antiAliasingMode)
                cv2.putText(image, f"Override: {str(trace.actionPerformed.wasRepeatOverride)}", (columnOneLeft, lineNineTop),
                            font, fontSize, fontColor, fontThickness, antiAliasingMode)

                cv2.putText(image, f"Code Execute: {str(trace.didCodeExecute)}", (columnTwoLeft, lineTwoTop),
                            font, fontSize, fontColor, fontThickness, antiAliasingMode)
                cv2.putText(image, f"New Branches: {str(trace.didNewBranchesExecute)}", (columnTwoLeft, lineThreeTop),
                            font, fontSize, fontColor, fontThickness, antiAliasingMode)

                cv2.putText(image, f"Network Traffic: {str(trace.hadNetworkTraffic)}", (columnTwoLeft, lineFourTop),
                            font, fontSize, fontColor, fontThickness, antiAliasingMode)
                cv2.putText(image, f"New Network Traffic: {str(trace.hadNewNetworkTraffic)}", (columnTwoLeft, lineFiveTop),
                            font, fontSize, fontColor, fontThickness, antiAliasingMode)

                cv2.putText(image, f"Screenshot Change: {str(trace.didScreenshotChange)}", (columnTwoLeft, lineSixTop),
                            font, fontSize, fontColor, fontThickness, antiAliasingMode)
                cv2.putText(image, f"New Screenshot: {str(trace.isScreenshotNew)}", (columnTwoLeft, lineSevenTop),
                            font, fontSize, fontColor, fontThickness, antiAliasingMode)

                cv2.putText(image, f"Cursor: {str(trace.cursor)}", (columnTwoLeft, lineEightTop), font,
                            fontSize, fontColor, fontThickness, antiAliasingMode)

                cv2.putText(image, f"Discounted Future Reward: {(discountedFutureReward):.3f}",
                            (columnThreeLeft, lineTwoTop), font, fontSize, fontColor, fontThickness,
                            antiAliasingMode)
                cv2.putText(image, f"Present Reward: {(presentReward):.3f}", (columnThreeLeft, lineThreeTop),
                            font, fontSize, fontColor, fontThickness, antiAliasingMode)

                cv2.putText(image, f"Branch Coverage: {(trace.cumulativeBranchCoverage * 100):.2f}%",
                            (columnThreeLeft, lineFourTop), font, fontSize, fontColor, fontThickness,
                            antiAliasingMode)

                cv2.putText(image, f"URL Change: {str(trace.didURLChange)}", (columnThreeLeft, lineFiveTop),
                            font, fontSize, fontColor, fontThickness, antiAliasingMode)
                cv2.putText(image, f"New URL: {str(trace.isURLNew)}", (columnThreeLeft, lineSixTop),
                            font, fontSize, fontColor, fontThickness, antiAliasingMode)

                cv2.putText(image, f"Had Log Output: {trace.hadLogOutput}", (columnThreeLeft, lineSevenTop),
                            font, fontSize, fontColor, fontThickness, antiAliasingMode)

                if trace.actionPerformed.predictedReward:
                    cv2.putText(image, f"Predicted Reward: {(trace.actionPerformed.predictedReward):.3f}", (columnThreeLeft, lineEightTop),
                                font, fontSize, fontColor, fontThickness, antiAliasingMode)

            def addBottomRewardChartToImage(image, trace):
                rewardChartFigure = plt.figure(figsize=(imageWidth / chartDPI, (bottomSize - self.config.debug_video_bottom_reward_chart_bottom_margin) / chartDPI), dpi=chartDPI)
                rewardChartAxes = rewardChartFigure.add_subplot(111)

                xCoords = numpy.array(range(len(presentRewards)))

                rewardChartAxes.set_ylim(ymin=self.config.debug_video_reward_min_value, ymax=self.config.debug_video_reward_max_value)

                rewardChartAxes.plot(xCoords, numpy.array(presentRewards) + numpy.array(discountedFutureRewards))

                rewardChartAxes.set_xticks(range(0, len(presentRewards), self.config.debug_video_bottom_reward_chart_x_tick_spacing))
                rewardChartAxes.set_xticklabels([str(n) for n in range(0, len(presentRewards), self.config.debug_video_bottom_reward_chart_x_tick_spacing)])
                rewardChartAxes.set_yticks([self.config.debug_video_reward_min_value, self.config.debug_video_reward_max_value])
                rewardChartAxes.set_yticklabels([f"{self.config.debug_video_reward_min_value:.2f}", f"{self.config.debug_video_reward_max_value:.2f}"])
                rewardChartAxes.set_title("Net Present Reward")
                rewardChartFigure.tight_layout()

                rewardChartHalfFrames = int(self.config['debug_video_bottom_reward_chart_frames'] / 2)

                rewardChartAxes.set_xlim(xmin=trace.frameNumber - rewardChartHalfFrames, xmax=trace.frameNumber + rewardChartHalfFrames)
                vline = rewardChartAxes.axvline(trace.frameNumber - 1, color='black', linewidth=self.config.debug_video_bottom_reward_chart_current_frame_line_width)
                hline = rewardChartAxes.axhline(0, color='grey', linewidth=1)

                # If we haven't already shown or saved the plot, then we need to
                # draw the figure first...
                rewardChartFigure.canvas.draw()

                # Now we can save it to a numpy array.
                rewardChart = numpy.fromstring(rewardChartFigure.canvas.tostring_rgb(), dtype=numpy.uint8, sep='')
                rewardChart = rewardChart.reshape(rewardChartFigure.canvas.get_width_height()[::-1] + (debugVideoImageChannels,))

                image[topSize + imageHeight:-self.config.debug_video_bottom_reward_chart_bottom_margin, leftSize:-rightSize] = rewardChart

                vline.remove()
                hline.remove()
                plt.close(rewardChartFigure)

            def addRightSideDebugCharts(plotImage, rawImage, trace):
                chartTopMargin = self.config.debug_video_neural_network_chart_top_margin

                neededFigures = 3 + len(self.actionsSorted) * 3

                squareSize = int(numpy.sqrt(neededFigures))

                if (squareSize * squareSize) < neededFigures:
                    squareSize += 1

                numColumns = squareSize
                numRows = squareSize

                mainColorMap = plt.get_cmap('inferno')
                greyColorMap = plt.get_cmap('gray')

                currentFig = 1

                mainFigure = plt.figure(
                    figsize=((rightSize) / chartDPI, (imageHeight + bottomSize + topSize - chartTopMargin) / chartDPI), dpi=chartDPI)

                rewardPredictionAxes = [
                    mainFigure.add_subplot(numColumns, numRows, actionIndex + currentFig)
                    for actionIndex, action in enumerate(self.actionsSorted)
                ]

                currentFig += len(rewardPredictionAxes)

                advantagePredictionAxes = [
                    mainFigure.add_subplot(numColumns, numRows, actionIndex + currentFig)
                    for actionIndex, action in enumerate(self.actionsSorted)
                ]

                currentFig += len(advantagePredictionAxes)

                actionProbabilityPredictionAxes = [
                    mainFigure.add_subplot(numColumns, numRows, actionIndex + currentFig)
                    for actionIndex, action in enumerate(self.actionsSorted)
                ]

                currentFig += len(actionProbabilityPredictionAxes)

                stampAxes = mainFigure.add_subplot(numColumns, numRows, currentFig)
                currentFig += 1

                stateValueAxes = mainFigure.add_subplot(numColumns, numRows, currentFig)
                currentFig += 1

                processedImage = DeepLearningAgent.processRawImageParallel(rawImage, self.config)

                rewardPixelMaskAxes = mainFigure.add_subplot(numColumns, numRows, currentFig)
                currentFig += 1
                rewardPixelMask = self.createRewardPixelMask(processedImage,
                                                             int(trace.actionPerformed.x * self.config['model_image_downscale_ratio']),
                                                             int(trace.actionPerformed.y * self.config['model_image_downscale_ratio'])
                                                             )
                rewardPixelCount = numpy.count_nonzero(rewardPixelMask)
                rewardPixelMaskAxes.imshow(rewardPixelMask, vmin=0, vmax=1, cmap=plt.get_cmap("gray"), interpolation="bilinear")
                rewardPixelMaskAxes.set_xticks([])
                rewardPixelMaskAxes.set_yticks([])
                rewardPixelMaskAxes.set_title(f"{rewardPixelCount} target pixels")

                # pixelActionMapAxes = mainFigure.add_subplot(numColumns, numRows, currentFig)
                # currentFig += 1
                pixelActionMap = self.createPixelActionMap(trace.actionPerformed.actionMapsAvailable, processedImage.shape[1], processedImage.shape[2])
                # actionPixelCount = numpy.count_nonzero(pixelActionMap)
                # pixelActionMapAxes.imshow(numpy.swapaxes(numpy.swapaxes(pixelActionMap, 0, 1), 1, 2) * 255, interpolation="bilinear")
                # pixelActionMapAxes.set_xticks([])
                # pixelActionMapAxes.set_yticks([])
                # pixelActionMapAxes.set_title(f"{actionPixelCount} action pixels")

                additionalFeature = self.prepareAdditionalFeaturesForTrace(trace)

                with torch.no_grad():
                    self.model.eval()
                    outputs = \
                        self.modelParallel({"image": self.variableWrapperFunc(torch.FloatTensor, numpy.array([processedImage])),
                                            "additionalFeature": self.variableWrapperFunc(torch.FloatTensor, [additionalFeature]),
                                            "pixelActionMaps": self.variableWrapperFunc(torch.FloatTensor, numpy.array([pixelActionMap])),
                                            "stepNumber": self.variableWrapperFunc(torch.FloatTensor, numpy.array([trace.frameNumber - 1])),
                                            "outputStamp": True,
                                            "computeExtras": False,
                                            "computeRewards": True,
                                            "computeActionProbabilities": True,
                                            "computeStateValues": True,
                                            "computeAdvantageValues": True
                                            })

                    totalRewardPredictions = numpy.array((outputs['presentRewards'] + outputs['discountFutureRewards']).data)
                    advantagePredictions = numpy.array((outputs['advantage']).data)
                    actionProbabilities = numpy.array((outputs['actionProbabilities']).data)
                    stateValue = numpy.array((outputs['stateValues'][0]).data)
                    stamp = outputs['stamp']

                for actionIndex, action in enumerate(self.actionsSorted):
                    actionY = actionProbabilities[0][actionIndex].max(axis=1).argmax(axis=0)
                    actionX = actionProbabilities[0][actionIndex, actionY].argmax(axis=0)

                    actionX = int(actionX / self.config["model_image_downscale_ratio"])
                    actionY = int(actionY / self.config["model_image_downscale_ratio"])

                    targetCircleCoords1 = skimage.draw.circle_perimeter(int(topSize + actionY),
                                                                               int(leftSize + actionX), self.config.debug_video_action_prediction_circle_1_radius,
                                                                               shape=[int(imageWidth + extraWidth),
                                                                                      int(imageHeight + extraHeight)])

                    targetCircleCoords2 = skimage.draw.circle_perimeter(int(topSize + actionY),
                                                                              int(leftSize + actionX), self.config.debug_video_action_prediction_circle_2_radius,
                                                                              shape=[int(imageWidth + extraWidth),
                                                                                     int(imageHeight + extraHeight)])
                    plotImage[targetCircleCoords1] = [self.config.debug_video_action_prediction_circle_color_r,
                                                             self.config.debug_video_action_prediction_circle_color_g,
                                                             self.config.debug_video_action_prediction_circle_color_b]
                    plotImage[targetCircleCoords2] = [self.config.debug_video_action_prediction_circle_color_r,
                                                             self.config.debug_video_action_prediction_circle_color_g,
                                                             self.config.debug_video_action_prediction_circle_color_b]

                for actionIndex, action in enumerate(self.actionsSorted):
                    maxValue = numpy.max(numpy.array(totalRewardPredictions[0][actionIndex]))

                    rewardPredictionAxes[actionIndex].set_xticks([])
                    rewardPredictionAxes[actionIndex].set_yticks([])

                    rewardPredictionsShrunk = skimage.measure.block_reduce(totalRewardPredictions[0][actionIndex], (squareSize, squareSize), numpy.max)

                    im = rewardPredictionAxes[actionIndex].imshow(rewardPredictionsShrunk, cmap=mainColorMap, interpolation="nearest", vmin=self.config.debug_video_reward_min_value, vmax=self.config.debug_video_reward_max_value)
                    rewardPredictionAxes[actionIndex].set_title(f"{action} {maxValue:.2f} reward")
                    mainFigure.colorbar(im, ax=rewardPredictionAxes[actionIndex], orientation='vertical')

                for actionIndex, action in enumerate(self.actionsSorted):
                    maxValue = numpy.max(numpy.array(advantagePredictions[0][actionIndex]))

                    advantagePredictionAxes[actionIndex].set_xticks([])
                    advantagePredictionAxes[actionIndex].set_yticks([])

                    advanagePredictionsShrunk = skimage.measure.block_reduce(advantagePredictions[0][actionIndex], (squareSize, squareSize), numpy.max)

                    im = advantagePredictionAxes[actionIndex].imshow(advanagePredictionsShrunk, cmap=mainColorMap, interpolation="nearest", vmin=self.config.debug_video_advantage_min_value, vmax=self.config.debug_video_advantage_max_value)
                    advantagePredictionAxes[actionIndex].set_title(f"{action} {maxValue:.2f} advantage")
                    mainFigure.colorbar(im, ax=advantagePredictionAxes[actionIndex], orientation='vertical')

                for actionIndex, action in enumerate(self.actionsSorted):
                    maxValue = numpy.max(numpy.array(actionProbabilities[0][actionIndex]))

                    actionProbabilityPredictionAxes[actionIndex].set_xticks([])
                    actionProbabilityPredictionAxes[actionIndex].set_yticks([])

                    actionProbabilityPredictionsShrunk = skimage.measure.block_reduce(actionProbabilities[0][actionIndex], (squareSize, squareSize), numpy.max)

                    im = actionProbabilityPredictionAxes[actionIndex].imshow(actionProbabilityPredictionsShrunk, cmap=mainColorMap, interpolation="nearest")
                    actionProbabilityPredictionAxes[actionIndex].set_title(f"{action} {maxValue:.3f} prob")
                    mainFigure.colorbar(im, ax=actionProbabilityPredictionAxes[actionIndex], orientation='vertical')

                stampAxes.set_xticks([])
                stampAxes.set_yticks([])
                stampImageWidth = self.config['additional_features_stamp_edge_size'] * self.config['additional_features_stamp_edge_size']
                stampImageHeight = self.config['additional_features_stamp_depth_size']

                stampIm = stampAxes.imshow(numpy.array(stamp.data[0]).reshape([stampImageWidth, stampImageHeight]), cmap=greyColorMap, interpolation="nearest", vmin=-2.5, vmax=2.5)
                mainFigure.colorbar(stampIm, ax=stampAxes, orientation='vertical')
                stampAxes.set_title("Memory Stamp")

                stateValueAxes.set_xticks([])
                stateValueAxes.set_yticks([])
                stateValueIm = stateValueAxes.imshow([stateValue], cmap=mainColorMap, interpolation="nearest", vmin=self.config.debug_video_reward_min_value, vmax=self.config.debug_video_reward_max_value)
                mainFigure.colorbar(stateValueIm, ax=stateValueAxes, orientation='vertical')
                stateValueAxes.set_title(f"State Value {float(stateValue[0]):.3f}")

                # ax.grid()
                mainFigure.tight_layout()
                mainFigure.canvas.draw()

                # Now we can save it to a numpy array and paste it into the image
                mainChart = numpy.fromstring(mainFigure.canvas.tostring_rgb(), dtype=numpy.uint8, sep='')
                mainChart = mainChart.reshape(mainFigure.canvas.get_width_height()[::-1] + (debugVideoImageChannels,))
                plotImage[chartTopMargin:, (-rightSize):] = mainChart
                plt.close(mainFigure)

            extraWidth = leftSize + rightSize
            extraHeight = topSize + bottomSize

            newImage = numpy.ones([imageHeight + extraHeight, imageWidth + extraWidth, debugVideoImageChannels]) * 255
            if hilight:
                newImage[:, :] = numpy.array([self.config.debug_video_hilight_background_color_r, self.config.debug_video_hilight_background_color_g, self.config.debug_video_hilight_background_color_b])
            newImage[topSize:-bottomSize, leftSize:-rightSize] = lastRawImage
            addDebugTextToImage(newImage, trace)
            addDebugCircleToImage(newImage, trace)
            addCropViewToImage(newImage, trace)
            if includeNetPresentRewardChart:
                addBottomRewardChartToImage(newImage, trace)
            if includeNeuralNetworkCharts:
                addRightSideDebugCharts(newImage, lastRawImage, trace)

            fileName = f"kwola-screenshot-{debugImageIndex:05d}.png"
            filePath = os.path.join(tempScreenshotDirectory, fileName)
            skimage.io.imsave(filePath, numpy.array(newImage, dtype=numpy.uint8))

            newImage = numpy.ones([imageHeight + extraHeight, imageWidth + extraWidth, debugVideoImageChannels]) * 255
            if hilight:
                newImage[:, :] = numpy.array([self.config.debug_video_hilight_background_color_r, self.config.debug_video_hilight_background_color_g, self.config.debug_video_hilight_background_color_b])
            addDebugTextToImage(newImage, trace)

            newImage[topSize:-bottomSize, leftSize:-rightSize] = rawImage
            if includeNetPresentRewardChart:
                addBottomRewardChartToImage(newImage, trace)
            if includeNeuralNetworkCharts:
                addRightSideDebugCharts(newImage, lastRawImage, trace)

            fileName = f"kwola-screenshot-{debugImageIndex + 1:05d}.png"
            filePath = os.path.join(tempScreenshotDirectory, fileName)
            skimage.io.imsave(filePath, numpy.array(newImage, dtype=numpy.uint8))

            print(datetime.now(), f"[{os.getpid()}]", "Completed debug image", fileName, flush=True)
        except Exception:
            traceback.print_exc()
            print(datetime.now(), f"[{os.getpid()}]", "Failed to create debug image!", flush=True)

    def createRewardPixelMask(self, processedImage, x, y):
        """
            This method takes a processed image, and returns a new mask array the same size of the image.
            This new array will show all of the pixels that should be rewarded for an action performed
            at the given x,y coordinates. Basically we use flood-fill to ensure that all nearby pixels
            to the action coordinates that are exactly the same color will get the same reward as the
            target pixel.

            This is done because UI elements usually have large areas of solid-color which will all act
            exactly the same when you click on it. So instead of just training the algo just one pixel
            at a time, we can instead train it to update that entire solid color block. This makes the
            algo train a ton faster.

            :param processedImage: This is the processed image object, as returned by processRawImageParallel
            :param x: The x coordinate of the action to be performed.
            :param y: The y coordinate of the action to be performed.
            :return: A numpy array, in the shape of [height, width] with 1's for all of the pixels that are
                     included in this reward mask, and 0's everywhere else.
        """
        # We use flood-segmentation on the original image to select which pixels we will update reward values for.
        # This works great on UIs because the elements always have big areas of solid-color which respond in the same
        # way.
        rewardPixelMask = skimage.segmentation.flood(numpy.array(processedImage[0], dtype=numpy.float32), (int(y), int(x)))

        return rewardPixelMask

    def prepareAdditionalFeaturesForTrace(self, trace):
        """
            This method computes the additional feature vector that will get injected into the model based on the given execution trace.

            :param trace: A kwola.datamodels.ExecutionTrace object

            :return: A numpy array containing the additional features vector
        """
        branchFeature = numpy.minimum(trace.startCumulativeBranchExecutionTrace, numpy.ones_like(trace.startCumulativeBranchExecutionTrace))
        decayingExecutionTraceFeature = numpy.array(trace.startDecayingExecutionTrace)
        additionalFeature = numpy.concatenate([branchFeature, decayingExecutionTraceFeature], axis=0)

        return additionalFeature


    def calculateTrainingCropPosition(self, centerX, centerY, imageWidth, imageHeight, nextStepCrop=False):
        """
            This method is used to calculate the coordinates for cropping the image for use in training.

            :param centerX: This is where the center of the cropped image should be
            :param centerY: This is where the center of the cropped image should be
            :param imageWidth: This is how large the original image is
            :param imageHeight: This is how large the original image is
            :param nextStepCrop: Whether we are cropping the regular image, or cropping the image used to do
                                 the next step calculation.
            :return: A tuple containing four integers representing the bounds of the crop, (left, top, right, bottom)
        """

        cropWidth = self.config['training_crop_width']
        cropHeight = self.config['training_crop_height']
        if nextStepCrop:
            cropWidth = self.config['training_next_step_crop_width']
            cropHeight = self.config['training_next_step_crop_height']

        # Find the left and top positions that are centered on the given center coordinates
        cropLeft = centerX - cropWidth / 2
        cropTop = centerY - cropHeight / 2

        # Here we make sure that the left and top positions actually fit within the image, and we adjust them
        # so that the cropped image stays within the bounds of the overall image
        cropLeft = max(0, cropLeft)
        cropLeft = min(imageWidth - cropWidth, cropLeft)
        cropLeft = int(cropLeft)

        cropTop = max(0, cropTop)
        cropTop = min(imageHeight - cropHeight, cropTop)
        cropTop = int(cropTop)

        # Right and bottom coordinates are a cinch once we know left and top
        cropRight = int(cropLeft + cropWidth)
        cropBottom = int(cropTop + cropHeight)

        # Return a tuple with all of the bounds
        return (cropLeft, cropTop, cropRight, cropBottom)


    def augmentProcessedImageForTraining(self, processedImage):
        """
            This method is used to apply augmentations to a given image prior to injecting it into the neural network.
            These random augmentations are just used to improve the generalization power of the neural network.
            You can call this function multiple times on the same processedImage and you will get different resulting
            augmented images.

            :param processedImage: This should be a numpy array containing the processed image data, as returned by
                                   DeepLearningAgent.processRawImageParallel
            :return: A new numpy array, with the same shape as the input processedImage, but now with data augmentations
                     applied.
        """
        # Add random noise
        augmentedImage = processedImage + numpy.random.normal(loc=0, scale=self.config['training_image_gaussian_noise_scale'], size=processedImage.shape)

        # Clip the bounds to between 0 and 1
        augmentedImage = numpy.maximum(numpy.zeros_like(augmentedImage), numpy.minimum(numpy.ones_like(augmentedImage), augmentedImage))

        return augmentedImage


    def prepareBatchesForExecutionSession(self, executionSession):
        """
            This function prepares individual samples so that they can be fed to the neural network. This method
            produces a python generator object, and that generator will yield a single dictionary
            object for each of the traces in the execution session. Each of those dictionaries will contain
            a set of keys that are mapped to numpy arrays.

            :param executionSession: A kwola.datamodels.ExecutionSession object for the session being prepared.
            :return: This method yields dictionaries containing a variety of keys mapped to numpy arrays.
        """
        processedImages = []

        executionTraces = []

        # In this section, we load the video and all of the execution traces from the disk
        # at the same time.
        videoPath = self.config.getKwolaUserDataDirectory("videos")
        for rawImage, traceId in zip(DeepLearningAgent.readVideoFrames(os.path.join(videoPath, f'{str(executionSession.id)}.mp4')), executionSession.executionTraces):
            trace = ExecutionTrace.loadFromDisk(traceId, self.config)
            # Occasionally if your doing a lot of R&D and killing the code a lot,
            # the software will save a broken file to disk. When this happens, you
            # will not be able to load the object and get a None value. Here we just
            # protect against this happening by skipping that frame and image entirely
            # and moving onto the next one. Its not perfect but it works for now.
            if trace is not None:
                processedImage = DeepLearningAgent.processRawImageParallel(rawImage, self.config)
                processedImages.append(processedImage)
                executionTraces.append(trace)

        # First compute the present reward at each time step
        presentRewards = DeepLearningAgent.computePresentRewards(executionTraces, self.config)

        # Create the decaying future execution trace for the prediction algorithm
        # The decaying future execution trace is basically a vector that describes
        # what is going to happen in the future. Its similar to the decaying branch
        # trace that is fed as an input to the algorithm. The difference is this.
        # The decaying branch trace shows what happened in the past, with the lines
        # of code that get executed set to 1 in the vector and then decaying thereafter.
        # The decaying future trace is exactly the same but in reverse - it provides
        # what is going to happen next after this trace. The lines of code which
        # execute immediately next are set to 1, and ones that execute further in the
        # future have some decayed value based on the decay rate. What this does is
        # provide an additional, highly supervised target for a secondary loss function.
        tracesReversed = list(copy.copy(executionTraces))
        tracesReversed.reverse()
        currentTrace = numpy.zeros_like(executionTraces[0].branchExecutionTrace)
        executionTraceDiscountRate = self.config['future_execution_trace_decay_rate']
        decayingFutureBranchTraces = []
        for trace in tracesReversed:
            decayingFutureBranchTrace = numpy.array(trace.branchExecutionTrace)
            currentTrace *= executionTraceDiscountRate
            currentTrace += numpy.minimum(decayingFutureBranchTrace, numpy.ones_like(decayingFutureBranchTrace))
            decayingFutureBranchTraces.append(decayingFutureBranchTrace)

        decayingFutureBranchTraces.reverse()

        # Here we construct a list containing the 'next' traces, that is, for every execution traces,
        # these are the execution traces and images that are immediately following.
        nextTraces = list(executionTraces)[1:]
        nextProcessedImages = list(processedImages)[1:]

        # Iterate over each trace along with all of the required data to compute the batch for that trace
        for trace, nextTrace, processedImage, nextProcessedImage, presentReward, decayingFutureBranchTrace in zip(executionTraces, nextTraces, processedImages, nextProcessedImages, presentRewards, decayingFutureBranchTraces):
            width = processedImage.shape[2]
            height = processedImage.shape[1]

            # Construct the additional feature vector. This is basically composed of the decaying past branch trace,
            # along with the cumulative branch vector containing all of the data
            additionalFeature = self.prepareAdditionalFeaturesForTrace(trace)

            # We do the same for the next trace
            nextAdditionalFeature = self.prepareAdditionalFeaturesForTrace(nextTrace)

            # Create the pixel action maps for both of the traces
            pixelActionMap = self.createPixelActionMap(trace.actionMaps, height, width)
            nextPixelActionMap = self.createPixelActionMap(nextTrace.actionMaps, height, width)

            # Here we provide a supervised target based on the cursor. Effectively, this is a one-hot encoding of which
            # cursor was found on the html element under the mouse for this trace
            cursorVector = [0] * len(self.cursors)
            if trace.cursor in self.cursors:
                cursorVector[self.cursors.index(trace.cursor)] = 1
            else:
                cursorVector[self.cursors.index("none")] = 1

            # This is another target used as a secondary loss function. In this case, instead of having the neural network
            # predict reward value directly, we instead of the neural network predict each of the component features that
            # are used to calculate that reward value. Its believed that breaking this apart gives the neural network
            # more insight into whats actually happening to drive these reward values, and hopefully makes it learn faster
            # as a result.
            executionFeatures = [
                trace.didActionSucceed,
                trace.didErrorOccur,
                trace.didNewErrorOccur,
                trace.didCodeExecute,
                trace.didNewBranchesExecute,
                trace.hadNetworkTraffic,
                trace.hadNewNetworkTraffic,
                trace.didScreenshotChange,
                trace.isScreenshotNew,
                trace.didURLChange,
                trace.isURLNew,
                trace.hadLogOutput,
            ]

            # We compute the reward pixel mask based on where the action was performed.
            rewardPixelMask = self.createRewardPixelMask(processedImage,
                                                         int(trace.actionPerformed.x * self.config['model_image_downscale_ratio']),
                                                         int(trace.actionPerformed.y * self.config['model_image_downscale_ratio'])
                                                         )

            # We down-sample some of the data points in the batch to be more compact.
            # We don't need a high precision for most of this data, so its better to be compact and save the ram
            # We also yield each sample individually instead of building up a list to save memory
            yield {
                "traceIds": [str(trace.id)],
                "processedImages": numpy.array([processedImage], dtype=numpy.float16),
                "additionalFeatures": numpy.array([additionalFeature], dtype=numpy.float16),
                "pixelActionMaps": numpy.array([pixelActionMap], dtype=numpy.uint8),
                "stepNumbers": numpy.array([trace.frameNumber - 1], dtype=numpy.int32),

                "nextProcessedImages": numpy.array([nextProcessedImage], dtype=numpy.float16),
                "nextAdditionalFeatures": numpy.array([nextAdditionalFeature], dtype=numpy.float16),
                "nextPixelActionMaps": numpy.array([nextPixelActionMap], dtype=numpy.uint8),
                "nextStepNumbers": numpy.array([nextTrace.frameNumber], dtype=numpy.uint8),

                "actionTypes": [trace.actionPerformed.type],
                "actionXs": numpy.array([int(trace.actionPerformed.x * self.config['model_image_downscale_ratio'])], dtype=numpy.int16),
                "actionYs": numpy.array([int(trace.actionPerformed.y * self.config['model_image_downscale_ratio'])], dtype=numpy.int16),
                "futureBranchTraces": numpy.array([decayingFutureBranchTrace], dtype=numpy.int8),

                "presentRewards": numpy.array([presentReward], dtype=numpy.float32),
                "rewardPixelMasks": numpy.array([rewardPixelMask], dtype=numpy.uint8),
                "executionFeatures": numpy.array([executionFeatures], dtype=numpy.uint8),
                "cursors": numpy.array([cursorVector], dtype=numpy.uint8)
            }

    def learnFromBatches(self, batches):
        """
            Runs backprop on the neural network with the given set of batches.
            Each of the batches will be processed separately, but only a single
            optimizer step will be taken. Therefore it is possible to have a larger
            effective batch size then what you can actually fit into memory,
            simply by passing in multiple batches.

            :param batches: A list of batches. Each batch should be in the same
                            structure as the return value of DeepLearningAgent.prepareBatchesForExecutionSession

            :return: A list of tuples, containing a result object for each of the batches.
                     The result tuples will contain all of the loss values in the following order:

                    (
                        totalRewardLoss,
                        presentRewardLoss,
                        discountedFutureRewardLoss,
                        stateValueLoss,
                        advantageLoss,
                        actionProbabilityLoss,
                        tracePredictionLoss,
                        predictedExecutionFeaturesLoss,
                        targetHomogenizationLoss,
                        predictedCursorLoss,
                        totalLoss,
                        totalRebalancedLoss,
                        batchReward,
                        sampleLosses
                    )
        """
        # Create some variables here to hold loss tensors and results for each batch
        totalLosses = []
        batchResultTensors = []

        # Zero out all of the gradients. We want to do this here because we are about to do
        # multiple backward passes before we use the optimizer to update the network parameters.
        # Therefore we zero it beforehand so we can accumulate the gradients from multiple
        # batches.
        self.optimizer.zero_grad()

        for batch in batches:
            # Here we create torch tensors out of literally all possible data we will need to do any calculations.
            # The reason its done all upfront like this is because this allows the code to pipeline the data it
            # is sending into the GPU. This ensures that all of the GPU calculations are done without any interruptions
            # due to the python coding needing to send data to the GPU
            rewardPixelMasks = self.variableWrapperFunc(torch.IntTensor, batch['rewardPixelMasks'])
            pixelActionMaps = self.variableWrapperFunc(torch.IntTensor, batch['pixelActionMaps'])
            nextStatePixelActionMaps = self.variableWrapperFunc(torch.IntTensor, batch['nextPixelActionMaps'])
            discountRate = self.variableWrapperFunc(torch.FloatTensor, [self.config['reward_discount_rate']])
            actionProbRewardSquareEdgeHalfSize = self.variableWrapperFunc(torch.IntTensor, [int(self.config['training_action_prob_reward_square_size'] / 2)])
            zeroTensor = self.variableWrapperFunc(torch.IntTensor, [0])
            oneTensor = self.variableWrapperFunc(torch.IntTensor, [1])
            oneTensorLong = self.variableWrapperFunc(torch.LongTensor, [1])
            oneTensorFloat = self.variableWrapperFunc(torch.FloatTensor, [1])
            stateValueLossWeightFloat = self.variableWrapperFunc(torch.FloatTensor, [self.config['loss_state_value_weight']])
            presentRewardLossWeightFloat = self.variableWrapperFunc(torch.FloatTensor, [self.config['loss_present_reward_weight']])
            discountedFutureRewardLossWeightFloat = self.variableWrapperFunc(torch.FloatTensor, [self.config['loss_discounted_future_reward_weight']])
            advantageLossWeightFloat = self.variableWrapperFunc(torch.FloatTensor, [self.config['loss_advantage_weight']])
            actionProbabilityLossWeightFloat = self.variableWrapperFunc(torch.FloatTensor, [self.config['loss_action_probability_weight']])
            executionFeatureLossWeightFloat = self.variableWrapperFunc(torch.FloatTensor, [self.config['loss_execution_feature_weight']])
            executionTraceLossWeightFloat = self.variableWrapperFunc(torch.FloatTensor, [self.config['loss_execution_trace_weight']])
            homogenizationLossWeightFloat = self.variableWrapperFunc(torch.FloatTensor, [self.config['loss_homogenization_weight']])
            cursorPredictionLossWeightFloat = self.variableWrapperFunc(torch.FloatTensor, [self.config['loss_cursor_prediction_weight']])
            widthTensor = self.variableWrapperFunc(torch.IntTensor, [batch["processedImages"].shape[3]])
            heightTensor = self.variableWrapperFunc(torch.IntTensor, [batch["processedImages"].shape[2]])
            presentRewardsTensor = self.variableWrapperFunc(torch.FloatTensor, batch["presentRewards"])
            processedImagesTensor = self.variableWrapperFunc(torch.FloatTensor, numpy.array(batch['processedImages']))
            additionalFeaturesTensor = self.variableWrapperFunc(torch.FloatTensor, batch['additionalFeatures'])
            stepNumberTensor = self.variableWrapperFunc(torch.FloatTensor, batch['stepNumbers'])
            nextProcessedImagesTensor = self.variableWrapperFunc(torch.FloatTensor, numpy.array(batch['nextProcessedImages']))
            nextAdditionalFeaturesTensor = self.variableWrapperFunc(torch.FloatTensor, numpy.array(batch['nextAdditionalFeatures']))
            nextStepNumbers = self.variableWrapperFunc(torch.FloatTensor, numpy.array(batch['nextStepNumbers']))

            # Only create the following tensors if the loss is actually enabled for them
            if self.config['enable_trace_prediction_loss']:
                futureBranchTracesTensor = self.variableWrapperFunc(torch.FloatTensor, batch['futureBranchTraces'])
            else:
                futureBranchTracesTensor = None

            if self.config['enable_execution_feature_prediction_loss']:
                executionFeaturesTensor = self.variableWrapperFunc(torch.FloatTensor, batch['executionFeatures'])
            else:
                executionFeaturesTensor = None

            if self.config['enable_cursor_prediction_loss']:
                cursorsTensor = self.variableWrapperFunc(torch.FloatTensor, batch['cursors'])
            else:
                cursorsTensor = None

            # Run the current images & states through the neural network and get
            # all of the various predictions
            currentStateOutputs = self.modelParallel({
                "image": processedImagesTensor,
                "additionalFeature": additionalFeaturesTensor,
                "pixelActionMaps": pixelActionMaps,
                "stepNumber": stepNumberTensor,
                "action_type": batch['actionTypes'],
                "action_x": batch['actionXs'],
                "action_y": batch['actionYs'],
                "outputStamp": False,
                "computeExtras": True,
                "computeActionProbabilities": True,
                "computeStateValues": True,
                "computeAdvantageValues": True,
                "computeRewards": True
            })

            # Here we just create some convenient local variables for each of the outputs
            # from the neural network
            presentRewardPredictions = currentStateOutputs['presentRewards']
            discountedFutureRewardPredictions = currentStateOutputs['discountFutureRewards']
            stateValuePredictions = currentStateOutputs['stateValues']
            advantagePredictions = currentStateOutputs['advantage']
            actionProbabilityPredictions = currentStateOutputs['actionProbabilities']

            with torch.no_grad():
                # Here we use the target network to get the predictions made on the next state.
                # This is part of a key mechanism in Q-learning, which is to calculate the discounted
                # reward value for my current action assuming that I take the best possible action
                # on the next step.
                nextStateOutputs = self.targetNetwork({
                    "image": nextProcessedImagesTensor,
                    "additionalFeature": nextAdditionalFeaturesTensor,
                    "pixelActionMaps": nextStatePixelActionMaps,
                    "stepNumber": nextStepNumbers,
                    "outputStamp": False,
                    "computeExtras": False,
                    "computeActionProbabilities": False,
                    "computeStateValues": False,
                    "computeAdvantageValues": False,
                    "computeRewards": True
                })

                # Again create some convenient local variables for the outputs from the target network.
                nextStatePresentRewardPredictions = nextStateOutputs['presentRewards']
                nextStateDiscountedFutureRewardPredictions = nextStateOutputs['discountFutureRewards']

            # Here we create a bunch of lists to store the loss tensors for each of the samples within the batch.
            # We process each sample separately just because it makes the math more straightforward
            totalSampleLosses = []
            stateValueLosses = []
            advantageLosses = []
            actionProbabilityLosses = []
            presentRewardLosses = []
            targetHomogenizationLosses = []
            discountedFutureRewardLosses = []

            # Here we just zip together all of the various data for each sample in this batch, so that we can iterate
            # over all of it at the same time and process each sample in the batch separately
            zippedValues = zip(range(len(presentRewardPredictions)), presentRewardPredictions, discountedFutureRewardPredictions,
                               nextStatePresentRewardPredictions, nextStateDiscountedFutureRewardPredictions,
                               rewardPixelMasks, presentRewardsTensor, stateValuePredictions, advantagePredictions,
                               batch['actionTypes'], batch['actionXs'], batch['actionYs'],
                               pixelActionMaps, actionProbabilityPredictions, batch['processedImages'])

            # Here we are just iterating over all of the relevant data and tensors for each sample in the batch
            for sampleIndex, presentRewardImage, discountedFutureRewardImage, \
                nextStatePresentRewardImage, nextStateDiscountedFutureRewardImage, \
                rewardPixelMask, presentReward, stateValuePrediction, advantageImage, \
                actionType, actionX, actionY, pixelActionMap, actionProbabilityImage, processedImage in zippedValues:

                # Here, we fetch out the reward and advantage images associated with the action that the AI actually
                # took in the trace. We then multiply by the reward pixel mask. This gives us a reward image that only
                # has values in the area covering the HTML element the algorithm actually touched with its action
                # at this step.
                presentRewardsMasked = presentRewardImage[self.actionsSorted.index(actionType)] * rewardPixelMask
                discountedFutureRewardsMasked = discountedFutureRewardImage[self.actionsSorted.index(actionType)] * rewardPixelMask
                advantageMasked = advantageImage[self.actionsSorted.index(actionType)] * rewardPixelMask

                # Here, we compute the best possible action we can take in the subsequent step from this one, and what is
                # its reward. This gives us the value for the discounted future reward, e.g. what is the reward that
                # the action we took in this sequence, could lead to in the future.
                nextStateBestPossibleTotalReward = torch.max(nextStatePresentRewardImage + nextStateDiscountedFutureRewardImage)
                discountedFutureReward = nextStateBestPossibleTotalReward * discountRate

                # Here we are basically calculating the target images. E.g., this is what we want the neural network to be predicting as outputs.
                # For the present reward, we want the neural network to predict the exact present reward value that we have for this execution trace.
                # For the discounted future reward, we use the above calculated value which is based on the best possible action it could take
                # in the next step after this one.
                # In both cases, the image is constructed with the same mask that the reward images were masked with above. This ensures that we
                # are only updating the values for the pixels of the html element the algo actually clicked on
                targetPresentRewards = torch.ones_like(presentRewardImage[self.actionsSorted.index(actionType)]) * presentReward * rewardPixelMask
                targetDiscountedFutureRewards = torch.ones_like(discountedFutureRewardImage[self.actionsSorted.index(actionType)]) * discountedFutureReward * rewardPixelMask

                # We basically do the same with the advantage to create the target advantage image, and again its multiplied by the same
                # pixel mask. The difference with advantage is that the advantage is updated to be the difference between the predicted reward
                # value for the action we took v.s. the average reward value no matter what action we take. E.g. its a measure of how much
                # better a particular action is versus the average action.
                targetAdvantage = ((presentReward.detach() + discountedFutureReward.detach()) - stateValuePrediction.detach())
                targetAdvantageImage = torch.ones_like(advantageImage[self.actionsSorted.index(actionType)]) * targetAdvantage * rewardPixelMask

                # Now to train the "actor" in the actor critic model, we have to do something different. Instead of
                # training the actor to predict how much better / worse particular actions are versus other actions,
                # now we just straight up train the actor to predict what is the best action it should take when its
                # in a given state. Therefore, we use the advantage calculations to determine what is the best action
                # to take. We then construct a target image which basically has a square of 1's on the location the
                # AI should click and 0's everywhere else.
                bestActionX, bestActionY, bestActionType = self.getActionInfoTensorsFromRewardMap(advantageImage.detach())
                actionProbabilityTargetImage = torch.zeros_like(actionProbabilityImage)
                bestLeft = torch.max(bestActionX - actionProbRewardSquareEdgeHalfSize, zeroTensor)
                bestRight = torch.min(bestActionX + actionProbRewardSquareEdgeHalfSize, widthTensor - 1)
                bestTop = torch.max(bestActionY - actionProbRewardSquareEdgeHalfSize, zeroTensor)
                bestBottom = torch.min(bestActionY + actionProbRewardSquareEdgeHalfSize, heightTensor - 1)
                actionProbabilityTargetImage[bestActionType, bestTop:bestBottom, bestLeft:bestRight] = 1
                actionProbabilityTargetImage[bestActionType] *= pixelActionMap[bestActionType]
                countActionProbabilityTargetPixels = actionProbabilityTargetImage[bestActionType].sum()
                # The max here is just for safety, if any weird bugs happen we don't want any NaN values or division by zero
                actionProbabilityTargetImage[bestActionType] /= torch.max(oneTensorFloat, countActionProbabilityTargetPixels)

                # The max here is just for safety, if any weird bugs happen we don't want any NaN values or division by zero
                countPixelMask = torch.max(oneTensorLong, rewardPixelMask.sum())

                # Now here we create tensors which represent the different between the predictions of the neural network and
                # our target values that were all calculated above.
                presentRewardLossMap = (targetPresentRewards - presentRewardsMasked) * pixelActionMap[self.actionsSorted.index(actionType)]
                discountedFutureRewardLossMap = (targetDiscountedFutureRewards - discountedFutureRewardsMasked) * pixelActionMap[self.actionsSorted.index(actionType)]
                advantageLossMap = (targetAdvantageImage - advantageMasked) * pixelActionMap[self.actionsSorted.index(actionType)]
                actionProbabilityLossMap = (actionProbabilityTargetImage - actionProbabilityImage) * pixelActionMap

                # Here we compute an average loss value for all pixels in the reward pixel mask.
                presentRewardLoss = presentRewardLossMap.pow(2).sum() / countPixelMask
                discountedFutureRewardLoss = discountedFutureRewardLossMap.pow(2).sum() / countPixelMask
                advantageLoss = advantageLossMap.pow(2).sum() / countPixelMask
                actionProbabilityLoss = actionProbabilityLossMap.abs().sum()
                # Additionally, we calculate a loss for the 'state' value, which is the average value the neural network
                # is expected to produce no matter what action it takes. We calculate a loss but the assumption is that
                # the network could never actually calculate this perfectly accurately. It just serves as a barometer
                # that allows us to calculate the advantage values.
                stateValueLoss = (stateValuePrediction - (presentReward.detach() + discountedFutureReward.detach())).pow(2)

                # Here we calculate the homogenization loss if it is enabled.
                if self.config['enable_homogenization_loss']:
                    pixelFeatureImage = currentStateOutputs['pixelFeatureMap'][sampleIndex]

                    # Target Homogenization loss - basically, all of the pixels for the masked area should produce similar features to each other and different features
                    # from other pixels.
                    targetHomogenizationDifferenceMap = ((pixelFeatureImage - pixelFeatureImage[:, actionY, actionX].unsqueeze(1).unsqueeze(1)) * rewardPixelMask).pow(2).mean(axis=0)
                    targetDifferentiationDifferenceMap = ((pixelFeatureImage - pixelFeatureImage[:, actionY, actionX].unsqueeze(1).unsqueeze(1)) * (1.0 - rewardPixelMask)).pow(2).mean(axis=0)
                    targetHomogenizationLoss = torch.abs((targetHomogenizationDifferenceMap.sum() / countPixelMask) - (targetDifferentiationDifferenceMap.sum() / (widthTensor * heightTensor - countPixelMask)))
                    targetHomogenizationLoss = targetHomogenizationLoss * homogenizationLossWeightFloat
                    targetHomogenizationLosses.append(targetHomogenizationLoss.unsqueeze(0))

                # Now we multiply each of the various losses by their weights. These weights are just
                # used to balance the losses against each other, since they have varying absolute magnitudes and
                # varying importance
                presentRewardLoss = presentRewardLoss * presentRewardLossWeightFloat
                discountedFutureRewardLoss = discountedFutureRewardLoss * discountedFutureRewardLossWeightFloat
                advantageLoss = advantageLoss * advantageLossWeightFloat
                actionProbabilityLoss = actionProbabilityLoss * actionProbabilityLossWeightFloat
                stateValueLoss = stateValueLoss * stateValueLossWeightFloat

                # Now we add a scalar tensor for each of the loss values into the lists
                presentRewardLosses.append(presentRewardLoss.unsqueeze(0))
                discountedFutureRewardLosses.append(discountedFutureRewardLoss.unsqueeze(0))
                advantageLosses.append(advantageLoss.unsqueeze(0))
                actionProbabilityLosses.append(actionProbabilityLoss.unsqueeze(0))
                stateValueLosses.append(stateValueLoss)
                totalSampleLosses.append(presentRewardLoss + discountedFutureRewardLoss + advantageLoss + actionProbabilityLoss)

            extraLosses = []

            # If the trace prediction loss is enabled, then we calculate it and add it to the list of extra losses.
            # These extra or secondary losses are just here to help stabilize / regularize the neural network and
            # help it to train faster.
            if self.config['enable_trace_prediction_loss']:
                tracePredictionLoss = (currentStateOutputs['predictedTraces'] - futureBranchTracesTensor).abs().mean() * executionTraceLossWeightFloat
                extraLosses.append(tracePredictionLoss.unsqueeze(0))
            else:
                tracePredictionLoss = zeroTensor

            # If the execution feature prediction is enabled, then we calculate the loss for it and add it to the list of extra losses.
            if self.config['enable_execution_feature_prediction_loss']:
                predictedExecutionFeaturesLoss = (currentStateOutputs['predictedExecutionFeatures'] - executionFeaturesTensor).abs().mean() * executionFeatureLossWeightFloat
                extraLosses.append(predictedExecutionFeaturesLoss.unsqueeze(0))
            else:
                predictedExecutionFeaturesLoss = zeroTensor

            # If the cursor prediction is enabled, then we calculate the loss for it and add it to the list of extra losses.
            if self.config['enable_cursor_prediction_loss']:
                predictedCursorLoss = (currentStateOutputs['predictedCursor'] - cursorsTensor).abs().mean() * cursorPredictionLossWeightFloat
                extraLosses.append(predictedCursorLoss.unsqueeze(0))
            else:
                predictedCursorLoss = zeroTensor

            # Here we calculate the mean value for all of the losses across all of the various samples
            presentRewardLoss = torch.mean(torch.cat(presentRewardLosses))
            discountedFutureRewardLoss = torch.mean(torch.cat(discountedFutureRewardLosses))
            stateValueLoss = torch.mean(torch.cat(stateValueLosses))
            advantageLoss = torch.mean(torch.cat(advantageLosses))
            actionProbabilityLoss = torch.mean(torch.cat(actionProbabilityLosses))

            # This is the final, total loss for all different loss functions across all the different samples
            totalRewardLoss = presentRewardLoss + discountedFutureRewardLoss + stateValueLoss + advantageLoss + actionProbabilityLoss

            # Add in the average homogenization loss if that is enabled
            if self.config['enable_homogenization_loss']:
                targetHomogenizationLoss = torch.mean(torch.cat(targetHomogenizationLosses))
                extraLosses.append(targetHomogenizationLoss.unsqueeze(0))
            else:
                targetHomogenizationLoss = zeroTensor

            # We do a check here because if there are no extra loss functions, then
            # torch will give us an error saying we are concatenating and summing an
            # empty tensor, which is true.
            if len(extraLosses) > 0:
                totalLoss = totalRewardLoss + torch.sum(torch.cat(extraLosses))
            else:
                totalLoss = totalRewardLoss

            totalRebalancedLoss = 0

            # Do the backward pass. This will accumulate gradient values for all of the
            # parameters in the neural network
            totalLoss.backward()

            # Add the total loss for this batch to the list of batch losses.
            totalLosses.append(totalLoss)

            batchResultTensors.append((
                presentRewardLoss,
                discountedFutureRewardLoss,
                stateValueLoss,
                advantageLoss,
                actionProbabilityLoss,
                tracePredictionLoss,
                predictedExecutionFeaturesLoss,
                targetHomogenizationLoss,
                predictedCursorLoss,
                totalRewardLoss,
                totalLoss,
                totalRebalancedLoss,
                totalSampleLosses,
                batch
            ))

        # Put a check in so that we don't do the optimizer step if there are NaNs in the loss
        if numpy.count_nonzero(numpy.isnan([totalLoss.data.item() for totalLoss in totalLosses])) == 0:
            # Now we use the optimizer to update all of the parameters of the neural network.
            # The optimizer will update based on all of the accumulated gradients from the loops above.
            self.optimizer.step()
        else:
            # This else statement should only happen if there is a significant error in the neural network
            # itself that is leading to NaN values in the results. So here, we print out all of the loss
            # values for all of the batchs to help you track down where the error is.
            print(datetime.now(), f"[{os.getpid()}]", "ERROR! NaN detected in loss calculation. Skipping optimization step.")
            for batchIndex, batchResult in batchResultTensors:
                presentRewardLoss, discountedFutureRewardLoss, stateValueLoss, \
                advantageLoss, actionProbabilityLoss, tracePredictionLoss, predictedExecutionFeaturesLoss, \
                targetHomogenizationLoss, predictedCursorLoss, totalRewardLoss, totalLoss, totalRebalancedLoss, \
                totalSampleLosses, batch = batchResult

                print(datetime.now(), f"[{os.getpid()}]", "Batch", batchIndex)
                print(datetime.now(), f"[{os.getpid()}]", "presentRewardLoss", float(presentRewardLoss.data.item()))
                print(datetime.now(), f"[{os.getpid()}]", "discountedFutureRewardLoss", float(discountedFutureRewardLoss.data.item()))
                print(datetime.now(), f"[{os.getpid()}]", "stateValueLoss", float(stateValueLoss.data.item()))
                print(datetime.now(), f"[{os.getpid()}]", "advantageLoss", float(advantageLoss.data.item()))
                print(datetime.now(), f"[{os.getpid()}]", "actionProbabilityLoss", float(actionProbabilityLoss.data.item()))
                print(datetime.now(), f"[{os.getpid()}]", "tracePredictionLoss", float(tracePredictionLoss.data.item()))
                print(datetime.now(), f"[{os.getpid()}]", "predictedExecutionFeaturesLoss", float(predictedExecutionFeaturesLoss.data.item()))
                print(datetime.now(), f"[{os.getpid()}]", "targetHomogenizationLoss", float(targetHomogenizationLoss.data.item()))
                print(datetime.now(), f"[{os.getpid()}]", "predictedCursorLoss", float(predictedCursorLoss.data.item()), flush=True)

            return

        batchResults = []

        # Now, we have to loop back over all of the batches, and prepare the result arrays we are going
        # to provide as return values
        for batchResult in batchResultTensors:
            presentRewardLoss, discountedFutureRewardLoss, stateValueLoss, \
            advantageLoss, actionProbabilityLoss, tracePredictionLoss, predictedExecutionFeaturesLoss, \
            targetHomogenizationLoss, predictedCursorLoss, totalRewardLoss, totalLoss, totalRebalancedLoss, \
            totalSampleLosses, batch = batchResult

            # We cast all of the torch tensors into Python float objects.
            # In the process, the tensors will all get moved from the GPU
            # into the CPU.
            totalRewardLoss = float(totalRewardLoss.data.item())
            presentRewardLoss = float(presentRewardLoss.data.item())
            discountedFutureRewardLoss = float(discountedFutureRewardLoss.data.item())
            stateValueLoss = float(stateValueLoss.data.item())
            advantageLoss = float(advantageLoss.data.item())
            actionProbabilityLoss = float(actionProbabilityLoss.data.item())
            tracePredictionLoss = float(tracePredictionLoss.data.item())
            predictedExecutionFeaturesLoss = float(predictedExecutionFeaturesLoss.data.item())
            targetHomogenizationLoss = float(targetHomogenizationLoss.data.item())
            predictedCursorLoss = float(predictedCursorLoss.data.item())
            totalLoss = float(totalLoss.data.item())
            if self.config['enable_loss_balancing']:
                totalRebalancedLoss = float(totalRebalancedLoss.data.item())
            else:
                totalRebalancedLoss = 0

            # Calculate the total present reward in the batch
            batchReward = float(numpy.sum(batch['presentRewards']))

            # Create a list which has the reward losses broken down by sample, instead of by loss type.
            # This is used in other code to prioritize which samples get trained on by selecting the
            # ones with the highest loss values more often then the ones with lower loss values.
            sampleLosses = [tensor.data.item() for tensor in totalSampleLosses]

            # Accumulate the giant tuple of values into the results list.
            batchResults.append((totalRewardLoss, presentRewardLoss, discountedFutureRewardLoss, stateValueLoss, advantageLoss, actionProbabilityLoss, tracePredictionLoss, predictedExecutionFeaturesLoss, targetHomogenizationLoss,
                                 predictedCursorLoss, totalLoss, totalRebalancedLoss, batchReward, sampleLosses))
        return batchResults

    @staticmethod
    def processRawImageParallel(rawImage, config):
        """
            This method takes a raw image, in regular RGB format directly as taken from the screenshot
            of the target application, and then processed it into the format that will be sent
            into the neural network.

            :param rawImage: A numpy array containing the original image being processed
            :param config: The global configuration object
            :return: A new numpy array, in the shape [1, height, width] containing the processed image data.
        """
        # Create local variables for convenince
        width = rawImage.shape[1]
        height = rawImage.shape[0]

        # Convert the RGB image into greyscale
        grey = skimage.color.rgb2gray(rawImage[:, :, :3])

        # Compute what the size of the image should look like after downscaling.
        shrunkWidth = int(width * config['model_image_downscale_ratio'])
        shrunkHeight = int(height * config['model_image_downscale_ratio'])

        # Make sure the image aligns to the nearest 8 pixels,
        # this is because the image gets downsampled and upsampled
        # within the neural network by 8x, so both the image width
        # and image height must perfectly divide 8 or else there
        # will be errors within the neural network.
        if (shrunkWidth % 8) > 0:
            shrunkWidth += 8 - (shrunkWidth % 8)

        if (shrunkHeight % 8) > 0:
            shrunkHeight += 8 - (shrunkHeight % 8)

        # Resize the image to the selected width and height
        shrunk = skimage.transform.resize(grey, (shrunkHeight, shrunkWidth), anti_aliasing=True)

        # Convert to a numpy array.
        processedImage = numpy.array([shrunk])

        # Round the float values down to 0. This minimizes the range of possible values
        # and can help make the runs more reproducible even in light of error due to the
        # video codec.
        processedImage = numpy.around(processedImage, decimals=2)

        return processedImage
