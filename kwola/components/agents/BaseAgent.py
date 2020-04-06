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


from ...datamodels.actions.ClickTapAction import ClickTapAction
from ...datamodels.actions.RightClickAction import RightClickAction
from ...datamodels.actions.TypeAction import TypeAction
from ...datamodels.actions.WaitAction import WaitAction
from ...datamodels.actions.ClearFieldAction import ClearFieldAction
import scipy.special
import random

class BaseAgent:
    """
        This is a base class for different implementations of the machine learning algorithms that are used to
        control the Environments and find bugs.
    """


    def __init__(self, config):
        self.actions = {
            "click": lambda x,y: ClickTapAction(type="click", x=x, y=y, times=1),
            "clear": lambda x, y: ClearFieldAction(type="clear", x=x, y=y),
        }

        self.actionBaseWeights = [
            config['random_weight_click'],
            config['random_weight_clear']
        ]

        self.actionProbabilityBoostKeywords = [
            [],
            []
        ]

        if config['email']:
            self.actions['typeEmail'] = lambda x,y: TypeAction(type="typeEmail", x=x, y=y, label="email", text=config['email'])
            self.actionBaseWeights.append(config['random_weight_type_email'])
            self.actionProbabilityBoostKeywords.append(["email", "user"])

        if config['password']:
            self.actions['typePassword'] = lambda x,y: TypeAction(type="typePassword", x=x, y=y, label="password", text=config['password'])
            self.actionBaseWeights.append(config['random_weight_type_password'])
            self.actionProbabilityBoostKeywords.append(["pass"])

        if config['name']:
            self.actions['typeName'] = lambda x,y: TypeAction(type="typeName", x=x, y=y, label="name", text=config['name'])
            self.actionBaseWeights.append(config['random_weight_type_name'])
            self.actionProbabilityBoostKeywords.append(["email", "user"])

        if config['paragraph']:
            self.actions['typeParagraph'] = lambda x,y: TypeAction(type="typeParagraph", x=x, y=y, label="paragraph", text=config['paragraph'])
            self.actionBaseWeights.append(config['random_weight_type_paragraph'])
            self.actionProbabilityBoostKeywords.append(["pass"])

        if config['enableRandomNumberCommand']:
            self.actions['typeNumber'] = lambda x,y: TypeAction(type="typeNumber", x=x, y=y, label="number", text=self.randomString('-.0123456789$%', random.randint(1, 5)))
            self.actionBaseWeights.append(config['random_weight_type_number'])
            self.actionProbabilityBoostKeywords.append(["num", "count", "int", "float"])

        if config['enableDoubleClickCommand']:
            self.actions['doubleClick'] = lambda x,y: ClickTapAction(type="double_click", x=x, y=y, times=2)
            self.actionBaseWeights.append(config['random_weight_double_click'])
            self.actionProbabilityBoostKeywords.append([])

        if config['enableRightClickCommand']:
            self.actions['rightClick'] = lambda x,y: RightClickAction(type="right_click", x=x, y=y)
            self.actionBaseWeights.append(config['random_weight_right_click'])
            self.actionProbabilityBoostKeywords.append([])

        if config['enableRandomBracketCommand']:
            self.actions['typeBrackets'] = lambda x,y: TypeAction(type="typeBrackets", x=x, y=y, label="brackets", text=self.randomString('{}[]()', random.randint(1, 3)))
            self.actionBaseWeights.append(config['random_weight_type_brackets'])
            self.actionProbabilityBoostKeywords.append([])

        if config['enableRandomMathCommand']:
            self.actions['typeMath'] = lambda x,y: TypeAction(type="typeMath", x=x, y=y, label="symbol", text=self.randomString('*=+<>', random.randint(1, 3)))
            self.actionBaseWeights.append(config['random_weight_type_math'])
            self.actionProbabilityBoostKeywords.append([])

        if config['enableRandomOtherSymbolCommand']:
            self.actions['typeOtherSymbol'] = lambda x,y: TypeAction(type="typeOtherSymbol", x=x, y=y, label="symbol", text=self.randomString('"\';:/?,!^&#@', random.randint(1, 3)))
            self.actionBaseWeights.append(config['random_weight_type_other_symbol'])
            self.actionProbabilityBoostKeywords.append([])

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

        self.actionsSorted = sorted(self.actions.keys())

    def randomString(self, chars, len):
        base = ""
        for n in range(len):
            base += str(random.choice(chars))
        return base

    def load(self):
        """
            Loads the agent from db / disk

            :return:
        """


    def save(self):
        """
            Saves the agent to the db / disk.

            :return:
        """


    def initialize(self, branchFeatureSize):
        """
        Initialize the agent for operating in the given environment.

        :param branchFeatureSize:
        :return:
        """

    def nextBestActions(self, stepNumber, rawImages, envActionMaps, additionalFeatures, recentActions):
        """
            Return the next best action predicted by the agent.
            :return:
        """
