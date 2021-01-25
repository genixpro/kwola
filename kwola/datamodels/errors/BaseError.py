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


from mongoengine import *
from ...components.utils.deunique import deuniqueString
import re
import datetime
import functools
import edlib

class BaseError(EmbeddedDocument):
    """
        This model is a base class for all different kinds of errors that can be detected by the Kwola engine.
    """

    meta = {'allow_inheritance': True}

    type = StringField()

    page = StringField()

    message = StringField()


    def computeHash(self):
        raise NotImplementedError()

    @staticmethod
    @functools.lru_cache(maxsize=1024)
    def computeReducedErrorComparisonMessage(message):
        return deuniqueString(message, deuniqueMode="error")

    @staticmethod
    def computeErrorMessageSimilarity(message, otherMessage):
        message = BaseError.computeReducedErrorComparisonMessage(message)
        otherMessage = BaseError.computeReducedErrorComparisonMessage(otherMessage)

        distanceScore = edlib.align(message, otherMessage)['editDistance'] / max(1, max(len(message), len(otherMessage)))
        return 1.0 - distanceScore

    def computeSimilarity(self, otherError):
        message = self.message
        otherMessage = otherError.message

        return BaseError.computeErrorMessageSimilarity(message, otherMessage)

    def isDuplicateOf(self, otherError):
        return self.computeSimilarity(otherError) >= 0.95
