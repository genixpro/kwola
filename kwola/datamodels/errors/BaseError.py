#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
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
