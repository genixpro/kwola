#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#


from .BaseError import BaseError
from mongoengine import *
import hashlib


class DotNetRPCError(BaseError):
    """
        This class represents errors specifically in dot net applications which use RPC calls
    """

    requestData = StringField()

    responseData = StringField()

    requestHeaders = DictField(StringField())

    responseHeaders = DictField(StringField())


    def computeHash(self):
        hasher = hashlib.sha256()
        hasher.update(bytes(self.requestData, "utf8"))
        hasher.update(bytes(str(self.responseData), "utf8"))

        return hasher.hexdigest()

    def generateErrorDescription(self):
        return f"Error in dot net RPC call. Request data: {self.requestData}\nResponse data: {self.responseData}"

