#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#


from .BaseError import BaseError
from mongoengine import *
import hashlib


class HttpError(BaseError):
    """
        This class represents errors between the frontend and the backend. This means that the backend send back an HTTP status code which
        is considered an error for the customers purposes.
    """

    path = StringField()

    statusCode = IntField()

    url = StringField()

    requestData = StringField()

    requestHeaders = DictField(StringField())

    responseHeaders = DictField(StringField())

    responseData = StringField()

    def computeHash(self):
        hasher = hashlib.sha256()
        hasher.update(bytes(self.path, "utf8"))
        hasher.update(bytes(str(self.statusCode), "utf8"))
        hasher.update(bytes(self.message, "utf8"))

        return hasher.hexdigest()

    def generateErrorDescription(self):
        return f"Error {self.statusCode} at {self.path}: {self.message}"