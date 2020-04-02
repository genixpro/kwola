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

    message = StringField()

    def computeHash(self):
        hasher = hashlib.md5()
        hasher.update(self.path)
        hasher.update(self.statusCode)
        hasher.update(self.message)

        return hasher.hexdigest()
