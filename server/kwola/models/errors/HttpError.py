from .BaseError import BaseError
from mongoengine import *


class HttpError(BaseError):
    """
        This class represents errors between the frontend and the backend. This means that the backend send back an HTTP status code which
        is considered an error for the customers purposes.
    """

    statusCode = IntField()

    message = StringField()

