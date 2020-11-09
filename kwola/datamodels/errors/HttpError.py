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

    def computeHash(self):
        hasher = hashlib.sha256()
        hasher.update(bytes(self.path, "utf8"))
        hasher.update(bytes(str(self.statusCode), "utf8"))
        hasher.update(bytes(self.message, "utf8"))

        return hasher.hexdigest()

    def generateErrorDescription(self):
        return f"Error {self.statusCode} at {self.path}: {self.message}"