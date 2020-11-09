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


class LogError(BaseError):
    """
        This class represents when something is logged at a log-level that is considered an error. The exact details are user definable but effectively this is an error
        that is only detected in the log files.
    """

    logLevel = StringField()



    def computeHash(self):
        hasher = hashlib.sha256()
        hasher.update(bytes(self.message, "utf8"))
        hasher.update(bytes(self.logLevel, "utf8"))

        return hasher.hexdigest()

    def generateErrorDescription(self):
        return self.message