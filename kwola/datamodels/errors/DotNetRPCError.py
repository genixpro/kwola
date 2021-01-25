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


class DotNetRPCError(BaseError):
    """
        This class represents errors specifically in dot net applications which use RPC calls
    """

    requestData = StringField()

    responseData = StringField()


    def computeHash(self):
        hasher = hashlib.sha256()
        hasher.update(bytes(self.requestData, "utf8"))
        hasher.update(bytes(str(self.responseData), "utf8"))

        return hasher.hexdigest()

    def generateErrorDescription(self):
        return f"Error in dot net RPC call. Request data: {self.requestData}\nResponse data: {self.responseData}"

