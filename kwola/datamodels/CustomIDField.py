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


import base64
import uuid
import mongoengine

class CustomIDField(mongoengine.fields.StringField):
    def __init__(self, **kwargs):
        kwargs['primary_key'] = True
        super(CustomIDField, self).__init__(**kwargs)

    def validate(self, value):
        return True

    def to_mongo(self, value):
        return str(value)

    def to_python(self, value):
        return str(value)

    @staticmethod
    def generateNewUUID(modelClass, config, minimumLength=4):
        currentLength = minimumLength
        generatedId = base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b'=').decode('ascii').replace("_", "").replace("-", "")[:currentLength].lower()
        while modelClass.loadFromDisk(generatedId, config, printErrorOnFailure=False) is not None:
            currentLength += 4
            generatedId = base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b'=').decode('ascii').replace("_", "").replace("-", "")[:currentLength].lower()

        return generatedId

