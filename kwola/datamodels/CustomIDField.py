#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
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

