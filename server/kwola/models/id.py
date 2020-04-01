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



def generateNewUUID(modelClass, minimumLength=4):
    currentLength = minimumLength
    id = base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b'=').decode('ascii').replace("_", "").replace("-", "")[:currentLength].lower()
    while modelClass.loadFromDisk(id) is not None:
        currentLength += 4
        id = base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b'=').decode('ascii').replace("_", "").replace("-", "")[:currentLength].lower()

    return id

