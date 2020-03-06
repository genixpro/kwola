from mongoengine import *
import datetime

class ApplicationModel(Document):

    name = StringField(required=True)

    url = StringField(required=True)


