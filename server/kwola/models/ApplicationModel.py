from mongoengine import *
import datetime

class ApplicationModel(Document):

    name = StringField(max_length=200, required=True)

    url = StringField(max_length=200, required=True)


