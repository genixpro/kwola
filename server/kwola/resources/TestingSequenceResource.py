from flask_restful import Resource, reqparse
from flask_jwt_extended import (create_access_token, create_refresh_token,
                                jwt_required, jwt_refresh_token_required, get_jwt_identity, get_raw_jwt)

from ..models.TestingSequenceModel import TestingSequenceModel
from ..tasks.RunTestingSequence import runTestingSequence
import json


class TestingSequencesGroup(Resource):
    def __init__(self):
        self.postParser = reqparse.RequestParser()
        self.postParser.add_argument('version', help='This field cannot be blank', required=False)
        self.postParser.add_argument('startTime', help='This field cannot be blank', required=False)
        self.postParser.add_argument('endTime', help='This field cannot be blank', required=False)
        self.postParser.add_argument('bugsFound', help='This field cannot be blank', required=False)
        self.postParser.add_argument('status', help='This field cannot be blank', required=False)

    def get(self, application_id):
        testingSequences = TestingSequenceModel.objects().to_json()

        return {"testing_sequences": json.loads(testingSequences)}

    def post(self, application_id):
        data = self.postParser.parse_args()


        newTestingSequence = TestingSequenceModel(
            version=data['version'],
            startTime=data['startTime'],
            endTime=data['endTime'],
            bugsFound=data['bugsFound'],
            status=data['status']
        )

        newTestingSequence.save()

        runTestingSequence.delay(str(newTestingSequence.id))

        return {}


class TestingSequencesSingle(Resource):
    def __init__(self):
        self.postParser = reqparse.RequestParser()
        self.postParser.add_argument('version', help='This field cannot be blank', required=True)
        self.postParser.add_argument('startTime', help='This field cannot be blank', required=True)
        self.postParser.add_argument('endTime', help='This field cannot be blank', required=True)
        self.postParser.add_argument('bugsFound', help='This field cannot be blank', required=True)
        self.postParser.add_argument('status', help='This field cannot be blank', required=True)

    def get(self, application_id, testing_sequence_id):
        application = TestingSequenceModel.objects(id=testing_sequence_id).limit(1)[0].to_json()

        return json.loads(application)
