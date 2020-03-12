from flask_restful import Resource, reqparse
from flask_jwt_extended import (create_access_token, create_refresh_token,
                                jwt_required, jwt_refresh_token_required, get_jwt_identity, get_raw_jwt)

from ..models.TestingSequenceModel import TestingSequenceModel
from ..tasks.RunTestingSequence import runTestingSequence
import json
import bson

class TestingSequencesGroup(Resource):
    def __init__(self):
        self.postParser = reqparse.RequestParser()
        self.postParser.add_argument('version', help='This field cannot be blank', required=False)
        self.postParser.add_argument('startTime', help='This field cannot be blank', required=False)
        self.postParser.add_argument('endTime', help='This field cannot be blank', required=False)
        self.postParser.add_argument('bugsFound', help='This field cannot be blank', required=False)
        self.postParser.add_argument('status', help='This field cannot be blank', required=False)

    def get(self):
        testingSequences = TestingSequenceModel.objects().order_by("-startTime").to_json()

        return {"testingSequences": json.loads(testingSequences)}

    def post(self):
        data = self.postParser.parse_args()


        newTestingSequence = TestingSequenceModel(
            version=data['version'],
            startTime=data['startTime'],
            endTime=data['endTime'],
            bugsFound=data['bugsFound'],
            status=data['status'],
            executionSessions=[],
            errors=[]
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

    def get(self, testing_sequence_id):
        testingSequence = TestingSequenceModel.objects(id=bson.ObjectId(testing_sequence_id)).limit(1)[0].to_json()

        return {"testingSequence": json.loads(testingSequence)}

