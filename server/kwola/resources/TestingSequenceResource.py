from flask_restful import Resource, reqparse
from flask_jwt_extended import (create_access_token, create_refresh_token,
                                jwt_required, jwt_refresh_token_required, get_jwt_identity, get_raw_jwt)

from ..models.TestingSequenceModel import TestingSequenceModel
import json


class TestingSequencesGroup(Resource):
    def __init__(self):
        self.postParser = reqparse.RequestParser()
        self.postParser.add_argument('name', help='This field cannot be blank', required=True)
        self.postParser.add_argument('url', help='This field cannot be blank', required=True)

    def get(self, application_id):
        testingSequences = TestingSequenceModel.objects().to_json()

        return {"testing_sequences": json.loads(testingSequences)}

    def post(self, application_id):
        data = self.postParser.parse_args()


        newTestingSequence = TestingSequenceModel(
            name=data['name'],
            url=data['url']
        )

        newTestingSequence.save()

        return {}

        # if not current_user:
        #     return {'message': 'User {} doesn\'t exist'.format(data['username'])}
        #
        # if data['username'] == current_user['username'] and data['password'] == current_user['password']:
        #     access_token = create_access_token(identity=data['username'])
        #     return {
        #         'token': access_token,
        #     }
        # else:
        #     return {'message': 'Wrong credentials'}


class TestingSequencesSingle(Resource):
    def __init__(self):
        self.postParser = reqparse.RequestParser()
        self.postParser.add_argument('name', help='This field cannot be blank', required=True)
        self.postParser.add_argument('url', help='This field cannot be blank', required=True)

    def get(self, application_id, testing_sequence_id):
        application = TestingSequenceModel.objects(id=testing_sequence_id).limit(1)[0].to_json()

        return json.loads(application)
