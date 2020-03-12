from flask_restful import Resource, reqparse
from flask_jwt_extended import (create_access_token, create_refresh_token,
                                jwt_required, jwt_refresh_token_required, get_jwt_identity, get_raw_jwt)

from ..models.TrainingStepModel import TrainingStep
import json
import math


class TrainingStepGroup(Resource):
    def __init__(self):
        self.postParser = reqparse.RequestParser()
        # self.postParser.add_argument('version', help='This field cannot be blank', required=False)
        # self.postParser.add_argument('startTime', help='This field cannot be blank', required=False)
        # self.postParser.add_argument('endTime', help='This field cannot be blank', required=False)
        # self.postParser.add_argument('bugsFound', help='This field cannot be blank', required=False)
        # self.postParser.add_argument('status', help='This field cannot be blank', required=False)

    def get(self):
        trainingSteps = TrainingStep.objects().order_by("-startTime").only("startTime", "id", "status", "averageLoss")

        for trainingStep in trainingSteps:
            if trainingStep.averageLoss is not None and math.isnan(trainingStep.averageLoss):
                trainingStep.averageLoss = None
                trainingStep.save()
                print("Got an unexpected nan!")

        return {"trainingSteps": json.loads(trainingSteps.to_json())}





class TrainingStepSingle(Resource):
    def __init__(self):
        self.postParser = reqparse.RequestParser()
        # self.postParser.add_argument('version', help='This field cannot be blank', required=True)
        # self.postParser.add_argument('startTime', help='This field cannot be blank', required=True)
        # self.postParser.add_argument('endTime', help='This field cannot be blank', required=True)
        # self.postParser.add_argument('bugsFound', help='This field cannot be blank', required=True)
        # self.postParser.add_argument('status', help='This field cannot be blank', required=True)

    def get(self, training_step_id):
        trainingStep = TrainingStep.objects(id=training_step_id).limit(1)[0]

        return {"trainingStep": json.loads(trainingStep.to_json())}

