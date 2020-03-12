from flask_restful import Resource, reqparse
from flask_jwt_extended import (create_access_token, create_refresh_token,
                                jwt_required, jwt_refresh_token_required, get_jwt_identity, get_raw_jwt)

from ..models.ExecutionTraceModel import ExecutionTrace
from ..tasks.RunTestingSequence import runTestingSequence
from flask import request
import json
import os
import flask
from kwola.config import config
import os.path


class ExecutionTraceGroup(Resource):
    def __init__(self):
        # self.postParser = reqparse.RequestParser()
        # self.postParser.add_argument('version', help='This field cannot be blank', required=False)
        # self.postParser.add_argument('startTime', help='This field cannot be blank', required=False)
        # self.postParser.add_argument('endTime', help='This field cannot be blank', required=False)
        # self.postParser.add_argument('bugsFound', help='This field cannot be blank', required=False)
        # self.postParser.add_argument('status', help='This field cannot be blank', required=False)
        pass

    def get(self):
        args = request.args
        executionSessionId = args['executionSessionId']

        executionTraces = ExecutionTrace.objects(executionSessionId=executionSessionId).order_by("-startTime").to_json()

        return {"executionTraces": json.loads(executionTraces)}



class ExecutionTraceSingle(Resource):
    def __init__(self):
        self.postParser = reqparse.RequestParser()
        # self.postParser.add_argument('version', help='This field cannot be blank', required=True)
        # self.postParser.add_argument('startTime', help='This field cannot be blank', required=True)
        # self.postParser.add_argument('endTime', help='This field cannot be blank', required=True)
        # self.postParser.add_argument('bugsFound', help='This field cannot be blank', required=True)
        # self.postParser.add_argument('status', help='This field cannot be blank', required=True)

    def get(self, execution_trace_id):
        executionTrace = ExecutionTrace.objects(id=execution_trace_id).limit(1)[0].to_json()

        return {"executionTrace": json.loads(executionTrace)}

#
#
# class ExecutionTraceImage(Resource):
#     def __init__(self):
#         self.postParser = reqparse.RequestParser()
#         # self.postParser.add_argument('version', help='This field cannot be blank', required=True)
#         # self.postParser.add_argument('startTime', help='This field cannot be blank', required=True)
#         # self.postParser.add_argument('endTime', help='This field cannot be blank', required=True)
#         # self.postParser.add_argument('bugsFound', help='This field cannot be blank', required=True)
#         # self.postParser.add_argument('status', help='This field cannot be blank', required=True)
#
#     def get(self, execution_trace_id):
#         videoFilePath = os.path.join(config.getKwolaUserDataDirectory("videos"), f'{str(execution_trace_id)}.mp4')
#
#         with open(videoFilePath, 'rb') as videoFile:
#             videoData = videoFile.read()
#
#         response = flask.make_response(videoData)
#         response.headers['content-type'] = 'application/octet-stream'
#         return response
#
#
