from flask import Flask
from flask_restful import Api
from flask_jwt_extended import JWTManager
from mongoengine import connect
from flask_cors import CORS

connect('kwola')

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'secretKey'
jwt = JWTManager(app)
api = Api(app)
CORS(app)

# import models
from .resources.ApplicationResource import ApplicationGroup, ApplicationSingle, ApplicationImage
from .resources.TestingSequenceResource import TestingSequencesGroup, TestingSequencesSingle
from .resources.ExecutionSessionResource import ExecutionSessionGroup, ExecutionSessionSingle, ExecutionSessionVideo
from .resources.ExecutionTraceResource import ExecutionTraceGroup, ExecutionTraceSingle
from .resources.TrainingSequenceResource import TrainingSequencesGroup, TrainingSequencesSingle
from .resources.TrainingStepResources import TrainingStepGroup, TrainingStepSingle

api.add_resource(ApplicationGroup, '/api/application')
api.add_resource(ApplicationSingle, '/api/application/<string:application_id>')
api.add_resource(ApplicationImage, '/api/application/<string:application_id>/image')


api.add_resource(TestingSequencesGroup, '/api/testing_sequences')
api.add_resource(TestingSequencesSingle, '/api/testing_sequences/<string:testing_sequence_id>')


api.add_resource(TrainingSequencesGroup, '/api/training_sequences')
api.add_resource(TrainingSequencesSingle, '/api/training_sequences/<string:training_sequence_id>')


api.add_resource(ExecutionSessionGroup, '/api/execution_sessions')
api.add_resource(ExecutionSessionSingle, '/api/execution_sessions/<string:execution_session_id>')
api.add_resource(ExecutionSessionVideo, '/api/execution_sessions/<string:execution_session_id>/video')


api.add_resource(ExecutionTraceGroup, '/api/execution_traces')
api.add_resource(ExecutionTraceSingle, '/api/execution_traces/<string:execution_trace_id>')


api.add_resource(TrainingStepGroup, '/api/training_steps')
api.add_resource(TrainingStepSingle, '/api/training_steps/<string:training_step_id>')


# api.add_resource(resources.TokenRefresh, '/refresh')
# api.add_resource(resources.SecretResource, '/api/secret/test')
