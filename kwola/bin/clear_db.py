#
#     Kwola is an AI algorithm that learns how to use other programs
#     automatically so that it can find bugs in them.
#
#     Copyright (C) 2020 Kwola Software Testing Inc.
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU Affero General Public License as
#     published by the Free Software Foundation, either version 3 of the
#     License, or (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
#


from kwola.models.ApplicationModel import ApplicationModel
from kwola.models.TrainingSequenceModel import TrainingSequence
from kwola.models.TestingStepModel import TestingStep
from kwola.models.ExecutionSessionModel import ExecutionSession
from kwola.models.ExecutionTraceModel import ExecutionTrace
from kwola.models.TrainingStepModel import TrainingStep
from mongoengine import connect


def main():
    ApplicationModel.objects().delete()
    TrainingSequence.objects().delete()
    TestingStep.objects().delete()
    ExecutionSession.objects().delete()
    ExecutionTrace.objects().delete()
    TrainingStep.objects().delete()

    print("Kwola Mongo database is now cleared")
