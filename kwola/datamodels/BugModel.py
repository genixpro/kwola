#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#


from .errors.BaseError import BaseError
from .errors.ExceptionError import ExceptionError
from .errors.HttpError import HttpError
from .errors.LogError import LogError
from .errors.DotNetRPCError import DotNetRPCError
from .actions.BaseAction import BaseAction
from .CustomIDField import CustomIDField
from .DiskUtilities import saveObjectToDisk, loadObjectFromDisk
from mongoengine import *
from kwola.components.utils.deunique import deuniqueString
from kwola.components.proxy.RewriteProxy import RewriteProxy

class BugModel(Document):
    meta = {
        'indexes': [
            ('owner', 'testingRunId',),
            ('owner', 'applicationId',),
            ('applicationId',),
            ('testingRunId', ),
        ]
    }

    id = CustomIDField()

    owner = StringField()

    applicationId = StringField()

    testingStepId = StringField()

    testingRunId = StringField(required=False)

    executionSessionId = StringField()

    creationDate = DateField()

    stepNumber = IntField()

    actionsPerformed = EmbeddedDocumentListField(BaseAction)

    error = EmbeddedDocumentField(BaseError)

    isMuted = BooleanField(default=False)

    mutedErrorId = StringField(default=None)

    # Deprecated / unused
    reproductionTraces = ListField(StringField())

    browser = StringField()

    userAgent = StringField()

    windowSize = StringField()

    isJavascriptError = BooleanField()

    # Deprecated
    severityScore = IntField()
    # Deprecated
    severityLevel = IntField()
    # Deprecated
    originalSeverityLevel = IntField()

    bugTypeSeverityScore = FloatField()

    codePrevalenceScore = FloatField()

    importanceLevel = IntField()

    originalImportanceLevel = IntField()

    status = StringField(enumerate=['new', 'triage', 'fix_in_progress', 'needs_testing', 'closed'], default="new")

    isBugNew = BooleanField()

    reproducible = BooleanField(default=False)

    canonicalPageUrl = StringField()

    def saveToDisk(self, config, overrideSaveFormat=None, overrideCompression=None):
        saveObjectToDisk(self, "bugs", config, overrideSaveFormat=overrideSaveFormat, overrideCompression=overrideCompression)


    @staticmethod
    def loadFromDisk(id, config, printErrorOnFailure=True):
        return loadObjectFromDisk(BugModel, id, "bugs", config, printErrorOnFailure=printErrorOnFailure)

    def generateBugText(self):
        return self.error.generateErrorDescription()

    def isDuplicateOf(self, otherBug):
        return self.error.isDuplicateOf(otherBug.error)

    def recomputeBugQualitativeFeatures(self):
        self.recomputeIsJavascriptError()
        self.recomputeBugTypeSeverityScore()
        self.recomputeImportanceLevel()

    def recomputeIsJavascriptError(self):
        self.isJavascriptError = False
        if isinstance(self.error, ExceptionError):
            self.isJavascriptError = True
        elif isinstance(self.error, LogError):

            javascriptErrorTexts = [
                'null',
                'RangeError',
                'ReferenceError',
                'SyntaxError',
                'TypeError',
                'undefined',
                'URIError',
                'Exception'
            ]

            for text in javascriptErrorTexts:
                if text in self.error.message:
                    self.isJavascriptError = True
                    break

    def recomputeBugTypeSeverityScore(self):
        if self.isJavascriptError:
            self.bugTypeSeverityScore = 1.0
        elif isinstance(self.error, DotNetRPCError):
            self.bugTypeSeverityScore = 0.8
        elif isinstance(self.error, HttpError) and self.error.statusCode >= 500:
            self.bugTypeSeverityScore = 0.8
        elif isinstance(self.error, HttpError) and (self.error.statusCode == 403 or self.error.statusCode == 401):
            self.bugTypeSeverityScore = 0.6
        elif isinstance(self.error, HttpError) and self.error.statusCode == 404:
            self.bugTypeSeverityScore = 0.4
        elif isinstance(self.error, LogError):
            self.bugTypeSeverityScore = 0.2
        else:
            self.bugTypeSeverityScore = 0.0

    def recomputeImportanceLevel(self):
        bugTypeWeight = 3.0
        codePrevelanceWeight = 1.0
        minimumSeverity = 1

        codePrevalence = self.codePrevalenceScore
        if codePrevalence is None:
            codePrevalence = 0
            codePrevelanceWeight = 0
            bugTypeWeight = 4.0

        self.importanceLevel = minimumSeverity + int(round(bugTypeWeight * (1.0 - self.bugTypeSeverityScore) + codePrevelanceWeight * (1.0 - codePrevalence) ))
        self.originalImportanceLevel = self.importanceLevel


    def recomputeCanonicalPageUrl(self):
        pageUrl = self.error.page

        if pageUrl is None or not pageUrl:
            self.canonicalPageUrl = ""
        else:
            self.canonicalPageUrl = RewriteProxy.canonicalizeUrl(pageUrl, substituteReferenceWrapperCharacters="[]")
