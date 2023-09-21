#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#

import datetime
from ...config.logger import getLogger
import os
import urllib.parse

class PathTracer:
    def __init__(self):
        self.seenPaths = set()
        self.recentPaths = set()
        self.mostRecentNetworkActivityTime = datetime.datetime.now()
        self.mostRecentNetworkActivityURL = ""
        self.mostRecentNetworkActivityEvent = ""

    def http_connect(self, flow):
        self.mostRecentNetworkActivityTime = datetime.datetime.now()
        self.mostRecentNetworkActivityURL = flow.request.url
        self.mostRecentNetworkActivityEvent = "http_connect"

    def requestheaders(self, flow):
        self.mostRecentNetworkActivityTime = datetime.datetime.now()
        self.mostRecentNetworkActivityURL = flow.request.url
        self.mostRecentNetworkActivityEvent = "requestheaders"

    def request(self, flow):
        parsed = urllib.parse.urlparse(flow.request.url)
        trackingPath = parsed.scheme + "://" + parsed.netloc + parsed.path

        self.seenPaths.add(trackingPath)
        self.recentPaths.add(trackingPath)
        self.mostRecentNetworkActivityTime = datetime.datetime.now()
        self.mostRecentNetworkActivityURL = flow.request.url
        self.mostRecentNetworkActivityEvent = "request"

    def responseheaders(self, flow):
        self.mostRecentNetworkActivityTime = datetime.datetime.now()
        self.mostRecentNetworkActivityURL = flow.request.url
        self.mostRecentNetworkActivityEvent = "responseheaders"

    def response(self, flow):
        self.mostRecentNetworkActivityTime = datetime.datetime.now()
        self.mostRecentNetworkActivityURL = flow.request.url
        self.mostRecentNetworkActivityEvent = "response"

    def error(self, flow):
        self.mostRecentNetworkActivityTime = datetime.datetime.now()
        self.mostRecentNetworkActivityURL = flow.request.url
        self.mostRecentNetworkActivityEvent = "error"

addons = [
    PathTracer()
]
