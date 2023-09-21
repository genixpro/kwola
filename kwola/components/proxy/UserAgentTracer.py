#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#

import datetime
from ...config.logger import getLogger
import os
import urllib.parse
import traceback

class UserAgentTracer:
    def __init__(self):
        self.lastUserAgent = ""

    def requestheaders(self, flow):
        try:
            if 'User-Agent' in flow.request.headers:
                self.lastUserAgent = flow.request.headers['User-Agent']
            elif 'user-agent' in flow.request.headers:
                self.lastUserAgent = flow.request.headers['user-agent']
        except Exception as e:
            getLogger().error(traceback.format_exc())

addons = [
    UserAgentTracer()
]
