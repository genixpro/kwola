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

import datetime
from ...datamodels.errors.HttpError import HttpError
from bs4 import BeautifulSoup
import json


class NetworkErrorTracer:
    def __init__(self):
        self.resetErrors()

    def resetErrors(self):
        self.errors = []

    def http_connect(self, flow):
        pass

    def requestheaders(self, flow):
        pass

    def request(self, flow):
        pass

    def responseheaders(self, flow):
        # Add this flow as an error if its 4xx or 5xx
        if flow.response.status_code >= 400:
            if "</html" in flow.response.data.content:
                # Parse response as html
                text = BeautifulSoup(flow.response.data.content).get_text()
            else:
                try:
                    data = json.loads(flow.response.data.content)
                    text = json.dumps(data, indent=4)
                except json.JSONDecodeError:
                    text = str(flow.response.data.content)

            self.errors.append(HttpError(type="http", path=flow.request.path, statusCode=flow.response.status_code, message=str(text)))



    def response(self, flow):
        pass

    def error(self, flow):
        pass

addons = [
    NetworkErrorTracer()
]
