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
from ...datamodels.errors.DotNetRPCError import DotNetRPCError
import json
import traceback
from ...config.config import getLogger
import re
import urllib.parse


class DotNetRPCErrorTracer:
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
        pass


    def response(self, flow):
        parsedURL = urllib.parse.urlparse(flow.request.url)

        if parsedURL.path.endswith("DirectHandler.ashx") and flow.response.status_code == 200:
            requestJSON = json.loads(flow.request.data.content)
            responseJSON = json.loads(flow.response.data.content)

            if isinstance(requestJSON, list):
                for rpcRequest, rpcResponse in zip(requestJSON, responseJSON):
                    self.analyzeRPC(rpcRequest, rpcResponse)
            elif isinstance(requestJSON, dict):
                self.analyzeRPC(requestJSON, responseJSON)

    def analyzeRPC(self, rpcRequest, rpcResponse):
        if rpcResponse['type'] != "rpc":
            message = ""
            if "exceptionmessage" in rpcResponse:
                message = f"Exception occurred in backend RPC call: {rpcResponse['exceptionmessage']}"
            elif "implementationexceptiontype" in rpcResponse:
                message = f"Exception occurred in backend RPC call: {rpcResponse['implementationexceptiontype']}"

            message += f"\nRequest data: {json.dumps(rpcRequest, indent=4)}\nResponse data: {json.dumps(rpcResponse, indent=4)}"

            message = message.strip()

            self.errors.append(
                DotNetRPCError(type="dotnetrpc",
                               message=message,
                               requestData=json.dumps(rpcRequest, indent=4),
                               responseData=json.dumps(rpcResponse, indent=4)
                               )
            )

    def error(self, flow):
        pass

addons = [
    DotNetRPCError()
]
