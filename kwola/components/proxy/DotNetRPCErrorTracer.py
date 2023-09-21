#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
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
                    self.analyzeRPC(rpcRequest, rpcResponse, flow)
            elif isinstance(requestJSON, dict):
                self.analyzeRPC(requestJSON, responseJSON, flow)

    def analyzeRPC(self, rpcRequest, rpcResponse, flow):
        if rpcResponse['type'] != "rpc":
            message = ""
            if "exceptionmessage" in rpcResponse:
                message = f"Exception occurred in backend RPC call: {rpcResponse['exceptionmessage']}"
            elif "implementationexceptiontype" in rpcResponse:
                message = f"Exception occurred in backend RPC call: {rpcResponse['implementationexceptiontype']}"

            message = message.strip()

            self.errors.append(
                DotNetRPCError(type="dotnetrpc",
                               message=message,
                               requestData=json.dumps(rpcRequest, indent=4),
                               responseData=json.dumps(rpcResponse, indent=4),
                               requestHeaders=dict(flow.request.headers),
                               responseHeaders=dict(flow.response.headers)
                              )
            )

    def error(self, flow):
        pass

addons = [
    DotNetRPCError()
]
