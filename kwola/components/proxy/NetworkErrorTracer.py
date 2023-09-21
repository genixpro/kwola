#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#

import datetime
from ...datamodels.errors.HttpError import HttpError
from bs4 import BeautifulSoup
import json
import traceback
from ...config.config import getLogger
import re


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
        pass


    def response(self, flow):
        try:
            # Add this flow as an error if its 4xx or 5xx
            if flow.response.status_code >= 400:
                if b"</html" in flow.response.data.content:
                    # Parse response as html
                    text = BeautifulSoup(flow.response.data.content, features="html.parser").get_text()

                    text = re.sub(re.compile(r"\s+"), " ", text)

                    if len(text) > 1024:
                        text = text[:1024] + " ... [message truncated due to excessive length]"
                else:
                    try:
                        data = json.loads(flow.response.data.content)
                        text = json.dumps(data, indent=4)
                    except json.JSONDecodeError:
                        text = str(flow.response.data.content)
                    except UnicodeDecodeError:
                        text = str(flow.response.data.content)

                self.errors.append(HttpError(
                    type="http",
                    path=flow.request.path,
                    statusCode=flow.response.status_code,
                    message=str(text),
                    url=flow.request.url,
                    requestHeaders=dict(flow.request.headers),
                    responseHeaders=dict(flow.response.headers),
                    requestData=str(flow.request.data.content),
                    responseData=str(text)
                ))
        except Exception as e:
            getLogger().error(traceback.format_exc())

    def error(self, flow):
        pass

addons = [
    NetworkErrorTracer()
]
