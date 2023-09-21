#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#

import socket
import urllib.parse
import traceback
from ...config.logger import getLogger
from mitmproxy import http

class BrowserFirewall:
    def __init__(self, config):
        self.config = config

    def http_connect(self, flow):
        pass


    def requestheaders(self, flow):
        try:
            if self.config["web_session_block_private_networks"]:
                parsed = urllib.parse.urlparse(flow.request.url)
                ip = socket.gethostbyname(parsed.netloc)
                if ip.startswith("192.168.") or \
                        ip.startswith("10.") or \
                        ip.startswith("127.0.0.1") or \
                        (ip.startswith("172.") and int(ip.split(".")[1]) >= 16 and int(ip.split(".")[1]) <= 31):
                    flow.response = http.HTTPResponse.make(
                        403,
                        b"Forbidden. Not allowed to access private networks.",
                        {"Content-Type": "text/html"}
                    )
        except Exception as e:
            getLogger().error(traceback.format_exc())

    def request(self, flow):
        pass

    def responseheaders(self, flow):
        pass

    def response(self, flow):
        pass

    def error(self, flow):
        pass

addons = [
    BrowserFirewall(None)
]
