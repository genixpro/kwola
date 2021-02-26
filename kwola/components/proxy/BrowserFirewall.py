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
