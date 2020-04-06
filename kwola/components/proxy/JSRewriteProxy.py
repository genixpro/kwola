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


from mitmproxy import ctx
import subprocess
import io
from datetime import datetime
from mitmproxy.script import concurrent
import hashlib
import os.path


class JSRewriteProxy:
    def __init__(self, config):
        self.config = config

        self.memoryCache = {}

    def getCacheFileName(self, fileHash, fileName):
        cacheFileName = os.path.join(self.config.getKwolaUserDataDirectory("javascript"), fileHash + "_" + fileName)
        return cacheFileName


    def findInCache(self, fileHash, fileName):
        cacheFileName = self.getCacheFileName(fileHash, fileName)
        if os.path.exists(cacheFileName):
            with open(cacheFileName, 'rb') as f:
                return f.read()


    def saveInCache(self, fileHash, fileName, data):
        cacheFileName = self.getCacheFileName(fileHash, fileName)
        with open(cacheFileName, 'wb') as f:
            return f.write(data)


    def request(self, flow):
        flow.request.headers['Accept-Encoding'] = 'identity'
        flow.request.headers['Transfer-Encoding'] = 'identity'


    def responseheaders(self, flow):
        """
            The full HTTP response has been read.
        """
        pass


    @concurrent
    def response(self, flow):
        """
            The full HTTP response has been read.
        """

        fileData = bytes(flow.response.data.content)
        hasher = hashlib.md5()
        hasher.update(bytes(flow.request.path, 'utf8'))
        hasher.update(fileData)

        # Ignore it if its a 304 not modified error. These are fine.
        if flow.response.status_code == 304:
            return

        fileHash = hasher.hexdigest()
        fileName = flow.request.path.split("/")[-1]
        try:
            if '.js' in fileName:
                cached = self.memoryCache.get(fileHash)
                if cached is None:
                    cached = self.findInCache(fileHash, fileName)
                    if cached is not None:
                        self.memoryCache[fileHash] = cached

                if cached is not None:
                    flow.response.data.headers['Content-Length'] = str(len(cached))
                    flow.response.data.content = cached
                    return

                filename = str(flow.request.path).split("/")[-1]

                result = subprocess.run(['babel','-f', filename, '--plugins', 'babel-plugin-kwola'], input=bytes(flow.response.data.content), capture_output=True)

                if result.returncode != 0:
                    print(datetime.now(), f"Error! Unable to install Kwola line-counting in the Javascript file {fileName}. Most likely this is because Babel thinks your javascript has invalid syntax.")
                    # print(result.stdout)
                    # print(result.stderr)
                else:
                    transformed = result.stdout

                    self.saveInCache(fileHash, fileName, transformed)
                    self.memoryCache[fileHash] = transformed

                    flow.response.data.headers['Content-Length'] = str(len(transformed))
                    flow.response.data.content = transformed
        except Exception as e:
            print(e)
