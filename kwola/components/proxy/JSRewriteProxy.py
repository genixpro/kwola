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


from datetime import datetime
from mitmproxy.script import concurrent
import hashlib
import os
import os.path
import subprocess
import base64
import traceback
import urllib.parse
import gzip


class JSRewriteProxy:
    def __init__(self, config):
        self.config = config

        self.memoryCache = {}

        self.knownResponseWrappers = [
            (b"""<!--/*--><html><body><script type="text/javascript"><!--//*/""",
             b"""""")
        ]

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


    def responseheaders(self, flow):
        """
            The full HTTP response has been read.
        """
        pass

    def computeHashes(self, data):
        """
            Computes two hashes for the given data. A short hash and a long hash.

            The long hash is a full md5 hash, encoded in base64 except with the extra 2 characters removed
            so its purely alphanumeric, although can vary in length.

            The short hash is a short, six character hash which helps uniquely identify the file when used
            alongside the filename. Its also purely alphanumeric and only in lowercase.

            @returns (longHash, shortHash) a tuple with two strings
        """
        hasher = hashlib.md5()
        hasher.update(data)

        base64ExtraCharacters = bytes("--", 'utf8')
        longHash = str(base64.b64encode(hasher.digest(), altchars=base64ExtraCharacters), 'utf8')
        longHash = longHash.replace("-", "")
        longHash = longHash.replace("=", "")

        shortHashLength = 6
        shortHash = longHash[::int(len(longHash)/shortHashLength)][:shortHashLength].lower()

        return longHash, shortHash


    @concurrent
    def response(self, flow):
        """
            The full HTTP response has been read.
        """

        # Ignore it if its a 304 not modified error. These are fine.
        if flow.response.status_code == 304:
            return

        longFileHash, shortFileHash = self.computeHashes(bytes(flow.response.data.content))
        fileName = urllib.parse.unquote(flow.request.path.split("/")[-1])
        if "?" in fileName:
            fileName = fileName.split("?")[0]
        if "#" in fileName:
            fileName = fileName.split("#")[0]

        try:
            if '.js' in fileName and not ".json" in fileName:
                cached = self.memoryCache.get(longFileHash)
                if cached is None:
                    cached = self.findInCache(shortFileHash, fileName)
                    if cached is not None:
                        self.memoryCache[longFileHash] = cached

                if cached is not None:
                    flow.response.data.headers['Content-Length'] = str(len(cached))
                    flow.response.data.content = cached
                    return

                fileNameForBabel = shortFileHash + "-" + fileName

                originalFileContents = bytes(flow.response.data.content)
                jsFileContents = bytes(flow.response.data.content).strip()

                gzipped = False
                # Special - check to see if the file might be gzipped
                if len(jsFileContents) > 10 and \
                    jsFileContents[0] == 0x1f and \
                        jsFileContents[1] == 0x8b:
                    try:
                        jsFileContents = gzip.decompress(jsFileContents)
                        gzipped = True
                    except gzip.BadGzipFile:
                        pass


                strictMode = False
                if jsFileContents.startswith(b"'use strict';") or jsFileContents.startswith(b'"use strict";'):
                    strictMode = True
                    jsFileContents = jsFileContents.replace(b"'use strict';", b"")
                    jsFileContents = jsFileContents.replace(b'"use strict";', b"")

                wrapperStart = b""
                wrapperEnd = b""
                for wrapper in self.knownResponseWrappers:
                    if jsFileContents.startswith(wrapper[0]) and jsFileContents.endswith(wrapper[1]):
                        jsFileContents = jsFileContents[len(wrapper[0]):-len(wrapper[1])]
                        wrapperStart = wrapper[0]
                        wrapperEnd = wrapper[1]

                sourceType = "script"
                if b"\nexport " in jsFileContents or b"\nimport " in jsFileContents:
                    sourceType = "module"

                result = subprocess.run(['babel', '-f', fileNameForBabel, '--plugins', 'babel-plugin-kwola', '--retain-lines', '--source-type', sourceType], input=jsFileContents, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                if result.returncode != 0:
                    print(datetime.now(), f"[{os.getpid()}]", f"Error! Unable to install Kwola line-counting in the Javascript file {fileName}. Most"
                                                              f" likely this is because Babel thinks your javascript has invalid syntax, or that"
                                                              f" babel is not working / not able to find the babel-plugin-kwola / unable to"
                                                              f" transpile the javascript for some other reason. See the following truncated"
                                                              f" output:", flush=True)

                    if len(result.stdout) > 0:
                        print(result.stdout[:250], flush=True)
                    else:
                        print("No data in standard output", flush=True)
                    if len(result.stderr) > 0:
                        print(result.stderr[:250], flush=True)
                    else:
                        print("No data in standard error output", flush=True)

                    self.saveInCache(shortFileHash, fileName, originalFileContents)
                    self.memoryCache[longFileHash] = originalFileContents
                else:
                    print(datetime.now(), f"[{os.getpid()}]", f"Successfully translated {fileName} with Kwola branch counting and event tracing.", flush=True)
                    transformed = wrapperStart + result.stdout + wrapperEnd

                    if strictMode:
                        transformed = b'"use strict";\n' + transformed

                    if gzipped:
                        transformed = gzip.compress(transformed, compresslevel=9)

                    self.saveInCache(shortFileHash, fileName, transformed)
                    self.memoryCache[longFileHash] = transformed

                    flow.response.data.headers['Content-Length'] = str(len(transformed))
                    flow.response.data.content = transformed
        except Exception as e:
            traceback.print_exc()
