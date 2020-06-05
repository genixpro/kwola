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
from ...config.logger import getLogger
from mitmproxy.script import concurrent
import hashlib
import os
import os.path
import subprocess
import base64
import traceback
import re
import urllib.parse
import gzip
import filetype
import json

class JSRewriteProxy:
    def __init__(self, config):
        self.config = config

        self.memoryCache = {}

        self.knownResponseWrappers = [
            (b"""<!--/*--><html><body><script type="text/javascript"><!--//*/""",
             b"""""")
        ]

    def getCacheFileName(self, fileHash, fileName):
        badChars = "%=~`!@#$^&*(){}[]\\|'\":;,<>/?+"
        for char in badChars:
            fileName = fileName.replace(char, "-")

        cacheFileName = os.path.join(self.config.getKwolaUserDataDirectory("javascript"), fileHash + "_" + fileName[:100])

        return cacheFileName


    def findInCache(self, fileHash, fileName):
        cacheFileName = self.getCacheFileName(fileHash, fileName)
        if os.path.exists(cacheFileName):
            try:
                with open(cacheFileName, 'rb') as f:
                    return f.read()
            except OSError:
                return

    def saveInCache(self, fileHash, fileName, data):
        cacheFileName = self.getCacheFileName(fileHash, fileName)
        try:
            with open(cacheFileName, 'wb') as f:
                return f.write(data)
        except FileExistsError:
            pass
        except OSError:
            pass


    def request(self, flow):
        flow.request.headers['Accept-Encoding'] = 'identity'


    @concurrent
    def responseheaders(self, flow):
        """
            The full HTTP response has been read.
        """

        # Check to see if there is an integrity verification header, if so, delete it
        headers = set(flow.response.headers.keys())
        #
        # if "Integrity" in headers:
        #     getLogger().info("Deleting an Integrity header")
        #     del flow.response.headers['Integrity']
        # if "integrity" in headers:
        #     getLogger().info("Deleting an Integrity header")
        #     del flow.response.headers['integrity']


    def computeHashes(self, data):
        """
            Computes two hashes for the given data. A short hash and a long hash.

            The long hash is a full md5 hash, encoded in base64 except with the extra 2 characters removed
            so its purely alphanumeric, although can vary in length.

            The short hash is a short, six character hash which helps uniquely identify the file when used
            alongside the filename. Its also purely alphanumeric and only in lowercase.

            @returns (longHash, shortHash) a tuple with two strings
        """
        hasher = hashlib.sha256()
        hasher.update(data)

        base64ExtraCharacters = bytes("--", 'utf8')
        longHash = str(base64.b64encode(hasher.digest(), altchars=base64ExtraCharacters), 'utf8')
        longHash = longHash.replace("-", "")
        longHash = longHash.replace("=", "")

        shortHashLength = 6
        shortHash = longHash[::int(len(longHash)/shortHashLength)][:shortHashLength].lower()

        return longHash, shortHash

    def decompressDataIfNeeded(self, data):
        gzipped = False

        data = bytes(data)

        kind = filetype.guess(data)
        mime = ''
        if kind is not None:
            mime = kind.mime

        # Decompress the file if it appears to be a gzip archive
        if mime == "application/gzip":
            try:
                data = gzip.decompress(data)
                gzipped = True
            except OSError:
                pass

        return data, gzipped

    def doesFlowLooksLikeJavascriptFile(self, flow):
        cleanedFileName = self.getCleanedFileName(flow)

        contentType = flow.response.headers.get('Content-Type')
        if contentType is None:
            contentType = flow.response.headers.get('content-type')

        jsMimeTypes = [
            "application/x-javascript",
            "application/javascript",
            "application/ecmascript",
            "text/javascript",
            "text/ecmascript"
        ]

        if ('_js' in cleanedFileName and not "_json" in cleanedFileName and not "_jsp" in cleanedFileName and not cleanedFileName.endswith("_css")) or str(contentType).strip().lower() in jsMimeTypes:
            fileContents, gzipped = self.decompressDataIfNeeded(flow.response.data.content)

            kind = filetype.guess(fileContents)
            mime = ''
            if kind is not None:
                mime = kind.mime

            # Next, check to see that we haven't gotten an image or something else that we should ignore. This happens, surprisingly.
            if mime.startswith("image/") or mime.startswith("video/") or mime.startswith("audio/") or mime.startswith("application/"):
                return False

            # For some reason, some websites send JSON data in files labelled as javascript files.
            # So we have to double check to make sure we aren't looking at JSON data
            try:
                json.loads(str(fileContents, 'utf8').lower())
                return False
            except json.JSONDecodeError:
                pass
            except UnicodeDecodeError:
                pass

            if fileContents.startswith(b"<html>"):
                return False

            return True
        else:
            return False

    def findMatchingJavascriptFilenameIgnoreKeyword(self, fileName):
        for ignoreKeyword in self.config['web_session_ignored_javascript_file_keywords']:
            if ignoreKeyword in fileName:
                return ignoreKeyword

        return None

    def doesFlowLooksLikeHTML(self, flow):
        cleanedFileName = self.getCleanedFileName(flow)

        if '_js' not in cleanedFileName and not "_json" in cleanedFileName and "_css" not in cleanedFileName:
            fileContents, gzipped = self.decompressDataIfNeeded(flow.response.data.content)

            kind = filetype.guess(fileContents)
            mime = ''
            if kind is not None:
                mime = kind.mime

            # Next, check to see that we haven't gotten an image or something else that we should ignore.
            if mime.startswith("image/") or mime.startswith("video/") or mime.startswith("audio/") or (mime.startswith("application/") and not mime.startswith("application/html")):
                return False

            try:
                stringFileContents = str(fileContents, 'utf8').lower()
            except UnicodeDecodeError:
                return False

            if "</html" in stringFileContents or "</body" in stringFileContents:
                return True
            else:
                return False
        else:
            return False

    def getCleanedFileName(self, flow):
        fileName = urllib.parse.unquote(flow.request.path.split("/")[-1])
        if "?" in fileName:
            fileName = fileName.split("?")[0]
        if "#" in fileName:
            fileName = fileName.split("#")[0]
        fileName = fileName.replace(".", "_")
        return fileName

    def rewriteJavascript(self, data, fileName, fileNameForBabel):
        jsFileContents = data.strip()
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

        result = subprocess.run(['babel', '-f', fileNameForBabel, '--plugins', 'babel-plugin-kwola', '--retain-lines', '--source-type', "script"], input=jsFileContents, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0 and "'import' and 'export' may appear only with" in str(result.stderr, 'utf8'):
            result = subprocess.run(['babel', '-f', fileNameForBabel, '--plugins', 'babel-plugin-kwola', '--retain-lines', '--source-type', "module"], input=jsFileContents, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            cutoffLength = 250

            kind = filetype.guess(data)
            mime = ''
            if kind is not None:
                mime = kind.mime

            getLogger().warning(f"[{os.getpid()}] Unable to install Kwola line-counting in the Javascript file {fileName}. Most"
                                                      f" likely this is because Babel thinks your javascript has invalid syntax, or that"
                                                      f" babel is not working / not able to find the babel-plugin-kwola / unable to"
                                                      f" transpile the javascript for some other reason. See the following truncated"
                                                      f" output:")

            if len(result.stdout) > 0:
                getLogger().warning(result.stdout[:cutoffLength])
            else:
                getLogger().warning("No data in standard output")
            if len(result.stderr) > 0:
                getLogger().warning(result.stderr[:cutoffLength])
            else:
                getLogger().warning("No data in standard error output")

            return data
        else:
            getLogger().info(f"[{os.getpid()}] Successfully translated {fileName} with Kwola branch counting and event tracing.")
            transformed = wrapperStart + result.stdout + wrapperEnd

            if strictMode:
                transformed = b'"use strict";\n' + transformed

            return transformed

    def rewriteHTML(self, data):
        stringData = str(data, 'utf8')

        # We want to strip out any "integrity" attributes that we see on html elements
        integrityRegex = re.compile(r"integrity\w*=\w*['\"]sha\d\d?\d?-[a-zA-Z0-9+/=]+['\"]")

        stringData = re.sub(integrityRegex, "", stringData)
        bytesData = bytes(stringData, "utf8")

        return bytesData

    @concurrent
    def response(self, flow):
        """
            The full HTTP response has been read.
        """

        # Ignore it if its not a 2xx
        if flow.response.status_code < 200 or flow.response.status_code >= 300:
            return

        longFileHash, shortFileHash = self.computeHashes(bytes(flow.response.data.content))
        fileName = self.getCleanedFileName(flow)

        cached = self.memoryCache.get(longFileHash)
        if cached is None:
            cached = self.findInCache(shortFileHash, fileName)
            if cached is not None:
                self.memoryCache[longFileHash] = cached

        if cached is not None:
            flow.response.data.headers['Content-Length'] = str(len(cached))
            flow.response.data.content = cached
            return

        try:
            originalFileContents = bytes(flow.response.data.content)

            if self.doesFlowLooksLikeJavascriptFile(flow):
                ignoreKeyword = self.findMatchingJavascriptFilenameIgnoreKeyword(fileName)
                if ignoreKeyword is None:
                    fileNameForBabel = shortFileHash + "-" + fileName

                    jsFileContents, gzipped = self.decompressDataIfNeeded(originalFileContents)

                    transformed = self.rewriteJavascript(jsFileContents, fileName, fileNameForBabel)

                    if gzipped:
                        transformed = gzip.compress(transformed, compresslevel=9)

                    self.saveInCache(shortFileHash, fileName, transformed)
                    self.memoryCache[longFileHash] = transformed

                    flow.response.data.headers['Content-Length'] = str(len(transformed))
                    flow.response.data.content = transformed
                else:
                    getLogger().info(f"[{os.getpid()}] Warning: Ignoring the javascript file '{fileName}' because it matches the javascript ignore keyword '{ignoreKeyword}'. "
                                                              f"This means that no learnings will take place on the code in this file. If this file is actually part of your "
                                                              f"application and should be learned on, then please modify your config file kwola.json and remove the ignore "
                                                              f"keyword '{ignoreKeyword}' from the variable 'web_session_ignored_javascript_file_keywords'. This file will be "
                                                              f"cached without Kwola line counting installed. Its faster to install line counting only in the files that need "
                                                              f"it.")

                    self.saveInCache(shortFileHash, fileName, originalFileContents)
                    self.memoryCache[longFileHash] = originalFileContents
            elif self.doesFlowLooksLikeHTML(flow):
                htmlFileContents, gzipped = self.decompressDataIfNeeded(originalFileContents)

                transformed = self.rewriteHTML(htmlFileContents)

                if gzipped:
                    transformed = gzip.compress(transformed, compresslevel=9)

                self.saveInCache(shortFileHash, fileName, transformed)
                self.memoryCache[longFileHash] = transformed

                flow.response.data.headers['Content-Length'] = str(len(transformed))
                flow.response.data.content = transformed
            else:
                self.saveInCache(shortFileHash, fileName, originalFileContents)
                self.memoryCache[longFileHash] = originalFileContents
        except Exception as e:
            getLogger().error(traceback.format_exc())
