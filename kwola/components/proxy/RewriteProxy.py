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


from ...config.logger import getLogger
from mitmproxy.script import concurrent
import os
import os.path
import traceback
import gzip
import filetype
import re
from ..plugins.base.ProxyPluginBase import ProxyPluginBase

class RewriteProxy:
    def __init__(self, config, plugins):
        self.config = config

        self.memoryCache = {}
        self.originalRewriteItemsBySize = {}
        self.plugins = plugins

    def getCacheFileName(self, fileHash, fileURL):
        fileName = ProxyPluginBase.getCleanedFileName(fileURL)

        fileNameSplit = fileName.split("_")

        if len(fileNameSplit) > 1:
            extension = "." + fileNameSplit[-1]
            fileNameRoot = "_".join(fileNameSplit[:-1])
        else:
            extension = ""
            fileNameRoot = fileName

        badChars = "%=~`!@#$^&*(){}[]\\|'\":;,<>/?+"
        for char in badChars:
            fileNameRoot = fileNameRoot.replace(char, "-")

        # Replace all unicode characters with -CODE-, with CODE being replaced by the unicode character code
        fileNameRoot = str(fileNameRoot.encode('ascii', 'xmlcharrefreplace'), 'ascii').replace("&#", "-").replace(";", "-")
        extension = str(extension.encode('ascii', 'xmlcharrefreplace'), 'ascii').replace("&#", "-").replace(";", "-")

        cacheFileName = os.path.join(self.config.getKwolaUserDataDirectory("proxy_cache"), fileNameRoot[:100] + "_" + fileHash + extension)

        return cacheFileName


    def findInCache(self, fileHash, fileURL):
        cacheFileName = self.getCacheFileName(fileHash, fileURL)
        if os.path.exists(cacheFileName):
            try:
                with open(cacheFileName, 'rb') as f:
                    return f.read()
            except OSError:
                return

    def saveInCache(self, fileHash, fileURL, data):
        cacheFileName = self.getCacheFileName(fileHash, fileURL)
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

    @concurrent
    def response(self, flow):
        """
            The full HTTP response has been read.
        """
        contentType = flow.response.headers.get('Content-Type')
        if contentType is None:
            contentType = flow.response.headers.get('content-type')

        # Don't attempt to transform if its not a 2xx, just let it pass through
        if flow.response.status_code < 200 or flow.response.status_code >= 300:
            for plugin in self.plugins:
                plugin.observeRequest(url=flow.request.url,
                                      statusCode=flow.response.status_code,
                                      contentType=contentType,
                                      headers=flow.request.headers,
                                      origFileData=bytes(flow.response.data.content),
                                      transformedFileData=bytes(flow.response.data.content),
                                      didTransform=False
                                      )
            return

        longFileHash, shortFileHash = ProxyPluginBase.computeHashes(bytes(flow.response.data.content))

        cached = self.memoryCache.get(longFileHash)
        if cached is None:
            cached = self.findInCache(shortFileHash, flow.request.url)
            if cached is not None:
                self.memoryCache[longFileHash] = cached

        if cached is not None:
            flow.response.data.headers['Content-Length'] = str(len(cached))
            flow.response.data.content = cached

            for plugin in self.plugins:
                plugin.observeRequest(url=flow.request.url,
                                      statusCode=flow.response.status_code,
                                      contentType=contentType,
                                      headers=flow.request.headers,
                                      origFileData=bytes(flow.response.data.content),
                                      transformedFileData=cached,
                                      didTransform=(len(flow.response.data.content) != len(cached)) # Cheap hack, should fix this to do a proper recording of whether
                                                                                                    # The transform happened
                                      )
            return

        fileURL = flow.request.url

        try:
            originalFileContents = bytes(flow.response.data.content)

            unzippedFileContents, gzipped = self.decompressDataIfNeeded(originalFileContents)

            chosenPlugin = None
            for plugin in self.plugins:
                if plugin.willRewriteFile(fileURL, contentType, unzippedFileContents):
                    chosenPlugin = plugin
                    break

            foundSimilarOriginal = False
            if chosenPlugin is not None:
                size = len(unzippedFileContents)
                if size not in self.originalRewriteItemsBySize:
                    self.originalRewriteItemsBySize[size] = []

                for sameSizedOriginal in self.originalRewriteItemsBySize[size]:
                    charsDifferent = 0
                    for chr, otherChr in zip(unzippedFileContents, sameSizedOriginal):
                        if chr != otherChr:
                            charsDifferent += 1
                    portionDifferent = charsDifferent / size
                    if portionDifferent < 0.20:
                        # Basically we are looking at what is effectively the same file with some minor differences.
                        # This is common with ad-serving, tracking tags and JSONP style responses.
                        foundSimilarOriginal = True
                        break

                if not foundSimilarOriginal:
                    self.originalRewriteItemsBySize[size].append(unzippedFileContents)

            if foundSimilarOriginal:
                # We don't translate it or save it in the cache. Just leave as is.
                getLogger().warning(f"Decided not to translate file {flow.request.url} because it looks extremely similar to a request we have already seen. This is probably a JSONP style response, and we don't translate these since they are only ever called once, but can clog up the system.")

                for plugin in self.plugins:
                    plugin.observeRequest(url=flow.request.url,
                                          statusCode=flow.response.status_code,
                                          contentType=contentType,
                                          headers=flow.request.headers,
                                          origFileData=originalFileContents,
                                          transformedFileData=originalFileContents,
                                          didTransform=False
                                          )
            elif chosenPlugin is not None:
                transformed = chosenPlugin.rewriteFile(fileURL, contentType, unzippedFileContents)

                if gzipped:
                    transformed = gzip.compress(transformed, compresslevel=9)

                self.saveInCache(shortFileHash, fileURL, transformed)
                self.memoryCache[longFileHash] = transformed

                flow.response.data.headers['Content-Length'] = str(len(transformed))
                flow.response.data.content = transformed

                for plugin in self.plugins:
                    plugin.observeRequest(url=flow.request.url,
                                          statusCode=flow.response.status_code,
                                          contentType=contentType,
                                          headers=flow.request.headers,
                                          origFileData=originalFileContents,
                                          transformedFileData=transformed,
                                          didTransform=True
                                          )

            else:
                self.saveInCache(shortFileHash, fileURL, originalFileContents)
                self.memoryCache[longFileHash] = originalFileContents

                for plugin in self.plugins:
                    plugin.observeRequest(url=flow.request.url,
                                          statusCode=flow.response.status_code,
                                          contentType=contentType,
                                          headers=flow.request.headers,
                                          origFileData=originalFileContents,
                                          transformedFileData=originalFileContents,
                                          didTransform=False
                                          )

        except Exception as e:
            getLogger().error(traceback.format_exc())
