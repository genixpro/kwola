from ..base.ProxyPluginBase import ProxyPluginBase
import filetype
import re
import json
import os
import subprocess
import re
from kwola.config.logger import getLogger



class JSRewriter(ProxyPluginBase):
    """
        Represents a plugin for the rewrite proxy
    """

    knownResponseWrappers = [
        (b"""<!--/*--><html><body><script type="text/javascript"><!--//*/""",
         b"""""")
    ]

    def __init__(self, config):
        self.config = config

        self.multipleBranchesCheckingRegex = re.compile(b"globalKwolaCounter_\\d{1,6}\\[1\\] \\+= 1;")

    def willRewriteFile(self, url, contentType, fileData):
        jsMimeTypes = [
            "application/x-javascript",
            "application/javascript",
            "application/ecmascript",
            "text/javascript",
            "text/ecmascript"
        ]

        cleanedFileName = self.getCleanedFileName(url)

        if ('_js' in url and not "_json" in url and not "_jsp" in url and not url.endswith("_css")) or str(contentType).strip().lower() in jsMimeTypes:
            kind = filetype.guess(fileData)
            mime = ''
            if kind is not None:
                mime = kind.mime

            # Next, check to see that we haven't gotten an image or something else that we should ignore. This happens, surprisingly.
            if mime.startswith("image/") or mime.startswith("video/") or mime.startswith("audio/") or mime.startswith("application/"):
                return False

            # For some reason, some websites send JSON data in files labelled as javascript files.
            # So we have to double check to make sure we aren't looking at JSON data
            try:
                json.loads(str(fileData, 'utf8').lower())
                return False
            except json.JSONDecodeError:
                pass
            except UnicodeDecodeError:
                pass

            if fileData.startswith(b"<html>"):
                return False

            ignoreKeyword = self.findMatchingJavascriptFilenameIgnoreKeyword(cleanedFileName)
            if ignoreKeyword is None:
                return True
            else:
                getLogger().info(
                    f"[{os.getpid()}] Warning: Ignoring the javascript file '{cleanedFileName}' because it matches the javascript ignore keyword '{ignoreKeyword}'. "
                    f"This means that no learnings will take place on the code in this file. If this file is actually part of your "
                    f"application and should be learned on, then please modify your config file kwola.json and remove the ignore "
                    f"keyword '{ignoreKeyword}' from the variable 'web_session_ignored_javascript_file_keywords'. This file will be "
                    f"cached without Kwola line counting installed. Its faster to install line counting only in the files that need "
                    f"it.")
                return False
        else:
            return False

    def checkIfRewrittenJSFileHasMultipleBranches(self, rewrittenJSFileData):
        # This method is used to check if the given javascript file, which has already been rewritten,
        # has multiple branches. It is common for old-school jsonp-style requests to use javascript
        # files with no branches and only a single function call. These can clog up the Kwola system
        # without actually delivering any value, since they are only called once.
        match = self.multipleBranchesCheckingRegex.search(rewrittenJSFileData)
        if match is None:
            return False
        else:
            return True


    def rewriteFile(self, url, contentType, fileData):
        jsFileContents = fileData.strip()

        strictMode = False
        if jsFileContents.startswith(b"'use strict';") or jsFileContents.startswith(b'"use strict";'):
            strictMode = True
            jsFileContents = jsFileContents.replace(b"'use strict';", b"")
            jsFileContents = jsFileContents.replace(b'"use strict";', b"")

        wrapperStart = b""
        wrapperEnd = b""
        for wrapper in JSRewriter.knownResponseWrappers:
            if jsFileContents.startswith(wrapper[0]) and jsFileContents.endswith(wrapper[1]):
                jsFileContents = jsFileContents[len(wrapper[0]):-len(wrapper[1])]
                wrapperStart = wrapper[0]
                wrapperEnd = wrapper[1]

        cleanedFileName = self.getCleanedFileName(url)
        longFileHash, shortFileHash = ProxyPluginBase.computeHashes(bytes(fileData))

        fileNameForBabel = shortFileHash + "_" + cleanedFileName

        result = subprocess.run(
            ['babel', '-f', fileNameForBabel, '--plugins', 'babel-plugin-kwola', '--retain-lines', '--source-type',
             "script"], input=jsFileContents, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0 and "'import' and 'export' may appear only with" in str(result.stderr, 'utf8'):
            result = subprocess.run(
                ['babel', '-f', fileNameForBabel, '--plugins', 'babel-plugin-kwola', '--retain-lines', '--source-type',
                 "module"], input=jsFileContents, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            cutoffLength = 250

            kind = filetype.guess(fileData)
            mime = ''
            if kind is not None:
                mime = kind.mime

            getLogger().warning(
                f"[{os.getpid()}] Unable to install Kwola line-counting in the Javascript file {url}. Most"
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

            return fileData
        else:
            # Check to see if the resulting file object had multiple branches
            if not self.checkIfRewrittenJSFileHasMultipleBranches(result.stdout):
                getLogger().warning(f"[{os.getpid()}] Ignoring the javascript file {url} because it looks like a JSONP-style request, or some other javascript "
                                    f"file without a significant number of code branches.")
                return fileData

            getLogger().info(
                f"[{os.getpid()}] Successfully translated {url} with Kwola branch counting and event tracing.")
            transformed = wrapperStart + result.stdout + wrapperEnd

            if strictMode:
                transformed = b'"use strict";\n' + transformed

            return transformed

    def findMatchingJavascriptFilenameIgnoreKeyword(self, fileName):
        for ignoreKeyword in self.config['web_session_ignored_javascript_file_keywords']:
            if ignoreKeyword in fileName:
                return ignoreKeyword

        return None



    def observeRequest(self, url, statusCode, contentType, headers, origFileData, transformedFileData, didTransform):
        pass



