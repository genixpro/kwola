import urllib.parse
import hashlib
import base64
import re


class ProxyPluginBase:
    """
        Represents a plugin for the rewrite proxy
    """

    rewriteMode = None

    def shouldHandleFile(self, resource, fileData):
        pass


    def getRewriteMode(self, resource, fileData, resourceVersion, priorResourceVersion):
        pass


    def rewriteFile(self, resource, fileData, resourceVersion, priorResourceVersion):
        pass


    @staticmethod
    def getCleanedURL(url):
        parts = urllib.parse.urlparse(url)

        path = parts.path
        if path == "" or path == "/":
            fileName = "root"
        else:
            if path.endswith("/"):
                path = path[:-1]

            fileName = path.replace("/", "_")

        fileName = parts.hostname + "_" + fileName

        fileName = fileName.replace(".", "_")
        fileName = re.sub(r"\W", "_", fileName)

        return fileName

    @staticmethod
    def computeHash(fileData):
        """
            Computes a hash for the file data.

            The hash is a full md5 hash, encoded in base64 except with the extra 2 characters removed
            so its purely alphanumeric, although can vary in length.

            @returns longHash as a string
        """
        hasher = hashlib.sha256()
        hasher.update(fileData)

        base64ExtraCharacters = bytes("--", 'utf8')
        longHash = str(base64.b64encode(hasher.digest(), altchars=base64ExtraCharacters), 'utf8')
        longHash = longHash.replace("-", "")
        longHash = longHash.replace("=", "")

        return longHash
