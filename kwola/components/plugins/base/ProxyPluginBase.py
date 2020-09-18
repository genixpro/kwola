import urllib.parse
import hashlib
import base64



class ProxyPluginBase:
    """
        Represents a plugin for the rewrite proxy
    """

    def willRewriteFile(self, url, contentType, fileData):
        pass


    def rewriteFile(self, url, contentType, fileData):
        pass


    def observeRequest(self, url, statusCode, contentType, headers, origFileData, transformedFileData, didTransform):
        pass


    @staticmethod
    def getCleanedFileName(path):
        fileName = urllib.parse.unquote(path.split("/")[-1])
        if "?" in fileName:
            fileName = fileName.split("?")[0]
        if "#" in fileName:
            fileName = fileName.split("#")[0]
        fileName = fileName.replace(".", "_")
        return fileName

    @staticmethod
    def computeHashes(fileData):
        """
            Computes two hashes for the given data. A short hash and a long hash.

            The long hash is a full md5 hash, encoded in base64 except with the extra 2 characters removed
            so its purely alphanumeric, although can vary in length.

            The short hash is a short, six character hash which helps uniquely identify the file when used
            alongside the filename. Its also purely alphanumeric and only in lowercase.

            @returns (longHash, shortHash) a tuple with two strings
        """
        hasher = hashlib.sha256()
        hasher.update(fileData)

        base64ExtraCharacters = bytes("--", 'utf8')
        longHash = str(base64.b64encode(hasher.digest(), altchars=base64ExtraCharacters), 'utf8')
        longHash = longHash.replace("-", "")
        longHash = longHash.replace("=", "")

        shortHashLength = 6
        shortHash = longHash[::int(len(longHash)/shortHashLength)][:shortHashLength].lower()

        return longHash, shortHash
