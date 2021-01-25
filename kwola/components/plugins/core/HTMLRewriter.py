from ..base.ProxyPluginBase import ProxyPluginBase
import filetype
import re


class HTMLRewriter(ProxyPluginBase):
    """
        Represents a plugin for the rewrite proxy
    """

    rewritePluginName = "html"
    rewritePluginShouldCheckForSimilarOriginals = False

    def __init__(self, config):
        self.config = config

        # We want to strip out any "integrity" attributes that we see on html elements
        self.integrityRegex = re.compile(r"integrity\w*=\w*['\"]sha\d\d?\d?-[a-zA-Z0-9+/=]+['\"]")


    def shouldHandleFile(self, resource, fileData):
        if '.js' not in resource.url and not ".json" in resource.url and ".css" not in resource.url:
            kind = filetype.guess(fileData)
            mime = ''
            if kind is not None:
                mime = kind.mime

            # Next, check to see that we haven't gotten an image or something else that we should ignore.
            if mime.startswith("image/") or mime.startswith("video/") or mime.startswith("audio/") or (
                    mime.startswith("application/") and not mime.startswith("application/html")):
                return False

            try:
                stringFileContents = str(fileData, 'utf8').lower()
            except UnicodeDecodeError:
                return False

            if "</html" in stringFileContents or "</body" in stringFileContents:
                return True
            else:
                return False
        else:
            return False


    def getRewriteMode(self, resource, fileData, resourceVersion, priorResourceVersion):
        return "integrity_attribute_replacement", None


    def rewriteFile(self, resource, fileData, resourceVersion, priorResourceVersion):
        stringData = str(fileData, 'utf8')

        stringData = re.sub(self.integrityRegex, "", stringData)
        bytesData = bytes(stringData, "utf8")

        return bytesData
