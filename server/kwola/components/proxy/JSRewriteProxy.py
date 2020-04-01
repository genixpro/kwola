from mitmproxy import ctx
import subprocess
import io
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
                    print("error")
                    print(result.stdout)
                    print(result.stderr)
                else:
                    transformed = result.stdout

                    self.saveInCache(fileHash, fileName, transformed)
                    self.memoryCache[fileHash] = transformed

                    flow.response.data.headers['Content-Length'] = str(len(transformed))
                    flow.response.data.content = transformed
        except Exception as e:
            print(e)
