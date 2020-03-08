from mitmproxy import ctx
import subprocess
import io
from mitmproxy.script import concurrent
import redis
import hashlib


class JSRewriteProxy:
    def __init__(self):
        self.cache = redis.Redis(db=2)

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
        try:
            if '.js' in flow.request.path:
                cached = self.cache.get(fileHash)

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

                    self.cache.set(fileHash, transformed)

                    flow.response.data.headers['Content-Length'] = str(len(transformed))
                    flow.response.data.content = transformed
        except Exception as e:
            print(e)


addons = [
    JSRewriteProxy()
]
