from mitmproxy import ctx
import subprocess
import io
from mitmproxy.script import concurrent


class JSRewriteProxy:
    def __init__(self):
        self.cache = {}

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

        try:
            if '.js' in flow.request.path:
                if flow.request.path in self.cache:
                    transformed = self.cache[flow.request.path]

                    flow.response.data.headers['Content-Length'] = str(len(transformed))
                    flow.response.data.content = transformed
                    return

                filename = str(flow.request.path).split("/")[-1]

                result = subprocess.run(['babel','-f', filename, '--plugins', 'babel-plugin-kwola'], input=bytes(flow.response.data.content), capture_output=True)

                if result.returncode != 0:
                    print("error")
                    print(result.stdout)
                    print(result.stderr)
                else:
                    transformed = result.stdout

                    self.cache[flow.request.path] = transformed

                    flow.response.data.headers['Content-Length'] = str(len(transformed))
                    flow.response.data.content = transformed
        except Exception as e:
            print(e)


addons = [
    JSRewriteProxy()
]
