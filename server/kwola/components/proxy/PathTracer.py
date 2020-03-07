from mitmproxy import ctx
import subprocess
import io
from mitmproxy.script import concurrent


class PathTracer:
    def __init__(self):
        self.seenPaths = set()
        self.recentPaths = set()

    def request(self, flow):
        self.seenPaths.add(flow.request.path)
        self.recentPaths.add(flow.request.path)


addons = [
    PathTracer()
]
