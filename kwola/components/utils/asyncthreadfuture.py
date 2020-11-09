import threading
import datetime
import time


class AsyncThreadFuture:
    """
        This class is a replacement for concurrent.futures. Unfortunately concurrent.futures.ThreadPoolExecutor does
        not reliably handle timeouts. Sometimes the executed code can exceed the timeout, by quite a lot, and still
        no timeout error is triggered. So we just made our own custom implementation to get around this problem. It
        uses a simple polling based mechanism instead of the fancier event based implementation in concurrent.futures,
        which depends on fancy kernel mechanics. Its not an overly performant solution, but its simple and works
        reliably. The performance isn't much of an issue unless you have a large number of AsyncThreadFuture's
        running in the same process, which in Kwola this is going to be like 10-20 at most. So not really an issue.
    """
    def __init__(self, func, args, timeout=None):
        self.timeout = timeout
        self.func = func
        self.args = args
        self.start = datetime.datetime.now()
        self.resultObj = None
        self.error = None
        self.complete = False
        self.failed = False
        self.mainThread = threading.Thread(target=self.main)
        self.mainThread.start()

    def main(self):
        try:
            result = self.func(*self.args)
            if not self.complete:
                self.resultObj = result
                self.complete = True
        except Exception as e:
            self.error = e
            self.failed = True
            self.complete = True

    def wait(self):
        while not self.complete:
            elapsed = (datetime.datetime.now() - self.start).total_seconds()
            if self.timeout is not None:
                if elapsed > self.timeout:
                    self.complete = True
                    raise TimeoutError()
                else:
                    time.sleep(0.25)

    def result(self):
        self.wait()

        if self.failed:
            raise self.error

        return self.resultObj
