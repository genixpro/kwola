import time
import os.path
import fcntl

class LockedFile(object):
    def __init__(self, filePath, mode):
        self.filePath = filePath
        self.fileName = filePath.split("/")[-1]
        self.folder = "/".join(filePath.split("/")[:-1])

        self.lockFile = os.path.join(self.folder, "." + self.fileName + ".lock")
        self.mode = mode

    def __enter__(self):
        if 'w' in self.mode:
            while True:
                try:
                    open(self.lockFile, "x")
                    break
                except FileExistsError:
                    time.sleep(0.05)

        if os.path.exists(self.filePath):
            self.readFile = open(self.filePath, 'r')
        else:
            try:
                self.readFile = open(self.filePath, 'x')
            except FileExistsError:
                self.readFile = open(self.filePath, 'r')

        fcntl.flock(self.readFile, fcntl.LOCK_EX)
        self.file = open(self.filePath, self.mode)
        return self.file

    def __exit__(self, type, value, traceback):
        if 'w' in self.mode:
            os.unlink(self.lockFile)

        fcntl.flock(self.readFile, fcntl.LOCK_UN)
        self.file.close()
        self.readFile.close()

