#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#


import time
import os.path
import sys
if sys.platform != "win32" and sys.platform != "win64":
    import fcntl

class LockedFile(object):
    def __init__(self, filePath, mode):
        self.filePath = filePath

        self.lockFile = LockedFile.getLockFilePath(filePath)
        self.mode = mode

    @staticmethod
    def getLockFilePath(filePath):
        fileName = os.path.split(filePath)[-1]
        folder = os.path.join(*os.path.split(filePath)[:-1])
        lockFile = os.path.join(folder, "." + fileName + ".lock")
        return lockFile

    @staticmethod
    def clearLockFile(filePath):
        path = LockedFile.getLockFilePath(filePath)
        if os.path.exists(path):
            os.unlink(path)

    def __enter__(self):
        if 'w' in self.mode:
            while True:
                try:
                    with open(self.lockFile, "x"):
                        pass
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

        if sys.platform != "win32" and sys.platform != "win64":
            fcntl.flock(self.readFile, fcntl.LOCK_EX)
        self.file = open(self.filePath, self.mode)
        return self.file

    def __exit__(self, type, value, traceback):
        if 'w' in self.mode:
            os.unlink(self.lockFile)

        if sys.platform != "win32" and sys.platform != "win64":
            fcntl.flock(self.readFile, fcntl.LOCK_UN)
        self.file.close()
        self.readFile.close()

