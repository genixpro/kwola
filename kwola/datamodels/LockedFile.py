#
#     Kwola is an AI algorithm that learns how to use other programs
#     automatically so that it can find bugs in them.
#
#     Copyright (C) 2020 Kwola Software Testing Inc.
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU Affero General Public License as
#     published by the Free Software Foundation, either version 3 of the
#     License, or (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
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

