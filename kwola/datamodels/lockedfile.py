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

