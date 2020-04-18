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

import subprocess


def testFfmpeg(verbose=True):
    """
        This function is used to test whether FFMPEG is working correctly.
    """

    if verbose:
        print("Testing running ffmpeg ...")

    result = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    failure = None

    if result.returncode != 0:
        if verbose:
            print(f"Error, process return code was not zero. It was {result.returncode}, indicating a failure. Please make sure ffmpeg is installed and is accessible from the console.")
        failure = True

    if failure:
        return False
    else:
        if verbose:
            print("Kwola was successfully able to run ffmpeg.")
        return True
