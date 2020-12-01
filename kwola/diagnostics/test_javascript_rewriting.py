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
import sys

def testJavascriptRewriting(verbose=True):
    """
        This is the entry for the selenium testing command.
    """

    if verbose:
        print("Running babel with babel-plugin-kwola on some test javascript")

    testFile = """
    let test = 0;
    
    if (test != 0)
    {
        console.log("foo");
    }
    """

    babelCmd = 'babel'
    if sys.platform == "win32" or sys.platform == "win64":
        babelCmd = 'babel.cmd'

    result = subprocess.run([babelCmd, '-f', "test.js", '--plugins', 'babel-plugin-kwola'], input=bytes(testFile, "utf8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    failure = None

    if result.returncode != 0:
        if verbose:
            print(f"Error, process return code was not zero. It was {result.returncode}, indicating a failure.")
        failure = True

    if "kwola" not in str(result.stdout):
        if verbose:
            print(f"Error, did not detect kwola transpiled javascript code from the output of babel. This indicates a failure. Output is:")
            print(str(result.stdout, 'utf8'))
            print(str(result.stderr, 'utf8'))
        failure = True

    if failure:
        return False
    else:
        if verbose:
            print("Kwola was able to successfully translate sample javascript code. It looks like NodeJS, babel, and babel-plugin-kwola are installed correctly.")
        return True
