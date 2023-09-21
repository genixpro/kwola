#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
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
