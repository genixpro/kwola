#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
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
