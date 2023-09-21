#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#

from ..diagnostics.test_ffmpeg import testFfmpeg


def main():
    """
        This is the entry for the selenium testing command.
    """

    success = testFfmpeg(verbose=True)
    if success:
        exit(0)
    else:
        exit(1)
