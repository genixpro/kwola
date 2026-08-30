#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#

from ..diagnostics.test_chromedriver import testChromedriver

def main():
    """
        Deprecated compatibility alias for the Playwright Chromium/Firefox diagnostic.
    """

    success = testChromedriver(verbose=True)
    if success:
        exit(0)
    else:
        exit(1)
