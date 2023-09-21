#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#

from ..diagnostics.test_chromedriver import testChromedriver

def main():
    """
        This is the entry for the command which tests your chrome & chromedriver installation to see if
        Selenium is able to interact with it successfully.
    """

    success = testChromedriver(verbose=True)
    if success:
        exit(0)
    else:
        exit(1)

