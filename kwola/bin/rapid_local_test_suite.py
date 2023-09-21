#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#

import unittest
from ..diagnostics.test_installation import testInstallation
from ..config.logger import getLogger, setupLocalLogging
import logging

def main():
    """
        This is the entry point for the Kwola secondary command that runs a rapid test suite that can be used to diagnose whether your Kwola installation is working.
    """
    setupLocalLogging()
    success = testInstallation(verbose=True)
    if not success:
        print(
            "There appears to be a problem with your Kwola installation or environment. Exiting.")
        exit(1)

    suite = unittest.defaultTestLoader.loadTestsFromName("kwola.tests.test_training_loop.TestTrainingLoop.test_kros3_all_actions")

    runner = unittest.TextTestRunner(verbosity=3)
    runner.run(suite)

