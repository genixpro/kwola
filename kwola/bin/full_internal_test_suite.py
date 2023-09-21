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
        This is the entry point for for the kwola full testing sequence.
    """
    setupLocalLogging()
    success = testInstallation(verbose=True)
    if not success:
        print(
            "There appears to be a problem with your Kwola installation or environment. Exiting.")
        exit(1)

    runner = unittest.TextTestRunner(verbosity=3)
    # runner.run(unittest.defaultTestLoader.discover("kwola.tests"))

    # runner.run(unittest.defaultTestLoader.loadTestsFromName("kwola.tests.test_rewrite_proxy.TestRewriteProxy"))
    # runner.run(unittest.defaultTestLoader.loadTestsFromName("kwola.tests.test_deunique_string.TestDeuniqueString"))
    runner.run(unittest.defaultTestLoader.loadTestsFromName("kwola.tests.test_html_saver.TestHTMLSaver"))
    # runner.run(unittest.defaultTestLoader.loadTestsFromName("kwola.tests.test_training_loop.TestTrainingLoop"))
    # runner.run(unittest.defaultTestLoader.loadTestsFromName("kwola.tests.test_end_to_end.TestEndToEnd"))
