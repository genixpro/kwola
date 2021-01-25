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
