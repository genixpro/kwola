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


from ..diagnostics.test_installation import testInstallation

def main():
    """
        This is the entry point for the Kwola secondary command, kwola_run_train_step.
    """
    success = testInstallation(verbose=True)
    if not success:
        print("There appears to be a problem with your Kwola installation or environment.")
        exit(1)
    else:
        print("Kwola was able to verify that your installation is working correctly. ")
        exit(0)
