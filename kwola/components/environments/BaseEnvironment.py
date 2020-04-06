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






class BaseEnvironment:
    """
        This is a base class for all different types of environments in ...

        This is not 'environment' from a devops perspective, with say credentials and configuration. A Kwola Environment
        is the interface that connects directly to the software that we are trying to test. E.g. if we are testing
        something on the web, the environment is what spins up the the headless browser to interact with the software.
        If we are just direct testing a URL given to us, then that would represent a different kind of environment.


    """

    pass # Nothing yet

