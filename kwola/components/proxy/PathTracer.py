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


class PathTracer:
    def __init__(self):
        self.seenPaths = set()
        self.recentPaths = set()

    def request(self, flow):
        # TODO: NEED A WAY TO SEPARATE OUT FLOWS FROM DIFFERENT TABS.
        self.seenPaths.add(flow.request.path)
        self.recentPaths.add(flow.request.path)


addons = [
    PathTracer()
]
