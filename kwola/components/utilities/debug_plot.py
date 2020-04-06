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


import matplotlib.pyplot as plt
import numpy


countDebugImageN = 0
uniqueDebugPlotTitles = set()
countDebugImageSets = 10

def showRewardImageDebug(image, title="", vmin=None, vmax=None):
    global countDebugImageN, uniqueDebugPlotTitles

    uniqueDebugPlotTitles.add(title)

    mainColorMap = plt.get_cmap('inferno')
    mainFigure = plt.figure(figsize=((10, 10)), dpi=100)
    imageAxes = mainFigure.add_subplot(1, 1, 1)

    imageAxes.set_xticks([])
    imageAxes.set_yticks([])
    im = imageAxes.imshow(image, cmap=mainColorMap, vmin=vmin, vmax=vmax)
    mainFigure.colorbar(im, ax=imageAxes, orientation='vertical')

    if title:
        imageAxes.set_title(title)

    # ax.grid()
    mainFigure.tight_layout()
    # mainFigure.canvas.draw()

    debugImageNumber = countDebugImageN
    countDebugImageN = (countDebugImageN + 1) % (len(uniqueDebugPlotTitles) * countDebugImageSets)

    mainFigure.savefig(f"/home/bradley/debug-{debugImageNumber}-{title}.png")
