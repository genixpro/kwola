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
