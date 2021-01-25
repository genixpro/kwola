
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

from .test_chromedriver import testChromedriver
from .test_ffmpeg import testFfmpeg
from .test_neural_network import testNeuralNetworkAllGPUs
from .test_javascript_rewriting import testJavascriptRewriting
import multiprocessing
import os
import pickle
import datetime

def testInstallation(verbose=True):
    """
        This function is used to quickly test the Kwola installation prior to running
        a command.
    """

    if os.path.exists(".test_installation_success"):
        with open(".test_installation_success", "rb") as f:
            testDate = pickle.load(f)

            if abs((datetime.datetime.now() - testDate).total_seconds()) < 7200:
                return True

    if verbose:
        print("Verifying your Kwola installation ... Please wait a moment")

    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    subprocessPool = multiprocessing.Pool(4)

    chromedriverWorking = subprocessPool.apply_async(testChromedriver, kwds={"verbose": False})
    ffmpegWorking = subprocessPool.apply_async(testFfmpeg, kwds={"verbose": False})
    neuralNetworkWorking = subprocessPool.apply_async(testNeuralNetworkAllGPUs, kwds={"verbose": False})
    jsRewritingWorking = subprocessPool.apply_async(testJavascriptRewriting, kwds={"verbose": False})

    chromedriverWorking = chromedriverWorking.get()
    ffmpegWorking = ffmpegWorking.get()
    neuralNetworkWorking = neuralNetworkWorking.get()
    jsRewritingWorking = jsRewritingWorking.get()

    subprocessPool.close()
    subprocessPool.join()

    if chromedriverWorking and ffmpegWorking and neuralNetworkWorking and jsRewritingWorking:
        if verbose:
            print("Everything in your Kwola installation appears to be working! Excellent.")

        with open(".test_installation_success", "wb") as f:
            pickle.dump(datetime.datetime.now(), f)

        return True
    else:
        if verbose:
            if not chromedriverWorking:
                print("Your Chrome & Chromedriver installation does not appear to be working. Please check your "
                      "Chrome & Chromedriver installation and ensure you can run kwola_test_chromedriver successfully.")
            if not ffmpegWorking:
                print("Your ffmpeg installation does not appear to be working. Please ensure you have ffmpeg installed"
                      "and it is accessible via PATH. You can test your installation using kwola_test_ffmpeg")
            if not neuralNetworkWorking:
                print("Something in your Pytorch / CUDA / NVIDIA / GPU installation. does not appear to be working."
                      " Please double check your installation of these tools and ensure you can run kwola_test_neural_network"
                      " successfully.")
            if not jsRewritingWorking:
                print("Something in your NodeJS / babel / babel-plugin-kwola installation does not appear to be working."
                      " Please double check your installation of these tools and ensure you can run kwola_test_javascript_rewriting"
                      " successfully.")
        return False
