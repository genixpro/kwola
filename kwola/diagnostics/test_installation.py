
#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
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
