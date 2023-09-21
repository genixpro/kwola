#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#

from ..diagnostics.test_neural_network import testNeuralNetworkAllGPUs


def main():
    """
        This is the entry for the neural network testing command.
    """

    success = testNeuralNetworkAllGPUs(verbose=True)
    if success:
        exit(0)
    else:
        exit(1)

