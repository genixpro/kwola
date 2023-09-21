#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
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
