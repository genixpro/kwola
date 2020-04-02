



class BaseEnvironment:
    """
        This is a base class for all different types of environments in Kwola.

        This is not 'environment' from a devops perspective, with say credentials and configuration. A Kwola Environment
        is the interface that connects directly to the software that we are trying to test. E.g. if we are testing
        something on the web, the environment is what spins up the the headless browser to interact with the software.
        If we are just direct testing a URL given to us, then that would represent a different kind of environment.


    """

    pass # Nothing yet

