

class KwolaError(Exception):
    pass




class ProxyVerificationFailed(KwolaError):
    pass




class AutologinFailure(KwolaError):
    def __init__(self, message, autologinMoviePath=None):
        super(AutologinFailure, self).__init__(message)

        self.autologinMoviePath = autologinMoviePath



