import logging

logger = logging.getLogger()

kwolaLoggingFormatString = '%(asctime)s %(levelname)s %(filename)s:%(lineno)d [%(process)d] %(message)s'
kwolaDateFormatString = "%b%d %H:%M:%S"

def setupLocalLogging():
    global logger
    logging.basicConfig(format=kwolaLoggingFormatString, datefmt=kwolaDateFormatString)
    logger.setLevel(logging.INFO)

def getLogger():
    global logger
    return logger

def setLogger(newLogger):
    global logger
    logger = newLogger


