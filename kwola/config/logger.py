import logging

logger = logging.getLogger()

def setupLocalLogging():
    logging.basicConfig(format='%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s', datefmt="%b%d %H:%M:%S")
    logger.setLevel(logging.INFO)

def getLogger():
    global logger
    return logger

def setLogger(newLogger):
    global logger
    logger = newLogger


