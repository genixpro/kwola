import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s', datefmt="%b%d %H:%M:%S")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def getLogger():
    global logger
    return logger

def setLogger(newLogger):
    global logger
    logger = newLogger


