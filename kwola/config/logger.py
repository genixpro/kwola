import logging

logger = logging.getLogger()

kwolaLoggingFormatString = '%(asctime)s %(levelname)s %(filename)s:%(lineno)d [%(process)d] %(message)s'
kwolaDateFormatString = "%b%d %H:%M:%S"

def setupLocalLogging(config=None):
    global logger
    logging.basicConfig(format=kwolaLoggingFormatString, datefmt=kwolaDateFormatString)
    logger.setLevel(logging.INFO)

    if config is not None:
        from .config import KwolaCoreConfiguration
        config = KwolaCoreConfiguration(config)
        if config['enable_google_cloud_logging']:
            import google.cloud.logging

            client = google.cloud.logging.Client()
            client.setup_logging()
            handler = client.get_default_handler()
            handler.setFormatter(logging.Formatter(kwolaLoggingFormatString, kwolaDateFormatString))

            logger.handlers = logger.handlers[0:1]

        if config['enable_slack_logging']:
            from kwolacloud.helpers.slack import SlackLogHandler
            logger.addHandler(SlackLogHandler())

def getLogger():
    global logger
    return logger

def setLogger(newLogger):
    global logger
    logger = newLogger


