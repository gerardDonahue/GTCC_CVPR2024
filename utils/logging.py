import logging


def configure_logging_format():
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    if logging.getLogger(__name__).hasHandlers():
        return logging.getLogger(__name__)

    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d] - %(message)s')
    console_handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger
