import logging
import os


class AIStackLogger:

    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger('AIStackLogger')
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(log_level)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)

    def warning(self, message):
        self.logger.warning(message)


LOG = AIStackLogger(log_level=os.getenv('LOG_LEVEL', logging.INFO))
