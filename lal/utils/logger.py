import logging
import logging.config


class LALLogger:

    def __init__(self, logger_name):
        """

        :param logger_name:
        """

        # create logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

        # create handler
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

        # format handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(handler)

        self.log = logger

    def info(self, msg):
        """
        This create a info message for the logger

        :param msg:
        :return:
        """
        self.log.info(msg)

    def warning(self, msg):
        """
        This creates a warning message for the logger

        :param msg:
        :return:
        """
        self.log.warning(msg)

    def log_error(self, funct):
        """
        We add a decorator on the funct parameter to allow for easy logging at the error level.

        :param funct: The function to be decorated
        :return:
        """

        logger = self.log

        def decorated(*args, **kwargs):
            try:
                return funct(*args, **kwargs)
            except Exception as e:
                logger.error(e, exc_info=True)
                logger.error("We are returning none.")

                return None

        # making sure the name of the funct did not change
        decorated.__name__ = funct.__name__

        return decorated
