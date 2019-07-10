import logging
import logging.config


import sys
import traceback
import threading
import multiprocessing
from logging import FileHandler as FH


# ============================================================================
# Define Log Handler
# ============================================================================
class CustomLogHandler(logging.Handler):
    """multiprocessing log handler

    This handler makes it possible for several processes
    to log to the same file by using a queue.

    """
    def __init__(self, fname):
        logging.Handler.__init__(self)

        self._handler = FH(fname)
        self.queue = multiprocessing.Queue(-1)

        thrd = threading.Thread(target=self.receive)
        thrd.daemon = True
        thrd.start()

    def setFormatter(self, fmt):
        logging.Handler.setFormatter(self, fmt)
        self._handler.setFormatter(fmt)

    def receive(self):
        while True:
            try:
                record = self.queue.get()
                self._handler.emit(record)
            except (KeyboardInterrupt, SystemExit):
                raise
            except EOFError:
                break
            except:
                traceback.print_exc(file=sys.stderr)

    def send(self, s):
        self.queue.put_nowait(s)

    def _format_record(self, record):
        if record.args:
            record.msg = record.msg % record.args
            record.args = None
        if record.exc_info:
            dummy = self.format(record)
            record.exc_info = None

        return record

    def emit(self, record):
        try:
            s = self._format_record(record)
            self.send(s)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

    def close(self):
        self._handler.close()
        logging.Handler.close(self)


class LALLogger:

    def __init__(self, logger_name, handler=None):
        """

        :param logger_name:
        :param handler:
        """

        # create logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

        # create handler
        if handler is None:
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

                raise e

        # making sure the name of the funct did not change
        decorated.__name__ = funct.__name__
        decorated.__doc__ = funct.__doc__

        return decorated
