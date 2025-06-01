import logging

import time
import os

root_dir = os.path.dirname(os.path.abspath(__file__))
default_dir = os.path.join(root_dir, "logs")
class Logger():
    def __init__(self,log_name,log_dir = None):
        if log_dir is None:
            log_dir = default_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.logger = logging.getLogger("logger")
        self.logger.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        log_file = os.path.join(log_dir, f"{log_name}.log")
        fh = logging.FileHandler(log_file, encoding="UTF-8")
        formator = logging.Formatter(fmt="%(asctime)s %(filename)s %(levelname)s %(message)s",
                                     datefmt="%Y/%m/%d %X")
        sh.setFormatter(formator)
        fh.setFormatter(formator)

        self.logger.addHandler(sh)
        self.logger.addHandler(fh)


if __name__ == '__main__':
    logprint = Logger("test").logger
    logprint.debug("------这是debug信息---")
    logprint.info("------这是info信息---")
    logprint.warning("------这是warning信息---")
    logprint.error("------这是error信息---")
    logprint.critical("------这是critical信息---")

