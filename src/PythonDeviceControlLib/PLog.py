# -*- coding:UTF-8 -*-
import logging
import os.path
import time

project_path = 'PythonDeviceControlLib'


class PLog(object):
    def __init__(self):
        current_path = os.path.dirname(os.path.abspath(project_path))
        path1 = current_path.split(project_path)
        path2 = [path1[0], project_path]
        path3 = ''
        new_name = path3.join(path2) + '/logs/'

        dir_time = time.strftime('%Y%m%d', time.localtime(time.time()))
        isExists = os.path.exists(new_name)
        if not isExists:
            os.makedirs(new_name)
            print(new_name + " directory created.")

        try:
            self.log = logging.getLogger()
            self.log.setLevel(logging.DEBUG)
            log_name = new_name + 'log-' + dir_time + '.log'

            fh = logging.FileHandler(log_name)
            fh.setLevel(logging.INFO)
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            formatter = logging.Formatter('[%(asctime)s] - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)

            self.log.addHandler(fh)
            self.log.addHandler(ch)
        except Exception as e:
            print("Output log %s" % e)

    def info(self, msg):
        self.log.info(msg)
        return

    def warning(self, msg):
        self.log.warning(msg)
        return

    def error(self, msg):
        self.log.error(msg)
        return


logger = PLog()

if __name__ == '__main__':
    logger.info('info')
    logger.warning('warning')
    logger.error('error')
