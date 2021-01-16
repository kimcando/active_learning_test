import logging
import time
import os
import openpyxl

class MyLogger(object):
    """
    logger
    """
    def __init__(self, args,logger_name, set_level=logging.INFO, needs_format=True, needs_file=True):
        """
        :param logger_name:
        :param set_level: DEBUG
        :param needs_format:
        :param needs_file:
        """
        self.args = args
        self.logger_name = logger_name
        self.logger = logging.getLogger(self.logger_name)
        self.set_level = set_level
        self.logger.setLevel(self.set_level)
        self.stream_handler = logging.StreamHandler()
        if needs_format:
            self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger.addHandler(self.stream_handler)

        if needs_file:
            path = os.getcwd()
            if os.path.isdir(path+f'/{self.args.logging_path}'):
                log_path = path+f'/{self.args.logging_path}/'
            else:
                os.mkdir(path+f'/{self.args.logging_path}')
                log_path = path + f'/{self.args.logging_path}/'
            self.file_handler = logging.FileHandler(log_path+self.logger_name+'.log')
            self.logger.addHandler(self.file_handler)
        self.init_log_notice()
    
    def info(self, msg):
        self.logger.info(msg)

    def init_data(self):
        return time.strftime('%c', time.localtime(time.time()))

    def init_log_notice(self):
        self.logger.info('\n')
        self.logger.info(f'{self.init_data()}')
        self.logger.info(f'{self.logger_name}\'s set level: {self.set_level}')
        self.logger.info(f'{self.logger_name} logger is ready')

    def save_config(self):
        # TODO
        # running file name
        # configuration file
        pass

class MyWorkBook(object):
    def __init__(self, args, config):
        self.wb = openpyxl.Workbook()
        self.args = args
        path = os.getcwd()
        if os.path.isdir(path+f'/{self.args.xlsx_path}'):
            self.base_path = path+f'/{self.args.xlsx_path}/'
        else:
            os.mkdir(path+f'/{self.args.xlsx_path}')
            self.base_path = path + f'/{self.args.xlsx_path}/'
    def save(self, name=None):

        name = self.base_path +'/'+self.args.file_name +'.xlsx'
        self.wb.save(filename=name)

# https://hamait.tistory.com/880
# mylogger = logging.getLogger("my")
# mylogger.setLevel(logging.INFO)
# stream_hander = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# mylogger.addHandler(stream_hander)
# file_handler = logging.FileHandler('my.log')
# mylogger.addHandler(file_handler)
# mylogger.info("server start!!!")