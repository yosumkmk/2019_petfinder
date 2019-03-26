# -*- coding: utf-8 -*-
import warnings
from logging import StreamHandler, Formatter, getLogger, FileHandler, DEBUG, INFO, ERROR
from logging.handlers import RotatingFileHandler
import os
import datetime
from pytz import timezone, utc

warnings.filterwarnings('ignore')

def set_logger(modname):
    BASE_DIR = os.path.realpath(os.path.dirname(__file__))
    LOG_DIR = os.path.join(BASE_DIR, 'logs')  # ログファイルディレクトリ

    # ログファイルディレクトリがなければ作成する
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    #################
    # logger 設定
    logger = getLogger(modname)
    logger.setLevel(DEBUG)
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(module)s][%(funcName)s] %(message)s', datefmt='%m-%d %H:%M:%S.%s,%03d')

    def customTime(*args):
        utc_dt = utc.localize(datetime.datetime.utcnow())
        my_tz = timezone("Asia/Tokyo")
        converted = utc_dt.astimezone(my_tz)
        return converted.timetuple()

    log_fmt.converter = customTime

    strm_handler = StreamHandler()
    strm_handler.setLevel(DEBUG)
    strm_handler.setFormatter(log_fmt)
    logger.addHandler(strm_handler)
    str_hour = str(datetime.datetime.now().hour) if datetime.datetime.now().hour >= 10 \
        else '0' + str(datetime.datetime.now().hour)
    str_minute = str(datetime.datetime.now().minute) if datetime.datetime.now().minute >= 10\
        else '0' + str(datetime.datetime.now().minute)

    rotate_file_handler = RotatingFileHandler(
        filename=os.path.join(LOG_DIR, '{}_{}{}_log.log'.format(str(datetime.datetime.now().date())
                                                                ,str_hour
                                                                ,str_minute)),
        maxBytes=1024 * 1024 * 5,
        backupCount=10
    )
    rotate_file_handler.setLevel(INFO)
    rotate_file_handler.setFormatter(log_fmt)
    logger.addHandler(rotate_file_handler)
    return logger

if __name__ == '__main__':
    pass