# -*- coding: utf-8 -*-
import warnings
import datetime
warnings.filterwarnings('ignore')

def str_datetime():
    str_hour = str(datetime.datetime.now().hour) if datetime.datetime.now().hour >= 10 \
        else '0' + str(datetime.datetime.now().hour)
    str_minute = str(datetime.datetime.now().minute) if datetime.datetime.now().minute >= 10\
        else '0' + str(datetime.datetime.now().minute)
    return str(datetime.datetime.now().date()) + '_' + str_hour + str_minute

if __name__ == '__main__':
    pass