# -*- coding: utf-8 -*-
import warnings
from src.util.log_util import set_logger
import pandas as pd
from src.util.util import str_datetime
warnings.filterwarnings('ignore')
from logging import StreamHandler, Formatter, getLogger, FileHandler, DEBUG, INFO, ERROR
import os
import data.submission

logger = set_logger(__name__)

def submit(submission, str_metric_score):
    file_name = str_datetime() + '_' + str_metric_score + '_submission.csv'
    logger.info('Prepare submission')
    submission.to_csv(os.path.join(data.submission.__path__[0], file_name), index=False)

def test():
    print(data.submission.__path__)

if __name__ == '__main__':
    test()
    pass