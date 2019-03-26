# -*- coding: utf-8 -*-
import warnings
from src.util.log_util import set_logger
import pandas as pd
warnings.filterwarnings('ignore')
from logging import StreamHandler, Formatter, getLogger, FileHandler, DEBUG, INFO, ERROR
import datetime
import shutil
import os
import src
import result_summary
import data
from src.util.util import str_datetime

logger = set_logger(__name__)


def insert_result_to_csv(score_df):
    try:
        summary_df = pd.read_csv(os.path.join(result_summary.__path__[0], 'summary_scores.csv'))
        next_index = len(summary_df) + 1
    except:
        summary_df = pd.DataFrame()
        next_index = 0
    score_df.index = [next_index]
    summary_df = pd.concat([summary_df, score_df], axis=0)
    summary_df.to_csv(os.path.join(result_summary.__path__[0], 'summary_scores.csv'), index=False)

def storage_src(str_metric_score, scores, second_score, feature_importance, comment=''):
    copy_dir_name = 'src_' + str_datetime() + '_' + str_metric_score
    scores = pd.DataFrame({'qwk':scores}, index=[0])
    scores['qwk_change'] = second_score
    scores['dir_name'] = copy_dir_name
    scores['datetime'] = datetime.datetime.now()
    scores['txt'] = comment
    scores.to_csv(os.path.join(os.path.dirname(__file__), 'scores.scv',), index=False)
    feature_importance.to_csv(os.path.join(os.path.dirname(__file__), 'feature_importance.scv'), index=False)
    origin_directory = src.__path__[0]
    insertion_directory = os.path.join(data.__path__[0], 'storage', copy_dir_name)
    shutil.copytree(origin_directory, insertion_directory)
    insert_result_to_csv(scores)


def test():
    # origin_directory = os.path.dirname(data.__path__)
    print(data.__path__[0])
    pass

if __name__ == '__main__':
    test()
    pass