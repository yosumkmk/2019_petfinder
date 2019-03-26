# -*- coding: utf-8 -*-
import warnings

warnings.filterwarnings('ignore')
from src.util.log_util import set_logger
from src.data.make_dataset import read_train_data, read_test_data, split_train_data
from src.features.build_features import process_data
from src.models.train_model import train_model
from src.submission.submit_data import submit
from src.result.summarize_result import storage_src
import numpy as np
import time

logger = set_logger(__name__)
metric = 'auc'
# params = {
#     'num_leaves': 10,
#     'max_bin': 119,
#     'min_data_in_leaf': 11,
#     'learning_rate': 0.02,
#     'min_sum_hessian_in_leaf': 0.00245,
#     'bagging_fraction': 1.0,
#     'bagging_freq': 5,
#     'feature_fraction': 0.05,
#     'lambda_l1': 4.972,
#     'lambda_l2': 2.276,
#     'min_gain_to_split': 0.65,
#     'max_depth': 14,
#     'save_binary': True,
#     'seed': 1337,
#     'feature_fraction_seed': 1337,
#     'bagging_seed': 1337,
#     'drop_seed': 1337,
#     'data_random_seed': 1337,
#     'objective': 'binary',
#     'boosting_type': 'gbdt',
#     'verbose': -1,
#     'metric': metric,
#     # 'is_unbalance': True,
#     'boost_from_average': False,
#     'num_threads': 10,
#     'device_type': 'gpu',
# }
params = dict(objective = "binary",
               boost="gbdt",
               metric="auc",
               boost_from_average="false",
               num_threads=10,
               learning_rate = 0.01,
               num_leaves = 13,
               max_depth=-1,
               tree_learner = "serial",
               feature_fraction = 0.05,
               bagging_freq = 5,
               bagging_fraction = 0.4,
               min_data_in_leaf = 80,
               min_sum_hessian_in_leaf = 10.0,
               verbosity = 1)

def main_submit():
    start_time = time.time()
    train = read_train_data(nrows=None)
    test = read_test_data()

    train, test = process_data(train, test)
    X = train.drop(['ID_code', 'target'], axis=1)
    y = train['target']
    X_test = test.drop(['ID_code'], axis=1)
    oof, predictions, scores, feature_importance = train_model(X, X_test, y, params, n_fold=10,
                                                               plot_feature_importance=True, model_type='lgb_sklearn')
    str_metric_score = metric + '_0' + str(int(scores['auc_score'].iloc[0] * 10000))
    submit(test, predictions, str_metric_score)
    comment = 'add 5 max min feature before standard scale'
    storage_src(str_metric_score, scores, feature_importance, comment)
    elapsed_time = time.time() - start_time
    print(elapsed_time)

def main_test():
    start_time = time.time()
    train = read_train_data(nrows=None)
    test = read_test_data()

    train, test = process_data(train, test)
    X = train.drop(['ID_code', 'target'], axis=1)
    y = train['target']
    X_test = test.drop(['ID_code'], axis=1)
    oof, predictions, scores, feature_importance = train_model(X, X_test, y, params, plot_feature_importance=True)
    str_metric_score = metric + '_0' + str(int(scores['auc_score'].iloc[0] * 10000))
    # submit(test, predictions, str_metric_score)
    comment = 'starter removed statistics feature, remove also 0 score, bagging_fraction1'
    # storage_src(str_metric_score, scores, feature_importance, comment)
    elapsed_time = time.time() - start_time
    print(elapsed_time)


if __name__ == "__main__":
    main_submit()
    # main_test()
