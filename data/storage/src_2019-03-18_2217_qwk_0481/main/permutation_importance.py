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
import lightgbm as lgb
from eli5.sklearn import PermutationImportance
import eli5
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
import pickle
import data.permutation_importance
import os

logger = set_logger(__name__)
metric = 'auc'
params = {
    'num_leaves': 10,
    'max_bin': 119,
    'min_data_in_leaf': 11,
    'learning_rate': 0.013,
    'min_sum_hessian_in_leaf': 0.00245,
    'bagging_fraction': 0.83,
    'bagging_freq': 5,
    'feature_fraction': 0.3,
    'lambda_l1': 4.972,
    'lambda_l2': 2.276,
    'min_gain_to_split': 0.65,
    'max_depth': 14,
    'save_binary': True,
    'seed': 1337,
    'feature_fraction_seed': 1337,
    'bagging_seed': 1337,
    'drop_seed': 1337,
    'data_random_seed': 1337,
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'metric': metric,
    # 'is_unbalance': True,
    # 'boost_from_average': False,
    'num_threads': 10,
    'device_type': 'gpu',
}

def main():
    start_time = time.time()
    train = read_train_data(nrows=None)
    test = read_test_data()

    train, test = process_data(train, test)
    X = train.drop(['ID_code', 'target'], axis=1)
    y = train['target']
    X_test = test.drop(['ID_code'], axis=1)
    model = lgb.LGBMClassifier(**params, n_estimators=20000, n_jobs=10)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=1000, early_stopping_rounds=200)
    perm = PermutationImportance(model, random_state=1).fit(X_valid, y_valid)
    eli_df = eli5.explain_weights_df(perm, feature_names=X.columns.tolist(), top=len(X.columns))
    eli_df.to_csv(os.path.join(data.permutation_importance.__path__[0], '0304_square_feature.csv'))
    elapsed_time = time.time() - start_time
    print(elapsed_time)

if __name__ == '__main__':
    main()
    pass