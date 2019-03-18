# -*- coding: utf-8 -*-
import gc
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

import datetime
import lightgbm as lgb
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
import os
import plotly.offline as py

import plotly.graph_objs as go
import plotly.tools as tls
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
from sklearn import metrics
import json
import ast
import time
from sklearn import linear_model
import eli5
from eli5.sklearn import PermutationImportance
import shap
from tqdm import tqdm_notebook
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import GenericUnivariateSelect, SelectPercentile, SelectKBest, f_classif, mutual_info_classif, RFE
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
from src.util.log_util import set_logger
import src.models
logger = set_logger(__name__)


def train_model(X, X_test, y, params, n_fold=5, shuffle_folds=True, model_type='lgb', plot_feature_importance=False,
                averaging='usual', model=None, folds_random_state=42):


    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    auc_scores = []
    f1_scores = []
    recall_scores = []
    precision_scores = []
    accuracy_scores = []
    feature_importance = pd.DataFrame()
    folds = StratifiedKFold(n_splits=n_fold, shuffle=shuffle_folds, random_state=folds_random_state)
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.loc[train_index], X.loc[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        if model_type == 'lgb':
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_valid, label=y_valid)

            model = lgb.train(params,
                              train_data,
                              num_boost_round=1000000,
                              valid_sets=[train_data, valid_data],
                              verbose_eval=1000,
                              early_stopping_rounds=3000)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration)

        if model_type == 'lgb_sklearn':
            model = lgb.LGBMClassifier(**params, n_estimators=1000000)
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)],
                      verbose=1000, early_stopping_rounds=3000)

            y_pred_valid = model.predict_proba(X_valid)[:,1]
            y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)[:,1]

        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X_train.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X_train.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X_train.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_train.columns), ntree_limit=model.best_ntree_limit)

        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            y_pred_valid = model.predict_proba(X_valid).reshape(-1, )
            score = roc_auc_score(y_valid, y_pred_valid)
            # print(f'Fold {fold_n}. AUC: {score:.4f}.')
            # print('')

            y_pred = model.predict_proba(X_test)[:, 1]

        if model_type == 'glm':
            model = sm.GLM(y_train, X_train, family=sm.families.Binomial())
            model_results = model.fit()
            model_results.predict(X_test)
            y_pred_valid = model_results.predict(X_valid).reshape(-1, )
            score = roc_auc_score(y_valid, y_pred_valid)

            y_pred = model_results.predict(X_test)

        if model_type == 'cat':
            model = CatBoostClassifier(iterations=20000, learning_rate=0.05, loss_function='Logloss', eval_metric='AUC', **params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict_proba(X_valid)[:, 1]
            y_pred = model.predict_proba(X_test)[:, 1]

        oof[valid_index] = y_pred_valid.reshape(-1, )

        f1 = 0
        best_t = 0
        for t in np.arange(0.1, 1, 0.05):
            valid_pr = (y_pred_valid > t).astype(int)
            valid_f1 = metrics.f1_score(y_valid, valid_pr)
            if valid_f1 > f1:
                f1 = valid_f1
                best_t = t

        t = best_t
        y_valid_pr = (y_pred_valid > t).astype(int)
        auc_scores.append(roc_auc_score(y_valid, y_pred_valid))
        f1_scores.append(f1_score(y_valid, y_valid_pr))
        precision_scores.append(precision_score(y_valid, y_valid_pr))
        recall_scores.append(recall_score(y_valid, y_valid_pr))
        accuracy_scores.append(accuracy_score(y_valid, y_valid_pr))

        if averaging == 'usual':
            prediction += y_pred
        elif averaging == 'rank':
            prediction += pd.Series(y_pred).rank().values

        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importance()
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_fold
    with open(os.path.join(src.models.__path__[0], 'model.pkl'), 'wb') as f:
        pickle.dump(model, f, protocol=2)

    scores = pd.DataFrame({
        'precision_score': np.mean(precision_scores),
        'recall_score': np.mean(recall_scores),
        'f1_score': np.mean(f1_scores),
        'accuracy_score': np.mean(accuracy_scores),
        'auc_score': np.mean(auc_scores),
                           }, index=[0])

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(auc_scores), np.std(auc_scores)))

    if model_type == 'lgb':
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
            plt.title('LGB Features (avg over folds)')
            plt.savefig('feature_importance.png')

            return oof, prediction, scores, feature_importance
        return oof, prediction, scores, feature_importance

    else:
        return oof, prediction, scores, feature_importance


if __name__ == '__main__':
    pass