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
from sklearn.metrics import accuracy_score, roc_auc_score
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

logger = set_logger(__name__)


def prediction(X, y, model, model_type='lgb'):
    scores = []

    if model_type == 'lgb':
        y_pred = model.predict(X, num_iteration=model.best_iteration)

    if model_type == 'xgb':
        y_pred = model.predict(xgb.DMatrix(X, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

    if model_type == 'sklearn':
        y_pred = model.predict_proba(X)[:, 1]

    if model_type == 'glm':
        y_pred = model.predict(X)

    if model_type == 'cat':
        y_pred = model.predict_proba(X)[:, 1]

    scores.append(roc_auc_score(y, y_pred))
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    return scores



if __name__ == '__main__':
    pass