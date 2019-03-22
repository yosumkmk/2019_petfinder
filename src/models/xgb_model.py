# -*- coding: utf-8 -*-
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from src.models.metrics import OptimizedRounder, quadratic_weighted_kappa
import joblib
import os
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

def run_xgb(params, X_train, X_test):
    n_splits = 10
    verbose_eval = 1000
    num_rounds = 60000
    early_stop = 500

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)

    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_splits))

    i = 0

    for train_idx, valid_idx in kf.split(X_train, X_train['AdoptionSpeed'].values):
        X_tr = X_train.iloc[train_idx, :]
        X_val = X_train.iloc[valid_idx, :]

        y_tr = X_tr['AdoptionSpeed'].values
        X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)

        y_val = X_val['AdoptionSpeed'].values
        X_val = X_val.drop(['AdoptionSpeed'], axis=1)

        d_train = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)
        d_valid = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, evals=watchlist,
                          early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=params)

        valid_pred = model.predict(xgb.DMatrix(X_val, feature_names=X_val.columns), ntree_limit=model.best_ntree_limit)
        test_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit=model.best_ntree_limit)

        oof_train[valid_idx] = valid_pred
        oof_test[:, i] = test_pred
        i += 1
        if params['booster'] == 'dart':
            feature_score = pd.DataFrame()
        else:
            feature_score = pd.DataFrame(model.get_fscore(), index=[0]).T.sort_values(0, ascending=False)
            feature_score.columns = ['fscore']
            feature_score.reset_index(inplace=True)
    return model, oof_train, oof_test, feature_score



def xgb_tuning(params, X_train, X_test):
    def run_xgb_tuning(params):
        nonlocal X_train, X_test
        n_splits = 10
        verbose_eval = 1000
        num_rounds = 60000
        early_stop = 500

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)

        oof_train = np.zeros((X_train.shape[0]))
        oof_test = np.zeros((X_test.shape[0], n_splits))

        i = 0

        for train_idx, valid_idx in kf.split(X_train, X_train['AdoptionSpeed'].values):
            X_tr = X_train.iloc[train_idx, :]
            X_val = X_train.iloc[valid_idx, :]

            y_tr = X_tr['AdoptionSpeed'].values
            X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)

            y_val = X_val['AdoptionSpeed'].values
            X_val = X_val.drop(['AdoptionSpeed'], axis=1)

            d_train = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)
            d_valid = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)

            watchlist = [(d_train, 'train'), (d_valid, 'valid')]
            model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, evals=watchlist,
                              early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=params)

            valid_pred = model.predict(xgb.DMatrix(X_val, feature_names=X_val.columns), ntree_limit=model.best_ntree_limit)
            test_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit=model.best_ntree_limit)

            oof_train[valid_idx] = valid_pred
            oof_test[:, i] = test_pred
            i += 1
            feature_score = pd.DataFrame(model.get_fscore(), index=[0]).T.sort_values(0, ascending=False)
            feature_score.columns = ['fscore']
            feature_score.reset_index(inplace=True)
        return model, oof_train, oof_test, feature_score

    def score(xgb_params):
        global model, oof_train, oof_test, feature_score, test_predictions, qwk, qwk_change
        model, oof_train, oof_test, feature_score = run_xgb_tuning(xgb_params)
        optR = OptimizedRounder()
        optR.fit(oof_train, X_train['AdoptionSpeed'].values)
        coefficients = optR.coefficients()
        valid_pred = optR.predict(oof_train, coefficients)
        qwk = quadratic_weighted_kappa(X_train['AdoptionSpeed'].values, valid_pred)
        print("QWK = ", qwk)

        coefficients_ = coefficients.copy()
        coefficients_[0] = 1.66
        coefficients_[1] = 2.13
        coefficients_[3] = 2.85
        train_predictions = optR.predict(oof_train, coefficients_).astype(np.int8)
        test_predictions = optR.predict(oof_test.mean(axis=1), coefficients_).astype(np.int8)

        valid_pred = optR.predict(oof_train, coefficients_)
        qwk_change = quadratic_weighted_kappa(X_train['AdoptionSpeed'].values, valid_pred)
        print("QWK_change = ", qwk_change)
        score_result = {
            'loss': -qwk_change,
            'status': STATUS_OK
        }
        return score_result

    trials = Trials()
    best = fmin(score, params, algo=tpe.suggest,
                trials=trials,
                max_evals=250)
    joblib.dump(trials, os.path.join('trials.pkl'), compress=True)
    return best


if __name__ == '__main__':
    pass