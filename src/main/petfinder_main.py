# -*- coding: utf-8 -*-
# #  Forked from [Baseline Modeling](https://www.kaggle.com/wrosinski/baselinemodeling)

# ## Added Image features from [Extract Image features from pretrained NN](https://www.kaggle.com/christofhenkel/extract-image-features-from-pretrained-nn)

# ## Added Image size features from [Extract Image Features](https://www.kaggle.com/kaerunantoka/extract-image-features)

import glob
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import os

# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import pickle
import random

from sklearn.preprocessing import MinMaxScaler

from src.data.make_dataset import read_train_data, read_test_data
from src.models.densenet import densenet_model, predict_using_img
from src.features.build_features import adopt_svd, agg_feature, merge_labels_breed, merge_labels_state, add_feature
from src.features.build_features import fill_and_drop_feature, fill_and_drop_feature_end, name_feature, metadata_color
from src.features.word_parse import extract_additional_features, parse_tfidf
from src.features.Img_parse import agg_img_feature
from src.models.xgb_model import run_xgb, xgb_tuning
from src.models.metrics import OptimizedRounder, quadratic_weighted_kappa
from src.result.summarize_result import storage_src
from src.submission.submit_data import submit
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import gensim
import input

np.random.seed(seed=1337)
random_state = 1337
warnings.filterwarnings('ignore')

split_char = '/'

xgb_params = {
    'eval_metric': 'rmse',
    'object': 'reg:squarederror',
    'seed': 1337,
    'n_estimators': 794,
    'max_depth': 7,
    'eta': 0.009,
    'gamma': 0.95,
    'subsample': 1,
    'min_child_weight': 5.5,
    'colsample_bytree': 0.5,
    'tree_method': 'gpu_hist',
    'device': 'gpu',
    'silent': 1,
    'booster': 'gbtree'
}

space_params = {
    'n_estimators': hp.quniform('n_estimators', 600, 1200, 1),
    'eta': hp.quniform('eta', 0.005, 0.02, 0.001),
    # A problem with max_depth casted to float instead of int with
    # the hp.quniform method.
    'max_depth': hp.choice('max_depth', np.arange(1, 14, dtype=int)),
    'min_child_weight': hp.quniform('min_child_weight', 1, 6, 0.5),
    'subsample': hp.quniform('subsample', 0.8, 1, 0.02),
    'gamma': hp.quniform('gamma', 0.8, 1, 0.01),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.2, 0.6, 0.01),
    'eval_metric': 'rmse',
    'object': 'reg:squarederror',
    # Increase this number if you have more cores. Otherwise, remove it and it will default
    # to the maxium number.
    'nthread': 10,
    'booster': 'gbtree',
    'tree_method': 'gpu_hist',
    'device': 'gpu',
    'silent': 1,
    'seed': random_state
}

img_size = 256
batch_size = 256

densenet_predict = False
exe_extract_additional_feature = False
adoption_shuffle = False
tuning_model = False
adoption_change =False

def storage_process(submission, str_metric_score, score, second_score, feature_score):
    submit(submission, str_metric_score)
    comment = 'metadata breed 2(1)'
    storage_src(str_metric_score, score, second_score, feature_score, comment, adoption_shuffle)


def main():
    train = read_train_data(path=os.path.join(input.__path__[0], 'petfinder-adoption-prediction/train/'))
    test = read_test_data(path=os.path.join(input.__path__[0], 'petfinder-adoption-prediction/test/'))
    model = gensim.models.KeyedVectors.load_word2vec_format(
        os.path.join(input.__path__[0], 'word2vec/GoogleNews-vectors-negative300.bin'), binary=True)
    train, test = name_feature(train, test, model)
    if adoption_shuffle:
        train['AdoptionSpeed'] = random.sample(train['AdoptionSpeed'].values.tolist(), len(train))
    if adoption_change:
        train['AdoptionSpeed'].replace({0:0, 1:7, 2:30, 3:90, 4:200}, inplace=True)

    if densenet_predict:
        dnet_model = densenet_model(weight_path=os.path.join(input.__path__[0], 'densenet-keras/DenseNet-BC-121-32-no-top.h5'))
        train_feats = predict_using_img(dnet_model,
                                        train,
                                        img_path=os.path.join(input.__path__[0], 'petfinder-adoption-prediction/train_images/'))
        test_feats = predict_using_img(dnet_model,
                                       test,
                                       img_path=os.path.join(input.__path__[0], 'petfinder-adoption-prediction/test_images/'))
        train_feats.to_pickle('densenet_train_predict.pkl')
        test_feats.to_pickle('densenet_test_predict.pkl')
    else:
        with open('./densenet_train_predict.pkl', 'rb') as f:
            train_feats = pickle.load(f)
        with open('./densenet_test_predict.pkl', 'rb') as f:
            test_feats = pickle.load(f)

    all_ids = pd.concat([train, test], axis=0, ignore_index=True, sort=False)[['PetID']]

    svd_col = adopt_svd(train_feats, test_feats)

    img_features = pd.concat([all_ids, svd_col], axis=1)

    labels_breed = pd.read_csv(os.path.join(input.__path__[0], 'petfinder-adoption-prediction/breed_labels.csv'))
    labels_color = pd.read_csv(os.path.join(input.__path__[0], 'petfinder-adoption-prediction/color_labels.csv'))
    labels_state = pd.read_csv(os.path.join(input.__path__[0], 'my_state_labels/my_state_labels.csv'))

    train_image_files = sorted(glob.glob(os.path.join(input.__path__[0], 'petfinder-adoption-prediction/train_images/*.jpg')))
    train_metadata_files = sorted(glob.glob(os.path.join(input.__path__[0], 'petfinder-adoption-prediction/train_metadata/*.json')))
    train_sentiment_files = sorted(glob.glob(os.path.join(input.__path__[0], 'petfinder-adoption-prediction/train_sentiment/*.json')))

    test_image_files = sorted(glob.glob(os.path.join(input.__path__[0], 'petfinder-adoption-prediction/test_images/*.jpg')))
    test_metadata_files = sorted(glob.glob(os.path.join(input.__path__[0], 'petfinder-adoption-prediction/test_metadata/*.json')))
    test_sentiment_files = sorted(glob.glob(os.path.join(input.__path__[0], 'petfinder-adoption-prediction/test_sentiment/*.json')))

    # Metadata:
    train_df_metadata = pd.DataFrame(train_metadata_files)
    train_df_metadata.columns = ['metadata_filename']
    train_df_sentiment = pd.DataFrame(train_sentiment_files)
    train_df_sentiment.columns = ['sentiment_filename']
    # Metadata:
    test_df_metadata = pd.DataFrame(test_metadata_files)
    test_df_metadata.columns = ['metadata_filename']
    test_df_sentiment = pd.DataFrame(test_sentiment_files)
    test_df_sentiment.columns = ['sentiment_filename']

    train_pet_ids = train.PetID.unique()
    test_pet_ids = test.PetID.unique()

    if exe_extract_additional_feature:
        dfs_train = Parallel(n_jobs=12, verbose=1)(
            delayed(extract_additional_features)(i, mode='train') for i in train_pet_ids)
        dfs_test = Parallel(n_jobs=12, verbose=1)(
            delayed(extract_additional_features)(i, mode='test') for i in test_pet_ids)
        train_dfs_sentiment = [x[0] for x in dfs_train if isinstance(x[0], pd.DataFrame)]
        train_dfs_metadata = [x[1] for x in dfs_train if isinstance(x[1], pd.DataFrame)]
        train_dfs_sentiment = pd.concat(train_dfs_sentiment, ignore_index=True, sort=False)
        train_dfs_metadata = pd.concat(train_dfs_metadata, ignore_index=True, sort=False)
        test_dfs_sentiment = [x[0] for x in dfs_test if isinstance(x[0], pd.DataFrame)]
        test_dfs_metadata = [x[1] for x in dfs_test if isinstance(x[1], pd.DataFrame)]
        test_dfs_sentiment = pd.concat(test_dfs_sentiment, ignore_index=True, sort=False)
        test_dfs_metadata = pd.concat(test_dfs_metadata, ignore_index=True, sort=False)
        train_dfs_metadata.to_pickle('train_dfs_metadata.pkl')
        train_dfs_sentiment.to_pickle('train_dfs_sentiment.pkl')
        test_dfs_metadata.to_pickle('test_dfs_metadata.pkl')
        test_dfs_sentiment.to_pickle('test_dfs_sentiment.pkl')

    else:
        with open('./train_dfs_metadata.pkl', 'rb') as f:
            train_dfs_metadata = pickle.load(f)
        with open('./train_dfs_sentiment.pkl', 'rb') as f:
            train_dfs_sentiment = pickle.load(f)
        with open('./test_dfs_metadata.pkl', 'rb') as f:
            test_dfs_metadata = pickle.load(f)
        with open('./test_dfs_sentiment.pkl', 'rb') as f:
            test_dfs_sentiment = pickle.load(f)

    # ### group extracted features by PetID:
    train_proc = agg_feature(train, train_dfs_metadata, train_dfs_sentiment, model, labels_breed)
    test_proc = agg_feature(test, test_dfs_metadata, test_dfs_sentiment, model, labels_breed)
    train_proc = merge_labels_breed(train_proc, labels_breed)
    test_proc = merge_labels_breed(test_proc, labels_breed)
    train_proc, test_proc = merge_labels_state(train_proc, test_proc, labels_state)
    train_proc = fill_and_drop_feature(train_proc)
    test_proc = fill_and_drop_feature(test_proc)
    train_proc = add_feature(train_proc)
    test_proc = add_feature(test_proc)

    X = pd.concat([train_proc, test_proc], ignore_index=True, sort=False)
    X_temp = X.copy()
    text_columns = ['Description', 'metadata_annots_top_desc', 'sentiment_entities']
    categorical_columns = ['main_breed_BreedName', 'second_breed_BreedName']
    to_drop_columns = ['PetID', 'Name', 'RescuerID']

    rescuer_count = X.groupby(['RescuerID'])['PetID'].count().reset_index()
    rescuer_count.columns = ['RescuerID', 'RescuerID_COUNT']

    X_temp = X_temp.merge(rescuer_count, how='left', on='RescuerID')

    for i in categorical_columns:
        try:
            X_temp.loc[:, i] = pd.factorize(X_temp.loc[:, i])[0]
        except:
            pass

    X_text = X_temp[text_columns]
    for i in X_text.columns:
        X_text.loc[:, i] = X_text.loc[:, i].fillna('none')

    X_temp['Length_Description'] = X_text['Description'].map(len)
    X_temp['Length_metadata_annots_top_desc'] = X_text['metadata_annots_top_desc'].map(len)
    X_temp['Lengths_sentiment_entities'] = X_text['sentiment_entities'].map(len)
    X_temp = parse_tfidf(X_temp, X_text)

    X_temp = X_temp.merge(img_features, how='left', on='PetID')

    agg_train_imgs = agg_img_feature(train_image_files)
    agg_test_imgs = agg_img_feature(test_image_files)
    agg_imgs = pd.concat([agg_train_imgs, agg_test_imgs], axis=0).reset_index(drop=True)
    X_temp = X_temp.merge(agg_imgs, how='left', on='PetID')

    # ### Drop ID, name and rescuerID
    X_temp = X_temp.drop(to_drop_columns, axis=1)

    X_train = X_temp.loc[np.isfinite(X_temp.AdoptionSpeed), :]
    X_test = X_temp.loc[~np.isfinite(X_temp.AdoptionSpeed), :]

    X_test = X_test.drop(['AdoptionSpeed'], axis=1)
    assert X_train.shape[0] == train.shape[0]
    assert X_test.shape[0] == test.shape[0]
    train_cols = X_train.columns.tolist()
    train_cols.remove('AdoptionSpeed')

    test_cols = X_test.columns.tolist()

    assert np.all(train_cols == test_cols)

    X_train_non_null = X_train.fillna(-1)
    X_test_non_null = X_test.fillna(-1)
    X_train_non_null.isnull().any().any(), X_test_non_null.isnull().any().any()

    X_train_non_null = fill_and_drop_feature_end(X_train_non_null)
    X_test_non_null = fill_and_drop_feature_end(X_test_non_null)

    X_train_non_null.to_csv('./X_train.csv')

    if tuning_model:
        best_params = xgb_tuning(space_params, X_train_non_null, X_test_non_null)
        print(best_params)
    else:
        model, oof_train, oof_test, feature_score = run_xgb(xgb_params, X_train_non_null, X_test_non_null)
        if adoption_change:
            mmscaler = MinMaxScaler()
            train_test_min = min(oof_train.min(), oof_test.min())
            train_test_max = min(oof_train.max(), oof_test.max())
            oof_train = (oof_train - train_test_min) / (train_test_max - train_test_min) * 4
            oof_test = (oof_test - train_test_min) / (train_test_max - train_test_min) * 4
            X_train['AdoptionSpeed'].replace({0:0, 7:1, 30:2, 90:3, 200:4}, inplace=True)

            optR = OptimizedRounder()
            optR.fit(oof_train, X_train['AdoptionSpeed'].values)
            coefficients = optR.coefficients()
            valid_pred = optR.predict(oof_train, coefficients)
            qwk = quadratic_weighted_kappa(X_train['AdoptionSpeed'].values, valid_pred)
            print("QWK = ", qwk)

            coefficients_ = coefficients.copy()
            coefficients_[0] = 0.75
            coefficients_[1] = 1.16
            coefficients_[2] = 1.7
            coefficients_[3] = 2.12
        else:
            optR = OptimizedRounder()
            optR.fit(oof_train, X_train['AdoptionSpeed'].values)
            coefficients = optR.coefficients()
            valid_pred = optR.predict(oof_train, coefficients)
            qwk = quadratic_weighted_kappa(X_train['AdoptionSpeed'].values, valid_pred)
            print("QWK = ", qwk)

            coefficients_ = coefficients.copy()
            coefficients_[0] = 1.62
            coefficients_[1] = 2.1
            coefficients_[3] = 2.9

        train_predictions = optR.predict(oof_train, coefficients_).astype(np.int8)
        test_predictions = optR.predict(oof_test.mean(axis=1), coefficients_).astype(np.int8)

        valid_pred = optR.predict(oof_train, coefficients_)
        qwk_change = quadratic_weighted_kappa(X_train['AdoptionSpeed'].values, valid_pred)
        print("QWK_change = ", qwk_change)
        submission = pd.DataFrame({'PetID': test['PetID'].values, 'AdoptionSpeed': test_predictions})
        submission.to_csv('submission.csv', index=False)
        str_metric_score = 'qwk' + '_0' + str(int(qwk * 100000))
        storage_process(submission, str_metric_score, qwk, qwk_change, feature_score)


if __name__ == '__main__':
    main()
