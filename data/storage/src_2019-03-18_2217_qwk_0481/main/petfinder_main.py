# -*- coding: utf-8 -*-
# #  Forked from [Baseline Modeling](https://www.kaggle.com/wrosinski/baselinemodeling)

# ## Added Image features from [Extract Image features from pretrained NN](https://www.kaggle.com/christofhenkel/extract-image-features-from-pretrained-nn)

# ## Added Image size features from [Extract Image Features](https://www.kaggle.com/kaerunantoka/extract-image-features)

import gc
import glob
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from joblib import Parallel, delayed
from tqdm import tqdm, tqdm_notebook
import cv2
import os

from src.data.make_dataset import read_train_data, read_test_data, resize_to_square, load_image
from src.models.densenet import densenet_model, predict_using_img
from src.features.build_features import adopt_svd, agg_feature, merge_labels_breed
from src.features.word_parse import extract_additional_features, PetFinderParser, parse_tfidf
from src.features.Img_parse import agg_img_feature
from src.models.xgb_model import run_xgb
from src.models.metrics import OptimizedRounder, quadratic_weighted_kappa
from src.result.summarize_result import storage_src
from src.submission.submit_data import submit

from data import input

np.random.seed(seed=1337)
warnings.filterwarnings('ignore')

split_char = '/'

img_size = 256
batch_size = 256

def storage_process(submission, str_metric_score, scores, feature_score):
    submit(submission, str_metric_score)
    comment = 'add 5 max min feature before standard scale'
    storage_src(str_metric_score, scores, feature_score, comment)

def main():
    train = read_train_data(path=os.path.join(input.__path__[0], 'petfinder-adoption-prediction/train/'))
    test = read_test_data(path=os.path.join(input.__path__[0], 'petfinder-adoption-prediction/test/'))
    dnet_model = densenet_model(weight_path=os.path.join(input.__path__[0], 'pre_trained_model/DenseNet-BC-121-32-no-top.h5'))
    train_feats = predict_using_img(dnet_model,
                                    train,
                                    img_path=os.path.join(input.__path__[0], 'petfinder-adoption-prediction/train_images/'))
    test_feats = predict_using_img(dnet_model,
                                    test,
                                    img_path=os.path.join(input.__path__[0], 'petfinder-adoption-prediction/test_images/'))

    all_ids = pd.concat([train, test], axis=0, ignore_index=True, sort=False)[['PetID']]

    svd_col = adopt_svd(train_feats, test_feats)

    img_features = pd.concat([all_ids, svd_col], axis=1)

    labels_breed = pd.read_csv(os.path.join(input.__path__[0], 'petfinder-adoption-prediction/breed_labels.csv'))
    labels_state = pd.read_csv(os.path.join(input.__path__[0], 'petfinder-adoption-prediction/color_labels.csv'))
    labels_color = pd.read_csv(os.path.join(input.__path__[0], 'petfinder-adoption-prediction/state_labels.csv'))

    train_image_files = sorted(glob.glob(os.path.join(input.__path__[0], 'petfinder-adoption-prediction/train_images/*.jpg')))
    train_metadata_files = sorted(glob.glob(os.path.join(input.__path__[0], 'petfinder-adoption-prediction/train_metadata/*.json')))
    train_sentiment_files = sorted(glob.glob(os.path.join(input.__path__[0], 'petfinder-adoption-prediction/train_sentiment/*.json')))

    test_image_files = sorted(glob.glob(os.path.join(input.__path__[0], 'petfinder-adoption-prediction/test_images/*.jpg')))
    test_metadata_files = sorted(glob.glob(os.path.join(input.__path__[0], 'petfinder-adoption-prediction/test_metadata/*.json')))
    test_sentiment_files = sorted(glob.glob(os.path.join(input.__path__[0], 'petfinder-adoption-prediction/test_sentiment/*.json')))
    train_df_ids = train[['PetID']]

    # Metadata:
    train_df_metadata = pd.DataFrame(train_metadata_files)
    train_df_metadata.columns = ['metadata_filename']
    train_metadata_pets = train_df_metadata['metadata_filename'].apply(lambda x: x.split(split_char)[-1].split('-')[0])
    train_df_metadata = train_df_metadata.assign(PetID=train_metadata_pets)
    pets_with_metadatas = len(np.intersect1d(train_metadata_pets.unique(), train_df_ids['PetID'].unique()))

    # Sentiment:
    train_df_ids = train[['PetID']]
    train_df_sentiment = pd.DataFrame(train_sentiment_files)
    train_df_sentiment.columns = ['sentiment_filename']
    train_sentiment_pets = train_df_sentiment['sentiment_filename'].apply(lambda x: x.split(split_char)[-1].split('.')[0])
    train_df_sentiment = train_df_sentiment.assign(PetID=train_sentiment_pets)
    print(len(train_sentiment_pets.unique()))

    pets_with_sentiments = len(np.intersect1d(train_sentiment_pets.unique(), train_df_ids['PetID'].unique()))
    print(f'fraction of pets with sentiment: {pets_with_sentiments / train_df_ids.shape[0]:.3f}')

    # Images:
    test_df_ids = test[['PetID']]
    print(test_df_ids.shape)

    # Metadata:
    test_df_metadata = pd.DataFrame(test_metadata_files)
    test_df_metadata.columns = ['metadata_filename']
    test_metadata_pets = test_df_metadata['metadata_filename'].apply(lambda x: x.split(split_char)[-1].split('-')[0])
    test_df_metadata = test_df_metadata.assign(PetID=test_metadata_pets)
    print(len(test_metadata_pets.unique()))

    pets_with_metadatas = len(np.intersect1d(test_metadata_pets.unique(), test_df_ids['PetID'].unique()))
    print(f'fraction of pets with metadata: {pets_with_metadatas / test_df_ids.shape[0]:.3f}')

    # Sentiment:
    test_df_sentiment = pd.DataFrame(test_sentiment_files)
    test_df_sentiment.columns = ['sentiment_filename']
    test_sentiment_pets = test_df_sentiment['sentiment_filename'].apply(lambda x: x.split(split_char)[-1].split('.')[0])
    test_df_sentiment = test_df_sentiment.assign(PetID=test_sentiment_pets)
    print(len(test_sentiment_pets.unique()))

    pets_with_sentiments = len(np.intersect1d(test_sentiment_pets.unique(), test_df_ids['PetID'].unique()))
    print(f'fraction of pets with sentiment: {pets_with_sentiments / test_df_ids.shape[0]:.3f}')

    debug = False
    train_pet_ids = train.PetID.unique()
    test_pet_ids = test.PetID.unique()

    if debug:
        train_pet_ids = train_pet_ids[:1000]
        test_pet_ids = test_pet_ids[:500]

    dfs_train = Parallel(n_jobs=12, verbose=1)(
        delayed(extract_additional_features)(i, mode='train') for i in train_pet_ids)

    train_dfs_sentiment = [x[0] for x in dfs_train if isinstance(x[0], pd.DataFrame)]
    train_dfs_metadata = [x[1] for x in dfs_train if isinstance(x[1], pd.DataFrame)]

    train_dfs_sentiment = pd.concat(train_dfs_sentiment, ignore_index=True, sort=False)
    train_dfs_metadata = pd.concat(train_dfs_metadata, ignore_index=True, sort=False)

    dfs_test = Parallel(n_jobs=12, verbose=1)(
        delayed(extract_additional_features)(i, mode='test') for i in test_pet_ids)

    test_dfs_sentiment = [x[0] for x in dfs_test if isinstance(x[0], pd.DataFrame)]
    test_dfs_metadata = [x[1] for x in dfs_test if isinstance(x[1], pd.DataFrame)]

    test_dfs_sentiment = pd.concat(test_dfs_sentiment, ignore_index=True, sort=False)
    test_dfs_metadata = pd.concat(test_dfs_metadata, ignore_index=True, sort=False)

    # ### group extracted features by PetID:
    train_proc = agg_feature(train, train_dfs_metadata, train_dfs_sentiment)
    test_proc = agg_feature(test, test_dfs_metadata, test_dfs_sentiment)
    train_proc = merge_labels_breed(train_proc, labels_breed)
    test_proc = merge_labels_breed(test_proc, labels_breed)

    X = pd.concat([train_proc, test_proc], ignore_index=True, sort=False)
    X_temp = X.copy()
    text_columns = ['Description', 'metadata_annots_top_desc', 'sentiment_entities']
    categorical_columns = ['main_breed_BreedName', 'second_breed_BreedName']
    to_drop_columns = ['PetID', 'Name', 'RescuerID']

    rescuer_count = X.groupby(['RescuerID'])['PetID'].count().reset_index()
    rescuer_count.columns = ['RescuerID', 'RescuerID_COUNT']

    X_temp = X_temp.merge(rescuer_count, how='left', on='RescuerID')

    for i in categorical_columns:
        X_temp.loc[:, i] = pd.factorize(X_temp.loc[:, i])[0]

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

    xgb_params = {
        'eval_metric': 'rmse',
        'seed': 1337,
        'eta': 0.0123,
        'subsample': 0.8,
        'colsample_bytree': 0.85,
        'tree_method': 'gpu_hist',
        'device': 'gpu',
        'silent': 1,
    }
    model, oof_train, oof_test, feature_score = run_xgb(xgb_params, X_train_non_null, X_test_non_null)

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

    submission = pd.DataFrame({'PetID': test['PetID'].values, 'AdoptionSpeed': test_predictions})
    submission.to_csv('submission.csv', index=False)
    str_metric_score = 'qwk' + '_0' + str(int(qwk * 1000))
    storage_process(submission, str_metric_score, qwk, feature_score)



if __name__ == '__main__':
    main()