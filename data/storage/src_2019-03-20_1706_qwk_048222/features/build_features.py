# -*- coding: utf-8 -*-
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import glob
from src.util.log_util import set_logger
from logging import StreamHandler, Formatter, getLogger, FileHandler, DEBUG, INFO, ERROR
from sklearn.preprocessing import StandardScaler
from src.data.make_dataset import read_permutation_importance
from sklearn.decomposition import TruncatedSVD
import json

logger = set_logger(__name__)


def fill_and_drop_feature(data):
    data.drop(['main_breed_BreedName', 'second_breed_BreedName'], axis=1, inplace=True)
    data[((data['main_breed_Type'] != data['second_breed_Type'])
          & (~np.isnan(data['second_breed_Type'])))]['main_breed_Type'] = \
        data[((data['main_breed_Type'] != data['second_breed_Type'])
              & (~np.isnan(data['second_breed_Type'])))]['second_breed_Type']
    data[((data['main_breed_Type'] != data['second_breed_Type'])
          & (~np.isnan(data['second_breed_Type'])))]['second_breed_Type'] = np.nan
    return data


def add_feature(data):
    data['mixed_Breed'] = -1
    data['mixed_Breed'] = np.where((data['Breed1'] != data['Breed2'])
                                   & ~np.isnan(data['Breed1'])
                                   & ~np.isnan(data['Breed2']), 1, data['mixed_Breed'])
    data['mixed_Breed'] = np.where((((data['Breed1'] == data['Breed2'])
                                     & ((data['Breed1'] == 370)))
                                    | ((data['Breed1'] == 370) & (np.isnan(data['Breed2'])))
                                    | ((data['Breed2'] == 370) & (np.isnan(data['Breed1']))))
                                   , 0, data['mixed_Breed'])
    data['mixed_Breed'] = np.where(((data['Breed1'] == data['Breed2'])
                                   & (data['Breed1'] != 370)), 2, data['mixed_Breed'])
    data['mixed_Breed'] = np.where((((data['Breed1'] != data['Breed2'])
                                     & (~np.isnan(data['Breed1']))
                                    & (~np.isnan(data['Breed2'])))
                                    & (((data['Breed1'] != 370) & (data['Breed2'] == 370))
                                    | ((data['Breed2'] == 370) & (data['Breed1'] == 370))))
                                   , 3, data['mixed_Breed'])
    data['State'] = data['State'].factorize()[0]

    return data


def agg_feature(data, data_metadata, data_sentiment):
    aggregates = ['sum', 'mean', 'var']
    sent_agg = ['sum']

    # data
    data_metadata_desc = data_metadata.groupby(['PetID'])['metadata_annots_top_desc'].unique()
    data_metadata_desc = data_metadata_desc.reset_index()
    data_metadata_desc[
        'metadata_annots_top_desc'] = data_metadata_desc[
        'metadata_annots_top_desc'].apply(lambda x: ' '.join(x))

    prefix = 'metadata'
    data_metadata_gr = data_metadata.drop(['metadata_annots_top_desc'], axis=1)
    for i in data_metadata_gr.columns:
        if 'PetID' not in i:
            data_metadata_gr[i] = data_metadata_gr[i].astype(float)
    data_metadata_gr = data_metadata_gr.groupby(['PetID']).agg(aggregates)
    data_metadata_gr.columns = pd.Index([f'{c[0]}_{c[1].upper()}' for c in data_metadata_gr.columns.tolist()])
    data_metadata_gr = data_metadata_gr.reset_index()

    data_sentiment_desc = data_sentiment.groupby(['PetID'])['sentiment_entities'].unique()
    data_sentiment_desc = data_sentiment_desc.reset_index()
    data_sentiment_desc[
        'sentiment_entities'] = data_sentiment_desc[
        'sentiment_entities'].apply(lambda x: ' '.join(x))

    prefix = 'sentiment'
    data_sentiment_gr = data_sentiment.drop(['sentiment_entities'], axis=1)
    for i in data_sentiment_gr.columns:
        if 'PetID' not in i:
            data_sentiment_gr[i] = data_sentiment_gr[i].astype(float)
    data_sentiment_gr = data_sentiment_gr.groupby(['PetID']).agg(sent_agg)
    data_sentiment_gr.columns = pd.Index([f'{c[0]}' for c in data_sentiment_gr.columns.tolist()])
    data_sentiment_gr = data_sentiment_gr.reset_index()
    # data merges:
    data_proc = data.copy()
    data_proc = data_proc.merge(data_sentiment_gr, how='left', on='PetID')
    data_proc = data_proc.merge(data_metadata_gr, how='left', on='PetID')
    data_proc = data_proc.merge(data_metadata_desc, how='left', on='PetID')
    data_proc = data_proc.merge(data_sentiment_desc, how='left', on='PetID')
    assert data_proc.shape[0] == data.shape[0]
    return data_proc


def merge_labels_breed(data_proc, labels_breed):
    data_breed_main = data_proc[['Breed1']].merge(labels_breed, how='left', left_on='Breed1', right_on='BreedID', suffixes=('', '_main_breed'))
    data_breed_main = data_breed_main.iloc[:, 2:]
    data_breed_main = data_breed_main.add_prefix('main_breed_')
    data_breed_second = data_proc[['Breed2']].merge(labels_breed, how='left', left_on='Breed2', right_on='BreedID', suffixes=('', '_second_breed'))
    data_breed_second = data_breed_second.iloc[:, 2:]
    data_breed_second = data_breed_second.add_prefix('second_breed_')
    data_proc = pd.concat([data_proc, data_breed_main, data_breed_second], axis=1)
    return data_proc


def adopt_svd(train_feats, test_feats):
    n_components = 32
    svd_ = TruncatedSVD(n_components=n_components, random_state=1337)

    features_df = pd.concat([train_feats, test_feats], axis=0)
    features = features_df[[f'pic_{i}' for i in range(256)]].values

    svd_col = svd_.fit_transform(features)
    svd_col = pd.DataFrame(svd_col)
    svd_col = svd_col.add_prefix('IMG_SVD_')
    return svd_col


def max_min_feature(train_df, test_df, idx):
    num_list = np.arange(5)
    for num in num_list:
        for df in [test_df, train_df]:
            use_values = df[idx].values
            df['sum_max_top' + str(num)] = 0
            df['sum_min_top' + str(num)] = 0
            sum_max_top = df['sum_max_top' + str(num)].values
            sum_min_top = df['sum_min_top' + str(num)].values
            for i in range(len(df)):
                sort_values = np.sort(use_values[i])
                sum_max_top[i] = sort_values[-(num + 1)]
                sum_min_top[i] = sort_values[num]
            df['sum_max_top' + str(num)] = sum_max_top
            df['sum_min_top' + str(num)] = sum_min_top
    return train_df, test_df


def id_match_feature(train_df, test_df, idx):
    train_df['ID_num'] = [int(train_df['ID_code'][x][6::]) for x in range(len(train_df))]
    test_df['ID_num'] = [int(test_df['ID_code'][x][5::]) for x in range(len(test_df))]
    for df in [test_df, train_df]:
        df['ID_match'] = np.sum(df[idx] == np.tile((df['ID_num'].values / 10 ** 4).reshape(-1, 1), (1, len(idx))), axis=1)
        df.drop('ID_num', inplace=True)


def round_feature(train_df, test_df, idx):
    for df in [test_df, train_df]:
        for feat in idx:
            df[feat] = np.round(df[feat], 3)
            df[feat] = np.round(df[feat], 3)


def outlier_distribution_categorize(train_df, test_df, idx):
    outlier_dict = {}
    for c in idx:
        count_df = np.round(train_df[c], 2).value_counts().sort_index()
        outlier = count_df[(
            #         ((count_df.diff().abs() > 50) & (count_df < 200)) |
            (count_df.diff().abs() > 130)
        )].index.tolist()
        outlier_dict[c] = outlier
    for k, l in outlier_dict.items():
        o_l = []
        if len(l) >= 1:
            min_o = min(l)
            max_o = max(l)
            if min_o < 0:
                o_l.append(min_o)
            if max_o > 0:
                o_l.append(max_o)
            for o in o_l:
                train_df[k + '_outlier_' + str(o)] = (np.round(train_df[k], 2) == o).astype(np.int64)
                test_df[k + '_outlier_' + str(o)] = (np.round(test_df[k], 2) == o).astype(np.int64)
    return train_df, test_df


def process_data(train_df, test_df):
    logger.info('Features engineering')
    idx = [c for c in train_df.columns if c not in ['ID_code', 'target']]
    train_df, test_df = max_min_feature(train_df, test_df, idx)
    scaler = StandardScaler()
    train_df[idx] = scaler.fit_transform(train_df[idx])
    test_df[idx] = scaler.transform(test_df[idx])
    # perm_imp = read_permutation_importance()
    # remove_features_weight = 0
    # remove_features = perm_imp[perm_imp.weight < -0.0002]
    # remove_columns.extend(remove_features.feature.tolist())
    # train_df.drop(columns=remove_columns, inplace=True)
    # test_df.drop(columns=remove_columns, inplace=True)
    train_df, test_df = outlier_distribution_categorize(train_df, test_df, idx)

    print('Train and test shape:', train_df.shape, test_df.shape)
    return train_df, test_df


if __name__ == '__main__':
    pass
