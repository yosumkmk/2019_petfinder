# -*- coding: utf-8 -*-
import warnings
from src.util.log_util import set_logger
import pandas as pd
warnings.filterwarnings('ignore')
import numpy as np
import os
import input
from keras.applications.densenet import preprocess_input

import cv2


logger = set_logger(__name__)


img_size = 256
batch_size = 256


def resize_to_square(im):
    old_size = im.shape[:2]
    ratio = float(img_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_im


def load_image(path, pet_id):
    image = cv2.imread(f'{path}{pet_id}-1.jpg')
    new_image = resize_to_square(image)
    new_image = preprocess_input(new_image)
    return new_image







#############
def read_train_data(path=input.__path__[0], nrows=None):
    logger.info('Input train_data')
    train_df = pd.read_csv(os.path.join(path, 'train.csv'), nrows=nrows)
    return train_df

def read_test_data(path=input.__path__[0]):
    logger.info('Input test_data')
    test_df = pd.read_csv(os.path.join(path, 'test.csv'))
    return test_df

def read_permutation_importance(name='0303_base_feature.csv'):
    return pd.read_csv(os.path.join(data.permutation_importance.__path__[0], name))

def split_train_data(train, split_rate):
    split_index = len(train) // split_rate
    return train[0:split_index], train[split_index]

def check_missing_data(df):
    flag=df.isna().sum().any()
    if flag==True:
        total = df.isnull().sum()
        percent = (df.isnull().sum())/(df.isnull().count()*100)
        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        data_type = []
        # written by MJ Bahmani
        for col in df.columns:
            dtype = str(df[col].dtype)
            data_type.append(dtype)
        output['Types'] = data_type
        return(np.transpose(output))
    else:
        return(False)

if __name__ == '__main__':
    pass