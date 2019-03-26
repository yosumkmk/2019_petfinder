# -*- coding: utf-8 -*-
import warnings
from PIL import Image
import os
import pandas as pd

warnings.filterwarnings('ignore')

def getSize(filename):
    st = os.stat(filename)
    return st.st_size


def getDimensions(filename):
    img_size = Image.open(filename).size
    return img_size

def agg_img_feature(data_image_files):
    data_df_imgs = img_parse(data_image_files)

    aggs = {
        'image_size': ['sum', 'mean', 'var'],
        'width': ['sum', 'mean', 'var'],
        'height': ['sum', 'mean', 'var'],
    }

    agg_train_imgs = data_df_imgs.groupby('PetID').agg(aggs)
    new_columns = [k + '_' + agg for k in aggs.keys() for agg in aggs[k]]
    agg_train_imgs.columns = new_columns
    agg_train_imgs = agg_train_imgs.reset_index()
    return agg_train_imgs



def img_parse(data_image_files):
    split_char= '/'
    data_df_imgs = pd.DataFrame(data_image_files)
    data_df_imgs.columns = ['image_filename']
    data_imgs_pets = data_df_imgs['image_filename'].apply(lambda x: x.split(split_char)[-1].split('-')[0])
    data_df_imgs = data_df_imgs.assign(PetID=data_imgs_pets)

    data_df_imgs['image_size'] = data_df_imgs['image_filename'].apply(getSize)
    data_df_imgs['temp_size'] = data_df_imgs['image_filename'].apply(getDimensions)
    data_df_imgs['width'] = data_df_imgs['temp_size'].apply(lambda x: x[0])
    data_df_imgs['height'] = data_df_imgs['temp_size'].apply(lambda x: x[1])
    data_df_imgs = data_df_imgs.drop(['temp_size'], axis=1)

    return data_df_imgs


if __name__ == '__main__':
    pass