# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
from keras.applications.densenet import preprocess_input, DenseNet121
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import keras.backend as K
from tqdm import tqdm
from src.data.make_dataset import load_image
import pandas as pd
import numpy as np

img_size = 256
batch_size = 256

def densenet_model(weight_path):
    inp = Input((256, 256, 3))
    backbone = DenseNet121(input_tensor=inp,
                           weights=weight_path,
                           include_top=False)



    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)
    x = AveragePooling1D(4)(x)
    out = Lambda(lambda x: x[:, :, 0])(x)
    m = Model(inp, out)
    return m

def predict_using_img(m, data, img_path):
    pet_ids = data['PetID'].values
    n_batches = len(pet_ids) // batch_size + 1

    features = {}
    for b in tqdm(range(n_batches)):
        start = b * batch_size
        end = (b + 1) * batch_size
        batch_pets = pet_ids[start:end]
        batch_images = np.zeros((len(batch_pets), img_size, img_size, 3))
        for i, pet_id in enumerate(batch_pets):
            try:
                batch_images[i] = load_image(img_path, pet_id)
            except:
                pass
        batch_preds = m.predict(batch_images)
        for i, pet_id in enumerate(batch_pets):
            features[pet_id] = batch_preds[i]

    data_feats = pd.DataFrame.from_dict(features, orient='index')
    data_feats.columns = [f'pic_{i}' for i in range(data_feats.shape[1])]
    data_feats = data_feats.reset_index()
    data_feats.rename({'index': 'PetID'}, axis='columns', inplace=True)
    return data_feats




if __name__ == '__main__':
    pass