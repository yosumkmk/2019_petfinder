# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import glob
from src.util.log_util import set_logger
from sklearn.decomposition import TruncatedSVD
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import input
import os

logger = set_logger(__name__)


def parse_tfidf(X_temp, X_text):
    n_components = 16
    text_features = []

    # Generate text features:
    for i in X_text.columns:
        # Initialize decomposition methods:
        print(f'generating features from: {i}')
        tfv = TfidfVectorizer(min_df=2, max_features=None,
                              strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',
                              ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1)
        svd_ = TruncatedSVD(
            n_components=n_components, random_state=1337)

        tfidf_col = tfv.fit_transform(X_text.loc[:, i].values)

        svd_col = svd_.fit_transform(tfidf_col)
        svd_col = pd.DataFrame(svd_col)
        svd_col = svd_col.add_prefix('TFIDF_{}_'.format(i))

        text_features.append(svd_col)

    text_features = pd.concat(text_features, axis=1)

    X_temp = pd.concat([X_temp, text_features], axis=1)

    for i in X_text.columns:
        X_temp = X_temp.drop(i, axis=1)
    return X_temp

class PetFinderParser(object):

    def __init__(self, debug=False):

        self.debug = debug
        self.sentence_sep = ' '

        self.extract_sentiment_text = False

    def open_json_file(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            json_file = json.load(f)
        return json_file

    def parse_sentiment_file(self, file):
        """
        Parse sentiment file. Output DF with sentiment features.
        """

        file_sentiment = file['documentSentiment']
        file_entities = [x['name'] for x in file['entities']]
        file_entities = self.sentence_sep.join(file_entities)

        file_sentences_sentiment = [x['sentiment'] for x in file['sentences']]

        file_sentences_sentiment = pd.DataFrame.from_dict(
            file_sentences_sentiment, orient='columns')
        file_sentences_sentiment_df = pd.DataFrame(
            {
                'magnitude_sum': file_sentences_sentiment['magnitude'].sum(axis=0),
                'score_sum': file_sentences_sentiment['score'].sum(axis=0),
                'magnitude_mean': file_sentences_sentiment['magnitude'].mean(axis=0),
                'score_mean': file_sentences_sentiment['score'].mean(axis=0),
                'magnitude_var': file_sentences_sentiment['magnitude'].var(axis=0),
                'score_var': file_sentences_sentiment['score'].var(axis=0),
            }, index=[0]
        )

        df_sentiment = pd.DataFrame.from_dict(file_sentiment, orient='index').T
        df_sentiment = pd.concat([df_sentiment, file_sentences_sentiment_df], axis=1)

        df_sentiment['entities'] = file_entities
        df_sentiment = df_sentiment.add_prefix('sentiment_')

        return df_sentiment

    def parse_metadata_file(self, file):
        """
        Parse metadata file. Output DF with metadata features.
        """

        file_keys = list(file.keys())

        if 'labelAnnotations' in file_keys:
            file_annots = file['labelAnnotations']
            file_top_score = np.asarray([x['score'] for x in file_annots]).mean()
            file_top_desc = [x['description'] for x in file_annots]
        else:
            file_top_score = np.nan
            file_top_desc = ['']

        file_colors = file['imagePropertiesAnnotation']['dominantColors']['colors']
        file_crops = file['cropHintsAnnotation']['cropHints']

        file_color_score = np.asarray([x['score'] for x in file_colors]).mean()
        file_color_pixelfrac = np.asarray([x['pixelFraction'] for x in file_colors]).mean()

        file_crop_conf = np.asarray([x['confidence'] for x in file_crops]).mean()

        if 'importanceFraction' in file_crops[0].keys():
            file_crop_importance = np.asarray([x['importanceFraction'] for x in file_crops]).mean()
        else:
            file_crop_importance = np.nan

        df_metadata = {
            'annots_score': file_top_score,
            'color_score': file_color_score,
            'color_pixelfrac': file_color_pixelfrac,
            'crop_conf': file_crop_conf,
            'crop_importance': file_crop_importance,
            'annots_top_desc': self.sentence_sep.join(file_top_desc)
        }

        df_metadata = pd.DataFrame.from_dict(df_metadata, orient='index').T
        df_metadata = df_metadata.add_prefix('metadata_')

        return df_metadata

def extract_additional_features(pet_id, mode='train'):
    pet_parser = PetFinderParser()
    sentiment_filename = os.path.join(input.__path__[0],
                                      f'petfinder-adoption-prediction/{mode}_sentiment/{pet_id}.json')
    try:
        sentiment_file = pet_parser.open_json_file(sentiment_filename)
        df_sentiment = pet_parser.parse_sentiment_file(sentiment_file)
        df_sentiment['PetID'] = pet_id
    except FileNotFoundError:
        df_sentiment = []

    dfs_metadata = []
    metadata_filenames = sorted(glob.glob(os.path.join(input.__path__[0],
                                                       f'petfinder-adoption-prediction/{mode}_metadata/{pet_id}*.json')))
    if len(metadata_filenames) > 0:
        for f in metadata_filenames:
            metadata_file = pet_parser.open_json_file(f)
            df_metadata = pet_parser.parse_metadata_file(metadata_file)
            df_metadata['PetID'] = pet_id
            dfs_metadata.append(df_metadata)
        dfs_metadata = pd.concat(dfs_metadata, ignore_index=True, sort=False)
    dfs = [df_sentiment, dfs_metadata]

    return dfs

if __name__ == '__main__':
    pass