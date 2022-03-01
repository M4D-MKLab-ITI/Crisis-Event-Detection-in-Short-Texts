# -*- coding: utf-8 -*-
"""Data Loader"""

import numpy as np
import pandas as pd
import os
import gensim

"""
Class responsible for loading functions such as,
loading models and datasets
"""
class DataLoader:
    """Data Loader class"""
    def __init__(self, config):
        self.config = config

    """
    Loads Crisis Lex dataset from data.csv file.
    data.csv is a file that was derived by transformations on the original CrisisLex data,
    with ease of use and reproducibility purposes in mind.
    """
    def load_data(self):
        return pd.read_csv(self.config.data["data_path"])

    def load_embeddings(self):
        # load Google's pre-trained w2v model
        return gensim.models.KeyedVectors.load_word2vec_format(self.config.data['emb_path'],
                                                               binary=True)

    def build_embedding_matrix(self, w2v_model, tokenizer):
        # build embedding matrix
        count = 0
        embedding_matrix = np.random.random((len(tokenizer.word_index) + 1, self.config.model['embedding_dim']))
        for word, i in tokenizer.word_index.items():
            try:
                vec = w2v_model.wv[word]
                count += 1
                embedding_matrix[i] = vec
            except KeyError:  # token is not in the corpus.
                continue
        print("Converted %d words (%d misses)" % (count, len(tokenizer.word_index) - count))
        return embedding_matrix
