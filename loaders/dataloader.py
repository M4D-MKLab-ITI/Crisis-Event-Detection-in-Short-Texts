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

    # BERT (balance out the binary dataset)
    def augment(self, x, y, encoder=None, majority_c=1):
        if self.config.data["setting"] == "info_type": majority_c = 5
        balanced_number = y[:, majority_c].sum()

        # list all augmented datasets
        list_of_files = [f.name for f in os.scandir(self.config.data["augmentation_path"])]

        """
        The following conditional statement is only for full reproducibility.
        In the results of the paper the multiclass augmentation was run in a later version
        where we did apply sorting to the list, whereas this is not the case for the binary
        setting.
        """
        if self.config.data["setting"] == "info_type": list_of_files.sort()

        for i, file in enumerate(list_of_files):
            if self.config.data["setting"] == "info_type":
                df = pd.read_csv(self.config.data["augmentation_path"] + "/" + file, sep='\t')
                augmented_texts = df.iloc[:, 1].to_numpy()
                labels = df.iloc[:, 0].to_numpy()
                label = labels[0]
                labels = encoder.transform(labels.reshape(-1, 1)).toarray()
            else:
                df = pd.read_csv(self.config.data["augmentation_path"] + "/" + file, sep='\t', header=None)
                augmented_texts = df.iloc[:, 0]
                label = 0
                labels = np.array([[1.0, 0.0] for dummy in augmented_texts])
            x = np.concatenate((x, augmented_texts), axis=0)
            y = np.concatenate((y, labels), axis=0)
            if y[:, label].sum() > balanced_number:
                x = x[:int(-(y[:, label].sum() - balanced_number))]
                y = y[:int(-(y[:, label].sum() - balanced_number)), :]
        """
        # discarding sam samples for complete balance
        if fn == "train":
            x = x[:-1804]
            y = y[:-1804, :]
        if fn == 'val':
            x = x[:-235]
            y = y[:-235, :]"""
        indices = np.random.permutation(len(x))
        return x[indices], y[indices]

