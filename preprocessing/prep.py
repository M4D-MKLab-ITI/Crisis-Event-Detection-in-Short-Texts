from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import sklearn
import numpy as np
import tensorflow as tf
from collections import Counter
import text

"""
All functionality related to preprocessing
"""


class Preprocessor:
    def __init__(self, config, data=None):
        self.data = data
        self.setting = config.data["setting"]
        self.tokenizer = None
        self.l_enc = None  # label encoder
        self.oh_enc = None  # one hot encoder
        self.maxim = None  # maximum number of words per tweet (used for padding)
        self.config = config

    def text_preprocessing(self, tweets, labels=None):
        return text.preprocessing(tweets)

    """
    Tokenizes and pads the input.
    if called only for test: train_data is None --> so it returns None xtrain var.
    """
    def tokens(self, test_data, train_data=None):
        tokenizer = self.tokenizer
        xtrain = None
        if train_data is not None:
            tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<OOV>')
            tokenizer.fit_on_texts(train_data)
            train_indexes = tokenizer.texts_to_sequences(train_data)
            if self.maxim == None: self.maxim = max([len(i) for i in train_indexes])
            xtrain = tf.keras.preprocessing.sequence.pad_sequences(train_indexes, maxlen=self.maxim)
            # self.to_pickle["tokenizer"] = tokenizer
            #self.to_pickle["maxim"] = self.maxim
            #self.de_serialize_inference_objects(False)
        test_indexes = tokenizer.texts_to_sequences(test_data)
        xtest = tf.keras.preprocessing.sequence.pad_sequences(test_indexes, maxlen=self.maxim)
        return tokenizer, xtrain, xtest

    """
    Balancing dataset with Random under/over sampling
    """
    def balancing(self, tweets, labels, strategy="undersampling"):
        tweets = tweets.reshape((-1, 1))
        labels = np.argmax(labels, axis=1)
        if strategy == "undersampling":
            undersample = RandomUnderSampler(sampling_strategy='majority')
            tweets, labels = undersample.fit_resample(tweets, labels)
        else:
            oversample = RandomOverSampler(sampling_strategy='minority')
            tweets, labels = oversample.fit_resample(tweets, labels)
        tweets = tweets.reshape((-1,))
        labels = tf.keras.utils.to_categorical(labels, num_classes=len(Counter(labels)), dtype='float32')
        return tweets, labels

    """
    Splits dataset in a stratified fashion
    """
    def splitting(self, tweets, labels):
        split = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1,
                                                               test_size=self.config.data["test_size"],
                                                               random_state=self.config.data['split_random_state'])
        train_idx, test_idx = list(split.split(tweets, labels))[0]
        xtrain = tweets[train_idx]
        ytrain = labels[train_idx]
        xtest = tweets[test_idx]
        ytest = labels[test_idx]
        return xtrain, ytrain, xtest, ytest

    def transform_crisis_lex(self):
        tweets = self.data.iloc[:, 1].to_numpy()  # tweets to nd array
        if self.setting == "info_source":
            labels = self.data.iloc[:, 2]
        elif self.setting == "info_type":
            labels = self.data.iloc[:, 3]
        else:
            labels = self.data.iloc[:, 4].to_numpy()  # informativeness is the label column in all other settings.
            for i, label in enumerate(labels):
                if label == "Not applicable":
                    labels[i] = "Not related"

        if self.setting == "binary":
            # convert label values: not related --> 0 , related --> 1
            for i, label in enumerate(labels):
                if label == "Not related":
                    labels[i] = 0
                else:
                    labels[i] = 1
        elif self.setting == "bombing" or self.setting == "shooting":
            for i, label in enumerate(labels):
                if self.setting in label:
                    labels[i] = 1
                else:
                    labels[i] = 0
        else:
            self.l_enc = LabelEncoder()
            self.l_enc.fit(labels)
            labels = self.l_enc.transform(labels)
        self.oh_enc = OneHotEncoder(handle_unknown='ignore')
        labels = self.oh_enc.fit_transform(labels.reshape(-1, 1))
        # self.to_pickle["l_enc"] = self.l_enc
        # self.to_pickle["oh_enc"] = self.oh_enc
        return tweets, labels
