from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf

"""
All functionality related to preprocessing
"""
class Preprocessor:
    def __init__(self, data, setting):
        self.data = data
        self.setting = setting
        self.l_enc = None
        self.oh_enc = None

    # if called only for test: train_data is None --> so it returns None xtrain var.
    def tokens(self, test_data, train_data=None):
        tokenizer = self.tokenizer
        maxim = self.maxim
        xtrain = None
        if train_data is not None:
            tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<OOV>')
            tokenizer.fit_on_texts(train_data)
            train_indexes = tokenizer.texts_to_sequences(train_data)
            if self.maxim == None: self.maxim = max([len(i) for i in train_indexes])
            xtrain = tf.keras.preprocessing.sequence.pad_sequences(train_indexes, maxlen=self.maxim)
            self.to_pickle["tokenizer"] = tokenizer
            self.to_pickle["maxim"] = self.maxim
            self.de_serialize_inference_objects(False)
        test_indexes = tokenizer.texts_to_sequences(test_data)
        xtest = tf.keras.preprocessing.sequence.pad_sequences(test_indexes, maxlen=self.maxim)
        return tokenizer, xtrain, xtest

    def transform_crisis_lex(self, data):
        tweets = data.iloc[:, 1].to_numpy()  # tweets to nd array
        if self.setting == "info_source":
            labels = data.iloc[:, 2]
        elif self.setting == "info_type":
            labels = data.iloc[:, 3]
        else:
            labels = data.iloc[:, 4].to_numpy()  # informativeness is the label column in all other settings.
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
        #self.to_pickle["l_enc"] = self.l_enc
        #self.to_pickle["oh_enc"] = self.oh_enc
        return tweets, labels
