# -*- coding: utf-8 -*-
import random
import numpy as np
import os
import tensorflow as tf
import keras
from keras.utils.vis_utils import plot_model
from . import architectures as arch



class Model:

    def __init__(self, config, emb_mat):
        """
        Model Class

        :param Any config: Configuration Class
        :param Any emb_mat: Embedding Matrix
        """
        self.experiment_name = config.data['experiment_name']
        self.augmentation = config.data['augmentation']

        self.base_model = None
        self.model = None
        self.model_name = config.model['architecture_name']
        self.n_class = config.get_output_size()
        self.embedding_dim = config.model['embedding_dim']
        self.sequence_len = config.model['seq_len']
        self.vocab_size = config.model['vocab_size']

        self.dataset = None
        self.batch_size = config.train['batch_size']
        self.epochs = config.train['epochs']
        self.early_stop = config.train['early_stop']  # patience
        self.optimizer = config.train['optimizer']['type']
        self.learning_rate = config.train['optimizer']['lr']
        self.loss = config.train['loss']
        self.embedding_matrix = emb_mat

        self.val_split = config.train['val_set']  # percentage

    def build_model(self):
        """
        builds model and adds it to the respective class property
        """
        model_name = self.model_name
        if model_name == 'MCNN':
            self.model = arch.mcnn(self.sequence_len, self.vocab_size, self.n_class,
                                  self.embedding_dim, self.embedding_matrix)
        elif model_name == 'MCNN-MA':
            self.model = arch.mcnn_ma(self.sequence_len, self.vocab_size, self.n_class,
                                      self.embedding_dim, self.embedding_matrix)
        elif model_name == 'AD-MCNN':
            self.model = arch.ad_mcnn(self.sequence_len, self.vocab_size, self.n_class,
                                  self.embedding_dim, self.embedding_matrix)
        elif model_name == 'AD-PGRU':
            self.model = arch.ad_pgru(self.sequence_len, self.vocab_size, self.n_class,
                                      self.embedding_dim, self.embedding_matrix)
        elif model_name == 'STACKED':
            self.model = arch.stacked_sae(self.sequence_len, self.vocab_size, self.n_class,
                                      self.embedding_dim, self.embedding_matrix)

    def set_seeds(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def vis_model(self):
        """visualizes and summarizes the model"""
        plot_model(self.model, to_file='experiments/' + self.experiment_name + '/model.png', show_shapes=True, show_layer_names=True)
        self.model.summary()

    def compile_model(self):
        """Compiles the model"""
        opt = self.opt_select()
        self.model.compile(optimizer=opt, loss=self.loss, metrics=['accuracy'])

    def training(self, x_train, y_train, xval, yval):
        # callbacks setup
        if self.early_stop:
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.early_stop)
            if self.augmentation:
                history = self.model.fit(x_train, y_train, batch_size=self.batch_size,
                                    epochs=self.epochs, validation_data=(xval, yval), verbose=1, shuffle=True,
                                    callbacks=[early_stop])
            else:
                history = self.model.fit(x_train, y_train, batch_size=self.batch_size,
                                     epochs=self.epochs, validation_split=self.val_split, verbose=1, shuffle=True,
                                     callbacks=[early_stop])
        else:
            history = self.model.fit(x_train, y_train, batch_size=self.batch_size,
                                     epochs=self.epochs, validation_split=self.val_split, verbose=1, shuffle=True)
        return history

    def opt_select(self):
        if self.optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer == 'adadelta':
            opt = keras.optimizers.Adadelta(learning_rate=self.learning_rate)
        elif self.optimizer == 'nadam':
            opt = keras.optimizers.Nadam(learning_rate=self.learning_rate)
        else:  # default case
            opt = keras.optimizers.Adam()
        return opt
