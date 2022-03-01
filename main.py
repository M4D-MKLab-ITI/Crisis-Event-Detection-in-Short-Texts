# -*- coding: utf-8 -*-
from configs.config import CFG
from model.model import Model
from loaders import dataloader
from utils.config import Config
from preprocessing import prep
from evaluation.evaluator import Eval
from utils import helpers

import tensorflow as tf
import numpy as np
import gensim.models
import random
import os
from tfdeterminism import patch
patch()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def op(model, evaluator, seeds,
             xtrain, ytrain, xtest, ytest):

    for i, seed in enumerate(seeds):
        with tf.Session() as sess:
            print("experiment " + str(i + 1) + " with seed " + str(seed))
            set_seeds(seed)
            # training and evaluation
            model = model.build_model()
            history = model.training(xtrain, ytrain)
            evaluator(xtest, ytest, history.history, seed=seed, model=model)
            helpers.graph(history.history, fig_name="plots/losses" + str(seed) + ".png")
        tf.reset_default_graph()
    if evaluator.exp_repeats != 1:
        evaluator.save_results()


def main_pipeline():
    """Builds model, loads data, trains and evaluates"""

    """setting up model configuration"""
    config = Config.from_json(CFG)

    """auto configuration"""
    config.create_folders()
    """n_classes = config.get_output_size()
    exp_repeats = config.get_number_of_experiments()
    """

    """loading dataset"""
    d_l = dataloader.DataLoader(config)
    data = d_l.load_data()

    """preprocessing pipeline"""
    preprocessor = prep.Preprocessor(config=config,
                                     data=data)
    tweets, labels = preprocessor.transform_crisis_lex()
    tweets_preped = preprocessor.text_preprocessing(tweets)
    xtrain, ytrain, xtest, ytest = preprocessor.splitting(tweets_preped, labels)
    if config.data["balancing"] != False:
        xtrain, ytrain = preprocessor.balancing(xtrain, ytrain)
    tokenizer, xtrain, xtest = preprocessor.tokens(train_data=xtrain, test_data=xtest)

    # update config
    config.set_sequence_len(xtrain=xtrain)
    config.set_vocabulary_size(size=len(tokenizer.word_index))

    """
    loading embedding vectors
    building embedding matrix
    """
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(config.data['emb_path'],
                                                                binary=True)
    embedding_matrix = d_l.build_embedding_matrix(w2v_model=w2v_model,
                                                  tokenizer=tokenizer)

    """training"""
    model = Model(config=config, emb_mat=embedding_matrix)
    evaluator = Eval(exp_repeats=config.get_number_of_experiments(),
                     n_class=config.get_output_size())
    seeds = config.train['seeds']
    op(model, evaluator, seeds,
             xtrain, ytrain,
             xtest, ytest)


if __name__ == '__main__':
    main_pipeline()
