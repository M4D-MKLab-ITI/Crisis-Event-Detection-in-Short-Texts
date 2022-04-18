# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
np.random.seed(54322)
import random
import os
from tfdeterminism import patch
patch()

from configs.config import CFG
from model.model_ import Model
from loaders import dataloader
from utils.config import Config
from preprocessing import prep
from evaluation.evaluator import Eval
from utils import helpers

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def op(config, embedding_matrix, evaluator, seeds, experiment_name,
             xtrain, ytrain, xtest, ytest, xval, yval):

    for i, seed in enumerate(seeds):
        with tf.Session() as sess:
            model_cls = Model(config=config, emb_mat=embedding_matrix)
            print("experiment " + str(i + 1) + " with seed " + str(seed))
            set_seeds(seed)
            # training and evaluation
            model_cls.build_model()
            model_cls.vis_model()
            model_cls.compile_model()
            history = model_cls.training(xtrain, ytrain, xval, yval)
            evaluator.evaluation(xtest, ytest, history.history, seed=seed, model=model_cls.model)
            helpers.graph(history.history, log_dir=experiment_name, fig_name="/plots/losses" + str(seed) + ".png")
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
    xtrain, ytrain, xtest, ytest, xval, yval = preprocessor.splitting(tweets_preped, labels)
    if config.data["balancing"] != False:
        xtrain, ytrain = preprocessor.balancing(xtrain, ytrain)
    elif config.data["augmentation"]:
        #f_name = "multiclass" if config.data["setting"] == "info_type" else "binary"
        xtrain, ytrain = d_l.augment(xtrain, ytrain, encoder=preprocessor.oh_enc)
    tokenizer, xtrain, xtest, xval = preprocessor.tokens(train_data=xtrain, test_data=xtest, val_data=xval)

    # update config
    config.set_sequence_len(xtrain=xtrain)
    config.set_vocabulary_size(size=len(tokenizer.word_index))

    """
    loading embedding vectors
    building embedding matrix
    """
    w2v_model = d_l.load_embeddings()
    embedding_matrix = d_l.build_embedding_matrix(w2v_model=w2v_model,
                                                  tokenizer=tokenizer)

    """training"""

    evaluator = Eval(exp_repeats=config.get_number_of_experiments(),
                     n_class=config.get_output_size(),
                     log_dir=config.data['experiment_name'])
    seeds = config.train['seeds']
    op(config, embedding_matrix,
       evaluator, seeds,
       config.data["experiment_name"],
       xtrain, ytrain,
       xtest, ytest,
       xval, yval)


if __name__ == '__main__':
    main_pipeline()
