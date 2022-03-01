# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG = {
    "data": {
        "experiment_name": "experiment",
        "data_path": "data/data.csv",
        "emb_path": "embeddings/GoogleNews-vectors-negative300.bin', binary=True",
        "setting": "binary",  # "binary", "info_type"
        "test_size": 0.2,
        "split_random_state": 20,
        "balancing": False,  # False, "oversampling", "undersampling"
    },
    "train": {
        "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "batch_size": 256,
        "epochs": 400,
        "early_stop": 10,  # int [0: no early stopping]
        "val_set": 0.1,  # *100% percent
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "loss": 'categorical_crossentropy'
    },
    "model": {
        "embedding_dim": 300,
        # TODO: add custom model architecture
    }
}
