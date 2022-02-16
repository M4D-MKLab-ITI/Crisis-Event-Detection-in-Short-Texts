# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG = {
    "data": {
        "path": "/home/pantelisk/projects/event/data/data.csv",
        "setting": "binary"  # "binary", "info_type"
    },
    "train": {
        "batch_size": 64,
        "early_stop": 10,
        "val_set": 5,  # percent
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
    },
    "model": {
        "input": [128, 128, 3],
        "up_stack": {
            "layer_1": 512,
            "layer_2": 256,
            "layer_3": 128,
            "layer_4": 64,
            "kernels": 3
        },
        "output": 3
    }
}
