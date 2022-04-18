# -*- coding: utf-8 -*-
"""Config class"""

import json
from . import helpers


class Config:
    def __init__(self, data, train, model):
        """
        Config class which contains data, train and model hyperparameters

        :param data:
        :param train:
        :param model:
        """
        self.data = data
        self.train = train
        self.model = model

        if self.data['setting'] == 'info_type':
            self.data['augmentation_path'] += "multiclass"
        else:
            self.data['augmentation_path'] += "binary"

    def get_number_of_experiments(self):
        return len(self.train['seeds'])

    def get_output_size(self):
        return 2 if self.data['setting'] == 'binary' else 7

    def set_sequence_len(self, xtrain):
        self.model['seq_len'] = xtrain.shape[1]

    def set_vocabulary_size(self, size):
        self.model['vocab_size'] = size

    @classmethod
    def from_json(cls, cfg):
        """Creates config from json"""
        params = json.loads(json.dumps(cfg))  #, object_hook=HelperObject)
        return cls(params['data'], params['train'], params['model'])

    def create_folders(self):
        experiment_name = self.data['experiment_name']
        helpers.crt_folder("experiments")
        helpers.crt_folder("experiments/" + experiment_name)
        helpers.crt_folder("experiments/" + experiment_name + "/reports")
        helpers.crt_folder("experiments/" + experiment_name + "/pred")
        helpers.crt_folder("experiments/" + experiment_name + "/plots")



class HelperObject(object):
    """Helper class to convert json into Python object"""
    def __init__(self, dict_):
        self.__dict__.update(dict_)
