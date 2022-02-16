# -*- coding: utf-8 -*-
""" main.py """

from configs.config import CFG
from model.model import Model
from loaders import dataloader
from utils.config import Config


def run():
    """Builds model, loads data, trains and evaluates"""
    config = Config.from_json(CFG)
    model = Model(config)


    def load_data(self):
        """Loads and Preprocess data """
        self.dataset, self.info = DataLoader().load_data(self.config.data)
        self._preprocess_data()

    model.load_data()
    model.build()
    model.train()
    model.evaluate()


if __name__ == '__main__':
    run()
