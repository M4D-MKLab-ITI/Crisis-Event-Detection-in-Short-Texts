# -*- coding: utf-8 -*-
"""Data Loader"""

import pandas as pd
import os

"""
Class responsible for loading functions such as,
loading models and datasets
"""
class DataLoader:
    """Data Loader class"""
    def __init__(self):
        pass

    """
    Loads Crisis Lex dataset from data.csv file.
    data.csv is a file that was derived by transformations on the original CrisisLex data,
    with ease of use and reproducibility purposes in mind.
    """
    @staticmethod
    def load_data(pth):
        return pd.read_csv(pth)
