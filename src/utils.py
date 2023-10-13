import os, sys
import numpy as np
import pandas as pd

import dill
import pickle

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_object:
            pickle.dump(obj, file_object)
    
    except Exception as e:
        raise CustomException(e, sys)

def load_object(self):
    try:
        with open(file_path, 'rb') as file_object:
            return pickle.load(file_object)

    except Exception as e:
        raise CustomException(e, sys)