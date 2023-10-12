import os, sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.logger import logging
from src.utils import save_object
from src.exception import CustomException

@dataclass
class DataValidationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataValidation:
    def __init__(self):
        self.data_validation_config = DataValidationConfig()

    def get_data_validation_object(self):
        try:
            pass


        except Exception as e:
            raise CustomException(e, sys)