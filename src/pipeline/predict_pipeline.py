import sys
import pandas as pd

from src.utils import load_object
from src.exception import CustomException

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts\model.pkl'
            model = load_object(file_path = model_path)
            pred = model.predict(features)

            return pred

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
        origin: str,
        destination: str,
        start_date: int,
        end_date: str,
        train_type: str,
        price: float,
        train_class: str,
        fare: str
    ):

        self.origin = origin
        self.destination = destination
        self.start_date = start_date
        self.end_date = end_date
        self.train_type = train_type
        self.price = price
        self.train_class = train_class
        self.fare = fare

    def get_data_as_frame(self):
        try:
            custom_data_input = {
                'origin': [self.origin],
                'destination': [self.destination],
                'start_date': [self.start_date],
                'end_date': [self.end_date],
                'train_type': [self.train_type],
                'price': [self.origin],
                'train_class': [self.train_class],
                'fare': [self.fare],
            }

            return pd.DataFrame(custom_data_input)

        except Exception as e:
            raise CustomException(e, sys)
