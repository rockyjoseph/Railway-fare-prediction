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
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
        origin: str,
        destination: str,
        start_day: int,
        start_minutes: int,
        start_hours: int,
        start_month: int,
        end_day: int,
        end_minutes: int,
        end_hours: int,
        end_month: int,
        train_type: str,
        train_class: str,
        fare: str
    ):

        self.origin = origin
        self.destination = destination
        self.start_minutes = start_minutes
        self.start_hours = start_hours
        self.start_month = start_month
        self.end_day = end_day
        self.end_minutes = end_minutes
        self.end_hours = end_hours
        self.end_month = end_month
        self.train_type = train_type
        self.train_class = train_class
        self.fare = fare 

    def get_data_as_frame(self):
        try:
            custom_data_input = {
                'origin': [self.origin],
                'destination': [self.destination],
                'start_day': [self.start_day],
                'start_minutes': [self.start_minutes],
                'start_hours': [self.start_hours],
                'start_month': [self.start_month],
                'end_day': [self.end_day],
                'end_hours': [self.end_hours],
                'end_minutes': [self.end_minutes],
                'end_month': [self.end_month],
                'train_type': [self.train_type],
                'train_class': [self.train_class],
                'fare': [self.fare]
            }

            return pd.DataFrame(custom_data_input)

        except Exception as e:
            raise CustomException(e, sys)