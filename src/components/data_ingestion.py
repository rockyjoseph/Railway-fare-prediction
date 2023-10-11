import os, sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass

from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the Data Ingestion module")

        try:
            df = pd.read_csv('notebook\data\train_price.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Data Ingestion Completed!")

            return self.ingestion_config.raw_data_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    module = DataIngestion()
    raw_data = module.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_data, test_data = data_transformation.initiate_data_transformation()
    print(train_data.shape)
    print(test_data.shape)