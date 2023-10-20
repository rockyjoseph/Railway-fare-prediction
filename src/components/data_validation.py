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
    preprocessor_object_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataValidation:
    def __init__(self):
        self.data_validation_config = DataValidationConfig()

    def get_data_validation_object(self):
        '''
            This function will return the preprocessing of features.
        '''
        try:
            categorical_columns = ['origin','destination','train_type','train_class','fare']

            cat_pipeline = Pipeline(
                steps = [
                    ('ohe', OneHotEncoder(drop='first', sparse=False))
                ]
            )

            logging.info(f'Categorical columns {categorical_columns}')

            preprocessor = ColumnTransformer(
                [
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ], remainder = 'passthrough'
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self, train_path, test_path):
        '''
            This function will clean the features in the data.
        '''
        try:
            # Reading the dataset
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Declaring objects and features.
            target_column_name = 'price'
            preprocessing_object = self.get_data_validation_object()
            categorical_features = ['origin','destination','train_type','train_class','fare']

            # Making different columns of datetime format
            train_df['start_month'] = pd.to_datetime(train_df.start_date).dt.month
            train_df['start_day'] = pd.to_datetime(train_df.start_date).dt.day
            train_df['start_hours'] = pd.to_datetime(train_df.start_date).dt.hour
            train_df['start_minutes'] = pd.to_datetime(train_df.start_date).dt.minute

            train_df['end_month'] = pd.to_datetime(train_df.end_date).dt.month
            train_df['end_day'] = pd.to_datetime(train_df.end_date).dt.day
            train_df['end_hours'] = pd.to_datetime(train_df.end_date).dt.hour
            train_df['end_minutes'] = pd.to_datetime(train_df.end_date).dt.minute

            # Making different columns of datetime format
            test_df['start_month'] = pd.to_datetime(test_df.start_date).dt.month
            test_df['start_day'] = pd.to_datetime(test_df.start_date).dt.day
            test_df['start_hours'] = pd.to_datetime(test_df.start_date).dt.hour
            test_df['start_minutes'] = pd.to_datetime(test_df.start_date).dt.minute

            test_df['end_month'] = pd.to_datetime(test_df.end_date).dt.month
            test_df['end_day'] = pd.to_datetime(test_df.end_date).dt.day
            test_df['end_hours'] = pd.to_datetime(test_df.end_date).dt.hour
            test_df['end_minutes'] = pd.to_datetime(test_df.end_date).dt.minute


            # Dropping the features end_date and start_date which is not of use.
            train_df.drop(columns=['start_date','end_date'], inplace=True)
            test_df.drop(columns=['start_date','end_date'], inplace=True)

            logging.info("Read the Train-Test data")
            logging.info("Starting Preprocessing process.....")

            ## Replacing price NAN values with 0.
            train_df['price'].fillna(0, inplace = True)

            # Replacing fare values in train_data
            train_df.loc[(train_df['price']==0) & (train_df['fare'] == 'Promo'), 'price'] = train_df[train_df['fare'] == 'Promo']['price'].median()
            train_df.loc[(train_df['price']==0) & (train_df['fare'] == 'Adulto ida'), 'price'] = train_df[train_df['fare'] == 'Adulto ida']['price'].median()
            train_df.loc[(train_df['price']==0) & (train_df['fare'] == 'Flexible'), 'price'] = train_df[train_df['fare'] == 'Flexible']['price'].median()
            train_df.loc[(train_df['price']==0) & (train_df['fare'] == 'Promo +'), 'price'] = train_df[train_df['fare'] == 'Promo +']['price'].median()
            train_df.loc[(train_df['price']==0) & (train_df['fare'] == 'Mesa '), 'price'] = train_df[train_df['fare'] == 'Mesa']['price'].median()

            # Dropping the remaining NULL values in train_data
            train_df.dropna(inplace=True)

            ## Replacing price NAN values with 0.
            test_df['price'].fillna(0, inplace = True)

            # Replacing fare values in test_data.
            test_df.loc[(test_df['price']==0) & (test_df['fare'] == 'Promo'), 'price'] = test_df[test_df['fare'] == 'Promo']['price'].median()
            test_df.loc[(test_df['price']==0) & (test_df['fare'] == 'Adulto ida'), 'price'] = test_df[test_df['fare'] == 'Adulto ida']['price'].median()
            test_df.loc[(test_df['price']==0) & (test_df['fare'] == 'Flexible'), 'price'] = test_df[test_df['fare'] == 'Flexible']['price'].median()
            test_df.loc[(test_df['price']==0) & (test_df['fare'] == 'Promo +'), 'price'] = test_df[test_df['fare'] == 'Promo +']['price'].median()
            test_df.loc[(test_df['price']==0) & (test_df['fare'] == 'Mesa '), 'price'] = test_df[test_df['fare'] == 'Mesa']['price'].median()

            # Dropping the remaining NULL values in test_data.
            test_df.dropna(inplace=True)

            # Applying train-test split.
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f'Applying preprocessing object on training dataframe and testing dataframe')

            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_object.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, target_feature_train_df
            ]

            test_arr = np.c_[
                input_feature_test_arr, target_feature_test_df
            ]

            save_object(
                file_path = self.data_validation_config.preprocessor_object_file_path,
                obj = preprocessing_object
            )

            logging.info(f"Saved Preprocessing Object!")

            return train_arr, test_arr, self.data_validation_config.preprocessor_object_file_path

        except Exception as e:
            raise CustomException(e, sys)