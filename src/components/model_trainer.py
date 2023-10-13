<<<<<<< HEAD
import os, sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from src.utils import save_object
from src.logger import logging
from src.exception import CustomException

from src.components.model_evaluation import ModelEvaluation

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.trained_model_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting train-test input data")

            X_train, X_test, y_train, y_test = (
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1],
            )

            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor()
            }

            algorithm = ModelEvaluation
            model_report = dict = algorithm.evaluate_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, models=models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            logging.info(f"Model for Training and Testing dataset")

            save_object(
                file_path = self.trained_model_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            score = r2_score(y_test, predicted)

            logging.info(f"Best model for prediction")

            return score

        except Exception as e:
            raise CustomException(e, sys)
=======
import os, sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from src.utils import save_object
from src.logger import logging
from src.exception import CustomException

from src.components.model_evaluation import ModelEvaluation

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.trained_model_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting train-test input data")

            X_train, X_test, y_train, y_test = (
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1],
            )

            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor()
            }

            algorithm = ModelEvaluation
            model_report = dict = algorithm.evaluate_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, models=models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            logging.info(f"Model for Training and Testing dataset")

            save_object(
                file_path = self.trained_model_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            score = r2_score(y_test, predicted)

            logging.info(f"Best model for prediction")

            return score

        except Exception as e:
            raise CustomException(e, sys)
>>>>>>> 23afb2523849890ff1628e1eb43f70afaee7f3b6
