<<<<<<< HEAD
import os, sys
from dataclasses import dataclass

from src.exception import CustomException
from sklearn.metrics import r2_score

@dataclass
class ModelEvaluation:
    def evaluate_model(X_train, X_test, y_train, y_test, models):
        try:
            report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                test_model_score = r2_score(y_test, y_pred)

                report[list(models.keys())[i]] = test_model_score
            
            return report

        except Exception as e:
            raise CustomException(e, sys)
=======
import os, sys
from dataclasses import dataclass

from src.exception import CustomException
from sklearn.metrics import r2_score

@dataclass
class ModelEvaluation:
    def evaluate_model(X_train, X_test, y_train, y_test, models):
        try:
            report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                test_model_score = r2_score(y_test, y_pred)

                report[list(models.keys())[i]] = test_model_score
            
            return report

        except Exception as e:
            raise CustomException(e, sys)
>>>>>>> 23afb2523849890ff1628e1eb43f70afaee7f3b6
