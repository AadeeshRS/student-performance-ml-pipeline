import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerconfig:
    trained_model_file_path=os.path.join('artifact','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerconfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )
            models = {
            "Linear Regression": LinearRegression(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "XGBoost Regressor":XGBRegressor(),
            "AdaBoost Regressor":AdaBoostRegressor(),
            "GradientBoost Regressor":GradientBoostingRegressor()
            }


            params = {
            "Linear Regression": {},

            "Decision Tree": {
                "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
            },

            "Random Forest Regressor": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
            },

            "GradientBoost Regressor": {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5],
            },

            "XGBoost Regressor": {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
            },

            "AdaBoost Regressor": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1, 1.0],
            },

            "K-Neighbors Regressor": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "p": [1, 2],
            },
        }


            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
            
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best Model found",sys)
            logging.info("Best model found")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            r2=r2_score(y_test,predicted)
            return {
                "best_model_name": best_model_name,
                "r2_score": r2
                    }
        except Exception as e:
            raise CustomException(e,sys)