import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', "model.pkl")

class Modeltrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params = {
                        "Random Forest": {
                            "n_estimators": [50, 100, 200],
                            "max_depth": [None, 10, 20, 30],
                            "min_samples_split": [2, 5, 10],
                            "min_samples_leaf": [1, 2, 4]
                        },
                        "Decision Tree": {
                            "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                            "max_depth": [None, 10, 20, 30],
                            "min_samples_split": [2, 5, 10],
                            "min_samples_leaf": [1, 2, 4]
                        },
                        "Gradient Boosting": {
                            "n_estimators": [50, 100, 200],
                            "learning_rate": [0.01, 0.1, 0.2],
                            "max_depth": [3, 5, 10],
                            "subsample": [0.8, 1.0]
                        },
                        "Linear Regression": {},  # No hyperparameters to tune
                        "K-Neighbors Regressor": {
                            "n_neighbors": [3, 5, 7, 10],
                            "weights": ["uniform", "distance"],
                            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
                        },
                        "XGB Regressor": {
                            "n_estimators": [50, 100, 200],
                            "learning_rate": [0.01, 0.1, 0.2],
                            "max_depth": [3, 5, 10],
                            "subsample": [0.8, 1.0],
                            "colsample_bytree": [0.8, 1.0]
                        },
                        "CatBoosting Regressor": {
                            "iterations": [50, 100, 200],
                            "depth": [4, 6, 10],
                            "learning_rate": [0.01, 0.1, 0.2],
                            "l2_leaf_reg": [1, 3, 5, 7]
                        },
                        "AdaBoost Regressor": {
                            "n_estimators": [50, 100, 200],
                            "learning_rate": [0.01, 0.1, 0.2],
                            "loss": ["linear", "square", "exponential"]
                        }
                    }

            
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models,param=params
            )
            
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException(f"No suitable model found with score {best_model_score:.2f}")
            
            logging.info(f"Best model found: {best_model_name} with score {best_model_score:.4f}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )  
            
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            
            return r2_square
        except Exception as e:
            raise CustomException(str(e), sys)
