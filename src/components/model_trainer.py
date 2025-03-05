import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from src.utils import save_object,evaluate_model

@dataclass
class Model_training_config:
    model_path : str = os.path.join("artifacts","model.pkl")

class Model_trainer:
    def __init__(self):
        self.model_training_config = Model_training_config()
    def initiate_training(self,train_data,test_data):
        try:
            logging.info("Read train and test data in Model Trainer")
            X_train , y_train, X_test, y_test = (train_data[:,:-1],train_data[:,-1],test_data[:,:-1],test_data[:,-1])
            models = {
                        "Linear Regression": LinearRegression(),
                        "Decision Tree": DecisionTreeRegressor(),
                        "Gradient Boosting": GradientBoostingRegressor(),
                        "Random Forest": RandomForestRegressor(),
                        "XGBRegressor": XGBRegressor(), 
                        "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                        "AdaBoost Regressor": AdaBoostRegressor()
                    }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            result = evaluate_model(X_train,y_train,X_test,y_test,models,params)
            #best_model_score = sorted(list(result.values()))[-1]
            #best_model_name = list(result.keys())[list(result.values()).index(best_model_score)]
            #best_model = models[best_model_name]
            value_list = list(result.values())
            best_model_score = 0.5
            for i in range(len(value_list)):
                if value_list[i][1] > best_model_score:
                    best_model_score = value_list[i][1]
                    best_model = value_list[i][0]
                    best_model_name = list(result.keys())[i]   
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found : {best_model_name}")
            logging.info("Saving the best model")
            save_object(self.model_training_config.model_path,best_model)
            
            score = r2_score(y_test,best_model.predict(X_test))
            return score
        except Exception as e:
            raise CustomException(e,sys)
        
        
    