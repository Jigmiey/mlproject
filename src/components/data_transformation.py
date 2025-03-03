import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np
from src.utils import save_object

@dataclass
class Data_transformation_config:
    preprocessor_obj_file_path : str = os.path.join("artifacts","processor.pkl")

class Data_transformation:
    def __init__(self):
        self.data_transformation_config = Data_transformation_config()
    def get_data_transformer_object(self):
        '''
        This function is responsible for Data Transformation
        '''
        try:
            num_features = ["writing_score","reading_score"]
            cat_features = ["gender","race_ethnicity","parental_level_of_education",
                            "lunch","test_preparation_course"]
            num_pipeline = Pipeline(
                steps=[("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())]
                
            )
            cat_pipeline = Pipeline(
                steps=[("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("Scaler",StandardScaler(with_mean=False))]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_features),
                    ("cat_pipeline",cat_pipeline,cat_features)       
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
        
    def initiate_data_transformation(self,train_path,test_path):
        logging.info("Data Transformation Started")
        try:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            logging.info("Read train and test data successfully")
            X_train = df_train.drop(columns=['math_score'],axis=1)
            y_train = df_train['math_score']
            X_test = df_test.drop(columns=['math_score'],axis=1)
            y_test = df_test['math_score']
            logging.info("Obtaining Pre processing object")
            processor = self.get_data_transformer_object()
            X_train = processor.fit_transform(X_train)
            X_test = processor.fit_transform(X_test)
            train_arr = np.c_[X_train,np.array(y_train)]
            test_arr = np.c_[X_test, np.array(y_test)]
            logging.info("Saving Pre processing object")
            save_object(self.data_transformation_config.preprocessor_obj_file_path,processor)
            logging.info("Preprocessing Object Saved")
            return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)
            
        except Exception as e :
            raise CustomException(e,sys)
            
        
        
        