import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
from src.components.data_ingestion import Data_ingestion_config, Data_ingestion
from src.components.data_transformation import Data_transformation_config
from src.components.model_trainer import Model_training_config
import pandas as pd
import numpy as np

class Inference:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model = load_object(Model_training_config.model_path)
            return model.predict(features)
        except Exception as e:
            raise CustomException(e,sys)
    def Custom_data(self,data):
        try:
            logging.info("Data receuved from the client !")
            column = ['gender', 'race_ethnicity', 'parental_level_of_education',
            'lunch', 'test_preparation_course', 'reading_score',
            'writing_score']
            data =[data] # getiing a 1d array and converting them into 2d array jus like pandas wants
            df = pd.DataFrame(data,columns=column)
            logging.info("Transformer object is being loaded") 
            processor = load_object(Data_transformation_config.preprocessor_obj_file_path)
            logging.info("Transformer object loaded succesfully")
            features = processor.transform(df)
            return features
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    data_ingestion = Data_ingestion()
    train_path, test_path=data_ingestion.data_ingestion()
    train_data = pd.read_csv(train_path)
    print(train_data.columns)
    processor = load_object(Data_transformation_config.preprocessor_obj_file_path)
    train_transformed_data = processor.transform(train_data)
    print(np.array([train_transformed_data[0]]))
    infer = Inference()
    data = train_data.loc[0,['gender', 'race_ethnicity', 'parental_level_of_education',
          'lunch', 'test_preparation_course', 'reading_score',
          'writing_score']].to_numpy()
    print(data)
    features = infer.Custom_data(data)
    print(infer.predict(features))
    