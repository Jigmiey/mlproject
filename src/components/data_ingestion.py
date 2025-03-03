from dataclasses import dataclass
import os
import sys
import pandas as pd
from src.logger import logging
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.components.data_transformation import Data_transformation
import numpy as np

@dataclass
class Data_ingestion_config:
    train_data_path : str = os.path.join("artifacts","train_data.csv")
    test_data_path : str = os.path.join("artifacts","test_data.csv")
    raw_data_path : str = os.path.join("artifacts","raw_data.csv")

class Data_ingestion:
    def __init__(self):
        self.data_ingestion_config = Data_ingestion_config()
    def data_ingestion(self):
        try:
            logging.info("Data Ingestion Initiated")
            df = pd.read_csv("notebook\data\stud.csv")
            train_data, test_data = train_test_split(df,test_size=0.2,random_state=42)
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            train_data.to_csv(self.data_ingestion_config.train_data_path)
            test_data.to_csv(self.data_ingestion_config.test_data_path)
            df.to_csv(self.data_ingestion_config.raw_data_path)
            logging.info("Data ingestion Completed")
            return (self.data_ingestion_config.train_data_path, self.data_ingestion_config.test_data_path)
            
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    data_ingestion= Data_ingestion()
    train_path, test_path = data_ingestion.data_ingestion()
    data_transformer = Data_transformation()
    train_arr,test_arr, pre_processor_object_path=data_transformer.initiate_data_transformation(train_path,test_path)
    print(f"Shape of train Array: {train_arr.shape}")
    print(f"Shape of test Array: {test_arr.shape}")
        
    
    