import os
import dill
from src.exception import CustomException
import sys
def save_object(file_path,object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as f:
            dill.dump(object,f)
    except Exception as e:
        raise CustomException(e,sys)
        
    
        
    