import os
import dill
from src.exception import CustomException
import sys
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
def save_object(file_path,object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as f:
            dill.dump(object,f)
    except Exception as e:
        raise CustomException(e,sys)
def evaluate_model(X_train, y_trian,X_test,y_test,models,params):
    model_list = list(models.values())
    try:
        result = {}
        for i in range(len(model_list)):
            model = model_list[i]
            param = params[list(models.keys())[i]]
            gs = GridSearchCV(model,param,cv=3)
            gs.fit(X_train,y_trian)
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_trian)
            score = r2_score(model.predict(X_test),y_test)
            result[list(models.keys())[i]]=[model,score]
        return result
    except Exception as e:
        raise CustomException(e,sys)
def load_object(file_path):
    with open(file_path,"rb") as file:
        return dill.load(file,file_path)
        
        
        
        
    
        
    