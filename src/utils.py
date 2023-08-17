import os
import sys
import dill

from sklearn.metrics import (classification_report
                             ,precision_score,roc_auc_score)
from sklearn.model_selection import GridSearchCV

from src.path_file import preprocessor_obj_file_path
from src.exception import CustomException


def save_object(file_path,obj):

    try:
        dir_path = os.path.dirname(preprocessor_obj_file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)



def evaluate_models(X_train,y_train,X_test,y_test,models,params):

    try:

        precision_report={}
        auc_score_report={}

        for model_name, model in models.items():

            para = params[model_name]

            gs= GridSearchCV(model,cv=10,param_grid=para,n_jobs=-1)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)
            y_pred_on_train = model.predict(X_train)   # predicted value from training data
            y_pred_on_test =  model.predict(X_test)     # predicted value from test data
            y_pred_prob_test =model.predict_proba(X_test)[:,1]     # predicted probability fromm test data



            testing_report = classification_report(y_test, y_pred_on_test, output_dict=True)
            # df_classification_report_test = pd.DataFrame(testing_report).transpose()
            auc_score_test = roc_auc_score(y_test, y_pred_prob_test)

            precision_report[model_name]={
                                    'zero':testing_report['0.0']['precision']
                                    ,'one':testing_report['1.0']['precision']}
            auc_score_report[model_name] = auc_score_test
        # print(precision_report)
        # print(auc_score_report)

        return auc_score_report,precision_report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):

    try:
        with open(file_path ,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)



