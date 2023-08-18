# importing general libraries

import sys


# importing machine learning algorithms
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# importing project libraries
import src.logger as logging
from src.exception import CustomException
from src.utils import save_object,evaluate_models


from sklearn.metrics import precision_score
from src.path_file import best_model_obj_file_path
class ModelTrainer:
    """
    This class shall  be used to train the model

    """

    def __init__(self,file_object):      # file_object is text file to save logs
        # self.preprocessor_config=PreprocessorConfig()
        self.file_object = file_object

    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):

        logging.log(self.file_object,
                    'Entered the initiate_model_trainer method of the ModelTrainer class')
        try:
            logging.log(self.file_object,
                        'splitting train array and test array in to X_train,y_train X_test,y_test')

            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            logging.log(self.file_object,
                        'splitting is successful')

            # creating dictionary of models
            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Catboost":CatBoostClassifier(),
                "XG Boost": XGBClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Adaboost": AdaBoostClassifier(),
                "Support Vector Classification": SVC(probability=True),
                "K Nearest Neighbor": KNeighborsClassifier()
            }

            params = {
                "Logistic Regression": {"penalty": ['l1', 'l2', 'elasticnet', 'None'],
                                        "C": [1, 2, 3, 4],
                                        "solver": ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
                                        "max_iter": [30, 50, 100, 200, 500]
                                        },
                "Decision Tree": {"criterion": ['gini', 'entropy', 'log_loss'],
                                  "max_depth": [3, 5, 8,16],
                                  "random_state": [42],
                                  "max_leaf_nodes": [8, 10, 20]
                                  },
                "Random Forest": {"n_estimators": [8, 16, 32, 64, 128, 256],
                                  "criterion": ['gini', 'entropy', 'log_loss'],
                                  "max_depth": [3, 5, 8, 16,32],
                                  "max_leaf_nodes": [5, 8,10,15,20]
                                  },
                "Catboost": {"depth": [8, 10, 20],
                             "iterations": [10,20,30,40, 50,60,70,80,90, 100],
                             "learning_rate": [0.01, 0.05, 0.1,0.001]
                             },
                "XG Boost": {"n_estimators": [8, 16, 32, 64, 128, 256],
                             "learning_rate": [0.01, 0.05, 0.1, 0.001],
                             "max_depth": [3, 5, 8, 16]
                             },
                "Gradient Boosting": {"loss": ['log_loss', 'exponential'],
                                      "subsample": [0.6, 0.7, 0.75, 0.85, 0.9, 1.0],
                                      "max_leaf_nodes": [8, 10, 20],
                                      "n_estimators": [8, 16, 32, 64, 128, 256],
                                      "learning_rate": [0.01, 0.05, 0.1, 0.001]
                                      },

                "Adaboost": {"n_estimators": [8, 16, 32, 64, 128, 256],
                             "learning_rate": [0.01, 0.05, 0.1, 0.001],
                             },
                "Support Vector Classification": {'max_iter': [30, 50, 100, 200, 500],
                                                  'C': [1, 2, 3, 4],
                                                  "kernel": ['linear', 'poly', 'rbf', 'sigmoid']
                                                  },
                "K Nearest Neighbor": {"n_neighbors": [5, 4, 3, 7, 8],
                                       }

            }


            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,
                                             X_test=X_test,y_test=y_test,models=models,params=params)

            # to get best model score
            best_model_auc_score = max(sorted(model_report[0].values()))

            # To get best model name from dict
            best_model_name = list(model_report[0].keys())[
                list(model_report[0].values()).index(best_model_auc_score)
            ]

            best_model =models[best_model_name]
            best_model_precision_score = model_report[1][best_model_name]

            logging.log(self.file_object,
                        f'best model has been chosen which is {best_model_name} with auc score : {best_model_auc_score} and precision score {best_model_precision_score}')


            logging.log(self.file_object,
                        'model training is succcessful of model_trainer method, exiting the ModelTrainer Class')

            save_object(  # saving preprocessor pickle file
                best_model_obj_file_path, obj=best_model
            )


            predicted =best_model.predict(X_test)

            precision = precision_score(y_test,predicted)

            return precision



        except Exception as e:

            logging.log(self.file_object,
                        'Exception occured in model_trainer method of the ModelTrainer class. Exception message: ' + str(
                            e))
            logging.log(self.file_object,
                        'model training is failed of model_trainer method, exiting the ModelTrainer Class')

            raise CustomException(e,sys)



