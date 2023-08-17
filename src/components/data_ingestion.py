# importing libraries
import os
import sys
import pandas as pd

# importing custom packages
import src.logger as logging
from src.exception import CustomException
from src.path_file import training_file,label_file


# saving file_name with path in variables for reading further

class Data_Getter:
    """
        This class shall  be used for obtaining the data from the source for training
    """

    def __init__(self,file_object):    # file_object is text file to save logs
        self.file_object = file_object


    def get_data(self):
        """
        Method Name: get_data
        Description: This method reads the data from source.
        Output: A pandas DataFrame.
        On Failure: Raise Exception

        """

        logging.log(self.file_object, 'Entered the get_data method of the Data_Getter class')
        try:
            X_data = pd.read_csv(training_file) # reading the training data file
            label_data = pd.read_csv(label_file)  # reading the label data file
            logging.log(self.file_object,'Data Load Successful.Exited the get_data method of the Data_Getter class')
            return X_data ,label_data
        except Exception as e:
            logging.log(self.file_object,'Exception occured in get_data method of the Data_Getter class. Exception message: '+str(e))
            logging.log(self.file_object,
                                   'Data Load Unsuccessful.Exited the get_data method of the Data_Getter class')
            raise CustomException(e,sys)

    def merge_data(self,X_data,label_data):
        """
        Method Name: merge_data
        Description: This method merge label data and independent data
        Output: A pandas DataFrame.
        On Failure: Raise Exception

        """
        logging.log(self.file_object, 'Entered the merge_data method of the Data_Getter class')
        try:
            data = X_data.merge(label_data,on='Ind_ID') #merging the training data and label data
            logging.log(self.file_object,
                               'Data merged Successful.Exited the merge_data method of the Data_Getter class')
            return data
        except Exception as e:
            logging.log(self.file_object,
                                   'Exception occured in merge_data method of the Data_Getter class. Exception message: ' + str(
                                       e))
            logging.log(self.file_object,
                                   'Data merged Unsuccessful.Exited the merge_data method of the Data_Getter class')
            raise CustomException(e,sys)
