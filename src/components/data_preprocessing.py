
# importing general libraries
import os
import sys
import pandas as pd
import numpy as np

# importing needed libraries
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest,f_classif

# importing custom packages
import src.logger as logging
from src.exception import CustomException
from src.utils import save_object
from src.path_file import missing_column_path,name_changed_path
from src.path_file import feature_extracted_data_path,preprocessor_obj_file_path
from src.path_file import test_arr_path,train_arr_path
from src.path_file import training_data_path,testing_data_path



class Preprocessor:
    """
    This class shall  be used to clean and transform the data before training

    """

    def __init__(self,file_object):      # file_object is text file to save logs
        # self.preprocessor_config=PreprocessorConfig()
        self.file_object = file_object

    def column_name_correction(self,data):
        """
        Method Name: column_name_correction
        Description: This method rename columns of pandas Dataframe in proper format
        Output: Returns data with new column names
        On Failure: Raise Exception

        """
        logging.log(self.file_object,
                    'Entered the column_name_correction method of the Preprocessor class')

        try:
            data.rename(columns={'Ind_ID': 'Ind_Id', 'GENDER': 'Gender','Propert_Owner': 'Property_Owner'
                ,'CHILDREN': 'Children', 'Annual_income': 'Annual_Income', 'EDUCATION': 'Education',
                                 'Marital_status': 'Marital_Status', 'Housing_type': 'Housing_Type'
                                , 'Birthday_count': 'Birthday_Count', 'Employed_days': 'Employed_Days'
                                , 'Mobile_phone': 'Mobile_Phone'
                                , 'EMAIL_ID': 'Email_Id', 'label': 'Label'}, inplace=True)
            logging.log(self.file_object,
                        'Renaming column is successful. Exited the column_name_correction method of the Preprocessor class')
            return data

        except Exception as e:
            logging.log(self.file_object,
                        'Exception occured in column_name_correction method of the Preprocessor class. Exception message: '+str(e))
            logging.log(self.file_object,
                        'Renaming column is unsuccessful.Exited the column_name_correction method of the Preprocessor class')
            raise CustomException(e,sys)

    def renaming_values(self,x:str): # taking values of series and converting names
        """
        Method Name: renaming_values
        Description: This method correct names in categorical columns.
        Output: A Dataframe which has all the correct names
        On Failure: Raise Exception

        """
        if (x == 'House / apartment') | (x == 'Single / not married') | (x == 'Secondary / secondary special'):
            x = x.split(' /')[0].strip()
        return x

    def value_name_change(self,data):

        """
        implementation of renaming_values function method

        """

        logging.log(self.file_object,
                    'Entered the value_name_change method of the Preprocessor class')
        try:
            # apply function works only on series not dataframe
            data_2=data.copy(deep=True)  # creating dataframe copy
            data_2['Housing_Type'] = data_2['Housing_Type'].apply(self.renaming_values)
            data_2['Marital_Status'] = data_2['Marital_Status'].apply(self.renaming_values)
            data_2['Education'] = data_2['Education'].apply(self.renaming_values)

            os.makedirs(os.path.dirname(name_changed_path),
                        exist_ok=True)  # making directory artifacts if not exist
            data_2.to_csv(name_changed_path, index=False,
                                       header=True)  # storing the changed name columns to csv file
            logging.log(self.file_object,
                    'value_name_change is successful.Data written to the name_changed_data.csv file. Exited the value_name_change method of the Preprocessor class')
            return data_2

        except Exception as e:
            logging.log(self.file_object,
                        'Exception occured in value_name_change method of the Preprocessor class. Exception message:  ' + str(
                            e))
            logging.log(self.file_object,
                        'value_name_change method is failed. Exited the value_name_change method of the Preprocessor class')
            raise CustomException(e, sys)


    def is_null_present(self,data):
        """
        Method Name: is_null_present
        Description: This method checks whether there are null values present in the pandas Dataframe or not.
        Output: returns the list of columns for which null values are present.
        On Failure: Raise Exception

        """
        logging.log(self.file_object,
                    'Entered the is_null_present method of the Preprocessor class')
        null_present = False
        cols_with_missing_values=[]  #empty list
        cols = data.columns          #column names in dataset

        try:
            null_counts=data.isnull().sum() # check for the count of null values per column
            for i in range(len(null_counts)):
                if null_counts[i]>0:
                    null_present=True
                    cols_with_missing_values.append(cols[i])

            if(null_present): # write the logs to see which columns have null values
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null['columns'] = data.columns
                dataframe_with_null['missing values count'] = np.asarray(data.isnull().sum())

                os.makedirs(os.path.dirname(missing_column_path),exist_ok=True)  #making directory artifacts if not exist
                dataframe_with_null.to_csv(missing_column_path,index=False,header=True) # storing the null column information to csv file
            logging.log(self.file_object,
                        'Finding missing values is successful.Data written to the missing_data_column.csv file. Exited the is_null_present method of the Preprocessor class')
            return cols_with_missing_values

        except Exception as e:
            logging.log(self.file_object,'Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(e))
            logging.log(self.file_object,'Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise CustomException(e,sys)


    def feature_extraction(self, data):
        """
        Method Name: feature_extraction
        Description: This method creates new features from the given data.
        Output: A Dataframe which has new features
        On Failure: Raise Exception

        """
        logging.log(self.file_object,
                    'Entered the feature_extraction method of the Preprocessor class')

        try:
            data['Year_Of_Experience'] = np.where(data['Employed_Days']<0,-round(data['Employed_Days']/365,1),0)
            data['Age'] = -round(data['Birthday_Count'] / 365, 0)
            data= data.drop(columns=['Ind_Id', 'Birthday_Count',
                                            'Employed_Days'],axis=1) # dropping irrelevant columns


            os.makedirs(os.path.dirname(feature_extracted_data_path), exist_ok=True)
            data.to_csv(feature_extracted_data_path, index=False,
                                header=True)  # storing the feature extracted data information to file
            logging.log(self.file_object,
                        'feature extraction is successful.Data written to the feature_extracted_data.csv file. Exited the feature_extraction method of the Preprocessor class')
            return data

        except Exception as e:
            logging.log(self.file_object,
                        'Exception occured in feature_extraction method of the Preprocessor class. Exception message:  ' + str(
                            e))
            logging.log(self.file_object,
                        'feature_extraction method is failed. Exited the feature_extraction method of the Preprocessor class')
            raise CustomException(e, sys)

    def split_data(self, data):
        """
        Method Name: split_data
        Description: This method split data in to training and test data
        Output: A pandas DataFrame.
        On Failure: Raise Exception

        """
        logging.log(self.file_object, 'Entered the split_data method of the Data_Getter class')
        try:
            train_set, test_set = train_test_split(data, test_size=0.33, random_state=42)
            os.makedirs(os.path.dirname(training_data_path), exist_ok=True)
            train_set.to_csv(training_data_path, index=False,
                             header=True)  # storing the training data set to train_set csv file
            os.makedirs(os.path.dirname(testing_data_path), exist_ok=True)
            test_set.to_csv(testing_data_path, index=False,
                            header=True)  # storing the testing data set to test_set csv file

            logging.log(self.file_object,
                        'Splitting of data is Successful.Exited the split_data method of the Data_Getter class')
            return train_set, test_set
        except Exception as e:
            logging.log(self.file_object,
                        'Exception occured in split_data method of the Data_Getter class. Exception message: ' + str(
                            e))
            logging.log(self.file_object,
                        'Splitting of data is Unsuccessful.Exited the merge_data method of the Data_Getter class')
            raise CustomException(e, sys)

    def get_data_transformer_object(self):

        """
        Method Name: get_data_transformer_object
        Description: This method is responsible for data transformation
        Output: An preprocessor object
        On Failure: Raise Exception

        """

        logging.log(self.file_object,
                    'Entered the get_data_transformer_object method of the Preprocessor class')

        try:

            num_col= ['Children','Mobile_Phone','Work_Phone',
           'Phone','Email_Id','Family_Members','Year_Of_Experience','Age']

            ann = [['Annual_Income']]

            ty_occu = [['Type_Occupation']]

            edu = [['Education']]

            cat_col = ['Gender','Car_Owner','Property_Owner',
                                'Type_Income','Marital_Status','Housing_Type']

            num_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                 ('scaling', StandardScaler(with_mean=False))])

            Ann_income_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                                        ('scaling', StandardScaler(with_mean=False))])

            Ty_Occup_pipe = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Other')),
                                      ('OHE', OneHotEncoder(drop='first', handle_unknown='ignore')),
                                      ('scaling', StandardScaler(with_mean=False))])

            edu_pipe = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Other')),
                                 ('ord_encoder', OrdinalEncoder(categories=[
                                     ['Lower secondary', 'Secondary', 'Incomplete higher', 'Higher education',
                                      'Academic degree']])),
                                 ('scaling', StandardScaler(with_mean=False))])

            cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                 ('OHE', OneHotEncoder(drop='first', handle_unknown='ignore')),
                                 ('scaling', StandardScaler(with_mean=False))])

            logging.log(self.file_object,
                        'preprocessor transformer initiating of get_data_transformer_object method in Preprocessor class')

            preprocessor = ColumnTransformer(transformers=[
                                    ("Ann_pipe", Ann_income_pipe, ann[0]),
                                    ("Ty_occu_pipe", Ty_Occup_pipe, ty_occu[0]),
                                    ("edu_pipe", edu_pipe, edu[0]),
                                    ("num_pipe", num_pipe, num_col),
                                    ("cat_pipe", cat_pipe, cat_col)])

            # feature_selection = SelectKBest()   # selecting features using SelectKbest
            #
            # feature_selection_transfomer = ColumnTransformer(transformers=[
            #         ("feature_selection_transfomer",feature_selection,slice(0,43))])  # Feature selection using selectKbest
            #
            # preprocessor = Pipeline([('transformer_1',transformer_1),
            #                         ('feature_selection_transfomer',feature_selection_transfomer)
            #                               ]
            #                              )

            return preprocessor

        except Exception as e:
            logging.log(self.file_object,
                        'Exception occured in get_data_transformer_object method of the Preprocessor class. Exception message:  ' + str(
                            e))
            logging.log(self.file_object,
                        'get_data_transformer_object method is failed. Exited the get_data_transformer_object method of the Preprocessor class')
            raise CustomException(e, sys)


    def initiate_data_transformation(self,train_path,test_path):

        logging.log(self.file_object,
                    'Entered the initiate_data_transformation method of the Preprocessor class')
        try:
            logging.log(self.file_object,
                        'reading training and testing data of initiate_data_transformation method in Preprocessor class')

            train_df=pd.read_csv(train_path)
            test_df =pd.read_csv(test_path)

            logging.log(self.file_object,
                        'reading training and testing data is successful of initiate_data_transformation method in Preprocessor class')
            logging.log(self.file_object,
                        'obtaining preprocessor object from get_data_transformer_object method')

            preprocessing_obj=self.get_data_transformer_object()   # calling the get_data_transformer method output as preprocessor object

            logging.log(self.file_object,
                        'obtaining preprocessor object successful from get_data_transformer_object method')

            target_column="Label"
            input_feature_train_df = train_df.drop(                          # defining input trainig data
                                            columns=[target_column],
                                            axis=1)
            target_feature_train_df = train_df[target_column]        # target column defining for training

            input_feature_test_df = test_df.drop(
                            columns=[target_column],axis=1)       # defining input testing data


            target_feature_test_df = test_df[target_column]         # target column defining for testing

            logging.log(self.file_object,
                    'Applying preprocessor object on training dataframe and testing dataframe in initiate_data_transformation method of Preprocessor Class')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)

            logging.log(self.file_object,
                        'Applying preprocessor object on training dataframe successful')

            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.log(self.file_object,
                        'Applying preprocessor object on testing dataframe successful')

            train_arr =np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            train_dataframe=pd.DataFrame(input_feature_test_arr )
            test_dataframe=pd.DataFrame(input_feature_test_arr)
            os.makedirs(os.path.dirname(train_arr_path), exist_ok=True)
            train_dataframe.to_csv(train_arr_path, index=False,
                            header=True)  # storing the testing data set to test_set csv file
            os.makedirs(os.path.dirname(test_arr_path), exist_ok=True)
            test_dataframe.to_csv(test_arr_path, index=False,
                             header=True)  # storing the testing data set to test_set csv file

            save_object(                                             # saving preprocessor pickle file
                preprocessor_obj_file_path,obj = preprocessing_obj
            )

            logging.log(self.file_object,
                        '"Saved preprocessing pickle file" of initiate_data_transformation method in Preprocessor Class')

            return (
                train_arr,test_arr,preprocessing_obj
            )
        except Exception as e:

            logging.log(self.file_object,
                        'Exception occured in initiate_data_transformation method of the Preprocessor class. Exception message:  ' + str(
                            e))
            logging.log(self.file_object,
                        'initiate_data_transformation method is failed. Exited the initiate_data_transformation method of the Preprocessor class')
            raise CustomException(e, sys)



