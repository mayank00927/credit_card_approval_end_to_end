import sys
import os
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.utils import load_object
from src.path_file import preprocessor_obj_file_path,best_model_obj_file_path

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):

        try:

            model_path=best_model_obj_file_path
            preprocessor_path=preprocessor_obj_file_path
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
                 Gender:str,
                 Car_Owner: str,
                 Property_Owner: str,
                 Children:int,
                 Annual_Income:int,
                 Type_Income :str,
                 Education:str,
                 Marital_Status:str,
                 Housing_Type:str,
                 Mobile_Phone:int,
                 Work_Phone:int,
                 Phone:int,
                 Email_Id:int,
                 Type_Occupation:str,
                 Family_Members:int,
                 Year_Of_Experience:int,
                 Age:int):

        self.Gender= Gender
        self.Car_Owner= Car_Owner
        self.Property_Owner= Property_Owner
        self.Children=Children
        self.Annual_Income=Annual_Income
        self.Type_Income=Type_Income
        self.Education=Education
        self.Marital_Status=Marital_Status
        self.Housing_Type=Housing_Type
        self.Mobile_Phone=Mobile_Phone
        self.Work_Phone=Work_Phone
        self.Phone=Phone
        self.Email_Id= Email_Id
        self.Type_Occupation=Type_Occupation
        self.Family_Members=Family_Members
        self.Year_Of_Experience=Year_Of_Experience
        self.Age=Age

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                'Gender':[self.Gender],
                'Car_Owner' :[self.Car_Owner],
                'Property_Owner':[self.Property_Owner],
                'Children':[self.Children],
                'Annual_Income':[self.Annual_Income],
                'Type_Income':[self.Type_Income],
                'Education':[self.Education],
                'Marital_Status':[self.Marital_Status],
                'Housing_Type':[self.Housing_Type],
                'Mobile_Phone':[self.Mobile_Phone],
                'Work_Phone':[self.Work_Phone],
                'Phone':[self.Phone],
                'Email_Id':[self.Email_Id],
                'Type_Occupation':[self.Type_Occupation],
                'Family_Members':[self.Family_Members],
                'Year_Of_Experience':[self.Year_Of_Experience],
                'Age':[self.Age]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e,sys)
