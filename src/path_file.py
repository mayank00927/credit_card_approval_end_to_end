import os

base_path =os.getcwd()   # base path

log_path = base_path + "/log_files/"   # path for log files to save

training_file = base_path + "/notebook/data/Credit_card.csv"     # training file path
label_file = base_path + "/notebook/data/Credit_card_label.csv"  # label file path

# after splitting training data path
training_data_path:str=os.path.join('artifacts',"train_set.csv")

# after splitting test data path
testing_data_path:str=os.path.join('artifacts',"test_set.csv")

# after applying preprocessor object train arrray path
train_arr_path:str=os.path.join('artifacts',"train_arr.csv")

# after applying preprocessor object test arrray path
test_arr_path:str=os.path.join('artifacts',"test_arr.csv")


# missing columns data path
missing_column_path:str=os.path.join('artifacts',"missing_data_column.csv")

# after doing feature extraction data path
feature_extracted_data_path: str = os.path.join('artifacts', "feature_extracted_data.csv")

# imputed_data_path: str = os.path.join('artifacts', "imputed_data.csv")

# after performing name changing -data path
name_changed_path: str = os.path.join('artifacts', "name_changed_data.csv")

# preprocessor object file path
preprocessor_obj_file_path= os.path.join('artifacts', "preprocessor.pkl")

# best model path after training and hyper-tuning
best_model_obj_file_path= os.path.join('artifacts', "best_model.pkl")