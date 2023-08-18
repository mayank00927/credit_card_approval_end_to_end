#importing cutom packages
from src.components.data_ingestion import Data_Getter as DG
from src.components.data_preprocessing import Preprocessor
from src.path_file import training_data_path,testing_data_path
from src.components.model_trainer import ModelTrainer
import warnings
warnings.filterwarnings('ignore')


dg=DG('training_logs.txt')
X_data,label= dg.get_data()
data  = dg.merge_data( X_data,label)


pre= Preprocessor("training_logs.txt")
data_2= pre.column_name_correction(data)
data_3=pre.value_name_change(data_2)
data_4=pre.feature_extraction(data_3)
pre.split_data(data_4)
train,test,processor=pre.initiate_data_transformation(training_data_path,testing_data_path)
MT = ModelTrainer("training_logs.txt")
MT.initiate_model_trainer(train, test,processor)
