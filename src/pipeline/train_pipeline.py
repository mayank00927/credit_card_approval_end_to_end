from application_logging import logger

class trainModel:

    def __init__(self):
        self.log_writer = logger.App_logger()
        self.file_object = open("Training_Logs/ModelTrainingLog.Txt",'a+')

    def trainingModel(self):
        # Logging the start of Training
        self.log_writer.log(self.file_object,'Start of Training')
        try:
            # Getting the data from the source
            data_getter=data_loader.Data_Getter(self.file_object,self.log_writer)
            data=data_getter.get_data()


            """doing  the data preprocessing"""

            preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)

            # create separate features and labels
            X,Y=preprocessor.separate_label_feature(data,label_column_name=default payment)

            #check if missing values are present in data
            is_null_present,cols_with_missing_values=preprocessor.is_null_present(X)

            # if missing values are there ,replace them appropriately.
            if(is_null_present):
                X=preprocessor.impute_missing_values(X,cols_with_missing_values)

            """ Applying the clustering approach """
            kmeans=KMeansClustering(self.file_object,self.log_writer) #object
            number_of_clusters=kmeans.elbow_plot(X)

            # Divide the data into clusters
            X=kmeans.create_clusters(X,number_of_clusters)

            #create a new column in the dataset consisting of the corresponding cluster
            X['Labels']= Y

            # getting the unique clusters from our dataset
            list_of_clusters=X['Cluster'].unique()
