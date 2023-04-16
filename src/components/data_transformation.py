import os
import sys
from src.logger import logging
from dataclasses import dataclass
from src.exception import CustomException
import pandas as pd 
import numpy as np
from sklearn.impute import SimpleImputer ## handeling missing values
from sklearn.preprocessing import  StandardScaler ## handaling feature scalling
from sklearn.preprocessing import OrdinalEncoder ## Ordinal Encoding
## Piplines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils import save_object



@dataclass
class DataTransformationconfig:
    preprocessr_obj_file_path = os.path.join("artifcats","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation Initiated ")

            ## saprate numerical and categorical features
            categorical_features = ['cut', 'color','clarity']
            numerical_features = ['carat', 'depth','table', 'x', 'y', 'z']
            
            
            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']


            logging.info("Pipline Initited")


            ## Numerical pipline
            num_pipline = Pipeline(
             steps = [
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler())
            ]
        )

            ## Catigorical Pipline
            cato_pipline = Pipeline(
            steps = [
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("ordinal",OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
            ("Scaler",StandardScaler())
            ]
        )

            preprossor = ColumnTransformer([

            ("num_pipline",num_pipline,numerical_features),
            ("cato_pipline",cato_pipline,categorical_features)

            ])

            return preprossor

            logging.info("Pipline Complited")

        except Exception as e:
            logging.info("Error in Data transformation")
            raise CustomException(e,sys)



    def initatie_data_transformation(self,train_path,test_path):
        try:
            # Read Train And Test Data
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Read Traning And Test Data Complited")
            logging.info(f'Train Dataframe Head : \n{train_data.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_data.head().to_string()}')


            logging.info("Obtaning Preprocessing object")

            preprocessing_obj = self.get_data_transformation_object()


            target_column_name = "price"
            drop_columns = [target_column_name,"id"]


            ## this is like X and Y
            input_features_train_data = train_data.drop(drop_columns,axis=1)
            traget_feature_train_data = train_data[target_column_name]

            ## this is like X and Y
            input_feature_test_data = test_data.drop(drop_columns,axis=1)
            traget_feature_test_data = test_data[target_column_name]

            ## Apply Transformation  using preprossing_obj on (xtrain and xtest)
            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_data)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_data)


            logging.info("Applying Preprossing Object on traning and testing datasets")



            ## convert in to array to be fast
            train_array = np.c_[input_feature_train_arr,np.array(traget_feature_train_data)]
            test_array = np.c_[input_feature_test_arr,np.array(traget_feature_test_data)]


            ## Calling Save_obj function to save pkl file
            save_object(self.data_transformation_config.preprocessr_obj_file_path,
            obj=preprocessing_obj
            )

            logging.info("Preprocessor Pikle File Saved")

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessr_obj_file_path
            )


        except Exception as e:
            logging.info("Exception Occured in the initatie_data_transformation")
            raise CustomException(e,sys)
