import sys
from typing import Union

import numpy as np
from imblearn.combine import SMOTETomek 
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from src.forest.constant import *
from src.forest.exception import ForestException
from src.forest.logger import logging
from src.forest.utils.main_utils import MainUtils
from src.forest.entity.config_entity import SimpleImputerConfig
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

class DataTransformation:
    def __init__(self):

        self.imputer_config = SimpleImputerConfig()

        self.utils = MainUtils()

    
    def get_data_transformer_object(self) -> object:
        """
        Method Name :   get_data_transformer_object
        Description :   This method creates and returns a data transformer object 
        
        Output      :   data transformer object is created and returned 
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info(
            "Entered get_data_transformer_object method of DataTransformation class"
        )

        try:
            logging.info("Got numerical, categorical, transformation columns from schema config")

            schema_info = self.utils.read_schema_config_file()
            
            num_features = schema_info['numerical_columns']
            
            categorical_features = schema_info['categorical_columns']

            logging.info("Created preprocessor object from ColumnTransformer")

            logging.info(
                "Exited get_data_transformer_object method of DataTransformation class"
            )
            numeric_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            preprocessor = ColumnTransformer(
                [
                    ("Numeric_Pipeline",numeric_pipeline,num_features)
            ]
            )

            logging.info("Created preprocessor object from ColumnTransformer")

            logging.info(
                "Exited get_data_transformer_object method of DataTransformation class"
            )

            return preprocessor

        except Exception as e:
            raise ForestException(e, sys) from e


    def initiate_data_transformation(
        self, train_set: DataFrame, test_set: DataFrame
    ) -> Union[np.ndarray, np.ndarray]:
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline 
        
        Output      :   data transformer object is created and returned 
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info(
            "Entered initiate_data_transformation method of Data_Transformation class"
        )
        try:

            preprocessor = self.get_data_transformer_object()

            logging.info("Got the preprocessor object")

            input_feature_train_df = train_set.drop(columns=[TARGET_COLUMN], axis=1)

            target_feature_train_df = train_set[TARGET_COLUMN]

            logging.info("Got train features and test features of Training dataset")

            input_feature_test_df = test_set.drop(columns=[TARGET_COLUMN], axis=1)

            target_feature_test_df = test_set[TARGET_COLUMN]

            logging.info("Got train features and test features of Testing dataset")

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe"
            )

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)

            logging.info(
                "Used the preprocessor object to fit transform the train features"
            )

            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[ input_feature_train_arr, np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            self.utils.save_object(PREPROCESSOR_OBJ_FILE_NAME, preprocessor)

            logging.info("Saved the preprocessor object")

            logging.info(
                "Exited initiate_data_transformation method of Data_Transformation class"
            )
            print("train", train_arr)
            print("test", test_arr)
            return train_arr, test_arr

        
        except Exception as e:
            raise ForestException(e, sys) from e