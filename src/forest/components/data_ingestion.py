import sys,os,shutil
from typing import Tuple
import numpy as np
import pandas as pd
from pandas import DataFrame
from zipfile import ZipFile
from src.forest.constant import *
from src.forest.exception import ForestException
from src.forest.logger import logging
from src.forest.utils.main_utils import MainUtils
from src.forest.entity.config_entity import DatabaseConfig
from src.forest.configuration.mongo_operations import MongoDBOperation
from sklearn.model_selection import train_test_split
from src.forest.entity.data_ingestion_config import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

        self.utils = MainUtils()

        self.mongo_op = MongoDBOperation()

        self.mongo_config = DatabaseConfig()

    def download_file(self):
        print(self.config)
        if not os.path.exists(self.config.local_data_file):
            logging.info("Download started...")
            shutil.copy(self.config.source_URL, self.config.root_dir)
        else:
            logging.info(f"Download File already exists")
    
    def extract_zipfile(self):
        file = ZipFile(self.config.local_data_file)
        
        if not os.path.exists(self.config.unzip_dir):
            MainUtils.create_directories([self.config.unzip_dir])
            file.extractall(path=self.config.unzip_dir)
        else:
            logging.info(f"File already exists of size")
    
    def get_dataframe(self):
        target_filepath = os.path.join(self.config.unzip_dir+'/covtype.csv')
        df = pd.read_csv(target_filepath)
        return df
    
    @staticmethod
    def split_data_as_train_test(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """
        Method Name :   split_data_as_train_test
        Description :   This method splits the dataframe into train set and test set based on split ratio 
        
        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")

        try:
            train_set, test_set = train_test_split(df, test_size=TRAIN_TEST_SPLIT_SIZE)

            logging.info("Performed train test split on the dataframe")

            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )

            return train_set, test_set

        except Exception as e:
            raise ForestException(e, sys) from e

    
    def initiate_data_ingestion(self):
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline 
        
        Output      :   train set and test set are returned as the artifacts of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            self.download_file()
            self.extract_zipfile()
            df = self.get_dataframe()

            _schema_config = self.utils.read_schema_config_file()

            df1 = df.drop(_schema_config["drop_columns"], axis=1)

            logging.info("Got the data from mongodb")

            train_set, test_set = self.split_data_as_train_test(df1)

            logging.info("Performed train test split on the dataset")

            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )

            return train_set, test_set
            
        except Exception as e:
            raise ForestException(e, sys) from e

