import sys
from typing import Tuple

from pandas import DataFrame

from src.forest.components.data_ingestion import DataIngestion
from src.forest.components.data_validation import DataValidation
from src.forest.components.data_transformation import DataTransformation
from src.forest.components.model_trainer import ModelTrainer
from src.forest.exception import ForestException
from src.forest.logger import logging
from src.forest.configuration.configuration import ConfigurationManager

class TrainPipeline:
    def __init__(self):
        pass

    def start_data_ingestion(self) -> Tuple[DataFrame, DataFrame]:
        logging.info("Entered the start_data_ingestion method of TrainPipeline class")

        try:
            logging.info("Getting the data from mongodb")

            config = ConfigurationManager()

            data_ingestion_config = config.get_data_ingestion_config()
        
            data_ingestion = DataIngestion(config=data_ingestion_config)

            train_data, test_set = data_ingestion.initiate_data_ingestion()

            logging.info("Got the train_set and test_set from mongodb")

            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )
        
            return train_data, test_set

        except Exception as e:
            raise ForestException(e, sys) from e

    @staticmethod
    def start_data_validation(train_set: DataFrame, test_set: DataFrame) -> bool:
        logging.info("Entered the start_data_validation method of TrainPipeline class")

        try:
            data_validation = DataValidation(train_set, test_set)

            data_validation_status = data_validation.initiate_data_validation()

            logging.info("Performed the data validation operation")

            logging.info(
                "Exited the start_data_validation method of TrainPipeline class"
            )

            return data_validation_status

        except Exception as e:
            raise ForestException(e, sys) from e
        
    @staticmethod
    def start_data_transformation(
        train_set: DataFrame, test_set: DataFrame
    ) -> Tuple[DataFrame, DataFrame]:
        logging.info(
            "Entered the start_data_transformation method of TrainPipeline class"
        )

        try:
            data_transformation = DataTransformation()

            train_set, test_set = data_transformation.initiate_data_transformation(
                train_set, test_set
            )

            logging.info(
                "Exited the start_data_transformation method of TrainPipeline class"
            )

            return train_set, test_set

        except Exception as e:
            raise ForestException(e, sys) from e
        
    @staticmethod
    def start_model_trainer(train_set: DataFrame, test_set: DataFrame) -> None:
        logging.info("Entered the start_model_trainer method of TrainPipeline class")

        try:
            model_trainer = ModelTrainer()

            model_trainer.initiate_model_trainer(train_set, test_set)

            logging.info("Exited the start_model_trainer method of TrainPipeline class")

        except Exception as e:
            raise ForestException(e, sys) from e
