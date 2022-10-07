import sys
from typing import Tuple

from pandas import DataFrame

from src.forest.components.data_ingestion import DataIngestion
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
