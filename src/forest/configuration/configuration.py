from src.forest.constant import CONFIG_FILE_PATH
from src.forest.utils.main_utils import MainUtils
from src.forest.logger import logging
from src.forest.entity.data_ingestion_config import DataIngestionConfig

class ConfigurationManager:
    def __init__(self,config_filepath = CONFIG_FILE_PATH):
        self.config = MainUtils.read_yaml_file(MainUtils, filename=config_filepath)

    def get_data_ingestion_config(self) -> DataIngestionConfig:

        config = self.config['data_ingestion']

        MainUtils.create_directories([config["root_dir"]])

        data_ingestion_config = DataIngestionConfig(
            root_dir = config["root_dir"],
            source_URL = config["source_URL"],
            local_data_file = config["local_data_file"],
            unzip_dir = config["unzip_dir"]
        )

        return data_ingestion_config