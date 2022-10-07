import os

from dotenv import load_dotenv

load_dotenv()

class DatabaseConfig:
    def __init__(self):
        self.DATABASE_NAME = "ineuron"

        self.COLLECTION_NAME = "forest"

        self.DB_URL = os.environ["DB_URL"]

    def get_database_config(self):
        return self.__dict__

class SimpleImputerConfig:
    def __init__(self):
        self.strategy = "constant"

        self.fill_value = 0

    def get_simple_imputer_config(self):
        return self.__dict__