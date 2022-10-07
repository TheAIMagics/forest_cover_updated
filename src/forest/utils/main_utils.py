import shutil
import sys
from typing import Dict, Tuple

import dill
import xgboost
import yaml
from pandas import DataFrame
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import all_estimators
from yaml import safe_dump

from src.forest.constant import MODEL_CONFIG_FILE, SCHEMA_CONFIG_FILE
from src.forest.exception import ForestException
from src.forest.logger import logging

class MainUtils:
    def __init__(self) -> None:
        pass

    def read_yaml_file(self, filename: str) -> dict:
        try:
            with open(filename, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)

        except Exception as e:
            raise ForestException(e, sys) from e

    def read_schema_config_file(self) -> dict:
        try:
            schema_config = self.read_yaml_file(SCHEMA_CONFIG_FILE)

            return schema_config

        except Exception as e:
            raise ForestException(e, sys) from e