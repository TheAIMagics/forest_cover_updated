from pathlib import Path

CONFIG_FILE_PATH = Path('config/config.yaml')

LOG_DIR = "logs"

LOG_FILE = "forest.log"

ARTIFACTS_DIR = "artifacts"

MODEL_SAVE_FORMAT = ".sav"

MODEL_FILE_NAME = "model"

PRED_DATA_CSV_FILE = "forest_pred_data.csv"

PREDICTIONS_FILE = "forest_predictions.csv"

MODEL_CONFIG_FILE = "config/model.yaml"

SCHEMA_CONFIG_FILE = "config/schema.yaml"

BEST_MODEL_PATH = ARTIFACTS_DIR + "/" + MODEL_FILE_NAME + MODEL_SAVE_FORMAT

BASE_MODEL_SCORE = 0.6

PREPROCESSOR_OBJ_FILE_NAME = ARTIFACTS_DIR + "/" + "forest_preprocessor.pkl"

TRAIN_TEST_SPLIT_SIZE = 0.2

TARGET_COLUMN = "class"

APP_HOST = "0.0.0.0"

APP_PORT = 8080