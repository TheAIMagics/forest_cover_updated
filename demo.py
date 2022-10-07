from asyncio.windows_utils import pipe
from src.forest.pipeline.train_pipeline import TrainPipeline

pipeline = TrainPipeline()

train_set, test_set = pipeline.start_data_ingestion()

print(pipeline.start_data_validation(train_set, test_set))
