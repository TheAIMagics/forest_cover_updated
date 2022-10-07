from src.forest.pipeline.train_pipeline import TrainPipeline

pipeline = TrainPipeline()

train_set, test_set = pipeline.start_data_ingestion()

print(train_set,test_set)