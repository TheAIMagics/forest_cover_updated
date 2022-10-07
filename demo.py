from src.forest.pipeline.train_pipeline import TrainPipeline

pipeline = TrainPipeline()

train_set, test_set = pipeline.start_data_ingestion()

is_validated = pipeline.start_data_validation(train_set, test_set)

if is_validated:
    train_set, test_set = pipeline.start_data_transformation(train_set, test_set)
