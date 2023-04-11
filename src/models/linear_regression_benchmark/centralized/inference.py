import skops.io as sio
from loguru import logger

from src.config import AIRPORTS, model_directory
from src.models.linear_regression_benchmark.centralized.train import (
    load_labels_and_features,
)

if __name__ == "__main__":
    logger.info("Loading labels and features")
    labels, features = load_labels_and_features(AIRPORTS)

    logger.info("Loading model")
    model = sio.load(
        model_directory / "centralized_linear_regression_benchmark.skops", trusted=True
    )

    logger.info("Predicting")
    model.predict(features)
