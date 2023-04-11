import pandas as pd
from loguru import logger

from src.config import data_directory


def load_airport_labels(airport: str, downsample: int = 10_000):
    logger.debug(f"Loading labels for {airport}")
    return pd.read_csv(
        data_directory / f"train_labels_{airport}.csv.bz2",
        parse_dates=["timestamp"],
        skiprows=lambda i: i % downsample,
        index_col=["gufi", "timestamp", "airport"],
    )
