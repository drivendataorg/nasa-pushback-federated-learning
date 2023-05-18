from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import data_directory


def load_airport_labels(airport: str, downsample: Optional[int] = 100_000):
    if downsample:

        def skiprows(i):
            if i == 0:
                return False
            return i % downsample

    else:
        skiprows = None

    logger.debug(f"Loading labels for {airport}")
    return pd.read_csv(
        data_directory / "raw" / f"phase2_train_labels_{airport}.csv.bz2",
        parse_dates=["timestamp"],
        skiprows=skiprows,
        index_col=["gufi", "timestamp", "airport"],
    )


def initialize_model(n_features: int = 11):
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, make_column_selector(dtype_exclude="object")),
            (
                "aircraft_engine_class",
                OneHotEncoder(categories=[["JET", "TURBO", "PISTON"]], handle_unknown="ignore"),
                ["aircraft_engine_class"],
            ),
            (
                "aircraft_type",
                OneHotEncoder(
                    categories=[["B738", "A319", "A321", "A320", "B772"]], handle_unknown="ignore"
                ),
                ["aircraft_type"],
            ),
        ]
    )

    model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LinearRegression())])

    # initialize the model and column transformer
    model.fit(
        pd.DataFrame(
            {
                "aircraft_engine_class": ["JET"],
                "aircraft_type": ["B738"],
                "n_jets": [0],
                "minutes_until_pushback": [0],
                "n_flights": [0],
            }
        ),
        [0.0],
    )
    model["classifier"].coef_ = np.zeros((1, n_features))
    model["classifier"].intercept_ = [0.0]

    return model
