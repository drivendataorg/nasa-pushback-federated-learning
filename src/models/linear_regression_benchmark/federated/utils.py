from datetime import timedelta
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import data_directory
from src.models.linear_regression_benchmark.centralized.train import estimate_pushback

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LinRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def get_model_parameters(model: LinearRegression) -> LinRegParams:
    """Returns the paramters of a sklearn LinearRegression model."""
    if model["classifier"].fit_intercept:
        params = [
            model["classifier"].coef_,
            model["classifier"].intercept_,
        ]
    else:
        params = [
            model["classifier"].coef_,
        ]
    return params


def set_model_params(model: LinearRegression, params: LinRegParams) -> LinearRegression:
    """Sets the parameters of a sklean LinearRegression model."""
    model["classifier"].coef_ = params[0]
    if model["classifier"].fit_intercept:
        model["classifier"].intercept_ = params[1]
    return model


def set_initial_params(model: LinearRegression):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LinearRegression documentation for more
    information.
    """
    n_classes = 1
    n_features = 38

    model["classifier"].coef_ = np.zeros((n_classes, n_features))
    if model["classifier"].fit_intercept:
        model["classifier"].intercept_ = np.zeros((n_classes,))


def load_public_airport_features(index: pd.Index, airport: str):
    """Loads the public features for a set of flights from one airport."""
    # initialize features with indices
    features = pd.DataFrame([], index=index)

    # # add airline as features
    # features["airline"] = features.index.get_level_values("gufi").str[:3]

    # get etd features
    logger.debug("Loading ETD features")
    etd = pd.read_csv(
        data_directory / airport / f"{airport}_etd.csv.bz2",
        parse_dates=["timestamp", "departure_runway_estimated_time"],
    )

    logger.debug("Merging ETD features")
    features["minutes_until_pushback"] = np.nan
    features["n_flights"] = np.nan

    for now in pd.to_datetime(features.index.get_level_values("timestamp")):
        now_features = features.xs(now, level="timestamp", drop_level=False).copy()
        now_etd = etd.loc[(etd.timestamp > now - timedelta(hours=30)) & (etd.timestamp <= now)]
        now_features = estimate_pushback(now_features, now_etd)

        # number of flights with ETD in time window
        now_gufis = pd.Series(now_etd.gufi.unique())
        now_features["n_flights"] = len(now_gufis)

        features.update(now_features)

    assert features.minutes_until_pushback.notnull().all()

    return features


def load_private_airline_airport_features(index: pd.Index, airline: str, airport: str):
    """Loads the private features for a set of flights from one airline and one airport."""
    features = pd.DataFrame([], index=index)

    # get mfs features
    logger.debug("Loading MFS features")
    mfs = (
        pd.read_csv(
            data_directory / airport / f"{airport}_mfs.csv.bz2",
            usecols=["gufi", "aircraft_engine_class", "aircraft_type"],
        )
        .drop_duplicates(subset=["gufi"], keep="first")
        .set_index("gufi")
    )

    features = features.merge(mfs, how="left", left_index=True, right_index=True, validate="1:1")

    # get etd features
    logger.debug("Loading ETD features")
    etd = pd.read_csv(
        data_directory / airport / f"{airport}_etd.csv.bz2",
        parse_dates=["timestamp", "departure_runway_estimated_time"],
    )

    features["n_jets"] = np.nan
    for now in pd.to_datetime(features.index.get_level_values("timestamp")):
        now_features = features.xs(now, level="timestamp", drop_level=False).copy()
        now_etd = etd.loc[(etd.timestamp > now - timedelta(hours=30)) & (etd.timestamp <= now)]
        now_gufis = pd.Series(now_etd.gufi.unique())

        # number of jets in time window
        now_features["n_jets"] = (
            mfs.loc[now_gufis.loc[now_gufis.isin(mfs.index)], "aircraft_engine_class"] == "JET"
        ).sum()

        features.update(now_features)

    return features


def get_airline_labels(labels: pd.DataFrame, airline: str) -> pd.DataFrame:
    logger.debug(f"Loading airline labels for {airline}")
    airlines = pd.Series(labels.index.get_level_values("gufi").str[:3], index=labels.index)
    return labels.loc[airlines == airline]


def initialize_model(features, labels):
    # train model
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

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=112
    )

    model.fit(x_train, y_train)

    return model
