from datetime import timedelta
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LinearRegression

from src.config import data_directory
from src.models.linear_regression_benchmark.centralized.train import get_etd_minutes_until_pushback

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
    """Sets initial parameters as zeros.

    Required since model params are uninitialized until `model.fit` is called, but the server asks
    for initial parameters from clients at launch. Refer to sklearn.linear_model.LinearRegression
    documentation for more information.
    """
    n_outputs = 1
    n_features = model["classifier"].n_features_in_

    model["classifier"].coef_ = np.zeros((n_outputs, n_features))
    if model["classifier"].fit_intercept:
        model["classifier"].intercept_ = np.zeros((n_outputs,))


def load_public_airport_features(index: pd.Index, airport: str):
    """Loads the public features for a set of flights from one airport.

    The public features are:
    - minutes_until_pushback: the latest ETD estimate for the flight
    - n_flights: the number of flights in the last 30 hours
    """
    # initialize an empty table with the desired gufi/timestamp index
    features = pd.DataFrame([], index=index)

    # get etd features
    logger.debug("Loading ETD features")
    etd = pd.read_csv(
        data_directory / "raw" / "public" / airport / f"{airport}_etd.csv.bz2",
        parse_dates=["timestamp", "departure_runway_estimated_time"],
    )

    logger.debug("Merging ETD features")
    features["minutes_until_pushback"] = np.nan
    features["n_flights"] = np.nan

    for now in pd.to_datetime(features.index.get_level_values("timestamp")):
        now_features = features.xs(now, level="timestamp", drop_level=False).copy()
        now_etd = etd.loc[(etd.timestamp > now - timedelta(hours=30)) & (etd.timestamp <= now)]
        now_features = get_etd_minutes_until_pushback(now_features, now_etd)

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
    logger.debug(f"Loading MFS features for {airline} @ {airport}")
    mfs = (
        pd.read_csv(
            data_directory / "raw" / "private" / airport / f"{airport}_{airline}_mfs.csv.bz2",
            usecols=["gufi", "aircraft_engine_class", "aircraft_type"],
        )
        .drop_duplicates(subset=["gufi"], keep="first")
        .set_index("gufi")
    )

    features = features.merge(mfs, how="left", left_index=True, right_index=True, validate="1:1")

    # get etd features
    logger.debug(f"Loading ETD features for {airline} @ {airport}")
    etd = pd.read_csv(
        data_directory / "raw" / "public" / airport / f"{airport}_etd.csv.bz2",
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
