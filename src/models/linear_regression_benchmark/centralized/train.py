from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd
import skops.io as sio
from loguru import logger
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import AIRPORTS, LOCAL_FINAL
from src.models.linear_regression_benchmark.utils import load_airport_labels

data_directory = LOCAL_FINAL / "public"


def estimate_pushback(now_features: pd.DataFrame, now_etd: pd.DataFrame) -> pd.Series:
    # get the latest ETD for each flight
    latest_now_etd = now_etd.groupby("gufi").last().departure_runway_estimated_time

    # merge the latest ETD with the flights we are predicting
    departure_runway_estimated_time = now_features.merge(
        latest_now_etd, how="left", on="gufi"
    ).departure_runway_estimated_time

    minutes_until_pushback = pd.Series(
        (
            departure_runway_estimated_time - now_features.index.get_level_values("timestamp")
        ).dt.total_seconds()
        / 60,
        name="minutes_until_pushback",
    )
    return now_features.drop(columns=["minutes_until_pushback"]).join(
        minutes_until_pushback, how="left", on="gufi", validate="1:1"
    )[["minutes_until_pushback"]]


def load_airport_features(index: pd.Index, airport: str):
    # initialize features with indices
    features = pd.DataFrame([], index=index)

    # get mfs features
    logger.debug("Loading MFS features")
    mfs = (
        pd.read_csv(
            data_directory / airport / f"{airport}_mfs.csv.bz2",
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

    logger.debug("Merging ETD features")
    features["minutes_until_pushback"] = np.nan
    features["n_flights"] = np.nan
    features["n_jets"] = np.nan

    for now in pd.to_datetime(features.index.get_level_values("timestamp")):
        now_features = features.xs(now, level="timestamp", drop_level=False).copy()
        now_etd = etd.loc[(etd.timestamp > now - timedelta(hours=30)) & (etd.timestamp <= now)]
        now_features = estimate_pushback(now_features, now_etd)

        # number of flights with ETD in time window
        now_gufis = pd.Series(now_etd.gufi.unique())
        now_features["n_flights"] = len(now_gufis)

        # number of jets in time window
        now_features["n_jets"] = (
            mfs.loc[now_gufis.loc[now_gufis.isin(mfs.index)], "aircraft_engine_class"] == "JET"
        ).sum()

        features.update(now_features)

    assert features.minutes_until_pushback.notnull().all()

    # get weather features
    logger.debug("Loading LAMP features")
    lamp = (
        pd.read_csv(
            data_directory / airport / f"{airport}_lamp.csv.bz2",
            parse_dates=["timestamp", "forecast_timestamp"],
        )
        .sort_values(["timestamp", "forecast_timestamp"])
        .drop_duplicates(subset=["forecast_timestamp"], keep="last")
        .drop(columns=["timestamp"])
        .set_index("forecast_timestamp")
    )
    features = (
        features.reset_index()
        .merge(
            lamp,
            how="left",
            left_on=features.index.get_level_values("timestamp").floor("H"),
            right_on="forecast_timestamp",
            validate="m:1",
        )
        .drop(columns=["forecast_timestamp"])
        .set_index(features.index.names)
    )

    return features


def train_model(features, labels):
    # train model
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ("selector", SelectPercentile(chi2, percentile=50)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, make_column_selector(dtype_exclude="object")),
            ("cat", categorical_transformer, make_column_selector(dtype_include="object")),
        ]
    )

    model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LinearRegression())])

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=112
    )

    model.fit(x_train, y_train)
    model.predict(x_train)
    model.score(x_test, y_test)

    return model


def load_labels_and_features(airports: List[str]):
    labels = []
    features = []
    for airport in airports:
        logger.info(f"Loading labels and features for {airport}")
        airport_labels = load_airport_labels(airport)
        airport_features = load_airport_features(airport_labels.index, airport)

        labels.append(airport_labels)
        features.append(airport_features)

    return pd.concat(labels), pd.concat(features)


def main():
    labels, features = load_labels_and_features(AIRPORTS[:2])

    logger.info("Training model")
    model = train_model(features, labels)

    sio.dump(model, "linear_regression_benchmark.skops")


if __name__ == "__main__":
    main()
