"""Centralized version of federated evaluation that does not use Flower."""
import numpy as np
import pandas as pd
from loguru import logger

from src.config import AIRLINES, AIRPORTS, data_directory, model_directory
from src.models.linear_regression_benchmark.federated.utils import (
    load_private_airline_airport_features,
    load_public_airport_features,
    set_model_params,
)
from src.models.linear_regression_benchmark.utils import initialize_model

if __name__ == "__main__":
    # load global model
    model = initialize_model()
    weights = np.load(model_directory / "federated_weights.npz")
    set_model_params(model, [weights["arr_0"], weights["arr_1"]])

    predictions = []
    submission_format = pd.read_csv(
        data_directory / "raw" / "submission_format.csv",
        parse_dates=["timestamp"],
        index_col=["gufi", "timestamp", "airport"],
    )
    submission_format["airline"] = submission_format.index.get_level_values("gufi").str[:3]

    for airport in AIRPORTS:
        logger.info(f"Predicting {airport}")
        for airline in AIRLINES:
            logger.info(f"Predicting {airline} @ {airport}")
            # subset submission format to just those flights from this airline
            partial_submission_format = (
                submission_format.xs(airport, level="airport", drop_level=False)
                .loc[submission_format.airline == airline]
                .copy()
            )

            # load the public features
            public_features = load_public_airport_features(partial_submission_format.index, airport)

            # load the private features
            private_features = load_private_airline_airport_features(
                partial_submission_format.index, airline, airport
            )

            # concatenate the public and private features
            features = pd.concat([public_features, private_features], axis=1)

            # generate predictions for this airline
            partial_submission_format["minutes_until_pushback"] = model.predict(features)
            predictions.append(partial_submission_format)

    predictions = pd.concat(predictions, axis=0)

    # re-order the predictions to match the submission format
    predictions = predictions.loc[submission_format.index].drop(columns=["airline"])
    predictions.to_csv(data_directory / "processed" / "submission.csv")
