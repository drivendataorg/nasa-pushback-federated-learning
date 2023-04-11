"""Centralized version of federated evaluation that does not use Flower."""
import pandas as pd
import skops.io as sio

from src.config import AIRLINES, AIRPORTS, data_directory
from src.models.linear_regression_benchmark.federated.utils import (
    load_private_airline_airport_features,
    load_public_airport_features,
)

# load global model
model = sio.load("linear_regression_benchmark.skops", trusted=True)

predictions = []
submission_format = pd.read_csv(data_directory / "submission_format.csv")
for airport in AIRPORTS:
    for airline in AIRLINES:
        # subset submission format to just those flights from this airline
        partial_submission_format = submission_format.loc[submission_format.airline == airline]

        # load the public features
        public_features = load_public_airport_features(partial_submission_format.index, airport)

        # load the private features
        private_features = load_private_airline_airport_features(
            partial_submission_format.index, airline, airport
        )

        # concatenate the public and private features
        features = pd.concat([public_features, private_features], axis=1)

        # generate predictions for this airline
        airline_predictions = model.predict(features)
        predictions.append(airline_predictions)


predictions = pd.concat(predictions, axis=0)

# re-order the predictions to match the submission format
predictions = predictions.loc[submission_format.index]
