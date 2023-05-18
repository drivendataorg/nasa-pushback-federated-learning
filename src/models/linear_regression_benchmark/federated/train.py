"""Federated training using Flower."""
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List

import flwr as fl
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error

from src.config import AIRLINES, AIRPORTS, model_directory
from src.models.linear_regression_benchmark.federated.utils import (
    get_airline_labels,
    get_model_parameters,
    load_private_airline_airport_features,
    load_public_airport_features,
    set_initial_params,
    set_model_params,
)
from src.models.linear_regression_benchmark.utils import (
    initialize_model,
    load_airport_labels,
)

AIRLINES = ["DAL", "EDV"]
AIRPORTS = ["KATL"]


def train_client_factory(airline: str):
    features = []
    labels = []
    for airport in AIRPORTS:
        airport_labels = load_airport_labels(airport)

        # get the airlines that had flights at this airport
        airlines = pd.Series(
            airport_labels.index.get_level_values("gufi").str[:3], index=airport_labels.index
        )
        if airline not in airlines.values:
            logger.warning(f"{airline} not present at {airport}.")
            continue

        airline_labels = get_airline_labels(airport_labels, airline)
        logger.debug(f"Loaded {len(airline_labels):,} data points for {airline} @ {airport}")
        public_features = load_public_airport_features(airline_labels.index, airport)
        private_features = load_private_airline_airport_features(
            airline_labels.index, airline, airport
        )

        airline_features = pd.concat([private_features, public_features], axis=1)

        features.append(airline_features)
        labels.append(airline_labels)

    features = pd.concat(features, axis=0)
    labels = pd.concat(labels, axis=0)

    model = initialize_model()
    set_initial_params(model)

    @dataclass
    class LinearRegressionTrainClient(fl.client.NumPyClient):
        """Linear regression flower client."""

        airline: str

        def get_parameters(self, config):
            return get_model_parameters(model)

        def fit(self, parameters, config):
            logger.debug(f"Fitting client {self.airline}")
            set_model_params(model, parameters)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(features, labels)

            model_parameters = get_model_parameters(model)
            logger.info(
                f"""{self.airline} training finished for round {config["server_round"]}. """
                f"""Parameters {model_parameters}"""
            )

            return model_parameters, len(labels), {}

        def evaluate(self, parameters, config):
            logger.debug(f"Evaluating client {self.airline}")
            set_model_params(model, parameters)
            predictions = model.predict(features)
            loss = mean_absolute_error(labels, predictions)
            logger.debug(f"{self.airline} loss = {loss:4.4f}")
            return loss, len(labels), {}

    return LinearRegressionTrainClient(airline)


def fit_config(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            logger.info(f"Saving round {server_round} aggregated_ndarrays")
            np.savez("federated_weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics


if __name__ == "__main__":
    client_resources = {
        "num_cpus": int(os.cpu_count() / 2) + 1,
    }

    strategy = SaveModelStrategy(
        min_available_clients=len(AIRLINES),
        on_fit_config_fn=fit_config,
    )

    fl.simulation.start_simulation(
        client_fn=train_client_factory,
        client_resources=client_resources,
        clients_ids=AIRLINES,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=1),
        ray_init_args={"include_dashboard": False},
    )
