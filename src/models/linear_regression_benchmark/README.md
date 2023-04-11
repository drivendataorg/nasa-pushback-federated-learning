# Federated learning benchmark

The goal here is to make a model that is simple, but still has some complexity that requires careful consideration of how to federate.

- Uses some private features: here, we use the private features from the MFS dataset.
- Use private features from other flights: It is important to demonstrate not only features for the predicted flight, but also features that come from other flights in the dataset at that time. Federation is simple when we are only computing features for the predicted flight—simply subset the rows of the features that match the GUFI for that predicted flight. More complicated is deriving features from all of the flights in the dataset while still obeying the rules of federation, i.e., only use flights from your airline. To that end, I created a simple feature that captures that behavior: `n_jets`. This feature is a count of the number of flights in the time window (those flights for which an ETD exists) that have jet engines (`mfs.aircraft_engine_class == "JET"`). In the centralized version, this is the true count of aircraft with jet engines. In the federated version, this is the count of aircraft _from my airline_ that have jet engines, so different airlines will come up with different values for the same time period.


1. Copy training dataset to the `data` directory at the root of the repository.

2. Run federated training. You might consider limiting the number of airports and airlines to speed up training.

```bash
python src/models/linear_regression_benchmark/train.py
```

3. Run federated evaluation.

```bash
python src/models/linear_regression_benchmark/test.py
```
