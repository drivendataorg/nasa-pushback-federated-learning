# Federated learning benchmark


## Motivation

The goal of this benchmark is to demonstrate a model that is simple, but still has some complexity that requires careful consideration of how to federate. In particular, the benchmark uses:

- Private features for the predicted flight: here, we use the `aircraft_engine_class` and `aircraft_type` from the MFS dataset.
- Private features from other flights: The idea is to use not only features for the predicted flight, but also features that from other flights. Federation is relatively simple when we are only computing features for the predicted flight—simply subset the rows of the features that match the GUFI for that predicted flight. More complicated is to derive features from all of the flights in the dataset while still obeying the rules of federation, i.e., only use flights from your airline. To that end, we construct a simple feature that captures that behavior: `n_jets`. This feature is a count of the number of flights in the time window (those flights for which an ETD exists) that have jet engines (`mfs.aircraft_engine_class == "JET"`). In the centralized version, this is simply the count of aircraft with jet engines. In the federated version, this is the count of aircraft _from my airline_ that have jet engines, so different airlines will come up with different values for the same time period.

This benchmark results in a single global model that aggregates weights for all airlines and airports. Of course, you might experiment with "personalization" where each airport/client fine-tunes the global model to account for the unique characteristics of its data.

For the sake of simplicity, this benchmark is intentionally not very good. It trains on a small subset of the data, and does not even use proper train and validation subsets. It is meant to serve as an example of model training using Flower, not as an example of machine learning best practices!

## How to run the benchmark

1. Copy training dataset to the following locations in the `data` directory at the root of the repository.


```
├── external
├── interim
├── processed
└── raw
    ├── private
    │   ├── KATL
    │   │   ├── KATL_AAL_mfs.csv.bz2  # contains rows for AAL airline, all columns
    │   │   ├── KATL_AAL_standtimes.csv.bz2  # contains rows for ALL, all columns
    │   │   ├── ...
    │   │   ├── KATL_UPS_mfs.csv.bz2  # contains rows for UPS airline, all columns
    │   │   └── KATL_UPS_standtimes.csv.bz2
    │   ├── KCLT
    │   ├── KDEN
    │   ├── KDFW
    │   ├── KJFK
    │   ├── KMEM
    │   ├── KMIA
    │   ├── KORD
    │   ├── KPHX
    │   └── KSEA
    ├── public
    │   ├── KATL
    │   │   ├── KATL_config.csv.bz2
    │   │   ├── KATL_etd.csv.bz2
    │   │   ├── KATL_first_position.csv.bz2
    │   │   ├── KATL_lamp.csv.bz2
    │   │   ├── KATL_mfs.csv.bz2  # only contains public columns [gufi, isdeparture]
    │   │   ├── KATL_runways.csv.bz2
    │   │   ├── KATL_standtimes.csv.bz2  # only contains public columns [gufi, timestamp, arrival_stand_actual_time]
    │   │   ├── KATL_tbfm.csv.bz2
    │   │   └── KATL_tfm.csv.bz2
    │   ├── KCLT
    │   ├── KDEN
    │   ├── KDFW
    │   ├── KJFK
    │   ├── KMEM
    │   ├── KMIA
    │   ├── KORD
    │   ├── KPHX
    │   └── KSEA
    ├── phase2_train_labels_KATL.csv.bz2
    ├── phase2_train_labels_KCLT.csv.bz2
    ├── phase2_train_labels_KDEN.csv.bz2
    ├── phase2_train_labels_KDFW.csv.bz2
    ├── phase2_train_labels_KJFK.csv.bz2
    ├── phase2_train_labels_KMEM.csv.bz2
    ├── phase2_train_labels_KMIA.csv.bz2
    ├── phase2_train_labels_KORD.csv.bz2
    ├── phase2_train_labels_KPHX.csv.bz2
    ├── phase2_train_labels_KSEA.csv.bz2
    └── submission_format.csv
```

1. Run federated training. While prototyping your code, you might consider limiting the number of airports and airlines to speed up training.

```bash
python src/models/linear_regression_benchmark/federated/train.py
```

1. Run federated evaluation.

```bash
python src/models/linear_regression_benchmark/federated/predict.py
```

We also include a centralized version of the same model for comparison. You can train it with:

```bash
python src/models/linear_regression_benchmark/centralized/train.py
```
