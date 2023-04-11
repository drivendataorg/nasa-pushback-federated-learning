# Pushback to the Future: Predict Pushback Time at US Airports (Phase 2)

This is the repository for Phase 2 of the **Pushback to the Future: Predict Pushback Time at US Airports**. The goal of Phase 2 is to explore federated learning techniques that enable independent *clients* (also known as parties, organizations, groups) to jointly train a global model without sharing or pooling data. In this case, the clients are **airlines**. Federated learning is a perfect match for the problem of pushback time prediction because airlines collect a lot of information that is relevant to pushback time, but too valuable or sensitive to share, like the number of passengers that have checked in for a flight or the number of bags that have been loaded onto a plane. Federated learning enables airlines to safely contribute the valuable and sensitive data they collect towards a centralized model. In Phase 2, we're asking you to translate your winning Phase 1 model into a model that can be trained in a federated manner. You will work with the same data from Phase 1, except it will be divided into "Public" or "non-federated" variables for which all rows are available to all airlines and "Private" or "federated" variables for which only rows corresponding to a flight that an airline operates are available to that airline.

In the spirit of experimentation, you will have some flexibility in how you translate your Phase 1 winning model during Phase 2. Do some strategies for combining the models from individual clients work better than others? Are there basic changes that your model needs to operate in a federated setting? We want to know! We will release more of the specific requirements as Phase 2 ramps up, but for now get started thinking about what it will take to turn your winning centralized model into an effective federated model! 

The exact variables airlines want to protect by federating are by definition too sensitive to release for a competition. Since we don't have access to the actual airline data that would be federated, we'll simulate the federated scenario by treating a subset of variables as if they are private.

## Data

In the development period, we will provide you with the training features and labels partitioned in a way that makes it straightforward to know which data is available to all airlines versus only available to specific airlines. We provide training data from the same 10 airports as tar archives of bzip2-compressed CSVs, one archive per airport. Each archive contains [training features](#train-features) and [labels](#train-labels) from the corresponding airport. Head to the [Data download](https://www.drivendata.org/competitions/218/competition-nasa-airport-pushback-phase2/data/) page to get the data, and read on to learn more about the data format.

<a id="limiting-predictions-to-the-most-active-airlines" href="#"></a>

<div class="alert alert-info">
<p><strong>Limiting predictions to the most active airlines</strong></p>
<p>The vast majority of flights are operated by a small subset of the total number of airlines present in the full dataset. For simplicity and efficiency, you will only be asked to predict for flights from 25 of the most active airlines. These are: AAL, AJT, ASA, ASH, AWI, DAL, EDV, EJA, ENY, FDX, FFT, GJS, GTI, JBU, JIA, NKS, PDT, QXE, RPA, SKW, SWA, SWQ, TPA, UAL, UPS.</p>
<p>Not all airlines will have flights out of all airports. If an airline does not operate any flights, the airline-specific private feature file will only contain a single header line. Also, note that although you will only be asked to make predictions for these airlines, you will have access to public data from all airlines (and of course, none of the private data from airlines that are not among these 25).</p>
</div>


<a id=train-features href="#"></a>

### Train features

The features used in Phase 2 will be the [same data used in Phase 1](https://www.drivendata.org/competitions/149/competition-nasa-airport-pushback/page/677/#features), with a few key differences:

- All public data (all airline data, available to all airlines) are stored in a `public` root directory.
- Public versions of the `mfs` and `standtimes` files do not include the private columns.
- All private data are stored in a `private` root directory. Each airline has its own version of the `mfs` and `standtimes` data containing the _public and private_ columns.

Here is an example of the training features within one airport archive, shown for KATL:

```
private
└── KATL
    ├── KATL_AAL_mfs.csv.bz2  # contains rows for AAL airline, all columns
    ├── KATL_AAL_standtimes.csv.bz2  # contains rows for ALL, all columns
    ├── ...
    ├── KATL_UPS_mfs.csv.bz2  # contains rows for UPS airline, all columns
    └── KATL_UPS_standtimes.csv.bz2

public
└── KATL
    ├── KATL_config.csv.bz2
    ├── KATL_etd.csv.bz2
    ├── KATL_first_position.csv.bz2
    ├── KATL_lamp.csv.bz2
    ├── KATL_mfs.csv.bz2  # only contains public columns [gufi, isdeparture]
    ├── KATL_runways.csv.bz2
    ├── KATL_standtimes.csv.bz2  # only contains public columns [gufi, timestamp, arrival_stand_actual_time]
    ├── KATL_tbfm.csv.bz2
    └── KATL_tfm.csv.bz2

```

When you extract all of the archives, you will end up with the training features in the following directory structure:

```
private
├── KATL
│   ├── KATL_AAL_mfs.csv.bz2  # contains rows for AAL airline, all columns
│   ├── KATL_AAL_standtimes.csv.bz2  # contains rows for ALL, all columns
│   ├── ...
│   ├── KATL_UPS_mfs.csv.bz2  # contains rows for UPS airline, all columns
│   └── KATL_UPS_standtimes.csv.bz2
├── KCLT
│ ...

public
├── KATL
│   ├── KATL_config.csv.bz2
│   ├── KATL_etd.csv.bz2
│   ├── KATL_first_position.csv.bz2
│   ├── KATL_lamp.csv.bz2
│   ├── KATL_mfs.csv.bz2  # only contains public columns [gufi, isdeparture]
│   ├── KATL_runways.csv.bz2
│   ├── KATL_standtimes.csv.bz2  # only contains public columns [gufi, timestamp, arrival_stand_actual_time]
│   ├── KATL_tbfm.csv.bz2
│   └── KATL_tfm.csv.bz2
├── KCLT
│ ...

```

**Public** MFS CSVs include the following public columns for _all_ flights:

- `gufi`
- `isdeparture`

**Private** MFS CSVs only include flights operated by the corresponding airline. They include all of the columns in the public MFS CSVs _and_ the private columns:

- `gufi`
- `aircraft_engine_class` (*private*)
- `aircraft_type` (*private*)
- `major_carrier` (*private*)
- `flight_type` (*private*)
- `isdeparture`

**Public** standtimes CSVs include the following public columns for _all_ flights:

- `gufi`
- `timestamp`
- `departure_stand_actual_time`


**Private** standtimes CSVs only include flights operated by the corresponding airline. They include all of the columns in the public standtimes CSVs _and_ the private columns:

- `gufi`
- `timestamp`
- `arrival_stand_actual_time` (*private*)
- `departure_stand_actual_time`


<a id="train-labels" href="#"></a>

### Train labels

The archives also include training labels to assist in the development of your model. You are not required to use these during training, but it could be a useful starting point. The Phase 2 training labels are identical to those [provided in the Prescreened Arena](https://www.drivendata.org/competitions/182/competition-nasa-airport-pushback-prescreened/data/), but subset to only the [selected airlines](#limiting-predictions-to-the-most-active-airlines). As in Phase 1, each file contains the actual minutes to pushback for a GUFI at a particular time. For example, here are the first five rows of `phase2_train_labels_KATL.csv.bz2`:

```csv
gufi,timestamp,airport,minutes_until_pushback
AAL1008.ATL.DFW.210403.1312.0051.TFM_TFDM,2021-04-03 19:30:00,KATL,114
AAL1008.ATL.DFW.210403.1312.0051.TFM_TFDM,2021-04-03 19:45:00,KATL,99
AAL1008.ATL.DFW.210403.1312.0051.TFM_TFDM,2021-04-03 20:00:00,KATL,84
AAL1008.ATL.DFW.210403.1312.0051.TFM_TFDM,2021-04-03 20:15:00,KATL,69
AAL1008.ATL.DFW.210403.1312.0051.TFM_TFDM,2021-04-03 20:30:00,KATL,54
```

## Federated learning

As discussed, the federated clients in Phase 2 are airlines. Airline is indicated by the first three letters of the GUFI, e.g., a flight with GUFI `AAL1007.DFW.MEM.211224.0015.0120.TFM` is operated by "AAL" (American Airlines) and a flight with GUFI `UAL2477.MEM.EWR.220710.1915.0167.TFM` is operated by "UAL" is (United Airlines). During training, you will simulate an AAL client that trains using its rows of the federated variables plus any of the public data and transmits model updates to a centralized server. A UAL client and clients for the other airlines will do the same. The centralized server can use different methods to weight and aggregate all of the individual model weights into a single global model. During prediction, each airline will use the final trained model to make pushback predictions for all of the flights it operates. Your federated learning approach will be scored on the same time period and as the test set from Phase 1—the only difference is that the federated variables can only be accessed by the airline that produced them.

### Federated learning with Flower

**We require you to use the [Flower](https://flower.dev/docs/index.html) federated learning framework to train your model.** Flower's design includes a few abstractions structured to match the general concepts in federated learning, which should help you focus on honing your algorithm without needing to reimplement the basics of federated learning from scratch.

You will be implementing your solution to work with the `Client` and `Strategy` APIs. `Client` classes hold the logic that is executed by the federation units that have access to their private data, while the `Server` class holds the logic for coordinating among the clients and aggregating results. For the server, Flower additionally separates the federated learning algorithm logic from the networking logic—the algorithm logic is handled by a `Strategy` class that the server dispatches to. This allows Flower to provide a default `Server` implementation which handles networking that generally does not need to be customized, while the `Strategy` can be swapped out or customized to handle different federated learning approaches. 

Rather than performing federated learning across multiple machines (as it would be in the real-world), you most likely will want to simulate federated learning on a single machine. For this, you can use [Flower's virtual client engine](https://flower.dev/docs/tutorial/Flower-1-Intro-to-FL-PyTorch.html#Using-the-Virtual-Client-Engine).

You may want to check out the [simple linear regression model](https://github.com/drivendataorg/nasa-pushback-federated-learning/blob/main/src/models/linear_regression_benchmark/README.md) in the competition repository for a worked example of federated training and inference on this dataset.

Flower has extensive documentation for how to get started:

- [Introduction to Federated Learning tutorial](https://flower.dev/docs/tutorial/Flower-1-Intro-to-FL-PyTorch.html) from Flower docs, which shows an example of using a `NumPyClient` with PyTorch
- [Configuring Clients how-to guide](https://flower.dev/docs/configuring-clients.html) from Flower docs
- [Implementing Strategies](https://flower.dev/docs/implementing-strategies.html)
- [`Client` abstract base class API reference](https://flower.dev/docs/apiref-flwr.html#flwr-client-client-apiref)
- [`Client` abstract base class source code](https://github.com/adap/flower/blob/v1.0.0/src/py/flwr/client/client.py#L32)
- [`NumPyClient` abstract base class API reference](https://flower.dev/docs/apiref-flwr.html#numpyclient)
- [`NumPyClient` abstract base class source code](https://github.com/adap/flower/blob/v1.0.0/src/py/flwr/client/numpy_client.py#L24)


## Evaluating your model

We want to be able to directly compare your Phase 1 centralized model with your Phase 2 federated model. To make an apples-to-apples comparison, we will want to evaluate both models on the exact same dataset: the held-out test set from Phase 1.

Your development submission will include inference code that should be capable of producing predictions from any dataset with the appropriate format. At the beginning of the Phase 2 evaluation period we will release the test features and you will run your inference code to make predictions.

It is important that during inference a prediction for a flight from an airline does not use private data from other airlines. How you achieve this is up to you. The simplest approach may be to evaluate outside of the Flower framework. A simple example of inference code might look like this:

```python
import pandas as pd
import skops.io as sio

from src.utils import (
    load_private_airline_features,
    load_public_features,
)

airports = ["KATL", "KCLT", ]  # ...
airlines = ["AAL", "AJT", ]  # ...

# load global model
model = sio.load("my_model.skops", trusted=True)

predictions = []
submission_format = pd.read_csv("submission_format.csv")

for airport in airports:
    for airline in airlines:
        # subset submission format to just those flights from this airline
        partial_submission_format = submission_format.loc[
            (submission_format.airport == airport) & (submission_format.airline == airline)
        ]

        # load the public features
        public_features = load_public_features(partial_submission_format.index, airport)

        # load the private features
        private_features = load_private_airline_features(
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
predictions.to_csv("submission.csv")

```

You will be responsible for implementing your own evaluation script. You will need to make sure that:

- Your prediction for a flight does not use private features from another airline. Be sure to check that the only private features you are using are those from the airline whose flight you are predicting.
- You only use data that was available from the previous 30 hours up through the time of estimation when generating predictions (just as in Phase 1). For example, if you are predicting pushback for a flight on December 31, 2022 at 12 pm, you can use any data that was available between 6am on 12/30 and 12 pm on 12/31. All features (except MFS, see below) include a `timestamp` column that indicates the time that that observation was made available to use for filtering.
- For the MFS data, which lacks a timestamp, the same rules apply as in Phase 1. That is, you may use MFS data for the current flight that predictions are being generated for, and you may look up metadata for GUFIs that you already know exist within the valid time window from other timestamped features. You may not use the MFS metadata to look for information about flights for which you do not already have a GUFI from timestamped features, and you may not analyze the entire MFS metadata file to, for example, directly incorporate the distribution of aircraft type, carriers, etc. into your solution.

We will verify that your code meets the requirements—clear documentation will speed up the verification process (and is generally appreciated!).
