# Congratulations and Welcome to Phase 2!

You developed one of the best submissions for Phase 1 of the **Pushback to the Future: Predict Pushback Time at US Airports** challenge and are now invited to help NASA explore the use of federated learning to improve pushback predictions!

The goal of Phase 2 is to explore federated learning techniques that enable independent *clients* (also known as parties, organizations, groups) to jointly train a global model without sharing or pooling data. In this case, the clients are **airlines**. Federated learning is a perfect match for the problem of pushback time prediction because airlines collect a lot of information that is relevant to pushback time, but too valuable or sensitive to share, like the number of passengers that have checked in for a flight or the number of bags that have been loaded onto a plane. Federated learning enables airlines to safely contribute the valuable and sensitive data they collect towards a centralized model. In Phase 2, we're asking you to translate your winning Phase 1 model into a model that can be trained in a federated manner. You will work with the same data from Phase 1, except it will be divided into "Public" or "non-federated" variables for which all rows are available to all airlines and "Private" or "federated" variables for which only rows corresponding to a flight that an airline operates are available to that airline.

In the spirit of experimentation, you will have some flexibility in how you translate your Phase 1 winning model during Phase 2. Do some strategies for combining the models from individual clients work better than others? Are there basic changes that your model needs to operate in a federated setting? We want to know! We will release more of the specific requirements as Phase 2 ramps up, but for now get started thinking about what it will take to turn your winning centralized model into an effective federated model! 

The exact variables airlines want to protect by federating are by definition too sensitive to release for a competition. Since we don't have access to the actual airline data that would be federated, we'll simulate the federated scenario by treating a subset of variables as if they are private. All of the federated variables for Phase 2 come from the Phase 1 `mfs` and `standtimes` datasets:

**Private mfs variables**

-   `aircraft_engine_class`
-   `aircraft_type`
-   `major_carrier`
-   `flight_type`

**Private standtimes variables**

-   `departure_stand_actual_time`

Each airline will have access to its federated variables and all airlines' non-federated variables. Airlines will not have access to federated variables from other airlines. These data constraints apply during training and prediction.

Airline is indicated by the first three letters of the GUFI in both dataset, e.g., a flight with GUFI `AAL1007.DFW.MEM.211224.0015.0120.TFM` is operated by "AAL" (American Airlines) and a flight with GUFI `UAL2477.MEM.EWR.220710.1915.0167.TFM` is operated by "UAL" is (United Airlines). During training, you will simulate an AAL client that trains using its rows of the federated variables plus any of the public data and transmits model updates to a centralized server. A UAL client and clients for the other airlines will do the same. The centralized server can use different methods to weight and aggregate all of the individual model weights into a single global model. During prediction, each airline will use the final trained model to make pushback predictions for all of the flights it operates. Your federated learning approach will be scored on the same time period and as the test set from Phase 1—the only difference is that the federated variables can only be accessed by the airline that produced them.

# Timeline

## Development period

The first period of Phase 2 will be a development period during which you will focus on translating your Phase 1 model to a federated model. We will release a training dataset partitioned in to public and private data for you to use. At the end of the development period, you will submit:

- Federated training code, inference code, and model assets as a ZIP archive
- A write-up documenting how you approached federating your model

## Evaluation period

Once you have submitted, we will release the test features to you so you can generate your test predictions. The test data will cover the same time period and flights as the Phase 1 test period (subsetting to only the selected airlines). The features will have the same format as the [train features](#train-features). We will also release a submission format demonstrating the exact rows and columns your submission must include.

You should not make any changes to your code or retrain your model at this point. If any changes are needed to run your code, they should be as minor as possible, and you will need to submit a detailed changelog to explain why they are needed. At the end of the evaluation period, you will submit:

- Test predictions matching the submission format CSV
- If any changes were needed to run your code, the updated code and detailed changelog as a ZIP archive

Phase 2 is intended to involve collaboration with NASA: they want to learn from your federated learning experiments and help you with any questions you have about the data.

# Data

In the development period, we will provide you with the following files:

- [Train features](#train-features) (`KATL.tar`, `KCLT.tar`, etc.)
- [Train labels](#train-labels) (`phase2_train_labels.tar`)


## Limiting predictions to the most active airlines

The vast majority of flights are operated by a small subset of the total number of airlines present in the full dataset. For simplicity and efficiency, you will only be asked to predict for flights from 25 of the most active airlines. These are: AAL, AJT, ASA, ASH, AWI, DAL, EDV, EJA, ENY, FDX, FFT, GJS, GTI, JBU, JIA, NKS, PDT, QXE, RPA, SKW, SWA, SWQ, TPA, UAL, UPS.

Note that not all airlines will have flights out of all airports. If an airline does not operate any flights, the airline-specific private feature file will not exist. Also, note that although you will only be asked to make predictions for these airlines, you will have access to public data from all airlines (and of course, none of the private data from airlines that are not among these 25).


## Train features

We will provide the features from the same 10 airports as tar archives of bzip2-compressed CSVs, one archive per airport. The features used in Phase 2 will be the [same data used in Phase 1](https://www.drivendata.org/competitions/149/competition-nasa-airport-pushback/page/677/#features), with a few key differences:

- All public data (all airline data, available to all airlines) are stored in a `public` root directory.
- Public versions of the `mfs` and `standtimes` files do not include the private columns.
- All private data are stored in a `private` root directory. Each airline has its own version of the `mfs` and `standtimes` data containing the _public and private_ columns.

Here is an example of the files within one airport archive, shown for KATL:

```
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

private
└── KATL
    ├── KATL_AAL_mfs.csv.bz2  # contains rows for AAL airline, all columns
    ├── KATL_AAL_standtimes.csv.bz2  # contains rows for ALL, all columns
    ├── ...
    ├── KATL_UPS_mfs.csv.bz2  # contains rows for UPS airline, all columns
    └── KATL_UPS_standtimes.csv.bz2
```

When you extract all of the tar archives, you will end up with the data in the following directory structure:

```
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

private
├── KATL
│   ├── KATL_AAL_mfs.csv.bz2  # contains rows for AAL airline, all columns
│   ├── KATL_AAL_standtimes.csv.bz2  # contains rows for ALL, all columns
│   ├── ...
│   ├── KATL_UPS_mfs.csv.bz2  # contains rows for UPS airline, all columns
│   └── KATL_UPS_standtimes.csv.bz2
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

## Train labels

We will also provide training labels (`phase2_train_labels.tar`) to assist in the development of your model. You are not required to use these during training, but it could be a useful starting point. The Phase 2 training labels are identical to those [provided in the Prescreened Arena](https://www.drivendata.org/competitions/182/competition-nasa-airport-pushback-prescreened/data/), but subset to only the [selected airlines](#limiting-predictions-to-the-most-active-airlines). The `phase2_train_labels.tar` archive contains the following files:

```
- phase2_train_labels_KATL.csv.bz2
- phase2_train_labels_KCLT.csv.bz2
- phase2_train_labels_KDEN.csv.bz2
- phase2_train_labels_KDFW.csv.bz2
- phase2_train_labels_KJFK.csv.bz2
- phase2_train_labels_KMEM.csv.bz2
- phase2_train_labels_KMIA.csv.bz2
- phase2_train_labels_KORD.csv.bz2
- phase2_train_labels_KPHX.csv.bz2
- phase2_train_labels_KSEA.csv.bz2
```

where each file contains the actual minutes to pushback for a GUFI at a particular time. Here are the first five rows of `phase2_train_labels_KATL.csv.bz2`:

```csv
gufi,timestamp,airport,minutes_until_pushback
AAL1008.ATL.DFW.210403.1312.0051.TFM_TFDM,2021-04-03 19:30:00,KATL,114
AAL1008.ATL.DFW.210403.1312.0051.TFM_TFDM,2021-04-03 19:45:00,KATL,99
AAL1008.ATL.DFW.210403.1312.0051.TFM_TFDM,2021-04-03 20:00:00,KATL,84
AAL1008.ATL.DFW.210403.1312.0051.TFM_TFDM,2021-04-03 20:15:00,KATL,69
AAL1008.ATL.DFW.210403.1312.0051.TFM_TFDM,2021-04-03 20:30:00,KATL,54
```

# Federated learning

## Training your model with Flower

We require you to use the [Flower](https://flower.dev/docs/index.html) federated learning framework to train your model. Flower has extensive documentation for how to get started:

- [Introduction to Federated Learning tutorial](https://flower.dev/docs/tutorial/Flower-1-Intro-to-FL-PyTorch.html) from Flower docs, which shows an example of using a `NumPyClient` with PyTorch
- [Configuring Clients how-to guide](https://flower.dev/docs/configuring-clients.html) from Flower docs
- [Implementing Strategies](https://flower.dev/docs/implementing-strategies.html)
- [`Client` abstract base class API reference](https://flower.dev/docs/apiref-flwr.html#flwr-client-client-apiref)
- [`Client` abstract base class source code](https://github.com/adap/flower/blob/v1.0.0/src/py/flwr/client/client.py#L32)
- [`NumPyClient` abstract base class API reference](https://flower.dev/docs/apiref-flwr.html#numpyclient)
- [`NumPyClient` abstract base class source code](https://github.com/adap/flower/blob/v1.0.0/src/py/flwr/client/numpy_client.py#L24)


## Flower's Client, Server, and Strategy Abstractions

Flower's design includes a few abstractions structured to match the general concepts in federated learning. You will be implementing your solution to work with the `Client` and `Strategy` APIs. 

`Client` classes hold the logic that is executed by the federation units that have access to their private data, while the `Server` class holds the logic for coordinating among the clients and aggregating results. For the server, Flower additionally separates the federated learning algorithm logic from the networking logic—the algorithm logic is handled by a `Strategy` class that the server dispatches to. This allows Flower to provide a default `Server` implementation which handles networking that generally does not need to be customized, while the `Strategy` can be swapped out or customized to handle different federated learning approaches. 

Rather than performing federated learning across multiple machines (as it would be in the real-world), you most likely will want to simulate federated learning on a single machine. For this, you can use [Flower's virtual client engine](https://flower.dev/docs/tutorial/Flower-1-Intro-to-FL-PyTorch.html#Using-the-Virtual-Client-Engine).

See the [simple linear regression model](https://github.com/drivendataorg/nasa-pushback-federated-learning/blob/main/src/models/linear_regression_benchmark/README.md) included in this repository for a worked example of federated training.

## Evaluating your model

We want to be able to directly compare your Phase 1 centralized model with your Phase 2 federated model. To make an apples-to-apples comparison, we will want to evaluate both models on the exact same dataset: the held-out test set from Phase 1.

What is important is that for your federated model, a prediction for a flight from an airline does not use private data from other airlines. How you achieve this is up to you. The simplest approach may be to evaluate outside of the Flower framework. A simple example of inference code might look like this:

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
- You only use data that was available from the previous 30 hours up through the time of estimation when generating predictions (just as in Phase 1). For example, if you are predicting pushback for a flight on December 31, 2022 at 12 pm, you can use any data that was available between 6am on 12/30 and 12 pm on 12/31. Most feature data includes a `timestamp` column that indicates the time that that observation was made available to use for filtering.

We will verify that your code meets the requirements—clear documentation will speed up the verification process (and is generally appreciated!).

# Forum

We will host a discussion forum for all of the Phase 2 participants and NASA team. This is a great place to post questions in comments for NASA or other participants to weigh in on. We already added any of the team members we know; please email [info@drivendata.org](mailto:info@drivendata.org?subject=Request%20to%20add%20user%20to%20pushback%20phase%202%20forum) to add DrivenData users.

# Office hours

NASA will hold office hours every other week for open discussion about the trials and tribulations of federated learning with this dataset. Office hours are scheduled for:

- Week of June 5th
- Week of June 19th
- Week of July 3rd

We'll announce the exact times on the forum. You are encouraged to reach out to NASA on the forum and in office hours as you develop your solutions. They are looking forward to hearing what you find out!
