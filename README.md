![ATD2 logo](images/ATD2_logo_animation.gif)
## Data Services

The ML-airport-data-services software is developed to provide common code used throughout the ML-airport suite of software. The software is built in python and leverages open-source libraries kedro, scikitlearn, MLFlow, and others. The software provides useful functions for development of pipelines including data query and save, data engineering, and data science.

## ML Airport Surface Model Background Information

ML-airport-data-services provides general-purpose code to several ML models that are tied together by the Airport Surface Model Orchestrator shown below.

![Airport Surface Model Orchestrator Diagram](images/orchestrator.png) 

The Airport Surface Model Orchestrator runs at fixed intervals and is driven from real-time Fused FAA System Wide Information Management (SWIM) data feeds and also pulls in additional D-ATIS airport configuration data and weather data. The data input manager within the Orchestrator prepares data for each ML service and data is exchanged between the Orchestrator and the ML services via API interfaces.

The input data sources for the individual ML prediction services are shown in the below diagram which also illustrates dependencies between the individual ML services.

![ML Model Diagram](images/ml_models.png)

The ML Airport Surface Model forms the building blocks of a cloud based predictive engine that alerts flight operators to pre-departure Trajectory Option Set (TOS) reroute opportunities within the terminal airspace. The ML Airport Surface Model was designed to be a scalable replacement for the capabilities provided by NASA's Surface Trajectory Based Operations (STBO) subsystem, which is a component of the fielded ATD2 Phase 3 System in the North Texas Metroplex. The STBO subsystem relies heavily upon detailed adaptation, which defines the physical constraints and encodes Subject Matter Expert knowledge within decision trees, and creates a costly bottleneck to scaling the pre-departure TOS digital reroute capability across the National Airspace System.


Data Services is part of a suite of softwares designed to model the airport surface:
- [ML-airport Airport Configuration Model](https://github.com/nasa/ML-airport-configuration)
- [ML-airport Arrival Runway Model](https://github.com/nasa/ML-airport-arrival-runway)
- [ML-airport Departure Runway Model](https://github.com/nasa/ML-airport-departure-runway)
- [ML-airport Taxi-In Model](https://github.com/nasa/ML-airport-taxi-in)
- [ML-airport Taxi-Out Model](https://github.com/nasa/ML-airport-taxi-out)
- [ML-airport Estimated-On-Time Model](https://github.com/nasa/ML-airport-estimated-ON)
- [ML-airport Data Services](https://github.com/nasa/ML-airport-data-services)

## Steps to start using this project

This repository does not contain a standalone project.
Code in this repository has been packaged to facilitate usage in environments for development and deployment of predictive models.
To use functionality in this repository in another project, it can be installed in the project's anaconda environment via a pip install of the packaged code available in the appropriate git repository.

Corresponding lines can also be added to the anaconda environment specification file.
For example, see the `conda.yaml` file in the `ML-airport-taxi-in` repository.

This repository also contains some general [MLflow](https://mlflow.org/docs/latest/index.html) and [Kedro](https://kedro.readthedocs.io) functionality.

## Description of some repository components

### Kedro extensions

The `kedro_extensions/io/sqlfile_dataset.py` file contains code defining the `SQLQueryFileDataSet` and `SQLQueryFileChunkedDataSet` data set types for use in Kedro data catalogs.
These enable users to specify a SQL file for executing parameterized queries and for breaking queries that cover long time periods in to time "chunks" that are less burdensome on the database responding to the queries.

### MLflow utilities

The `mlflow_utils/mlflow_utils.py` file contains some utility functions for working with MLflow and the MLflow server.
For example, some of these use the MLflow API to get the most recently registered model of a certain type.

### Custom scikit-learn ColumnTransformers or Pipelines

Several files contain classes that extend the scikit-learn base `Pipeline` or `BaseEstimator` or `TransformerMixin` classes and can be used when creating scikit-learn `ColumnTransformer` or `Pipeline` objects that incorporate fairly sophisticated feature calculations along with prediction models.

#### FilterPipeline

The `FilterPipeline.py` file implements the `FilterPipeline` class, a noteworthy child class of `sklearn.pipeline.Pipeline`.
It is a wrapper class that applies consistent filtering criteria for *rows* when doing model fitting and prediction.

### Best SWIM ETA

The `build_swim_eta` function in `swim_based_eta.py` contains rules to select the "best" ETA from candidate ETAs available in SWIM (i.e., TFMS ETA, TBFM ETA, TBFM STA).

### Time series utilities

The `time_series_sampling.py` file contains some functions that are useful for time series data.
For example, the `_count_in_ts` function computes a time series representation of expected counts, given frequently-updated predictions of some future event.

### Data Services Utilities

The `data_services_utils.py` file contains some functions that are useful for lining up the data to match what the live nas-fuser/nas-model system does. For example, there is an `add_flight_timeout` function that times out a flight after not receiving a final time after a pre-configured amount of time.

### Compute Surface Counts

The `compute_surface_counts.py` file computes arrival, departure, and total counts of flights on the surface of the airport at the time of landing for each arrival in the dataset.

### Gate Occupied at Landing

The `gate_occupied_at_landing_proxy.py` file computes whether or not the assigned gate is occupied at the time of landing.


## Copyright and Notices

The ML-airport-data-services code is released under the [NASA Open Source Agreement Version 1.3 license](license.pdf)

The “ATD-2 Machine Learning Airport Surface Model: Data Services” software also makes use of the following 3rd party Open Source software:
- scp - under GNU Library or Lesser General Public License (LGPL) - Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
- paramiko - under GNU Library or Lesser General Public License (LGPL) - Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>

Software provided by GNU is released under the [GNU LESSER GENERAL PUBLIC LICENSE](license_GNU.pdf)


## Notices

Copyright © 2021 United States Government as represented by the
Administrator of the National Aeronautics and Space Administration. All
Rights Reserved.

### Disclaimers

No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY
WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY,
INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE
WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM
INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE,
OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE
SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN
ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS,
RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS
RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY
DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE,
IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity: RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS
AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND
SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF
THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES,
EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM
PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT
SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES
GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR
RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR
ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS
AGREEMENT.
