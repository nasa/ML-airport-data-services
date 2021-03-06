# Kedro Test

General-purpose code for testing ML pipelines. The pipeline tested 
(model.pkl) is downloaded from the selected mlflow server using the provided
 run id and experiment name. 
Usage of FilterPipeline inclusion/exclusion rules is recommended to ensure all tests pass. See how FilterPipeline is levereged in [airport configuration prediction code](https://github.com/nasa/ML-airport-configuration/blob/main/src/airport_config_prediction/pipelines/data_science/nodes.py)


## Configuration
Configuration parameters need to be properly defined in parameters.yml before running the tests. For example, see [parameters.yml](https://github.com/nasa/ML-airport-configuration/blob/main/conf/base/parameters.yml) in airport configuration prediction model 
### Input Features Format

Tests  for each feature are configured dynamically. The following format is expected:
````
inputs:
  feature_name:
    core: 
    type:
    constraints:
      min: 
      max:     
    encoder: 
    series: 
````
Parameters definition:
- *Core*: True/False
- *type*: categorical/numeric/datetime/bool
- *constraints*: defines the constrains for the selected type
    - *min*: minimum valid value for numeric type
    - *max*: maximum valid value for numeric type
- *encoder*: encoder name if any. None otherwise
- *series*: True/False, used for time series features, e.g. wind_15, wind_30 ...

### Tests Parameters

The following parameters need to be defined in parameters.yml:

````
unit_tests:
  run_id: 
  input_data:
    nodes_to_run:
      pipeline: 
      pipeline_path: 
      nodes_name: 
      nodes_output: 
#    catalog_item:
````
Parameters definition:
- *run_id*: run id in mlflow for model to be tested. The *model.pkl* file for the selected run id will be downloaded and tested. 
- *input_data*: defines how to obtain the input data, *X*, needed to run and test the model. Either the *catalog_item* field or the *nodes_to_run* fields needs to be defined under *input_data*
    - *catalog_item*: catalog entry used as input to the model, e.g. *de_data_set@PKL*
    - *nodes_to_run*
        - *pipeline*: name of the pipeline in which the nodes to run are located
        - *pipeline_path*: path of the selected pipeline
        - *nodes_name*: nodes that will be executed sequentially to obtain *X*
        - *nodes_output*: name of the output generated by the last node in *nodes_name*, i.e. *X*

### MLflow Parameters
The following parameters needs to be defined in parameters.yml:
````
mlflow:
  tracking_uri: 
  experiment_name: 
````
Parameters definition:
- *tracking_uri*: uri of the mlflow server where the pipeline of interest is registered
- *experiment_name*: experiment name where the run storing the pipeline of interest is located


## Execution
### Kedro Test Run
The tests need to be executed from the top level of the project directory used to train the model to be tested. This is necessary  to ensure that the source code needed to obtain *X* is available. To launch the tests run:
````
kedro test path/pipeline.py 
````

being path the appropiate path to execute the kedro_test/pipeline.py file in data_services.

The tests can also be run in PyCharm using the pytest interface, which allows to see progress, tests passed/failed, current test and other info. 
Other IDEs might provide similar functionalities. 

### Mlflow Run

The output of the tests is logged in mlflow. 
An experiment with the experiment name defined in parameters.yml plus "_test" is created if it does not exist.
 A new run is logged containing the configuration parameters for the tests and metrics indicating whether each of the tests passed or failed.


## Test Files Overview

### pipeline.py
Defines the tests to be executed. Additional tests can be added by defining other test functions in pipeline.py  

### conftest.py
Provides the inputs necessary to run the tests in pipeline.py. 
It loads the model to be tested and the data needed to run the model and generate the predictions.
